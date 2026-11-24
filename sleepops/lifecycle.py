"""Fit, track, and evaluate a candidate against a pinned Sleep-EDF archive."""

from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import platform
import shutil
import tomllib

import numpy as np

from eeg_data import normalize
from eeg_model import load_checkpoint, verify_causality
from sleepops.dataset import load_dataset
from sleepops.files import read_json, sha256, utc_now, write_json
from sleepops.monitor import build_reference
from sleepops.registry import CHECKPOINT, RELEASE_FILES, VERSION_PATTERN, load_bundle, promote, validate_bundle
from sleepops.training import TrainingConfig, evaluate, fit, init_training


def configure_tracking(runtime, experiment=None):
    import mlflow

    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    custom_uri = os.getenv("MLFLOW_TRACKING_URI")
    uri = custom_uri or f"sqlite:///{(runtime / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if experiment is not None:
        if not custom_uri and mlflow.get_experiment_by_name(experiment) is None:
            client = mlflow.MlflowClient()
            try:
                client.create_experiment(experiment, artifact_location=(runtime / "mlartifacts").as_uri())
            except mlflow.exceptions.MlflowException:
                if mlflow.get_experiment_by_name(experiment) is None:
                    raise
        mlflow.set_experiment(experiment)
    return mlflow


def source_fingerprint():
    root = Path(__file__).resolve().parent.parent
    paths = [root / name for name in (
        "eeg_data.py", "eeg_model.py", "sleepops/dataset.py", "sleepops/monitor.py",
        "sleepops/training.py", "sleepops/lifecycle.py",
    )]
    return hashlib.sha256("\n".join(f"{path.relative_to(root).as_posix()} {sha256(path)}" for path in paths).encode()).hexdigest()


def log_epoch(mlflow, row):
    mlflow.log_metrics({key: value for key, value in row.items() if key != "epoch"}, step=row["epoch"])


def score_metrics(report):
    return {f"{report['split']}.{task}.{method}.{metric}": value
            for task in ("state", "forecast") for method, values in report[task].items()
            for metric, value in values.items() if isinstance(value, (int, float))}


def store_candidate(args, *, imported=False):
    import sklearn
    import torch

    with args.config.open("rb") as source:
        settings = tomllib.load(source)
    config = TrainingConfig(**settings["training"])
    if not imported and args.epochs is not None:
        config = replace(config, max_epochs=args.epochs)
    if (min(config.max_epochs, config.patience, config.batch_size) < 1
            or not 0 <= config.dropout < 1 or config.learning_rate <= 0 or config.weight_decay < 0):
        raise ValueError("Invalid training configuration.")
    data, manifest = load_dataset(args.archive, args.lock)
    archive_hash = manifest["archive_sha256"]
    source_hash = source_fingerprint()
    origin = "imported_cs495_checkpoint" if imported else "sleepops_training"
    mlflow = configure_tracking(args.runtime, settings["tracking"]["experiment"])
    with mlflow.start_run(run_name="cs495-import" if imported else "cnn-transformer", tags={
        "dataset_sha256": archive_hash, "source_sha256": source_hash, "origin": origin,
    }) as run:
        run_id = run.info.run_id
        output = args.runtime / "runs" / run_id
        output.mkdir(parents=True)
        checkpoint = output / CHECKPOINT
        if imported:
            original = read_json(args.artifacts / "training_log.json")
            if original["archive_sha256"] != archive_hash:
                raise ValueError("Imported training log belongs to a different archive.")
            shutil.copyfile(args.artifacts / CHECKPOINT, checkpoint)
            write_json(output / "training_log.json", original)
            device = init_training(original["config"]["seed"], args.device)
            training_log = original
            for row in original["history"]:
                log_epoch(mlflow, row)
        else:
            file, device, training_log = fit(data, args.archive, output, config, args.device,
                                             on_epoch=lambda row: log_epoch(mlflow, row))
            assert file == checkpoint

        model, mean, std, payload = load_checkpoint(checkpoint, device)
        if (payload["archive_sha256"] != archive_hash or payload["best_epoch"] != training_log["best_epoch"]
                or payload["validation_score"] != training_log["best_validation_mean"]):
            raise ValueError("Checkpoint, dataset, and training log disagree.")
        for_split = data.arrays["forecast_split"] == "validation"
        samples = normalize(data.arrays["X_forecast"][for_split][:3], mean, std)
        causal_delta = verify_causality(model, torch.from_numpy(samples).to(device))
        validation = evaluate(data, checkpoint, output, device, split="validation")
        validation["causality_max_delta"] = causal_delta
        write_json(output / "validation_metrics.json", validation)
        write_json(output / "dataset.json", manifest)
        write_json(output / "drift_reference.json", build_reference(data, archive_hash))
        metadata = {"version": 1, "run_id": run_id, "origin": origin, "created_at": utc_now(),
                    "archive_sha256": archive_hash, "source_sha256": source_hash,
                    "file_sha256": {name: sha256(output / name) for name in RELEASE_FILES if name != "metadata.json"}}
        write_json(output / "metadata.json", metadata)
        validate_bundle(output)
        mlflow.log_params({**(training_log["config"] if imported else asdict(config)),
                           "selected_epoch": payload["best_epoch"], "training_hardware": training_log["hardware"],
                           "training_torch_version": training_log["torch_version"],
                           "evaluation_device": str(device), "python": platform.python_version(),
                           "numpy": np.__version__, "sklearn": sklearn.__version__})
        mlflow.log_metrics({**score_metrics(validation), "causality_max_delta": causal_delta,
                            "selected_validation_loss": payload["validation_score"]})
        mlflow.log_artifacts(str(output), artifact_path="run")

    print(f"Run ID {run_id}\nRun files {output}", flush=True)
    if args.promote:
        promote_run(args, run_id)
    return run_id


def find_run(runtime, run_id):
    if not VERSION_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid run ID.")
    return Path(runtime) / "runs" / run_id


def promote_run(args, run_id):
    output = find_run(args.runtime, run_id)
    if read_json(output / "dataset.json") != read_json(args.lock):
        raise ValueError("Candidate was trained on a different locked dataset.")
    path = promote(args.registry, output, read_json(args.policy))
    print(f"Active model release {path.name}")


def evaluate_release(args):
    output = find_run(args.runtime, args.run)
    _, _, _, payload, metadata, _ = load_bundle(output)
    data, manifest = load_dataset(args.archive, args.lock)
    if read_json(output / "dataset.json") != manifest:
        raise ValueError("Evaluation dataset differs from the model's training archive.")
    device = init_training(495, args.device)
    report = evaluate(data, output / CHECKPOINT, output, device, split="test")
    report["run_id"] = metadata["run_id"]
    report["state_threshold"] = payload["state_threshold"]
    write_json(output / "test_metrics.json", report)
    mlflow = configure_tracking(args.runtime)
    client = mlflow.MlflowClient()
    for name, value in score_metrics(report).items():
        client.log_metric(args.run, name, value)
    for name in ("test_metrics.json", "heldout_forecast.png"):
        client.log_artifact(args.run, str(output / name), artifact_path="run")
    print(f"Held-out test metrics {output / 'test_metrics.json'}")
