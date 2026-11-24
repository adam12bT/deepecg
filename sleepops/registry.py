"""Validation gates and an immutable, checksum-verified model registry."""

from math import isfinite
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from sleepops.files import read_json, sha256, utc_now, write_json


CHECKPOINT = "best_checkpoint.pt"
RELEASE_FILES = (CHECKPOINT, "dataset.json", "validation_metrics.json", "drift_reference.json", "metadata.json")
VERSION_PATTERN = re.compile(r"[a-f0-9]{32}")


def check_quality(report, policy):
    if report["split"] != "validation":
        raise ValueError("Promotion requires metrics from the validation split.")
    state = report["state"]
    forecast = report["forecast"]
    model = state["shared_model"] | forecast["shared_model"]
    baseline = state["logistic_regression"] | forecast["training_class_distribution"]
    metrics = ("balanced_accuracy", "macro_f1", "multiclass_log_loss", "within_hour_brier")
    for name in metrics:
        if not all(isfinite(value) for value in (model[name], baseline[name])):
            raise ValueError(f"Nonfinite validation metric {name}.")
    checks = {
        "minimum_validation_subjects": report["subjects"] >= policy["min_subjects"],
        "balanced_accuracy": model["balanced_accuracy"] >= policy["min_balanced_accuracy"],
        "macro_f1": model["macro_f1"] >= policy["min_macro_f1"],
        "multiclass_log_loss": model["multiclass_log_loss"] <= policy["max_forecast_log_loss"],
        "within_hour_brier": model["within_hour_brier"] <= policy["max_forecast_brier"],
        "causality": 0 <= report["causality_max_delta"] <= policy["max_causality_delta"],
        "detection_baseline": all(model[m] >= baseline[m] for m in metrics[:2]),
        "forecast_baseline": all(model[m] <= baseline[m] for m in metrics[2:]),
    }
    return {"passed": all(checks.values()), "checks": checks, "policy": policy}


def validate_bundle(path):
    path = Path(path)
    metadata = read_json(path / "metadata.json")
    identifier = metadata["run_id"]
    if metadata["version"] != 1 or not VERSION_PATTERN.fullmatch(identifier):
        raise ValueError("Invalid model run identifier.")
    expected = set(RELEASE_FILES) - {"metadata.json"}
    hashes = metadata["file_sha256"]
    if set(hashes) != expected or any(sha256(path / name) != hashes[name] for name in expected):
        raise ValueError("The release bundle has an invalid file checksum.")
    data = read_json(path / "dataset.json")
    validation = read_json(path / "validation_metrics.json")
    reference = read_json(path / "drift_reference.json")
    digests = {item["archive_sha256"] for item in (metadata, data, validation, reference)}
    if len(digests) != 1 or validation["subjects"] != data["splits"]["validation"]["subjects"]:
        raise ValueError("Dataset provenance differs across release files.")
    return metadata, validation, reference


def load_bundle(path):
    from eeg_model import load_checkpoint

    metadata, validation, reference = validate_bundle(path)
    model, mean, scale, checkpoint = load_checkpoint(Path(path) / CHECKPOINT)
    threshold = checkpoint["state_threshold"]
    if (checkpoint["archive_sha256"] != metadata["archive_sha256"]
            or checkpoint["best_epoch"] != validation["selected_epoch"] or not 0 < threshold < 1):
        raise ValueError("Checkpoint provenance or sleep threshold disagrees with the release.")
    return model, mean, scale, checkpoint, metadata, reference


def release_path(registry, version):
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Invalid release ID.")
    path = Path(registry) / "releases" / version
    if not path.is_dir():
        raise FileNotFoundError(f"Unknown promoted release {version}.")
    return path


def activate(registry, version):
    path = release_path(registry, version)
    metadata, _, _ = validate_bundle(path)
    if metadata["run_id"] != version or not read_json(path / "promotion.json")["passed"]:
        raise ValueError("Release does not have a passing promotion decision.")
    write_json(Path(registry) / "champion.json", {"release_id": version, "activated_at": utc_now()})
    return path


def promote(registry, run, policy):
    run = Path(run)
    metadata, validation, _ = validate_bundle(run)
    decision = check_quality(validation, policy)
    if not decision["passed"]:
        failed = [key for key, passed in decision["checks"].items() if not passed]
        raise ValueError(f"Promotion rejected by validation gates: {', '.join(failed)}")
    load_bundle(run)
    releases = Path(registry) / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    version = metadata["run_id"]
    destination = releases / version
    with TemporaryDirectory(dir=releases, prefix=".staging-") as temp:
        staging = Path(temp) / "release"
        staging.mkdir()
        for name in RELEASE_FILES:
            shutil.copyfile(run / name, staging / name)
        write_json(staging / "promotion.json", {**decision, "promoted_at": utc_now()})
        validate_bundle(staging)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir() or read_json(destination / "metadata.json") != metadata:
                raise
    return activate(registry, version)


def champion(registry):
    version = read_json(Path(registry) / "champion.json")["release_id"]
    path = release_path(registry, version)
    if not read_json(path / "promotion.json")["passed"]:
        raise ValueError("The active release lacks a successful promotion decision.")
    return path
