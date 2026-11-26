"""Gradio controls for training a SleepOps candidate on a private Space."""

import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock, Thread

import gradio as gr

from deploy.huggingface import cloud_train
from deploy.huggingface.release import ARCHIVE, checked_repo_id
from deploy.training_space.datasets import DatasetStore
from deploy.training_space.models import READ_ERRORS, ModelCatalog, check_checkpoint, compare, details, model_rows
from sleepops.files import read_json, sha256


class Trainer:
    def __init__(self, dvc_store, output, bucket, versions=None):
        self.output = output
        self.bucket = checked_repo_id(bucket)
        self.datasets = DatasetStore(dvc_store, versions if versions is not None else Path(
            os.getenv("SLEEPOPS_DATASETS_VOLUME", "/sleepops-datasets")), self.bucket)
        self.lock = Lock()
        self.running = False
        self.phase = "idle"
        self.result = None
        self.dataset = "base"
        self.message = "Ready to train."

    def status(self):
        with self.lock:
            return self.message

    def snapshot(self):
        with self.lock:
            return {"phase": self.phase, "running": self.running, "result": self.result, "dataset": self.dataset}

    def start(self, epochs, evaluate_test, dataset="base"):
        if epochs is None or not 1 <= epochs <= 20 or int(epochs) != epochs:
            raise gr.Error("Choose a whole number from 1 to 20.")
        try:
            archive_object, dataset_lock = self.datasets.location(dataset)
            if not archive_object.is_file() or not self.output.is_dir():
                raise FileNotFoundError
        except (OSError, ValueError, KeyError):
            raise gr.Error("The selected dataset or private candidate storage is unavailable.") from None

        with self.lock:
            if self.running:
                return self.message
            self.running = True
            self.phase = "restoring"
            self.result = None
            self.dataset = dataset
            self.message = "Restoring the DVC-pinned training archive…"
            Thread(target=self.train, args=(archive_object, dataset_lock, int(epochs), evaluate_test), daemon=True).start()
            return self.message

    def set_message(self, phase, message, result=None):
        with self.lock:
            self.message = message
            self.phase = phase
            self.result = result

    def train(self, archive_object, dataset_lock, epochs, evaluate_test):
        try:
            with TemporaryDirectory(prefix="sleepops-training-") as temp:
                workspace = Path(temp)
                archive = workspace / ARCHIVE
                lock = workspace / "dataset.json"
                shutil.copyfile(archive_object, archive)
                shutil.copyfile(dataset_lock, lock)
                if sha256(archive) != read_json(lock)["archive_sha256"]:
                    raise ValueError("The archive checksum differs from the selected dataset lock.")
                self.set_message("fitting", "Training on CPU and selecting a checkpoint by validation loss…")
                command = [sys.executable, "-m", "deploy.huggingface.cloud_train", "--runtime", str(workspace),
                           "fit", "--archive", str(archive), "--lock", str(lock), "--epochs", str(epochs)]
                if evaluate_test:
                    command.append("--evaluate-test")
                subprocess.run(command, check=True)
                [run] = (workspace / "runs").iterdir()
                run_id = run.name
                log = read_json(run / "training_log.json")
                gates = read_json(run / "cloud_run.json")["validation_gates"]
                self.set_message("saving", "Saving and verifying the private candidate…")
                candidate = self.output / run_id
                candidate.mkdir()
                for name in cloud_train.candidate_files(evaluate_test):
                    shutil.copyfile(run / name, candidate / name)
                    if sha256(run / name) != sha256(candidate / name):
                        raise ValueError(f"Stored file failed checksum verification: {name}")
                result = "passed" if gates["passed"] else "did not pass"
                url = f"https://huggingface.co/buckets/{self.bucket}/tree/training-runs/{run_id}"
            completed = len(log["history"])
            self.set_message("complete", f"Training completed after **{completed} {'epoch' if completed == 1 else 'epochs'}**. "
                             f"Selected epoch **{log['best_epoch']}** with validation loss "
                             f"**{log['best_validation_mean']:.5f}**. Validation gates {result}.\n\n"
                             f"[Candidate checkpoint and reports]({url})", {
                                 "id": run_id, "epochs": completed, "epoch": log["best_epoch"],
                                 "loss": log["best_validation_mean"], "gatesPassed": gates["passed"], "url": url,
                             })
        except Exception:
            logging.exception("SleepOps training failed")
            self.set_message("failed", "Training could not finish. See the Space logs for details.")
        finally:
            with self.lock:
                self.running = False


def default_catalog(trainer):
    return ModelCatalog(trainer.output, Path(os.getenv("SLEEPOPS_REGISTRY_VOLUME", "/sleepops-registry")),
                        Path(os.getenv("SLEEPOPS_REPORTS_VOLUME", "/sleepops-reports")), trainer.bucket,
                        os.getenv("SLEEPOPS_PREDICTION_SPACE", f"{trainer.bucket.split('/')[0]}/assistance"))


def make_app(trainer, catalog=None):
    catalog = catalog if catalog is not None else default_catalog(trainer)

    def refresh(selected, a, b):
        versions = catalog.versions()
        choices = [(version.label, version.key) for version in versions]
        keys = [value for _, value in choices]
        first = keys[0] if keys else None
        second = next((version.key for version in versions if version.key != first and version.gates["passed"]),
                      keys[1] if len(keys) > 1 else first)
        selections = (selected if selected in keys else first, a if a in keys else first, b if b in keys else second)
        visible = [gr.update(choices=choices, value=value) for value in selections]
        return model_rows(versions), *visible

    def inspect(key):
        if key is None:
            return "No model selected.", []
        try:
            return details(catalog.version(key))
        except READ_ERRORS:
            return "The selected model's report could not be read. Refresh the catalog.", []

    def check(key):
        if key is None:
            return "Select a model first."
        try:
            return check_checkpoint(catalog.version(key))
        except (*READ_ERRORS, RuntimeError, AssertionError):
            logging.exception("SleepOps checkpoint check failed")
            return "Checkpoint check failed. See the Space logs for details."

    def compare_selection(a, b, split):
        if a is None or b is None:
            return "Choose two model versions to compare.", []
        try:
            return compare(catalog.version(a), catalog.version(b), split, trainer.datasets.anchor)
        except READ_ERRORS:
            return "A model report could not be read. Refresh the catalog.", []

    runs_url = f"https://huggingface.co/buckets/{trainer.bucket}/tree/training-runs"
    with gr.Blocks(title="SleepOps Admin") as app:
        with gr.Row(equal_height=False):
            gr.Markdown("# SleepOps Admin\nReview model versions, compare evaluation results, and start training.",
                        elem_classes=["admin-heading"])
            refresh_button = gr.Button("Refresh models", size="sm", scale=0)
        with gr.Tabs():
            with gr.Tab("Models"):
                model_table = gr.Dataframe(value=[], interactive=False, label="Model registry", headers=[
                    "Release ID", "Role", "Selected epoch", "Validation gates", "Stage macro F1", "Balanced accuracy",
                    "Forecast log loss", "Held-out report",
                ])
                with gr.Row():
                    selected = gr.Dropdown(label="Inspect model", scale=3)
                    verify = gr.Button("Check checkpoint", scale=1)
                detail_status = gr.Markdown()
                detail_table = gr.Dataframe(value=[], interactive=False, label="Model metrics", headers=[
                    "Metric", "Validation", "Held-out test",
                ])
                check_status = gr.Markdown()
            with gr.Tab("Compare models"):
                with gr.Row():
                    model_a = gr.Dropdown(label="Model A")
                    model_b = gr.Dropdown(label="Model B")
                    split = gr.Radio([("Validation", "validation"), ("Held-out test", "test")], value="validation",
                                     label="Evaluation split")
                comparison_status = gr.Markdown()
                comparison = gr.Dataframe(value=[], interactive=False, label="Comparison", headers=[
                    "Metric", "Model A", "Model B", "Difference", "Better result",
                ])
            with gr.Tab("Training"):
                with gr.Row():
                    epochs = gr.Number(value=20, minimum=1, maximum=20, precision=0, label="Maximum epochs")
                    evaluate = gr.Checkbox(value=False, label="Evaluate on held-out subjects")
                start = gr.Button("Start training", variant="primary")
                status = gr.Markdown(trainer.status)
                gr.Markdown(f"[Browse private training runs]({runs_url})")

        selectors = [selected, model_a, model_b]
        refresh_button.click(refresh, inputs=selectors, outputs=[model_table, *selectors], api_name="models")
        app.load(refresh, inputs=selectors, outputs=[model_table, *selectors], api_visibility="private")
        selected.change(inspect, inputs=selected, outputs=[detail_status, detail_table], api_name="model_details")
        verify.click(check, inputs=selected, outputs=check_status, api_name="check_model")
        gr.on([model_a.change, model_b.change, split.change], compare_selection, inputs=[model_a, model_b, split],
              outputs=[comparison_status, comparison], api_name="compare_models")
        start.click(trainer.start, inputs=[epochs, evaluate], outputs=status, api_name="train")
        gr.Timer(3).tick(trainer.status, outputs=status, api_name="training_status", queue=False)
    return app


def default_trainer():
    return Trainer(Path(os.getenv("SLEEPOPS_DVC_VOLUME", "/sleepops-dvc")),
                   Path(os.getenv("SLEEPOPS_CANDIDATES_VOLUME", "/sleepops-candidates")),
                   os.environ["SLEEPOPS_HF_BUCKET"])


def main():
    make_app(default_trainer()).launch(share=False, theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
                             css=".gradio-container { max-width: 1360px !important; margin: auto; }"
                                 ".admin-heading { border-left: 3px solid var(--color-accent); padding-left: 1rem; }")


if __name__ == "__main__":
    main()
