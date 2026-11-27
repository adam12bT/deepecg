"""Serve the React admin and the existing Gradio clients from the same Space."""

import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr
import numpy as np
from pydantic import BaseModel, Field

from deploy.training_space.app import default_catalog, default_trainer, make_app
from deploy.training_space.datasets import DatasetTooLarge, MAX_UPLOAD, chart_metadata
from deploy.training_space.models import READ_ERRORS, check_checkpoint, comparison, details, metric
from deploy.training_space.promotion import PromotionQueue
from eeg_data import CLASS_LABELS, EPOCH_SEC, FEATURE_NAMES, HISTORY, STAGE_NAMES
from eeg_model import SleepStageForecastPipeline, predict_window, predict_with_stages
from sleepops.files import read_json, sha256
from sleepops.registry import load_bundle


class TrainingRequest(BaseModel):
    epochs: int = Field(ge=1, le=20, strict=True)
    evaluateTest: bool = Field(default=False, strict=True)
    dataset: str = Field(default="base", pattern=r"^(base|[0-9a-f]{64})$")


class DeploymentRequest(BaseModel):
    candidate: str = Field(pattern=r"^[0-9a-f]{32}$")


def model_summary(version):
    return {
        "key": version.key,
        "id": version.metadata["run_id"],
        "label": version.label,
        "deployed": version.key == "deployed",
        "createdAt": version.metadata["created_at"],
        "epoch": version.validation["selected_epoch"],
        "accuracy": metric(version.validation, "state", "balanced_accuracy"),
        "stageMacroF1": metric(version.validation, "stages", "macro_f1"),
        "logLoss": metric(version.validation, "forecast", "multiclass_log_loss"),
        "gatesPassed": version.gates["passed"],
        "hasTest": version.test is not None,
        "url": version.url,
    }


def create_app(trainer, catalog, dist):
    app = FastAPI(title="SleepOps Admin", docs_url=None, redoc_url=None, openapi_url=None)
    deployment = PromotionQueue(catalog, trainer.datasets, trainer.output)

    def version(key):
        try:
            return catalog.version(key)
        except READ_ERRORS:
            raise HTTPException(404, "This model or its reports are unavailable. Refresh the registry.") from None

    @app.get("/api/admin/catalog")
    def get_catalog():
        return {"models": [model_summary(item) for item in catalog.versions()],
                "bucket": f"https://huggingface.co/buckets/{trainer.bucket}/tree/training-runs",
                "dvc": f"https://huggingface.co/buckets/{trainer.bucket}/tree/dvc-store",
                "prediction": f"https://huggingface.co/spaces/{catalog.prediction_space}"}

    @app.get("/api/admin/models/{key}")
    def get_model(key: str):
        item = version(key)
        return {**model_summary(item), "dataset": item.metadata["archive_sha256"],
                "checks": item.gates["checks"], "metrics": details(item)[1]}

    @app.get("/api/admin/models/{key}/behavior-options")
    def behavior_options(key: str):
        item = version(key)
        try:
            data, _ = trainer.datasets.load(item.metadata["archive_sha256"])
        except (*READ_ERRORS, AssertionError):
            raise HTTPException(404, "The model's versioned EEG archive is unavailable.") from None
        a = data.arrays
        if "y_stage" not in a:
            raise HTTPException(409, "This model's dataset has no five-stage labels.")
        recordings = []
        rows = np.flatnonzero(a["state_split"] == "test")
        for record in sorted(set(a["state_record_id"][rows].tolist())):
            record_rows = rows[a["state_record_id"][rows] == record]
            order = record_rows[np.argsort(a["state_epoch_end_sec"][record_rows])]
            times = a["state_epoch_end_sec"][order]
            valid_ends = [int(times[index]) for index in range(HISTORY - 1, len(times))
                          if np.all(np.diff(times[index - HISTORY + 1:index + 1]) == EPOCH_SEC)]
            if valid_ends:
                recordings.append({"id": str(record), "windowEndsSec": valid_ends})
        return {"recordings": recordings, "stageNames": list(STAGE_NAMES),
                "forecastBins": list(CLASS_LABELS), "historyEpochs": HISTORY, "epochSeconds": EPOCH_SEC}

    @app.get("/api/admin/models/{key}/behavior")
    def model_behavior(key: str, record: Annotated[str, Query(min_length=1, max_length=80)],
                       end_sec: Annotated[int, Query(ge=1)]):
        item = version(key)
        try:
            data, _ = trainer.datasets.load(item.metadata["archive_sha256"])
            a = data.arrays
            rows = np.flatnonzero((a["state_record_id"] == record) & (a["state_split"] == "test"))
            order = rows[np.argsort(a["state_epoch_end_sec"][rows])]
            times = a["state_epoch_end_sec"][order]
            position = int(np.searchsorted(times, end_sec))
            if position >= len(times) or int(times[position]) != end_sec or position < HISTORY - 1:
                raise ValueError
            indices = order[position - HISTORY + 1:position + 1]
            if not np.all(np.diff(times[position - HISTORY + 1:position + 1]) == EPOCH_SEC):
                raise ValueError
            window = a["X_state"][indices]
            truth = a["y_stage"][indices]
        except HTTPException:
            raise
        except (*READ_ERRORS, AssertionError, IndexError, ValueError):
            raise HTTPException(404, "Choose a consecutive 10-epoch window from a held-out recording.") from None

        try:
            model, mean, scale, checkpoint, _, _ = load_bundle(item.path)
            if isinstance(model, SleepStageForecastPipeline):
                asleep, forecast, stages = predict_with_stages(model, mean, scale, window)
                stage_probabilities = stages.tolist()
                stage_names = list(STAGE_NAMES)
            else:
                asleep, forecast = predict_window(model, mean, scale, window)
                stage_probabilities, stage_names = [], []
        except (*READ_ERRORS, RuntimeError, ValueError) as error:
            raise HTTPException(409, f"Could not run this checkpoint: {error}") from None

        is_asleep = bool(asleep[-1] >= checkpoint["state_threshold"])
        predicted_stages = ([stage_names[int(np.argmax(row))] for row in stage_probabilities]
                            if stage_probabilities else [])
        forecast_result = None
        if is_asleep:
            forecast_result = {"withinHourProbability": float(1 - forecast[-1]),
                               "bins": [{"interval": label, "probability": float(probability)}
                                        for label, probability in zip(CLASS_LABELS, forecast)]}
        return {"modelId": item.metadata["run_id"], "recordingId": record,
                "endSec": end_sec, "windowStartSec": end_sec - HISTORY * EPOCH_SEC,
                "state": "asleep" if is_asleep else "awake", "asleepProbability": float(asleep[-1]),
                "stageNames": stage_names, "stageProbabilities": stage_probabilities,
                "predictedStages": predicted_stages,
                "observedStages": [STAGE_NAMES[int(label)] for label in truth],
                "forecast": forecast_result}

    @app.post("/api/admin/models/{key}/check")
    def verify(key: str):
        item = version(key)
        try:
            return {"message": check_checkpoint(item).replace("**", "")}
        except (*READ_ERRORS, RuntimeError, AssertionError):
            logging.exception("SleepOps checkpoint check failed")
            raise HTTPException(409, "Checkpoint check failed. See the Space logs for details.") from None

    @app.get("/api/admin/comparison")
    def get_comparison(a: str, b: str, split: Literal["validation", "test"]):
        message, rows, comparable = comparison(version(a), version(b), split, trainer.datasets.anchor)
        return {"message": message, "rows": rows, "comparable": comparable}

    @app.get("/api/admin/deployment")
    def deployment_status():
        return deployment.status()

    @app.post("/api/admin/deployment", status_code=202)
    def deploy_best(payload: DeploymentRequest):
        try:
            return deployment.enqueue(payload.candidate)
        except ValueError as error:
            raise HTTPException(409, str(error)) from None
        except OSError:
            logging.exception("Could not queue a SleepOps deployment")
            raise HTTPException(503, "Private deployment storage is unavailable.") from None

    @app.get("/api/admin/datasets")
    def dataset_versions():
        return {**chart_metadata(), "versions": trainer.datasets.list()}

    @app.get("/api/admin/datasets/{key}/recordings")
    def recordings(key: str, split: Literal["train", "validation", "test"]):
        try:
            return {"recordings": trainer.datasets.recordings(key, split)}
        except (*READ_ERRORS, AssertionError):
            raise HTTPException(404, "This dataset version is unavailable.") from None

    @app.get("/api/admin/datasets/{key}/preview")
    def preview(key: str, record: Annotated[str, Query(max_length=80)], feature: str,
                start: Annotated[int, Query(ge=0)] = 0):
        if feature not in FEATURE_NAMES:
            raise HTTPException(422, "Choose an EEG feature from the dataset.")
        try:
            return trainer.datasets.preview(key, record, feature, start)
        except (*READ_ERRORS, AssertionError):
            raise HTTPException(404, "This recording or dataset version is unavailable.") from None

    @app.get("/api/admin/datasets/{key}/archive")
    def dataset_archive(key: str):
        try:
            archive, lock = trainer.datasets.location(key)
            if sha256(archive) != read_json(lock)["archive_sha256"]:
                raise ValueError("Dataset checksum mismatch.")
        except READ_ERRORS:
            raise HTTPException(404, "The locked dataset archive is unavailable.") from None
        return FileResponse(archive, filename="sleep_edf_wake_data.npz", media_type="application/octet-stream",
                            headers={"Cache-Control": "private, no-store"})

    @app.post("/api/admin/datasets", status_code=201)
    def add_dataset(parent: Annotated[str, Form(pattern=r"^(base|[0-9a-f]{64})$")],
                    name: Annotated[str, Form(max_length=80)],
                    file: Annotated[UploadFile | None, File()] = None,
                    url: Annotated[str | None, Form(max_length=2048)] = None):
        has_url = bool((url or "").strip())
        if (file is None and not has_url) or (file is not None and has_url):
            raise HTTPException(422, "Choose one prepared archive file or one direct HTTPS link.")
        if file is not None and not (file.filename or "").lower().endswith(".npz"):
            raise HTTPException(422, "Upload a prepared EEG feature archive in NPZ format.")
        if file is not None and file.size is not None and file.size > MAX_UPLOAD:
            raise HTTPException(413, "The upload exceeds 24 MiB.")
        try:
            if file is not None:
                return trainer.datasets.add(parent, name, file.file)
            return trainer.datasets.add_url(parent, name, url)
        except DatasetTooLarge as error:
            raise HTTPException(413, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        except (AssertionError, KeyError, TypeError, IndexError, OverflowError):
            raise HTTPException(422, "The uploaded arrays do not match the recorded labels and causal EEG windows.") from None
        except OSError:
            logging.exception("Could not store dataset version")
            raise HTTPException(503, "The dataset could not be saved to private storage.") from None

    @app.get("/api/admin/training")
    def training_status():
        return trainer.snapshot()

    @app.post("/api/admin/training", status_code=202)
    def train(payload: TrainingRequest):
        try:
            trainer.start(payload.epochs, payload.evaluateTest, payload.dataset)
        except gr.Error as error:
            raise HTTPException(409, error.message) from None
        return trainer.snapshot()

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/admin-static", StaticFiles(directory=dist), name="admin-static")
    return gr.mount_gradio_app(app, make_app(trainer, catalog), path="/")


def main():
    import uvicorn

    trainer = default_trainer()
    dist = Path(os.getenv("SLEEPOPS_ADMIN_DIST", "frontend/admin/dist"))
    uvicorn.run(create_app(trainer, default_catalog(trainer), dist),
                host=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
                port=int(os.getenv("GRADIO_SERVER_PORT", "7860")))


if __name__ == "__main__":
    main()
