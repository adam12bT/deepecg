"""FastAPI inference service for the active model release."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
import torch

from eeg_data import CLASS_LABELS, EPOCH_SEC, FEATURE_NAMES, HISTORY, N_FEATURES
from eeg_model import predict_window
from sleepops.monitor import DriftMonitor
from sleepops.registry import CHECKPOINT, champion, load_bundle


NumericFeature = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1e6, le=1e6)]
FeatureRow = Annotated[list[NumericFeature], Field(min_length=N_FEATURES, max_length=N_FEATURES)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: Annotated[list[FeatureRow], Field(min_length=HISTORY, max_length=HISTORY)]


class BinProbability(BaseModel):
    interval: str
    probability: float


class Forecast(BaseModel):
    within_hour_probability: float
    bins: list[BinProbability]


class Prediction(BaseModel):
    release_id: str
    state: Literal["awake", "asleep"]
    state_threshold: float
    asleep_probabilities: list[float]
    forecast: Forecast | None


def create_app(registry=None, *, minimum_windows=100, alert_psi=0.25):
    registry = Path(registry or os.getenv("SLEEPOPS_REGISTRY", "registry"))
    collectors = CollectorRegistry()
    requests = Counter("sleepops_prediction_requests", "Prediction requests by outcome", ("outcome",),
                       registry=collectors)
    latency = Histogram("sleepops_prediction_seconds", "Validated prediction handling time", registry=collectors)
    decisions = Counter("sleepops_predictions", "Successful predictions by state and release", ("state", "release"),
                        registry=collectors)
    windows_seen = Gauge("sleepops_drift_windows", "Successful feature windows monitored", registry=collectors)
    psi = Gauge("sleepops_feature_psi", "Population Stability Index against training EEG", ("feature",),
                registry=collectors)

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(int(os.getenv("SLEEPOPS_TORCH_THREADS", "2")))
        path = champion(registry)
        model, mean, scale, checkpoint, metadata, reference = load_bundle(path)
        if metadata["run_id"] != path.name:
            raise RuntimeError("The champion pointer disagrees with its release metadata.")
        app.state.model_input = (model, mean, scale)
        app.state.drift = DriftMonitor(reference, minimum_windows=minimum_windows, alert_psi=alert_psi)
        app.state.details = {"release_id": path.name, "archive_sha256": metadata["archive_sha256"],
                             "checkpoint_sha256": metadata["file_sha256"][CHECKPOINT],
                             "selected_epoch": checkpoint["best_epoch"], "state_threshold": checkpoint["state_threshold"]}
        if build_id := os.getenv("SLEEPOPS_BUILD_ID"):
            app.state.details["build_id"] = build_id
        yield

    app = FastAPI(title="SleepOps EEG Inference", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        if request.url.path == "/v1/predict":
            requests.labels("invalid_input").inc()
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": item["loc"], "msg": item["msg"]} for item in error.errors()
        ]})

    @app.get("/health/live")
    def live():
        return {"status": "live"}

    @app.get("/health/ready")
    def ready(request: Request):
        return {"status": "ready", "release_id": request.app.state.details["release_id"]}

    @app.get("/model")
    def model_info(request: Request):
        return request.app.state.details

    @app.get("/v1/schema")
    def feature_schema():
        return {"shape": [HISTORY, N_FEATURES], "epoch_seconds": EPOCH_SEC,
                "feature_names": FEATURE_NAMES, "window_order": "oldest to most recent completed epoch",
                "forecast_bins": CLASS_LABELS}

    @app.post("/v1/predict", response_model=Prediction)
    def predict(body: PredictionRequest, request: Request):
        started = perf_counter()
        serving = request.app.state
        details = serving.details
        version = details["release_id"]
        threshold = details["state_threshold"]
        window = np.asarray(body.features, dtype=np.float32)
        try:
            p_sleep, probabilities = predict_window(*serving.model_input, window)
            if not np.isfinite(p_sleep).all() or not np.isfinite(probabilities).all():
                raise RuntimeError("Nonfinite model output.")
        except ValueError as error:
            requests.labels("invalid_input").inc()
            raise HTTPException(422, detail=str(error)) from error
        except RuntimeError as error:
            requests.labels("prediction_error").inc()
            raise HTTPException(500, detail="Prediction could not be completed.") from error
        finally:
            latency.observe(perf_counter() - started)
        state = "asleep" if p_sleep[-1] >= threshold else "awake"
        forecast = None
        if state == "asleep":
            forecast = Forecast(within_hour_probability=1 - float(probabilities[-1]), bins=[
                BinProbability(interval=name, probability=float(probability))
                for name, probability in zip(CLASS_LABELS, probabilities)
            ])
        serving.drift.observe(window)
        decisions.labels(state, version).inc()
        requests.labels("success").inc()
        return Prediction(release_id=version, state=state, state_threshold=threshold,
                          asleep_probabilities=p_sleep.tolist(), forecast=forecast)

    @app.get("/monitor/drift")
    def drift(request: Request):
        state = request.app.state
        return {"release_id": state.details["release_id"], **state.drift.snapshot()}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics(request: Request):
        summary = request.app.state.drift.snapshot()
        windows_seen.set(summary["windows"])
        for feature in summary["features"]:
            if feature["psi"] is not None:
                psi.labels(feature["name"]).set(feature["psi"])
        return Response(generate_latest(collectors), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
