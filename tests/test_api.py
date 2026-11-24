from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from eeg_data import FEATURE_NAMES, normalize
from sleepops.api import create_app
from sleepops.registry import champion, load_bundle


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.serving_assets
def test_health_predictions_schema_and_telemetry(monkeypatch):
    monkeypatch.setenv("SLEEPOPS_BUILD_ID", "test-build-marker")
    registry = ROOT / "registry"
    path = champion(registry)
    model, mean, scale, details, _, _ = load_bundle(path)
    with np.load(ROOT / "artifacts" / "demo_windows.npz", allow_pickle=False) as data:
        examples = data["windows"][::24]
    with TestClient(create_app(registry, minimum_windows=2)) as client:
        with torch.inference_mode():
            state, forecasts = model(torch.from_numpy(normalize(examples, mean, scale)))
            p_sleep = state.sigmoid().numpy()
            p_forecasts = forecasts.softmax(-1).numpy()
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").json()["release_id"] == path.name
        deployed = client.get("/model").json()
        assert deployed["archive_sha256"] == details["archive_sha256"]
        assert deployed["build_id"] == "test-build-marker"
        assert len(deployed["checkpoint_sha256"]) == 64
        schema = client.get("/v1/schema").json()
        assert schema["shape"] == [10, 11] and schema["feature_names"] == list(FEATURE_NAMES)

        for state_name, row in (("awake", p_sleep[:, -1].argmin()), ("asleep", p_sleep[:, -1].argmax())):
            response = client.post("/v1/predict", json={"features": examples[row].tolist()})
            assert response.status_code == 200
            body = response.json()
            assert body["state"] == state_name and body["release_id"] == path.name
            np.testing.assert_allclose(body["asleep_probabilities"], p_sleep[row], atol=1e-6)
            if state_name == "awake":
                assert body["forecast"] is None
            else:
                forecast = body["forecast"]
                np.testing.assert_allclose([bin["probability"] for bin in forecast["bins"]], p_forecasts[row], atol=1e-6)
                assert abs(forecast["within_hour_probability"] - (1 - p_forecasts[row, -1])) < 1e-6

        for bad in ([], [["abc"] * 11] * 10, [[1e99] * 11] * 10):
            response = client.post("/v1/predict", json={"features": bad})
            assert response.status_code == 422
            assert all("input" not in error for error in response.json()["detail"])
        report = client.get("/monitor/drift").json()
        assert report["windows"] == 2 and len(report["features"]) == 11
        metrics = client.get("/metrics").text
        assert 'sleepops_prediction_requests_total{outcome="success"} 2.0' in metrics
        assert 'sleepops_prediction_requests_total{outcome="invalid_input"} 3.0' in metrics
        assert "sleepops_prediction_seconds_bucket" in metrics and "sleepops_feature_psi" in metrics
