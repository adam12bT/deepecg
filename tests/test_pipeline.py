from pathlib import Path
import shutil

import numpy as np
import pytest

from sleepops.dataset import load_dataset
from sleepops.files import read_json, write_json
from sleepops.monitor import DriftMonitor, build_reference
from sleepops.registry import activate, champion, check_quality, promote, validate_bundle


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registry"
POLICY = ROOT / "configs" / "promotion.json"


@pytest.fixture(scope="module")
def locked_data():
    return load_dataset(ROOT / "sleep_edf_wake_data.npz", ROOT / "configs" / "dataset.json")


@pytest.mark.dataset
def test_locked_splits_and_forecast_counts(locked_data):
    data, manifest = locked_data
    assert sum(split["subjects"] for split in manifest["splits"].values()) == 78
    assert sum(split["recordings"] for split in manifest["splits"].values()) == 153
    assert manifest["splits"]["test"]["state_epoch_counts"] == [13337, 27123]
    assert sum(manifest["splits"]["test"]["forecast_class_counts"]) == 6365
    assert np.all(data.forecast_classes[data.arrays["forecast_wake_onset_sec"] == -1] == 12)


@pytest.mark.dataset
def test_rejects_wrong_data_version(tmp_path, locked_data):
    _, manifest = locked_data
    lock = tmp_path / "lock.json"
    write_json(lock, {**manifest, "archive_sha256": "f" * 64})
    with pytest.raises(ValueError, match="Archive SHA-256"):
        load_dataset(ROOT / "sleep_edf_wake_data.npz", lock)


@pytest.mark.dataset
def test_feature_drift_is_aggregate_and_requires_enough_windows(locked_data):
    data, manifest = locked_data
    reference = build_reference(data, manifest["archive_sha256"])
    train = data.arrays["state_split"] == "train"
    assert all(sum(row["counts"]) == int(train.sum()) for row in reference["features"])
    monitor = DriftMonitor(reference, minimum_windows=100)
    windows = np.random.default_rng(0).choice(data.arrays["X_state"][train], (100, 10))
    for window in windows[:-1]:
        monitor.observe(window)
    assert monitor.snapshot()["features"][0]["psi"] is None
    monitor.observe(windows[-1])
    assert not any(feature["alert"] for feature in monitor.snapshot()["features"])
    for _ in windows:
        monitor.observe(np.full((10, 11), 1e3))
    report = monitor.snapshot()
    assert report["windows"] == 200 and report["epoch_observations"] == 2000
    assert all(feature["alert"] for feature in report["features"])


@pytest.mark.serving_assets
def test_quality_gate_only_accepts_passing_validation_metrics():
    _, report, _ = validate_bundle(champion(REGISTRY))
    policy = read_json(POLICY)
    assert check_quality(report, policy)["passed"]
    with pytest.raises(ValueError, match="validation split"):
        check_quality({**report, "split": "test"}, policy)
    decision = check_quality(report, {**policy, "min_balanced_accuracy": 1.0})
    assert not decision["passed"] and not decision["checks"]["balanced_accuracy"]


@pytest.mark.serving_assets
def test_atomic_promotion_rollback_and_tamper_detection(tmp_path):
    source = champion(REGISTRY)
    registry = tmp_path / "registry"
    policy = read_json(POLICY)
    first = promote(registry, source, policy)
    second_run = shutil.copytree(source, tmp_path / "candidate")
    metadata = read_json(second_run / "metadata.json")
    second_id = "1" * 32 if metadata["run_id"] != "1" * 32 else "2" * 32
    write_json(second_run / "metadata.json", {**metadata, "run_id": second_id})
    with pytest.raises(ValueError, match="Promotion rejected"):
        promote(registry, second_run, {**policy, "min_subjects": 1000})
    assert champion(registry) == first
    second = promote(registry, second_run, policy)
    assert champion(registry) == second
    activate(registry, first.name)
    assert champion(registry) == first
    with (second / "drift_reference.json").open("ab") as file:
        file.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        activate(registry, second.name)
    assert champion(registry) == first
