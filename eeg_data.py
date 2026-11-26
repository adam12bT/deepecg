"""Sleep-EDF archive validation and causal window construction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


EPOCH_SEC = 30
HISTORY = 10
N_FEATURES = 11
N_CLASSES = 13
N_STAGES = 5
STAGE_NAMES = ("Wake", "N1", "N2", "N3", "REM")
SPLITS = ("train", "validation", "test")
BANDS = ("delta", "theta", "alpha", "sigma", "beta")
FEATURE_NAMES = tuple(f"{band}_{kind}_power" for kind in ("log", "relative") for band in BANDS) + (
    "log_rms",
)
CLASS_LABELS = tuple(f"{i * 5}–{(i + 1) * 5} min" for i in range(12)) + ("No onset <60 min",)


@dataclass
class EEGData:
    arrays: dict[str, np.ndarray]
    state_runs: list[np.ndarray]
    state_windows: dict[str, np.ndarray]
    forecast_classes: np.ndarray


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_archive(path: str | Path, verbose: bool = True) -> EEGData:
    with np.load(path, allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    if verbose:
        print("Archive keys, shapes, and data types")
        for key, value in a.items():
            print(f"  {key:38} {str(value.shape):17} {value.dtype}")

    metadata = {
        "epoch_sec": EPOCH_SEC, "history_epochs": HISTORY,
        "wake_minimum_sec": 120, "horizon_min": 60, "sampling_hz": 100,
        "channel": "EEG Fpz-Cz",
    }
    for key, expected in metadata.items():
        require(key in a and a[key].shape == () and a[key].item() == expected,
                f"Expected archive metadata {key}={expected!r}.")
    require(tuple(a.get("feature_names", ())) == FEATURE_NAMES, "Unexpected feature names or order.")
    require(a["schema_version"].shape == (), "Invalid archive schema version.")
    schema_version = int(a["schema_version"].item())
    require(schema_version in {1, 2}, "Unsupported archive schema version.")
    has_stages = "y_stage" in a
    require(has_stages == (schema_version == 2), "Schema version and sleep-stage labels disagree.")

    record_ids = a["record_id"]
    subjects = a["record_subject_id"]
    splits = a["record_split"]
    require(record_ids.ndim == 1 and subjects.shape == splits.shape == record_ids.shape,
            "Manifest identifiers must be aligned one-dimensional arrays.")
    require(len(np.unique(record_ids)) == len(record_ids), "Duplicate manifest recording ID.")
    require(set(splits) == set(SPLITS), "Missing or unexpected manifest split.")
    subject_sets = {split: set(subjects[splits == split]) for split in SPLITS}
    require(sum(map(len, subject_sets.values())) == len(set(subjects)), "A subject crosses the splits.")
    manifest = dict(zip(record_ids, zip(subjects, splits)))

    for task, target, feature_shape in (
        ("state", "y_state", (N_FEATURES,)),
        ("forecast", "y_forecast_min", (HISTORY, N_FEATURES)),
    ):
        y = a[target]
        x = a[f"X_{task}"]
        require(y.ndim == 1 and x.shape == (len(y), *feature_shape) and np.isfinite(x).all(),
                f"Unexpected {task} feature shape or nonfinite value.")
        record, subject, split, end = (a[f"{task}_{field}"] for field in (
            "record_id", "subject_id", "split", "epoch_end_sec",
        ))
        require(all(arr.shape == y.shape for arr in (record, subject, split, end)),
                f"Misaligned {task} metadata.")
        require(set(split) == set(SPLITS), f"Missing or unexpected {task} split.")
        require(end.dtype.kind in "iu" and np.all((end > 0) & (end % EPOCH_SEC == 0)),
                f"Invalid {task} epoch-end timestamps.")
        for recording in np.unique(record):
            rows = record == recording
            require(recording in manifest, f"Unknown recording {recording}.")
            expected_subject, expected_split = manifest[recording]
            require(np.all(subject[rows] == expected_subject) and np.all(split[rows] == expected_split),
                    f"Manifest disagrees with {task} rows for {recording}.")
            require(len(np.unique(end[rows])) == rows.sum(), f"Duplicate {task} epoch for {recording}.")

    require(set(a["state_record_id"]) == set(record_ids), "State rows do not cover the manifest.")
    require(np.isin(a["y_state"], [0, 1]).all(), "State labels must be Wake=0 or Sleep=1.")
    if has_stages:
        stage = a["y_stage"]
        require(stage.shape == a["y_state"].shape and np.isin(stage, np.arange(N_STAGES)).all(),
                "Sleep-stage labels must align with state rows and use Wake, N1, N2, N3, or REM.")
        require(np.array_equal(a["y_state"], stage != 0),
                "Binary sleep labels must agree with the detailed sleep-stage labels.")
    classes = forecast_targets(a)
    runs, windows = build_state_windows(a)
    validate_forecast_histories(a)
    data = EEGData(a, runs, windows, classes)
    if verbose:
        print(f"\n{len(record_ids)} recordings / {len(np.unique(subjects))} people; disjoint subject splits")
        print("Counts in order Wake, Sleep for state and class 0 through 12 for forecast")
        for split in SPLITS:
            state = a["state_split"] == split
            forecast = a["forecast_split"] == split
            endpoints = a["y_state"][windows[split][:, -1]]
            print(f"{split}: {len(subject_sets[split])} people, {int((splits == split).sum())} recordings")
            print(f"  State epochs {state.sum():,}: {np.bincount(a['y_state'][state].astype(int), minlength=2)}")
            print(f"  State 10-epoch windows {len(endpoints):,}, by end label: "
                  f"{np.bincount(endpoints.astype(int), minlength=2)}")
            print(f"  Forecast windows {forecast.sum():,}: {np.bincount(classes[forecast], minlength=N_CLASSES)}")
        print("Verified every forecast history, target bin, onset sentinel, and observed sustained Wake.")
    return data


def forecast_targets(a):
    minutes = a["y_forecast_min"]
    end = a["forecast_epoch_end_sec"]
    onset = a["forecast_wake_onset_sec"]
    require(onset.shape == minutes.shape and onset.dtype.kind in "iu", "Invalid forecast onset array.")
    require(np.isfinite(minutes).all() and np.all((minutes >= 0) & (minutes <= 60)),
            "Forecast minutes must be in [0, 60].")
    require(np.all(end >= HISTORY * EPOCH_SEC), "Forecast history starts before the recording.")
    observed = onset >= 0
    delta = onset[observed].astype(np.int64) - end[observed].astype(np.int64)
    require(np.all((delta >= 0) & (delta < 3600) & (delta % EPOCH_SEC == 0)),
            "Observed onsets must be on the epoch grid and before 60 minutes.")
    require(np.all(onset[~observed] == -1) and np.all(minutes[~observed] == 60),
            "Capped targets must have sentinel -1 and target 60 minutes.")
    np.testing.assert_allclose(minutes[observed], delta / 60, rtol=0, atol=1e-6)
    classes = np.floor_divide(minutes, 5).astype(np.int64)
    expected = np.full(len(minutes), N_CLASSES - 1, dtype=np.int64)
    expected[observed] = delta // 300
    np.testing.assert_array_equal(classes, expected)
    return classes


def build_state_windows(a):
    record, end, split = (a[f"state_{field}"] for field in ("record_id", "epoch_end_sec", "split"))
    ordered = np.lexsort((end, record))
    boundaries = (record[ordered[1:]] != record[ordered[:-1]]) | (np.diff(end[ordered]) != EPOCH_SEC)
    runs = np.split(ordered, np.flatnonzero(boundaries) + 1)
    grouped = {name: [] for name in SPLITS}
    for run in runs:
        if len(run) >= HISTORY:
            grouped[split[run[0]]].append(sliding_window_view(run, HISTORY))
    windows = {name: np.concatenate(parts) if parts else np.empty((0, HISTORY), dtype=np.int64)
               for name, parts in grouped.items()}
    for name, indices in windows.items():
        require(np.all(record[indices] == record[indices[:, :1]])
                and np.all(np.diff(end[indices], axis=1) == EPOCH_SEC)
                and np.all(split[indices] == name), f"An invalid {name} state window was constructed.")
        require(set(a["y_state"][indices[:, -1]]) == {0, 1},
                f"The {name} state windows must end on both Wake and Sleep epochs.")
    return runs, windows


def validate_forecast_histories(a):
    """Compare archived histories and check that observed onsets have four Wake epochs."""
    for record in np.unique(a["forecast_record_id"]):
        state = np.flatnonzero(a["state_record_id"] == record)
        state = state[np.argsort(a["state_epoch_end_sec"][state])]
        times = a["state_epoch_end_sec"][state]
        labels = a["y_state"][state]
        rows = np.flatnonzero(a["forecast_record_id"] == record)
        end = a["forecast_epoch_end_sec"][rows]
        positions = np.searchsorted(times, end)[:, None] - np.arange(HISTORY - 1, -1, -1)
        require(np.all((positions >= 0) & (positions < len(times))), f"Missing forecast history in {record}.")
        np.testing.assert_array_equal(times[positions], end[:, None] - EPOCH_SEC * np.arange(HISTORY - 1, -1, -1))
        require(np.all(labels[positions[:, -1]] == 1), f"A forecast history ends on Wake in {record}.")
        np.testing.assert_allclose(a["X_forecast"][rows], a["X_state"][state[positions]], rtol=1e-6, atol=1e-7)

        onset = a["forecast_wake_onset_sec"][rows]
        scored_times = sliding_window_view(times, 5)
        scored_labels = sliding_window_view(labels, 5)
        transitions = (np.diff(scored_times, axis=1) == EPOCH_SEC).all(axis=1) & (
            scored_labels == [1, 0, 0, 0, 0]
        ).all(axis=1)
        # The preceding sleep epoch ends exactly at the first Wake epoch's start.
        wake_starts = np.append(scored_times[transitions, 0].astype(np.int64), np.iinfo(np.int64).max)
        next_wake = wake_starts[np.searchsorted(wake_starts, end)]
        expected = np.where(next_wake - end < 3600, next_wake, -1)
        np.testing.assert_array_equal(onset, expected, err_msg=f"Sustained Wake mismatch in {record}")


def evaluation_histories(data, split):
    """One row per scored epoch, using the longest available causal history up to 10."""
    a = data.arrays
    windows, positions, rows = [], [], []
    for run in data.state_runs:
        if a["state_split"][run[0]] != split:
            continue
        end = np.arange(len(run))
        start = np.maximum(end - HISTORY + 1, 0)
        lookup = start[:, None] + np.arange(HISTORY)
        observed = lookup <= end[:, None]
        # A safe index for gathering. The unused suffix is zeroed after scaling.
        windows.append(run[np.minimum(lookup, end[:, None])])
        positions.append(end - start)
        rows.append(run)
    indices = np.concatenate(windows)
    last = np.concatenate(positions)
    targets = np.concatenate(rows)
    np.testing.assert_array_equal(np.sort(targets), np.flatnonzero(a["state_split"] == split))
    np.testing.assert_array_equal(indices[np.arange(len(targets)), last], targets)
    return indices, last, targets


def fit_normalization(data):
    a = data.arrays
    train = a["X_state"][a["state_split"] == "train"].astype(np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def normalize(x, mean, scale):
    return (np.asarray(x, dtype=np.float32) - mean) / scale
