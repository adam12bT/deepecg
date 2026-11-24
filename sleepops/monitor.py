"""Population Stability Index from aggregate feature counts."""

from threading import Lock

import numpy as np

from eeg_data import FEATURE_NAMES, HISTORY


def build_reference(data, archive_sha256):
    a = data.arrays
    train = a["X_state"][a["state_split"] == "train"]
    features = []
    for name, column in zip(FEATURE_NAMES, train.T):
        cuts = np.unique(np.quantile(column, np.arange(1, 10) / 10))
        bins = np.searchsorted(cuts, column, side="right")
        counts = np.bincount(bins, minlength=len(cuts) + 1)
        features.append({"name": name, "cuts": cuts.tolist(), "counts": counts.tolist()})
    return {"version": 1, "archive_sha256": archive_sha256, "source": "unique train state epochs",
            "features": features}


class DriftMonitor:
    def __init__(self, reference, *, minimum_windows=100, alert_psi=0.25):
        if reference["version"] != 1 or [item["name"] for item in reference["features"]] != list(FEATURE_NAMES):
            raise ValueError("Unexpected drift-reference schema or feature order.")
        if minimum_windows < 1 or not np.isfinite(alert_psi) or alert_psi <= 0:
            raise ValueError("Drift thresholds must be positive.")
        self.cuts = [np.asarray(item["cuts"], dtype=float) for item in reference["features"]]
        self.reference = [np.asarray(item["counts"], dtype=np.int64) for item in reference["features"]]
        for cuts, counts in zip(self.cuts, self.reference):
            if (cuts.ndim != 1 or not np.isfinite(cuts).all() or not (np.diff(cuts) > 0).all()
                    or counts.shape != (len(cuts) + 1,) or np.any(counts < 0) or counts.sum() == 0):
                raise ValueError("Invalid drift-reference histogram.")
        self.counts = [np.zeros_like(counts) for counts in self.reference]
        self.windows = 0
        self.minimum_windows = minimum_windows
        self.alert_psi = alert_psi
        self.lock = Lock()

    def observe(self, window):
        with self.lock:
            for column, cuts, counts in zip(window.T, self.cuts, self.counts):
                bins = np.searchsorted(cuts, column, side="right")
                counts += np.bincount(bins, minlength=len(counts))
            self.windows += 1

    def snapshot(self):
        with self.lock:
            windows = self.windows
            counts = [item.copy() for item in self.counts]
        features = []
        for name, expected, observed in zip(FEATURE_NAMES, self.reference, counts):
            psi = None
            if windows >= self.minimum_windows:
                reference = (expected + 0.5) / (expected.sum() + 0.5 * len(expected))
                current = (observed + 0.5) / (observed.sum() + 0.5 * len(observed))
                psi = float(np.sum((current - reference) * np.log(current / reference)))
            features.append({"name": name, "psi": psi, "alert": psi is not None and psi >= self.alert_psi})
        return {"windows": windows, "epoch_observations": HISTORY * windows,
                "minimum_windows": self.minimum_windows, "alert_psi": self.alert_psi, "features": features}
