"""Check the data contract and build a content-addressed dataset manifest."""

import hashlib
from pathlib import Path

import numpy as np

from eeg_data import FEATURE_NAMES, N_CLASSES, SPLITS, load_archive
from sleepops.files import read_json, sha256


def load_dataset(archive, lock=None):
    archive = Path(archive)
    locked = read_json(lock) if lock is not None else None
    fingerprint = sha256(archive)
    if locked is not None and locked["archive_sha256"] != fingerprint:
        raise ValueError(f"Archive SHA-256 does not match {lock}.")
    data = load_archive(archive, verbose=False)
    if sha256(archive) != fingerprint:
        raise ValueError("Dataset archive changed while loading it.")
    a = data.arrays
    manifest = {"version": 1, "archive_sha256": fingerprint, "bytes": archive.stat().st_size,
                "feature_names": list(FEATURE_NAMES), "splits": {}}
    for split in SPLITS:
        records = a["record_split"] == split
        state = a["state_split"] == split
        forecast = a["forecast_split"] == split
        subjects = sorted(set(a["record_subject_id"][records]))
        manifest["splits"][split] = {
            "subjects": len(subjects), "recordings": int(records.sum()),
            "subject_ids_sha256": hashlib.sha256("\n".join(subjects).encode()).hexdigest(),
            "state_epoch_counts": np.bincount(a["y_state"][state], minlength=2).tolist(),
            "state_windows": len(data.state_windows[split]),
            "forecast_class_counts": np.bincount(data.forecast_classes[forecast], minlength=N_CLASSES).tolist(),
        }
    if locked is not None and locked != manifest:
        raise ValueError(f"Dataset contract differs from {lock}.")
    return data, manifest
