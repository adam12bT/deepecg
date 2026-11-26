"""Browse the pinned EEG archive and append recordings as private DVC versions."""

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from threading import Lock
from zipfile import BadZipFile, ZipFile

import numpy as np
from numpy.lib import format as npy

from deploy.huggingface.release import ARCHIVE, DATASET_PREFIX, object_key
from eeg_data import CLASS_LABELS, EPOCH_SEC, FEATURE_NAMES, SPLITS, require
from sleepops.dataset import load_dataset
from sleepops.files import read_json, sha256, utc_now, write_json


MAX_UPLOAD = 24 * 1024 * 1024
MAX_ARCHIVE = 48 * 1024 * 1024
MAX_EXPANDED = 192 * 1024 * 1024
PREVIEW_EPOCHS = 120
SHA256 = re.compile(r"[0-9a-f]{64}")
MD5 = re.compile(r"[0-9a-f]{32}")


def load_upload(path, expected):
    """Validate the ZIP and each NPY header before allocating NumPy arrays."""
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            uploaded = {entry.filename[:-4] for entry in entries}
            allowed = set(expected)
            if "y_stage" not in allowed:
                allowed.add("y_stage")
            require(len(entries) == len(uploaded) and uploaded in (set(expected), allowed)
                    and {f"{key}.npy" for key in uploaded} == {entry.filename for entry in entries},
                    "Upload a complete SleepOps NPZ archive, optionally with five-class stage labels.")
            require(sum(entry.file_size for entry in entries) <= MAX_EXPANDED, "The expanded archive is too large.")
            for entry in entries:
                with archive.open(entry) as member:
                    version = npy.read_magic(member)
                    require(version in {(1, 0), (2, 0)}, "Unsupported NumPy array header.")
                    reader = npy.read_array_header_1_0 if version == (1, 0) else npy.read_array_header_2_0
                    shape, _, dtype = reader(member)
                    require(not dtype.hasobject and math.prod(shape) * dtype.itemsize == entry.file_size - member.tell(),
                            "The upload contains an invalid or object-valued array.")
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (BadZipFile, EOFError, OSError, NotImplementedError) as error:
        raise ValueError("The attachment could not be read as a NumPy NPZ archive.") from error


def group(key):
    if key == "record_id" or key.startswith("record_"):
        return "record"
    if key.startswith("state_") or key in {"X_state", "y_state", "y_stage"}:
        return "state"
    if key.startswith("forecast_") or key in {"X_forecast", "y_forecast_min"}:
        return "forecast"
    return None


def append_recordings(base, incoming):
    originals = dict(zip(base["record_id"], base["record_subject_id"]))
    records = incoming["record_id"]
    subjects = incoming["record_subject_id"]
    record_count = len(records)
    require(records.ndim == subjects.ndim == 1 and len(subjects) == record_count
            and records.dtype.kind == subjects.dtype.kind == "U", "Invalid recording manifest.")
    require(len(set(records)) == record_count, "Duplicate recording IDs in the upload.")
    require(all(value and value.strip() == value and len(value) <= 80 for value in (*records, *subjects)),
            "Use nonempty recording and subject IDs of at most 80 characters.")
    require(all(originals[record] == subject for record, subject in zip(records, subjects) if record in originals),
            "A recording ID was reassigned to a different subject.")
    new_records = {record for record in records if record not in originals}
    if not new_records:
        require("y_stage" in incoming and "y_stage" not in base,
                "There are no new recording IDs or stage labels to add.")
        for key, original in base.items():
            if key != "schema_version":
                require(np.array_equal(original, incoming[key]),
                        f"A stage-label upload cannot change existing data in {key}.")
        stages = incoming["y_stage"]
        require(stages.shape == base["y_state"].shape and np.isin(stages, np.arange(5)).all(),
                "Provide one Wake, N1, N2, N3, or REM label for each existing scored epoch.")
        require(np.array_equal(base["y_state"], stages != 0),
                "The stage labels disagree with the existing sleep/wake labels.")
        upgraded = {**base, "y_stage": stages, "schema_version": np.asarray(2, dtype=np.int64)}
        return upgraded, {"recordings": 0, "subjects": 0, "stateEpochs": 0,
                          "forecastWindows": 0, "skippedRecordings": 0, "stageLabels": len(stages)}
    new_rows = np.isin(records, list(new_records))
    require(("y_stage" in incoming) == ("y_stage" in base),
            "The upload and selected dataset must both include five-class stage labels, or both omit them.")
    held_out = {subject.casefold() for subject in base["record_subject_id"][base["record_split"] != "train"]}
    require(not held_out.intersection(subject.casefold() for subject in subjects[new_rows]),
            "A new recording belongs to an existing validation or test subject.")
    require(set(records).issuperset(incoming["state_record_id"]) and set(records).issuperset(incoming["forecast_record_id"]),
            "Some upload rows refer to recordings missing from the manifest.")
    masks = {name: np.isin(incoming[f"{name}_record_id"], list(new_records)) for name in ("state", "forecast")}
    masks["record"] = new_rows
    require(set(incoming["state_record_id"][masks["state"]]) == new_records,
            "Every new recording needs scored sleep–wake epochs.")

    merged = {}
    for key, original in base.items():
        value = incoming[key]
        task = group(key)
        if task is None:
            require(np.array_equal(original, value), f"The upload has incompatible metadata for {key}.")
            merged[key] = original
            continue
        mask = masks[task]
        compatible = (original.dtype.kind == value.dtype.kind or
                      {original.dtype.kind, value.dtype.kind} <= {"i", "u"})
        require(compatible and value.shape == (len(mask), *original.shape[1:]), f"Invalid shape or dtype for {key}.")
        added = value[mask]
        if key == f"{task}_split":
            added = np.full(len(added), "train", dtype=original.dtype)
        elif original.dtype.kind != "U":
            converted = added.astype(original.dtype)
            if original.dtype.kind in "iu":
                require(np.array_equal(converted, added), f"Some values in {key} exceed the supported range.")
            added = converted
        merged[key] = np.concatenate((original, added))
    return merged, {"recordings": len(new_records), "subjects": len(set(subjects[new_rows]) - set(originals.values())),
                    "stateEpochs": int(masks["state"].sum()), "forecastWindows": int(masks["forecast"].sum()),
                    "skippedRecordings": record_count - len(new_records),
                    "stageLabels": int(masks["state"].sum()) if "y_stage" in incoming else 0}


class DatasetStore:
    def __init__(self, dvc_store, versions, bucket, base_lock=Path("configs/dataset.json")):
        self.dvc_store = dvc_store
        self.versions = versions
        self.bucket = bucket
        self.base_lock = base_lock
        self.guard = Lock()

    def location(self, key):
        if key == "base":
            return self.dvc_store / object_key(ARCHIVE), self.base_lock
        if not SHA256.fullmatch(key):
            raise ValueError("Unknown dataset version.")
        directory = self.versions / "releases" / key
        info = read_json(directory / "version.json")
        lock = directory / "dataset.json"
        digest = info["md5"]
        if info["id"] != key or not MD5.fullmatch(digest) or read_json(lock)["archive_sha256"] != key:
            raise ValueError("The dataset version record is invalid.")
        return self.versions / "files" / "md5" / digest[:2] / digest[2:], lock

    @lru_cache(maxsize=2)
    def load(self, key):
        return load_dataset(*self.location(key))

    def anchor(self, archive_hash):
        base = read_json(self.base_lock)["archive_sha256"]
        if archive_hash == base:
            return base
        if SHA256.fullmatch(archive_hash):
            directory = self.versions / "releases" / archive_hash
            if (directory / "version.json").is_file():
                info = read_json(directory / "version.json")
                if info["id"] == archive_hash and info["heldOutAnchor"] == base:
                    return base
        return None

    def summary(self, key):
        _, lock = self.location(key)
        manifest = read_json(lock)
        if key == "base":
            info = {"id": "base", "name": "Original Sleep-EDF", "parent": None, "createdAt": None, "added": None}
            url = f"https://huggingface.co/buckets/{self.bucket}/tree/dvc-store"
        else:
            entry = read_json(lock.with_name("version.json"))
            info = {field: entry[field] for field in ("id", "name", "parent", "createdAt", "added")}
            url = f"https://huggingface.co/buckets/{self.bucket}/tree/{DATASET_PREFIX}/releases/{key}"
        return {**info, "hash": manifest["archive_sha256"], "bytes": manifest["bytes"], "splits": manifest["splits"],
                "url": url}

    def list(self):
        original = self.summary("base")
        location = self.versions / "releases"
        entries = []
        if location.is_dir():
            for path in location.iterdir():
                if SHA256.fullmatch(path.name) and (path / "version.json").is_file():
                    try:
                        entries.append(self.summary(path.name))
                    except (OSError, ValueError, KeyError):
                        continue
        return [original, *sorted(entries, key=lambda entry: entry["createdAt"], reverse=True)]

    def recordings(self, key, split):
        data, _ = self.load(key)
        a = data.arrays
        return [{"id": record, "subject": subject} for record, subject, group_name in
                zip(a["record_id"].tolist(), a["record_subject_id"].tolist(), a["record_split"].tolist())
                if group_name == split]

    def preview(self, key, record, feature, start):
        data, _ = self.load(key)
        a = data.arrays
        rows = np.flatnonzero(a["state_record_id"] == record)
        require(bool(len(rows)), "Unknown recording in this dataset version.")
        times = a["state_epoch_end_sec"]
        ordered = rows[np.argsort(times[rows])]
        start = min(start, max(0, len(rows) - PREVIEW_EPOCHS))
        selected = ordered[start:start + PREVIEW_EPOCHS]
        return {"id": record, "start": start, "total": len(rows), "epochSeconds": EPOCH_SEC,
                "timeSeconds": times[selected].tolist(), "states": a["y_state"][selected].tolist(),
                "values": a["X_state"][selected, FEATURE_NAMES.index(feature)].tolist()}

    def add(self, parent, name, source):
        require(self.versions.is_dir(), "The private dataset storage volume is unavailable.")
        name = name.strip()
        require(bool(name) and len(name) <= 80 and name.isprintable(), "Enter a dataset name of at most 80 characters.")
        with self.guard, TemporaryDirectory(prefix="sleepops-dataset-upload-") as tmp:
            work = Path(tmp)
            uploaded = work / "upload.npz"
            with uploaded.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    require(target.tell() + len(chunk) <= MAX_UPLOAD, "The upload exceeds 24 MiB.")
                    target.write(chunk)
            base, current = self.load(parent)
            incoming = load_upload(uploaded, base.arrays)
            merged, added = append_recordings(base.arrays, incoming)
            archive = work / ARCHIVE
            np.savez_compressed(archive, **merged)
            require(archive.stat().st_size <= MAX_ARCHIVE, "The merged dataset exceeds 48 MiB.")
            _, manifest = load_dataset(archive)
            require(all({key: value for key, value in manifest["splits"][split].items() if key != "stage_counts"}
                        == {key: value for key, value in current["splits"][split].items() if key != "stage_counts"}
                        for split in ("validation", "test")),
                    "The existing held-out cohorts must be preserved.")
            key = manifest["archive_sha256"]
            release = self.versions / "releases" / key
            marker = release / "version.json"
            if marker.is_file():
                try:
                    return self.summary(key)
                except (OSError, ValueError, KeyError):
                    pass
            with archive.open("rb") as stream:
                digest = hashlib.file_digest(stream, "md5").hexdigest()
            remote = self.versions / "files" / "md5" / digest[:2] / digest[2:]
            remote.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, remote)
            require(sha256(remote) == key, "The stored archive checksum could not be verified.")
            release.mkdir(parents=True, exist_ok=True)
            write_json(work / "dataset.json", manifest)
            pointer = f"{ARCHIVE}.dvc"
            (work / pointer).write_text(f"outs:\n- md5: {digest}\n  size: {manifest['bytes']}\n"
                                        f"  hash: md5\n  path: {ARCHIVE}\n", encoding="utf-8")
            write_json(work / "version.json", {"id": key, "name": name, "parent": parent, "createdAt": utc_now(),
                       "added": added, "md5": digest, "heldOutAnchor": read_json(self.base_lock)["archive_sha256"]})
            for file in ("dataset.json", pointer, "version.json"):
                shutil.copyfile(work / file, release / file)
            return self.summary(key)


def chart_metadata():
    return {"features": FEATURE_NAMES, "forecastLabels": CLASS_LABELS,
            "stageNames": ("Wake", "N1", "N2", "N3", "REM"),
            "splits": SPLITS, "maxUploadBytes": MAX_UPLOAD,
            "previewEpochs": PREVIEW_EPOCHS}
