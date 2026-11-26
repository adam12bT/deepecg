"""Rank eligible models and queue explicit deployment requests in private storage."""

from functools import lru_cache
import json
from math import isfinite
from threading import Lock
from time import time_ns
from uuid import uuid4

from deploy.training_space.models import READ_ERRORS, comparison
from sleepops.files import read_json, sha256, utc_now
from sleepops.registry import CHECKPOINT


def checkpoint_loss(path, digest, report_digest):
    if sha256(path) != digest:
        raise ValueError("The checkpoint checksum changed.")
    report = path.with_name("validation_metrics.json")
    if sha256(report) != report_digest:
        raise ValueError("The validation report checksum changed.")
    score = float(read_json(report)["forecast"]["shared_model"]["multiclass_log_loss"])
    if not isfinite(score) or score < 0:
        raise ValueError("The checkpoint has an invalid validation loss.")
    return score


cached_loss = lru_cache(maxsize=24)(checkpoint_loss)


class PromotionQueue:
    def __init__(self, catalog, datasets, output):
        self.catalog = catalog
        self.datasets = datasets
        self.path = output / "deployment"
        self.lock = Lock()

    def current_request(self):
        latest = max((self.path / "requests").glob("*.json"), default=None)
        if latest is None:
            return None
        request = read_json(latest)
        try:
            status = read_json(self.path / "status" / f"{request['id']}.json")
        except FileNotFoundError:
            status = {"phase": "queued"}
        return {**request, **status}

    def best(self):
        try:
            active = self.catalog.version("deployed")
            active_loss = cached_loss(active.path / CHECKPOINT, active.metadata["file_sha256"][CHECKPOINT],
                                      active.metadata["file_sha256"]["validation_metrics.json"])
        except (*READ_ERRORS, RuntimeError):
            return None, None
        best = None
        for model in self.catalog.versions():
            if model.key == "deployed" or not model.gates["passed"] or model.metadata["run_id"] == active.metadata["run_id"]:
                continue
            try:
                if not comparison(active, model, "validation", self.datasets.anchor)[2]:
                    continue
                loss = cached_loss(model.path / CHECKPOINT, model.metadata["file_sha256"][CHECKPOINT],
                                   model.metadata["file_sha256"]["validation_metrics.json"])
            except (*READ_ERRORS, RuntimeError):
                continue
            if loss < active_loss and (best is None or loss < best["loss"]):
                best = {"id": model.metadata["run_id"], "loss": loss, "dataset": model.dataset["archive_sha256"],
                        "epoch": model.validation["selected_epoch"]}
        return {"id": active.metadata["run_id"], "loss": active_loss}, best

    def status(self):
        active, candidate = self.best()
        return {"active": active, "candidate": candidate, "request": self.current_request()}

    def enqueue(self, candidate_id):
        with self.lock:
            previous = self.current_request()
            if previous and previous["phase"] in {"queued", "running"}:
                raise ValueError("A deployment request is already in progress.")
            if not self.path.parent.is_dir():
                raise ValueError("Private deployment storage is unavailable.")
            active, best = self.best()
            if best is None or best["id"] != candidate_id:
                raise ValueError("The best eligible candidate changed. Refresh and review the current result.")
            request = {"id": uuid4().hex, "candidate": candidate_id, "expectedRelease": active["id"],
                       "dataset": best["dataset"], "requestedAt": utc_now()}
            requests = self.path / "requests"
            requests.mkdir(parents=True, exist_ok=True)
            path = requests / f"{time_ns():020d}-{request['id']}.json"
            path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            return {**request, "phase": "queued"}
