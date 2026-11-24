"""Read and compare the mounted prediction release and training candidates."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from sleepops.files import read_json
from sleepops.registry import VERSION_PATTERN, check_quality, release_path


METRICS = (
    ("Balanced accuracy", "state", "balanced_accuracy", True),
    ("Macro F1", "state", "macro_f1", True),
    ("Forecast log loss", "forecast", "multiclass_log_loss", False),
    ("Within-hour Brier score", "forecast", "within_hour_brier", False),
)
READ_ERRORS = (OSError, ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class Version:
    key: str
    path: Path
    metadata: dict
    dataset: dict
    validation: dict
    test: dict | None
    gates: dict
    url: str

    @property
    def label(self):
        kind = "Deployed" if self.key == "deployed" else "Candidate"
        return f"{kind} · {self.metadata['run_id'][:12]}"


def metric(report, task, name):
    value = report[task]["shared_model"][name]
    if not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"Invalid model metric {name}.")
    return value


def display(value, percent):
    return f"{value:.2%}" if percent else f"{value:.4f}"


def matching_report(report, split, metadata, dataset):
    return (report["split"] == split and report["archive_sha256"] == metadata["archive_sha256"]
            and report.get("run_id", metadata["run_id"]) == metadata["run_id"]
            and report["subjects"] == dataset["splits"][split]["subjects"])


class ModelCatalog:
    def __init__(self, candidates, registry, reports, bucket, prediction_space):
        self.candidates = candidates
        self.registry = registry
        self.reports = reports
        self.bucket = bucket
        self.prediction_space = prediction_space

    def version(self, key):
        deployed = key == "deployed"
        if deployed:
            run_id = read_json(self.registry / "champion.json")["release_id"]
            path = release_path(self.registry, run_id)
            url = f"https://huggingface.co/spaces/{self.prediction_space}/tree/main/registry/releases/{run_id}"
            test_path = self.reports / "test_metrics.json"
        else:
            if not isinstance(key, str) or not VERSION_PATTERN.fullmatch(key):
                raise ValueError("Choose a model from the catalog.")
            run_id = key
            path = self.candidates / key
            if read_json(path / "cloud_run.json")["run_id"] != run_id:
                raise ValueError("The candidate has no matching completion record.")
            url = f"https://huggingface.co/buckets/{self.bucket}/tree/training-runs/{run_id}"
            test_path = path / "test_metrics.json"

        metadata = read_json(path / "metadata.json")
        dataset = read_json(path / "dataset.json")
        validation = read_json(path / "validation_metrics.json")
        if (metadata["run_id"] != run_id or metadata["archive_sha256"] != dataset["archive_sha256"]
                or not matching_report(validation, "validation", metadata, dataset)):
            raise ValueError("The evaluation report and checkpoint provenance disagree.")
        gates = check_quality(validation, read_json(Path("configs/promotion.json")))
        test = None
        if test_path.is_file():
            report = read_json(test_path)
            if (report.get("run_id") == run_id and matching_report(report, "test", metadata, dataset)
                    and report.get("selected_epoch") == validation["selected_epoch"]):
                test = report
        return Version(key, path, metadata, dataset, validation, test, gates, url)

    def versions(self):
        releases, candidates = [], []
        try:
            releases.append(self.version("deployed"))
        except READ_ERRORS:
            pass
        if self.candidates.is_dir():
            for entry in self.candidates.iterdir():
                if VERSION_PATTERN.fullmatch(entry.name) and (entry / "cloud_run.json").is_file():
                    try:
                        candidates.append(self.version(entry.name))
                    except READ_ERRORS:
                        continue
        return releases + sorted(candidates, key=lambda item: item.metadata["created_at"], reverse=True)


def model_rows(versions):
    return [[version.metadata["run_id"], "Deployed" if version.key == "deployed" else "Candidate",
             version.validation["selected_epoch"], "Passed" if version.gates["passed"] else "Review",
             display(metric(version.validation, "state", "balanced_accuracy"), True),
             display(metric(version.validation, "forecast", "multiclass_log_loss"), False),
             "Available" if version.test else "—"] for version in versions]


def details(version):
    validation, test = version.validation, version.test
    rows = [[label, display(metric(validation, task, key), percent),
             display(metric(test, task, key), percent) if test else "—"] for label, task, key, percent in METRICS]
    status = "Passed all validation gates." if version.gates["passed"] else (
        "Review validation gates for " + ", ".join(name.replace("_", " ")
                                                   for name, passed in version.gates["checks"].items() if not passed) + ".")
    return (f"**{version.label}** · Selected epoch {validation['selected_epoch']} · "
            f"Dataset `{version.metadata['archive_sha256'][:12]}`\n\n{status} "
            f"[Open checkpoint and reports]({version.url})", rows)


def comparison(left, right, split, anchor=None):
    if split not in {"validation", "test"}:
        raise ValueError("Choose a supported evaluation split.")
    a, b = getattr(left, split), getattr(right, split)
    left_hash, right_hash = (version.dataset["archive_sha256"] for version in (left, right))
    same_source = left_hash == right_hash
    if not same_source and anchor:
        left_anchor = anchor(left_hash)
        same_source = bool(left_anchor and left_anchor == anchor(right_hash))
    same_cohort = same_source and left.dataset["splits"][split] == right.dataset["splits"][split]
    comparable = bool(a and b and same_cohort and all(
        a[name] == b[name] for name in ("subjects", "state_epochs", "forecast_windows")))
    if comparable:
        message = f"Same {split} split with {a['subjects']} subjects. Difference is B minus A."
    elif not a or not b:
        missing = " and ".join(label for label, report in (("A", a), ("B", b)) if not report)
        message = f"Evaluation report unavailable for model {missing}."
    else:
        message = "Different datasets or evaluation cohorts. Compare the individual scores in context."

    rows = []
    for label, task, key, percent in METRICS:
        first = metric(a, task, key) if a else None
        second = metric(b, task, key) if b else None
        difference, better = "—", "—"
        if comparable:
            delta = round((second - first) * 100, 2) if percent else round(second - first, 4)
            difference = f"{delta:+.2f} pp" if percent else f"{delta:+.4f}"
            better = "Tie" if not delta else "B" if (delta > 0) == percent else "A"
        rows.append([label, display(first, percent) if a else "—", display(second, percent) if b else "—",
                     difference, better])
    return message, rows, comparable


def compare(left, right, split, anchor=None):
    message, rows, _ = comparison(left, right, split, anchor)
    return f"**A** {left.label} · **B** {right.label}  \n{message}", rows


def check_checkpoint(version):
    import numpy as np
    import torch

    from eeg_model import HISTORY, N_CLASSES, N_FEATURES, predict_window, verify_causality
    from sleepops.registry import load_bundle

    torch.set_num_threads(1)
    model, mean, scale, payload, _, _ = load_bundle(version.path)
    state, forecast = predict_window(model, mean, scale, np.tile(mean, (HISTORY, 1)))
    if (state.shape != (HISTORY,) or forecast.shape != (N_CLASSES,) or not np.isfinite(state).all()
            or not np.isfinite(forecast).all() or not np.isclose(forecast.sum(), 1.0)):
        raise ValueError("Unexpected inference output.")
    delta = verify_causality(model, torch.zeros(1, HISTORY, N_FEATURES))
    return (f"**Check passed for {version.label}.** Checksums and normalization verified. "
            f"CPU inference returned {len(state)} sleep probabilities and {len(forecast)} forecast probabilities. "
            f"Sleep threshold {payload['state_threshold']:.2f}. Causality check difference {delta:.2g}.")
