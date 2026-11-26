"""Shared-model fitting and split-specific evaluation for SleepOps."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
from itertools import zip_longest
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from eeg_data import (
    CLASS_LABELS, FEATURE_NAMES, HISTORY, N_CLASSES, evaluation_histories, fit_normalization, normalize,
)
from eeg_model import SleepWakeModel, load_checkpoint, verify_causality


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 495
    max_epochs: int = 20
    patience: int = 5
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    dropout: float = 0.1


def init_training(seed, requested_device="auto"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested_device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Keep float32 evaluation comparable across full windows and short prefixes.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def training_loaders(data, mean, scale, config, device):
    a = data.arrays
    state_indices = data.state_windows["train"]
    state_x = normalize(a["X_state"][state_indices], mean, scale)
    state_y = a["y_state"][state_indices].astype(np.float32)
    forecast = a["forecast_split"] == "train"
    forecast_x = normalize(a["X_forecast"][forecast], mean, scale)
    forecast_y = data.forecast_classes[forecast]
    # Similar numbers of batches allow both losses on almost every optimizer step.
    forecast_batch_size = max(1, math.ceil(len(forecast_y) / math.ceil(len(state_y) / config.batch_size)))
    shuffle_loader = partial(DataLoader, shuffle=True, pin_memory=device.type == "cuda")
    state = shuffle_loader(
        TensorDataset(torch.from_numpy(state_x), torch.from_numpy(state_y)),
        batch_size=config.batch_size, generator=torch.Generator().manual_seed(config.seed),
    )
    forecast = shuffle_loader(
        TensorDataset(torch.from_numpy(forecast_x), torch.from_numpy(forecast_y)),
        batch_size=forecast_batch_size, generator=torch.Generator().manual_seed(config.seed + 1),
    )
    print(f"Train batches: state {len(state)} × up to {config.batch_size}, "
          f"forecast {len(forecast)} × up to {forecast_batch_size}")
    return state, forecast


def train_epoch(model, loaders, optimizer, device, use_bfloat16):
    model.train()
    sums = torch.zeros(2, device=device)
    counts = [0, 0]
    for state, forecast in zip_longest(*loaders):
        batches = [batch[0] for batch in (state, forecast) if batch is not None]
        x = torch.cat(batches).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
        with precision:
            state_logits, forecast_logits = model(x)
            losses = []
            state_count = 0
            if state is not None:
                labels = state[1].to(device, non_blocking=True)
                state_count = len(labels)
                loss = F.binary_cross_entropy_with_logits(state_logits[:state_count], labels)
                losses.append(loss)
                sums[0] += loss.detach() * labels.numel()
                counts[0] += labels.numel()
            if forecast is not None:
                labels = forecast[1].to(device, non_blocking=True)
                loss = F.cross_entropy(forecast_logits[state_count:], labels)
                losses.append(loss)
                sums[1] += loss.detach() * labels.numel()
                counts[1] += labels.numel()
            joint_loss = torch.stack(losses).mean()
        joint_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return tuple((sums.cpu() / torch.tensor(counts)).tolist())


def evaluation_inputs(data, split, mean, scale):
    a = data.arrays
    indices, last, rows = evaluation_histories(data, split)
    state_x = normalize(a["X_state"][indices], mean, scale)
    state_x[np.arange(HISTORY) > last[:, None]] = 0
    frows = np.flatnonzero(a["forecast_split"] == split)
    return state_x, last, rows, normalize(a["X_forecast"][frows], mean, scale), frows


@torch.inference_mode()
def evaluation_logits(model, inputs, device, batch_size=1024):
    model.eval()
    state_x, last, _, forecast_x, _ = inputs
    state, forecast = [], []
    for start in range(0, len(state_x), batch_size):
        stop = start + batch_size
        x = torch.as_tensor(state_x[start:stop], device=device)
        logits, _ = model(x)
        positions = torch.as_tensor(last[start:stop], device=device)
        state.append(logits.gather(1, positions[:, None]).squeeze(1).cpu())
    for start in range(0, len(forecast_x), batch_size):
        x = torch.as_tensor(forecast_x[start:start + batch_size], device=device)
        _, logits = model(x)
        forecast.append(logits.cpu())
    return torch.cat(state), torch.cat(forecast)


def fit(data, archive_path, output_dir, config, requested_device="auto", on_epoch=None):
    if "y_stage" in data.arrays:
        from sleepops.stage_pipeline import fit as fit_two_stage

        return fit_two_stage(data, archive_path, output_dir, config, requested_device, on_epoch)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = init_training(config.seed, requested_device)
    mean, scale = fit_normalization(data)
    loaders = training_loaders(data, mean, scale, config, device)
    val_inputs = evaluation_inputs(data, "validation", mean, scale)
    a = data.arrays
    val_state_y = torch.as_tensor(a["y_state"][val_inputs[2]], dtype=torch.float32)
    val_forecast_y = torch.from_numpy(data.forecast_classes[val_inputs[4]])
    model = SleepWakeModel(dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                  fused=device.type == "cuda")
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Training on {hardware}, {'BF16 autocast' if use_bfloat16 else 'float32'}, "
          f"{sum(t.numel() for t in model.parameters()):,} trainable parameters")
    example = torch.as_tensor(val_inputs[0][val_inputs[1] == HISTORY - 1][:3], device=device)
    initial_change = verify_causality(model, example)
    print(f"Initial causal suffix test: max earlier-logit change {initial_change:.3g}")

    best = float("inf")
    best_epoch = 0
    checkpoint = output_dir / "best_checkpoint.pt"
    archive_hash = sha256_file(archive_path)
    history = []
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        train_state, train_forecast = train_epoch(model, loaders, optimizer, device, use_bfloat16)
        state_logits, forecast_logits = evaluation_logits(model, val_inputs, device)
        val_state = F.binary_cross_entropy_with_logits(state_logits, val_state_y).item()
        val_forecast = F.cross_entropy(forecast_logits, val_forecast_y).item()
        score = (val_state + val_forecast) / 2
        if not np.isfinite([train_state, train_forecast, score]).all():
            raise FloatingPointError("Training or validation loss became nonfinite.")
        history.append({"epoch": epoch, "train_bce": train_state, "train_ce": train_forecast,
                        "val_bce": val_state, "val_ce": val_forecast, "val_mean": score,
                        "seconds": round(time.perf_counter() - started, 2)})
        print(f"{epoch:2d} | training BCE {train_state:.5f} CE {train_forecast:.5f} | "
              f"validation BCE {val_state:.5f} CE {val_forecast:.5f} mean {score:.5f} | "
              f"{history[-1]['seconds']:.1f} s", flush=True)
        if on_epoch is not None:
            on_epoch(history[-1])
        if score < best:
            best, best_epoch = score, epoch
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({
                "version": 1, "dropout": config.dropout, "model_state": model.state_dict(),
                "feature_names": list(FEATURE_NAMES), "feature_mean": torch.from_numpy(mean),
                "feature_scale": torch.from_numpy(scale), "state_threshold": 0.5,
                "best_epoch": best_epoch, "validation_score": best, "archive_sha256": archive_hash,
                "test_records": a["record_id"][a["record_split"] == "test"].tolist(),
            }, temporary)
            temporary.replace(checkpoint)
        if epoch - best_epoch >= config.patience:
            print(f"Early stopping after {config.patience} epochs without validation improvement.")
            break

    model, _, _, _ = load_checkpoint(checkpoint, device=device)
    change = verify_causality(model, example)
    print(f"Selected epoch {best_epoch}, validation mean {best:.5f}; "
          f"checkpoint causal suffix test max earlier-logit change {change:.3g}")
    log = {
        "config": asdict(config), "hardware": hardware, "torch_version": str(torch.__version__),
        "precision": "bfloat16 autocast" if use_bfloat16 else "float32", "history": history,
        "best_epoch": best_epoch, "best_validation_mean": best, "archive_sha256": archive_hash,
        "checkpoint_causality_max_change": change,
    }
    (output_dir / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return checkpoint, device, log


def detection_metrics(y, predicted):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def forecast_metrics(y, probabilities):
    return {
        "multiclass_log_loss": float(log_loss(y, probabilities, labels=np.arange(N_CLASSES))),
        "within_hour_brier": float(np.mean(((1 - probabilities[:, -1]) - (y != N_CLASSES - 1)) ** 2)),
    }


def plot_forecast(probabilities, prior, true_class, record, end_seconds):
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    bins = np.arange(N_CLASSES)
    ax.axvspan(true_class - 0.48, true_class + 0.48, color="#d0ecda", label="Observed class")
    ax.bar(bins - 0.18, probabilities, 0.36, color="#2359a1", label="CNN–Transformer")
    ax.bar(bins + 0.18, prior, 0.36, color="#8396a5", label="Training class distribution")
    ax.set(xticks=bins, xticklabels=CLASS_LABELS, ylabel="Probability",
           title=f"Held-out recording {record} · prediction at {end_seconds / 60:g} minutes")
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    return fig


def evaluate(data, checkpoint, output_dir, device, *, split):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["version"] == 2:
        from sleepops.stage_pipeline import evaluate as evaluate_two_stage

        return evaluate_two_stage(data, checkpoint, output_dir, device, split=split)
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation requires the validation or test split.")
    a = data.arrays
    model, mean, scale, payload = load_checkpoint(checkpoint, device)
    output_dir = Path(output_dir)
    inputs = evaluation_inputs(data, split, mean, scale)
    state_rows, frows = inputs[2], inputs[4]
    state_logits, forecast_logits = evaluation_logits(model, inputs, device)
    y_state = a["y_state"][state_rows]
    state_pred = (state_logits.sigmoid().numpy() >= payload["state_threshold"]).astype(int)
    train = a["state_split"] == "train"
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(normalize(a["X_state"][train], mean, scale), a["y_state"][train])
    baseline_state = logistic.predict(normalize(a["X_state"][state_rows], mean, scale))

    y_forecast = data.forecast_classes[frows]
    p_forecast = forecast_logits.double().softmax(dim=-1).numpy()
    train_counts = np.bincount(data.forecast_classes[a["forecast_split"] == "train"], minlength=N_CLASSES)
    prior = train_counts / train_counts.sum()
    baseline_forecast = np.broadcast_to(prior, p_forecast.shape)

    included = a["record_split"] == split
    results = {
        "split": split,
        "archive_sha256": payload["archive_sha256"], "selected_epoch": payload["best_epoch"],
        "subjects": len(set(a["record_subject_id"][included])),
        "recordings": int(included.sum()),
        "state_epochs": int(len(state_rows)), "state_short_history_epochs": int((inputs[1] < HISTORY - 1).sum()),
        "forecast_windows": int(len(frows)),
        "awakenings_within_hour": int((y_forecast != N_CLASSES - 1).sum()),
        "state": {"shared_model": detection_metrics(y_state, state_pred),
                  "logistic_regression": detection_metrics(y_state, baseline_state)},
        "forecast": {"shared_model": forecast_metrics(y_forecast, p_forecast),
                     "training_class_distribution": forecast_metrics(y_forecast, baseline_forecast)},
        "forecast_train_class_counts": train_counts.tolist(),
    }
    if split == "test":
        # Choose by recording ID and time before considering the predictions.
        records = a["forecast_record_id"][frows]
        record = min(records)
        locations = np.flatnonzero(records == record)
        locations = locations[np.argsort(a["forecast_epoch_end_sec"][frows[locations]])]
        i = locations[len(locations) // 2]
        row = frows[i]
        end = int(a["forecast_epoch_end_sec"][row])
        fig = plot_forecast(p_forecast[i], prior, int(y_forecast[i]), record, end)
        fig.savefig(output_dir / "heldout_forecast.png", dpi=160)
        plt.close(fig)
        results["plot_example"] = {
            "record_id": str(record), "window_end_sec": end, "observed_minutes": float(a["y_forecast_min"][row]),
            "observed_class": int(y_forecast[i]), "probabilities": p_forecast[i].tolist(),
        }
    return results
