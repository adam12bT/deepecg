"""Two-stage sleep classification and awakening forecasting."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from eeg_data import (
    FEATURE_NAMES, HISTORY, N_CLASSES, N_FEATURES, N_STAGES, STAGE_NAMES,
    fit_normalization, normalize,
)
from eeg_model import SleepStageClassifier, SleepStageForecastPipeline, load_checkpoint, verify_causality
from sleepops.files import sha256
from sleepops.training import detection_metrics, forecast_metrics, init_training, plot_forecast


STAGE_FOLDS = 5


def stage_weights(labels, device):
    counts = np.bincount(labels, minlength=N_STAGES)
    if np.any(counts == 0):
        raise ValueError("Every sleep stage must have training examples.")
    weights = np.sqrt(counts.sum() / (N_STAGES * counts))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_stage_epoch(model, loader, optimizer, weights, device):
    model.train()
    total = 0.0
    seen = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), labels, weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        seen += len(labels)
    return total / seen


@torch.inference_mode()
def stage_logits(model, features, device, batch_size=2048):
    model.eval()
    batches = [
        model(torch.as_tensor(features[start:start + batch_size], device=device)).cpu()
        for start in range(0, len(features), batch_size)
    ]
    return torch.cat(batches) if batches else torch.empty((0, N_STAGES))


def fit_stage_model(features, labels, epochs, config, device, validation=None, *, log=False):
    model = SleepStageClassifier(config.dropout).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)),
                        batch_size=config.batch_size, shuffle=True, generator=generator,
                        pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, fused=device.type == "cuda")
    weights = stage_weights(labels, device)
    history = []
    best_epoch = epochs
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train_stage_epoch(model, loader, optimizer, weights, device)
        if validation is None:
            history.append(train_loss)
            if log:
                print(f"stage epoch {epoch:2d} | training weighted CE {train_loss:.5f}", flush=True)
            continue
        val_x, val_y = validation
        logits = stage_logits(model, val_x, device)
        val_loss = F.cross_entropy(logits, torch.from_numpy(val_y)).item()
        history.append({"epoch": epoch, "train_ce": train_loss, "val_ce": val_loss})
        if log:
            print(f"stage epoch {epoch:2d} | training weighted CE {train_loss:.5f} | "
                  f"validation CE {val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_loss, history


@torch.inference_mode()
def stage_probabilities(model, features, device):
    return stage_logits(model, features, device).softmax(dim=-1).numpy().astype(np.float32)


def forecast_history_indices(data, forecast_rows):
    a = data.arrays
    state_lookup = {
        (str(record), int(end)): index
        for index, (record, end) in enumerate(zip(a["state_record_id"], a["state_epoch_end_sec"]))
    }
    output = np.empty((len(forecast_rows), HISTORY), dtype=np.int64)
    offsets = np.arange(HISTORY - 1, -1, -1) * 30
    for position, row in enumerate(forecast_rows):
        record = str(a["forecast_record_id"][row])
        ends = int(a["forecast_epoch_end_sec"][row]) - offsets
        try:
            output[position] = [state_lookup[(record, int(end))] for end in ends]
        except KeyError as error:
            raise ValueError("A forecast EEG window has no matching scored sleep-stage epochs.") from error
    return output


def forecast_inputs(data, split, mean, scale, state_probabilities):
    a = data.arrays
    rows = np.flatnonzero(a["forecast_split"] == split)
    indices = forecast_history_indices(data, rows)
    eeg = normalize(a["X_forecast"][rows], mean, scale)
    stages = state_probabilities[indices]
    return np.concatenate((eeg, stages), axis=-1), rows


def out_of_fold_stage_predictions(features, labels, subjects, epochs, config, device):
    subject_count = len(np.unique(subjects))
    if subject_count < 2:
        raise ValueError("Cross-fitting sleep-stage features needs at least two training subjects.")
    splits = GroupKFold(n_splits=min(STAGE_FOLDS, subject_count))
    output = np.empty((len(labels), N_STAGES), dtype=np.float32)
    for fold, (training, held_out) in enumerate(splits.split(features, labels, groups=subjects), 1):
        classifier, _, _, _ = fit_stage_model(features[training], labels[training], epochs, config, device)
        output[held_out] = stage_probabilities(classifier, features[held_out], device)
        print(f"Generated out-of-fold sleep-stage probabilities for fold {fold}.", flush=True)
    return output


def fit(data, archive_path, output_dir, config, requested_device="auto", on_epoch=None):
    a = data.arrays
    if "y_stage" not in a:
        raise ValueError("This archive has only binary sleep labels. Rerun the preprocessing notebook for stage labels.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = init_training(config.seed, requested_device)
    mean, scale = fit_normalization(data)
    normalized = normalize(a["X_state"], mean, scale)
    training = np.flatnonzero(a["state_split"] == "train")
    validation = np.flatnonzero(a["state_split"] == "validation")
    target = a["y_stage"].astype(np.int64)
    training = training[target[training] >= 0]
    validation = validation[target[validation] >= 0]
    train_x, train_y = normalized[training], target[training]
    val_x, val_y = normalized[validation], target[validation]
    stage_model, stage_epoch, stage_val_loss, stage_history = fit_stage_model(
        train_x, train_y, config.max_epochs, config, device, (val_x, val_y), log=True,
    )
    selected_stage_epochs = len(stage_history)
    if stage_history and stage_epoch < selected_stage_epochs:
        selected_stage_epochs = stage_epoch
    stage_train_probs = out_of_fold_stage_predictions(
        train_x, train_y, a["state_subject_id"][training], selected_stage_epochs, config, device,
    )
    final_stage, _, _, _ = fit_stage_model(train_x, train_y, selected_stage_epochs, config, device)
    state_probs = np.zeros((len(target), N_STAGES), dtype=np.float32)
    state_probs[training] = stage_train_probs
    predicted_rows = np.concatenate((validation, np.flatnonzero(a["state_split"] == "test")))
    state_probs[predicted_rows] = stage_probabilities(final_stage, normalized[predicted_rows], device)

    forecast_x, forecast_rows = forecast_inputs(data, "train", mean, scale, state_probs)
    forecast_y = data.forecast_classes[forecast_rows]
    val_forecast_x, val_forecast_rows = forecast_inputs(data, "validation", mean, scale, state_probs)
    val_forecast_y = data.forecast_classes[val_forecast_rows]
    dataset = DataLoader(TensorDataset(torch.from_numpy(forecast_x), torch.from_numpy(forecast_y)),
                         batch_size=config.batch_size, shuffle=True,
                         generator=torch.Generator().manual_seed(config.seed + 1),
                         pin_memory=device.type == "cuda")
    model = SleepStageForecastPipeline(config.dropout).to(device)
    model.stage_classifier.load_state_dict(final_stage.state_dict())
    for parameter in model.stage_classifier.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.forecaster.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, fused=device.type == "cuda")
    validation_tensor = torch.from_numpy(val_forecast_y).to(device)
    validation_x_tensor = torch.from_numpy(val_forecast_x).to(device)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    probe = torch.as_tensor(val_x[:3, None, :].repeat(HISTORY, axis=1), device=device)
    causal_change = verify_causality(model, probe)
    print(f"Training the awakening forecaster on {hardware}; stage classifier selected epoch {stage_epoch}.",
          flush=True)

    archive_hash = sha256(archive_path)
    best = float("inf")
    best_epoch = 0
    checkpoint = output_dir / "best_checkpoint.pt"
    history = []
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        examples = 0
        for features, labels in dataset:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.forecaster(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.forecaster.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            examples += len(labels)
        train_loss = loss_sum / examples
        model.eval()
        with torch.inference_mode():
            val_logits = torch.cat([
                model.forecaster(features).cpu()
                for features in validation_x_tensor.split(1024)
            ])
        val_loss = F.cross_entropy(val_logits, validation_tensor.cpu()).item()
        history_row = {"epoch": epoch, "train_stage_ce": stage_history[stage_epoch - 1]["train_ce"],
                       "train_forecast_ce": train_loss, "val_stage_ce": stage_val_loss,
                       "val_forecast_ce": val_loss, "val_mean": (stage_val_loss + val_loss) / 2,
                       "seconds": round(time.perf_counter() - started, 2)}
        history.append(history_row)
        print(f"{epoch:2d} | stage CE {stage_val_loss:.5f} | training forecast CE {train_loss:.5f} | "
              f"validation forecast CE {val_loss:.5f} | {history_row['seconds']:.1f} s", flush=True)
        if on_epoch is not None:
            on_epoch(history_row)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({
                "version": 2, "dropout": config.dropout, "model_state": model.state_dict(),
                "feature_names": list(FEATURE_NAMES), "stage_names": list(STAGE_NAMES),
                "feature_mean": torch.from_numpy(mean), "feature_scale": torch.from_numpy(scale),
                "stage_best_epoch": stage_epoch, "best_epoch": best_epoch, "state_threshold": 0.5,
                "validation_score": best, "archive_sha256": archive_hash,
                "test_records": a["record_id"][a["record_split"] == "test"].tolist(),
            }, temporary)
            temporary.replace(checkpoint)
        if epoch - best_epoch >= config.patience:
            break

    change = verify_causality(load_checkpoint(checkpoint, device)[0], probe)
    log = {
        "config": asdict(config), "hardware": hardware, "torch_version": str(torch.__version__),
        "precision": "float32", "stage_history": stage_history, "history": history,
        "stage_best_epoch": stage_epoch, "best_epoch": best_epoch, "best_validation_mean": best,
        "archive_sha256": archive_hash, "stage_oof_folds": min(STAGE_FOLDS, len(np.unique(a["state_subject_id"][training]))),
        "checkpoint_causality_max_change": change,
    }
    (output_dir / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return checkpoint, device, log


def evaluate(data, checkpoint, output_dir, device, *, split):
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation requires the validation or test split.")
    a = data.arrays
    if "y_stage" not in a:
        raise ValueError("This archive does not contain sleep-stage labels.")
    model, mean, scale, payload = load_checkpoint(checkpoint, device)
    if payload["version"] != 2:
        raise ValueError("This checkpoint does not contain the two-stage classifier.")
    state_rows = np.flatnonzero((a["state_split"] == split) & (a["y_stage"] >= 0))
    state_x = normalize(a["X_state"][state_rows], mean, scale)
    probabilities = stage_probabilities(model.stage_classifier, state_x, device)
    stage_y = a["y_stage"][state_rows].astype(int)
    state_y = a["y_state"][state_rows].astype(int)
    predicted_stage = probabilities.argmax(axis=1)
    predicted_state = (1 - probabilities[:, 0] >= payload["state_threshold"]).astype(int)
    train = (a["state_split"] == "train") & (a["y_stage"] >= 0)
    from sklearn.linear_model import LogisticRegression

    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(normalize(a["X_state"][train], mean, scale), a["y_state"][train])
    binary_predictions = logistic.predict(normalize(a["X_state"][state_rows], mean, scale))
    all_stage_probs = stage_probabilities(model.stage_classifier, normalize(a["X_state"], mean, scale), device)
    augmented, forecast_rows = forecast_inputs(data, split, mean, scale, all_stage_probs)
    forecast_y = data.forecast_classes[forecast_rows]
    with torch.inference_mode():
        logits = torch.cat([
            model.forecaster(torch.as_tensor(augmented[start:start + 1024], device=device)).cpu()
            for start in range(0, len(augmented), 1024)
        ])
    forecast_prob = logits.double().softmax(dim=-1).numpy()
    counts = np.bincount(data.forecast_classes[a["forecast_split"] == "train"], minlength=N_CLASSES)
    prior = counts / counts.sum()
    baseline = np.broadcast_to(prior, forecast_prob.shape)
    included = a["record_split"] == split
    result = {
        "split": split, "archive_sha256": payload["archive_sha256"], "selected_epoch": payload["best_epoch"],
        "subjects": len(set(a["record_subject_id"][included])), "recordings": int(included.sum()),
        "state_epochs": len(state_rows), "state_short_history_epochs": 0, "forecast_windows": len(forecast_rows),
        "awakenings_within_hour": int((forecast_y != N_CLASSES - 1).sum()),
        "stages": {"names": list(STAGE_NAMES),
                   "log_loss": float(F.cross_entropy(torch.from_numpy(np.log(
                       np.maximum(probabilities, np.finfo(np.float32).tiny))),
                       torch.from_numpy(stage_y)).item()),
                   "balanced_accuracy": float(balanced_accuracy_score(stage_y, predicted_stage)),
                   "macro_f1": float(f1_score(stage_y, predicted_stage, labels=np.arange(N_STAGES),
                                               average="macro", zero_division=0)),
                   "confusion_matrix": confusion_matrix(stage_y, predicted_stage, labels=np.arange(N_STAGES)).tolist(),
                   "class_counts": np.bincount(stage_y, minlength=N_STAGES).tolist()},
        "state": {"shared_model": detection_metrics(state_y, predicted_state),
                  "logistic_regression": detection_metrics(state_y, binary_predictions)},
        "forecast": {"shared_model": forecast_metrics(forecast_y, forecast_prob),
                     "training_class_distribution": forecast_metrics(forecast_y, baseline),
                     "train_class_counts": counts.tolist()},
    }
    if split == "test":
        records = a["forecast_record_id"][forecast_rows]
        record = min(records)
        locations = np.flatnonzero(records == record)
        locations = locations[np.argsort(a["forecast_epoch_end_sec"][forecast_rows[locations]])]
        index = locations[len(locations) // 2]
        row = forecast_rows[index]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_forecast(forecast_prob[index], prior, int(forecast_y[index]), record,
                            int(a["forecast_epoch_end_sec"][row]))
        fig.savefig(output_dir / "heldout_forecast.png", dpi=160)
        import matplotlib.pyplot as plt

        plt.close(fig)
        result["plot_example"] = {
            "record_id": str(record), "window_end_sec": int(a["forecast_epoch_end_sec"][row]),
            "observed_minutes": float(a["y_forecast_min"][row]), "observed_class": int(forecast_y[index]),
            "probabilities": forecast_prob[index].tolist(),
        }
    return result
