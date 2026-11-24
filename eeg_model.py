"""PyTorch backbone, task heads, and checkpoint inference for the EEG pipeline."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eeg_data import FEATURE_NAMES, HISTORY, N_CLASSES, N_FEATURES, normalize


class CausalEncoderLayer(nn.Module):
    def __init__(self, width=64, heads=4, feedforward=128, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attn_out = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(feedforward, width),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, steps, width = x.shape
        qkv = self.qkv(self.attn_norm(x)).reshape(batch, steps, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, steps, width)
        x = x + self.residual_dropout(self.attn_out(attention))
        return x + self.residual_dropout(self.ff(self.ff_norm(x)))


class SleepWakeModel(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        width = 64
        kernel = 3
        self.left_padding = kernel - 1
        self.conv = nn.Conv1d(N_FEATURES, width, kernel_size=kernel)
        self.position = nn.Embedding(HISTORY, width)
        self.encoder = nn.Sequential(*(CausalEncoderLayer(dropout=dropout) for _ in range(2)))
        self.final_norm = nn.LayerNorm(width)
        self.state_head = nn.Linear(width, 1)
        self.forecast_head = nn.Linear(width, N_CLASSES)

    def encode(self, x):
        if x.ndim != 3 or x.shape[-1] != N_FEATURES or not 1 <= x.shape[1] <= HISTORY:
            raise ValueError(f"Expected (batch, 1 to {HISTORY}, {N_FEATURES}) EEG features.")
        x = F.pad(x.transpose(1, 2), (self.left_padding, 0))
        x = F.gelu(self.conv(x)).transpose(1, 2)
        x = x + self.position(torch.arange(x.shape[1], device=x.device))
        return self.final_norm(self.encoder(x))

    def forward(self, x):
        embeddings = self.encode(x)
        state_logits = self.state_head(embeddings).squeeze(-1)
        forecast_logits = self.forecast_head(embeddings[:, -1, :])
        return state_logits, forecast_logits


@torch.inference_mode()
def verify_causality(model, x):
    """Perturb each possible suffix and compare untouched prefix predictions."""
    was_training = model.training
    model.eval()
    try:
        state, forecast = model(x)
        embeddings = model.encode(x)
        assert embeddings.shape == (len(x), HISTORY, 64)
        assert state.shape == (len(x), HISTORY) and forecast.shape == (len(x), N_CLASSES)
        max_change = 0.0
        for cut in range(1, HISTORY):
            modified = x.clone()
            modified[:, cut:] = -7.0 * x[:, cut:] + 3.0
            changed, _ = model(modified)
            torch.testing.assert_close(state[:, :cut], changed[:, :cut], rtol=1e-5, atol=1e-5)
            max_change = max(max_change, (state[:, :cut] - changed[:, :cut]).abs().max().item())
            prefix_state, prefix_forecast = model(x[:, :cut])
            torch.testing.assert_close(state[:, :cut], prefix_state, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(
                model.forecast_head(embeddings[:, cut - 1]), prefix_forecast, rtol=1e-5, atol=1e-5,
            )
        return max_change
    finally:
        model.train(was_training)


def load_checkpoint(path: str | Path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["version"] != 1 or tuple(payload["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Unsupported checkpoint version or feature order.")
    mean = payload["feature_mean"].numpy()
    scale = payload["feature_scale"].numpy()
    if (mean.shape != (N_FEATURES,) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid checkpoint normalization statistics.")
    model = SleepWakeModel(dropout=payload["dropout"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval(), mean, scale, payload


@torch.inference_mode()
def predict_window(model, mean, scale, window):
    raw = np.asarray(window, dtype=np.float32)
    if raw.shape != (HISTORY, N_FEATURES) or not np.isfinite(raw).all():
        raise ValueError(f"Provide a finite {HISTORY} × {N_FEATURES} raw feature matrix.")
    with np.errstate(over="ignore", invalid="ignore"):
        x = normalize(raw, mean, scale)
    if not np.isfinite(x).all():
        raise ValueError("Feature values are outside the supported numeric range.")
    batch = torch.as_tensor(x[None], device=next(model.parameters()).device)
    state, forecast = model(batch)
    return state[0].sigmoid().cpu().numpy(), forecast[0].softmax(dim=-1).cpu().numpy()
