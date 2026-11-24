import numpy as np
import torch

from eeg_model import SleepWakeModel, predict_window, verify_causality


def test_model_shapes_probabilities_and_causality_without_private_assets():
    torch.manual_seed(495)
    model = SleepWakeModel().eval()
    window = np.random.default_rng(495).normal(size=(10, 11)).astype(np.float32)
    mean = np.zeros(11, dtype=np.float32)
    scale = np.ones(11, dtype=np.float32)
    sleep, forecast = predict_window(model, mean, scale, window)
    assert sleep.shape == (10,) and np.all((sleep >= 0) & (sleep <= 1))
    assert forecast.shape == (13,)
    np.testing.assert_allclose(forecast.sum(), 1.0, atol=1e-6)
    assert verify_causality(model, torch.from_numpy(window[None])) < 1e-5
