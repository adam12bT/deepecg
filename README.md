---
title: SleepOps EEG
emoji: 🌙
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# CS495 Sleep–Wake Segmentation and Awakening Forecasting from EEG

The project also runs as **SleepOps**, with tracked training, a validation-gated registry, a FastAPI service, and Prometheus monitoring. See the [MLOps guide](MLOPS.md) for commands to train, serve, promote, and roll back a release.

The [Hugging Face guide](HUGGING_FACE.md) covers the hosted app, private training-data bucket, and DVC commands for restoring the archive.

## Reproduce training in Colab

1. Open `Sleep_Wake_Training_Colab.ipynb` and upload `sleep_edf_wake_data.npz` to `/content`. Set `ARCHIVE_PATH` in the first code cell if you store it elsewhere.
2. Run all cells. Choose a GPU runtime for faster training. The notebook also runs on a CPU runtime. Its default run uses seed 495, 20 maximum epochs, and validation-based early stopping.
3. Download `artifacts/best_checkpoint.pt`, `artifacts/demo_windows.npz`, `artifacts/training_log.json`, `artifacts/test_metrics.json`, and `artifacts/heldout_forecast.png` before ending the Colab session.

To regenerate the archive, run `Sleep_EDF_Wake_Up_Prediction_Colab.ipynb` with `MAX_SUBJECTS = None`. It downloads the public Sleep Cassette recordings and exports `/content/sleep_edf_wakeup/sleep_edf_wake_data.npz`. The training notebook detects that path. The project archive was regenerated with the supplied notebook after the original export could not be located. It preserves the preprocessing notebook’s seed-42 subject assignments.

## Run the Streamlit demo

Use Python 3.11 or newer. Keep the `artifacts` directory beside `demo.py`, `eeg_data.py`, and `eeg_model.py`.

```sh
python -m pip install -r requirements.txt
python -m streamlit run demo.py
```

Select a held-out recording and window end, or upload one raw, unscaled feature window as a numeric `.npy` or a headerless `.csv`. It must have 10 rows and 11 columns. Put the oldest 30-second epoch first and end the last row at the prediction time. Use the same EEG channel and per-epoch features as the preprocessing notebook, in this column order.

```text
delta_log_power, theta_log_power, alpha_log_power, sigma_log_power, beta_log_power,
delta_relative_power, theta_relative_power, alpha_relative_power, sigma_relative_power, beta_relative_power,
log_rms
```

The demo loads the training means and scales from the checkpoint. It shows the ten sleep probabilities and the latest state decision. When the final epoch crosses the 0.5 sleep threshold, it shows the full forecast and the probability of a sustained awakening within the hour.

## Evaluation files

- `artifacts/test_metrics.json` records the held-out metrics, confusion matrices, sample counts, and plot probabilities.
- `artifacts/heldout_forecast.png` compares the forecast and training-class prior for one window from an unseen recording.
- `artifacts/training_log.json` records the training and validation losses and the selected checkpoint epoch.

Detection is evaluated once per scored epoch around the sleep period. Forecasting is evaluated on eligible windows whose final epoch is expert-scored Sleep and whose outcome can be determined from the available annotations.

## Results from the full run

The notebook ran locally on an NVIDIA GeForce RTX 3060 Laptop GPU with PyTorch 2.6.0+cu124. Validation selected epoch 4 of 9 with mean loss 1.0100. Perturbing later epochs changed earlier sleep logits by at most 0. The test split contained 16 people and 32 recordings.

| Task and method | Balanced accuracy | Macro F1 | Multiclass log loss | Within-hour Brier |
|---|---:|---:|---:|---:|
| Detection · shared model | 0.8920 | 0.9048 | | |
| Detection · logistic regression | 0.8565 | 0.8579 | | |
| Forecast · shared model | | | 1.8072 | 0.2307 |
| Forecast · training class distribution | | | 1.8551 | 0.2523 |

Detection counted 40,460 unique test epochs. Confusion matrix rows are true labels and columns are predictions, both in Wake, Sleep order.

- Shared model `[[10850, 2487], [803, 26320]]`
- Logistic regression `[[10700, 2637], [2423, 24700]]`

Forecasting was scored on 6,365 test windows. The probability plot in the full project bundle shows recording `SC4071`. The CPU path was smoke-tested end to end for one epoch using six subjects drawn from the original training split.

## Data attribution

The features and held-out demo windows are derived from Bob Kemp’s Sleep-EDF Database Expanded, version 1.0.0, made available under the Open Data Commons Attribution License v1.0. [PhysioNet dataset and citations](https://physionet.org/content/sleep-edfx/1.0.0/) · [Dataset DOI](https://doi.org/10.13026/C2X676) · [License terms](https://opendatacommons.org/licenses/by/1-0/)
