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

The project also runs as **SleepOps**, with a React EEG dashboard, tracked training, a validation-gated registry, a FastAPI service, and Prometheus monitoring. See the [MLOps guide](MLOPS.md) for commands to train, serve, promote, and roll back a release.

The [Hugging Face guide](HUGGING_FACE.md) covers the hosted app, private training-data bucket, and DVC commands for restoring the archive.

## Train the two-stage model

The checked-in archive is schema v1 and has binary wake/sleep labels only. It cannot train the detailed sleep-stage classifier until the expert stage labels are added.

1. Open `Sleep_EDF_Wake_Up_Prediction_Colab.ipynb`. Its default path asks for the existing schema-v1 archive, downloads the expert hypnograms, and adds five-class labels. It preserves the existing EEG features, forecast targets, recording metadata, and subject splits. Download the schema-v2 archive from Colab.
2. For Colab training, open `Sleep_Stage_Awakening_Training_Colab.ipynb`, upload the new archive, and run all cells. Choose a GPU runtime for faster training; CPU is supported. The notebook streams losses and downloads the checkpoint with training, validation, test, and probability-plot reports.
3. For hosted training, open **Data** and add the schema-v2 archive as a new version of the original dataset. Choose a local `.npz` file or import it using a direct public HTTPS download link. Then use **Train a model** and select the version marked **five sleep stages**. The resulting checkpoint is a candidate until it passes validation gates and is deployed.

Set `UPGRADE_EXISTING_ARCHIVE = False` only when rebuilding EEG features from raw EDF recordings. That route downloads the public PSG files as well as hypnograms.

## Run the React dashboard

Start the bundled dashboard and API with `docker compose up --build --wait`, then open `http://127.0.0.1:8501`. For local development, use Node 24 and Python 3.11 or newer. Start the API from the project root.

```sh
python -m pip install -e '.[api]'
python -m uvicorn sleepops.api:app --host 127.0.0.1 --port 8000
```

In a second terminal, run `npm ci --prefix frontend` and `npm run dev --prefix frontend`. In PowerShell, use `npm.cmd` if script execution policy blocks `npm`.

Select a held-out recording and window end, or upload one raw, unscaled feature window as a numeric `.npy` or a headerless `.csv`. It must have 10 rows and 11 columns. Put the oldest 30-second epoch first and end the last row at the prediction time. Use the same EEG channel and per-epoch features as the preprocessing notebook, in this column order.

```text
delta_log_power, theta_log_power, alpha_log_power, sigma_log_power, beta_log_power,
delta_relative_power, theta_relative_power, alpha_relative_power, sigma_relative_power, beta_relative_power,
log_rms
```

The demo loads the training means and scales from the checkpoint. A v2 checkpoint also shows the five-class stage timeline. When the final epoch crosses the 0.5 sleep threshold, it shows the forecast and the probability of a sustained awakening within the hour. Every inference window contains only ten completed epochs ending at prediction time.

## Evaluation files

- `artifacts/test_metrics.json` records the held-out metrics, confusion matrices, sample counts, and plot probabilities.
- `artifacts/heldout_forecast.png` compares the forecast and training-class prior for one window from an unseen recording.
- `artifacts/training_log.json` records the training and validation losses and the selected checkpoint epoch.

Detection is evaluated once per scored epoch around the sleep period. Forecasting is evaluated on eligible windows whose final epoch is expert-scored Sleep and whose outcome can be determined from the available annotations.

## Existing binary-model baseline

These actual results are from the existing v1 binary CNN–Transformer, not from the new stage-aware model. The run used an NVIDIA GeForce RTX 3060 Laptop GPU with PyTorch 2.6.0+cu124. Validation selected epoch 4 of 9 with mean loss 1.0100. Perturbing later epochs changed earlier sleep logits by at most 0. The test split contained 16 people and 32 recordings.

| Task and method | Balanced accuracy | Macro F1 | Multiclass log loss | Within-hour Brier |
|---|---:|---:|---:|---:|
| Detection · shared model | 0.8920 | 0.9048 | | |
| Detection · logistic regression | 0.8565 | 0.8579 | | |
| Forecast · shared model | | | 1.8072 | 0.2307 |
| Forecast · training class distribution | | | 1.8551 | 0.2523 |

Detection counted 40,460 unique test epochs. Confusion matrix rows are true labels and columns are predictions, both in Wake, Sleep order.

- Shared model `[[10850, 2487], [803, 26320]]`
- Logistic regression `[[10700, 2637], [2423, 24700]]`

Forecasting was scored on 6,365 test windows. The probability plot in the project bundle shows recording `SC4071`. The CPU path for v1 was smoke-tested for one epoch using six subjects from the original training split. The stage-aware v2 model has not been trained or evaluated yet; its held-out metrics will be produced when the training notebooks or Admin trainer run on a schema-v2 archive.

## Data attribution

The features and held-out demo windows are derived from Bob Kemp’s Sleep-EDF Database Expanded, version 1.0.0, made available under the Open Data Commons Attribution License v1.0. [PhysioNet dataset and citations](https://physionet.org/content/sleep-edfx/1.0.0/) · [Dataset DOI](https://doi.org/10.13026/C2X676) · [License terms](https://opendatacommons.org/licenses/by/1-0/)
