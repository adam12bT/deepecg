# Private SleepOps training Space

The private React admin dashboard lists the deployed release and completed candidates. The model check validates bundle checksums and normalization, then runs synthetic CPU inference and a causality check. Use **Compare models** to compare matched validation cohorts or held-out reports. **Model behavior** replays a selected checkpoint on consecutive windows from held-out test recordings and compares predicted stages against scored labels; awakening probabilities are shown only when the model predicts sleep. The **Training** view can use the DVC-pinned base archive or a version with five-class annotations. It saves each selected checkpoint, normalization, loss history, and evaluation reports under `training-runs/` in the private bucket. The complete stage-aware checkpoint is also tracked through `stage_aware_model.pt.dvc`. Production promotion follows the existing validation-gated release workflow. The separate Gradio SDK option uses the same trainer and model catalog.

From the SleepOps project root, use an authenticated Hugging Face CLI session with the `huggingface_hub` version in `deploy/huggingface/requirements.txt`.

```sh
python -m deploy.training_space.publish publish \
  --space adamnouassida5/sleepops-training \
  --bucket adamnouassida5/sleepops-eeg-data
```

CPU Basic requires a PRO account for a new Gradio Space. An eligible free account with an available ZeroGPU Space slot can add `--hardware zero-a10g`. The app trains on CPU on both configurations. [Hugging Face compute Space requirements](https://huggingface.co/docs/hub/spaces-overview#creating-a-new-space)

To replace an existing authorized Docker Space on CPU Basic, use `reuse-docker` with its `--space` ID and the same `--bucket` flag. The deployment preserves the previous commit on a rollback branch and verifies that the Admin Space remains private. Add `--prediction-space adamnouassida5/assistance` to either deployment command to mount the prediction Space's registry and reports read-only for model comparison; the prediction Space can be public or private.

The publisher mounts the `dvc-store` bucket subfolder read-only and `training-runs` read/write. Hugging Face supplies those volumes. Completed candidates persist across Space restarts. A run interrupted by a Space restart can be started again from the app.

## Explore and extend the dataset

Open **Data** to review the split sizes, state and forecast distributions, and 30-second EEG feature values for a recording. Select a base version and upload a feature NPZ exported by `Sleep_EDF_Wake_Up_Prediction_Colab.ipynb`. Use stable subject and recording IDs. Existing recordings are skipped, and new recordings for current validation or test subjects are rejected. The uploaded additions go to training.

Each saved version writes its dataset lock and `.dvc` pointer to `dataset-versions/releases/<archive-sha256>/` in the private bucket. The corresponding DVC cache object lives under `dataset-versions/files/md5/`. The Training view lets you choose the exact dataset version for the next run. The stage-aware pipeline checkpoint and its manifest are tracked separately in `stage_aware_model.pt.dvc` and `configs/stage_aware_model.json`; it remains a candidate until validation gates pass.

## Notebook

The **Notebook** tab runs Python cells in a browser worker with NumPy, pandas, and Matplotlib available. It imports and exports Jupyter `.ipynb` files and keeps the editable draft in browser storage where available. Kernel memory and attached files last for the current page session. Use **Open in notebook** in the recording explorer to populate `recording`. The selected dataset manifest is available as `dataset`. **Attach dataset file** makes the verified archive available at `dataset_path` for NumPy analysis. Use the hosted Training view or the linked Colab notebook for PyTorch training.
