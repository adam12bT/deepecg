# SleepOps on Hugging Face

The deployment uses a private CPU Docker Space for the React dashboard and FastAPI service. DVC versions the prepared EEG training archive and the promoted checkpoint. A separate private Hugging Face Storage Bucket holds both files. The Space upload contains their DVC pointers and checksums, the source code, the serving checkpoint, and the held-out demo windows.

## Deploy from this repository

Run these commands with Python 3.11 and a Hugging Face login authorized to write to the intended Space and bucket. Creating a personal Docker Space currently requires Hugging Face PRO. The deployment selects CPU Basic hardware. [Hugging Face Spaces plans and hardware](https://huggingface.co/docs/hub/en/spaces-overview#creating-a-new-space)

```sh
python -m pip install -r deploy/huggingface/requirements.txt
hf auth login
python deploy/huggingface/release.py deploy --wait-seconds 900
```

The default resources are `sleepops-cs495` and `sleepops-eeg-data` in your Hugging Face namespace. Pass `--space NAMESPACE/SPACE` or `--bucket NAMESPACE/BUCKET` for another destination. You can use `publish-data`, `publish-model`, and `publish-space` separately. The deployment restores both DVC objects through a fresh cache and compares their SHA-256 hashes. It verifies that the running Space has the new build ID, active checkpoint checksum, and training archive checksum, and that a prediction succeeds.

Open the Space page to use the demo. On the Space app host, API endpoints are prefixed with `/api`, including `/api/docs`, `/api/health/ready`, `/api/v1/predict`, and `/api/metrics`. The model is loaded from the shipped registry release when the app starts. Training and MLflow commands remain available in the repository. See the [MLOps guide](MLOPS.md).

## Continuous deployment

GitHub Actions runs `.github/workflows/ci.yml` from the source repository. For public GitHub hosting, create a source export with `python deploy/huggingface/release.py stage --github-source --output PATH_TO_EMPTY_DIRECTORY`. The export includes code, DVC pointers, release metadata, and notebooks with cleared outputs. Pull requests run code tests with synthetic data. On the deployment branch, the trusted job restores the private DVC files, regenerates the demo windows, runs the full tests, and smoke-tests the containers before publishing the Space.

The intended GitHub source is `adam12bT/deepecg`. Configure the following repository settings.

| Name | Repository setting | Value |
|---|---|---|
| `SLEEPOPS_DEPLOY_BRANCH` | Actions variable | `sleepops` for a separate deployment branch. Defaults to `main`. |
| `SLEEPOPS_HF_SPACE` | Actions variable | `adamnouassida5/assistance` after the owner approves reusing it |
| `SLEEPOPS_HF_BUCKET` | Actions variable | `adamnouassida5/sleepops-eeg-data` |
| `HF_DEPLOY_TOKEN` | Actions secret | Hugging Face token with access to write the private Space and bucket |

The first deployment seeds the private bucket with the local archive and checkpoint. For later releases, seed new DVC objects before pushing updated pointers and locks to the deployment branch. The job restores and validates the exact dataset and checkpoint, re-publishes their verified DVC objects, updates the Docker Space, and waits for live checks. [GitHub Actions secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets)

### Cloud training

Run **SleepOps cloud training** on the deployment branch from the GitHub Actions tab. The manual launch button appears when the workflow exists on the repository's default branch. Cloud training reuses the bucket setting and `HF_DEPLOY_TOKEN` above. It restores the DVC-pinned archive and saves each candidate's weights, normalizer, loss history, validation comparison, and optional held-out test report in `training-runs/<run-id>/` in the private bucket. The production checkpoint is versioned independently through its existing DVC pointer and release workflow. [GitHub manual workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)

### Reuse the existing `assistance` Space

The following command is for the owner-approved takeover of the existing app. It preserves the old remote revision on a `before-sleepops-…` branch, verifies private visibility before uploading, and replaces the working tree in one commit. The commit is conditional on the old revision still being the head. To roll back in a clone of the Space, revert the SleepOps deployment commit and push the revert to `main`.

```sh
python deploy/huggingface/release.py deploy --space adamnouassida5/assistance --bucket adamnouassida5/sleepops-eeg-data --reuse-existing --make-private --wait-seconds 900
```

## Restore the dataset and best model

After cloning the Space repository and signing in, the Hugging Face CLI login is enough for this helper. It downloads the pinned DVC object and uses `dvc pull` against a fresh local cache before moving the verified archive into the checkout.

```sh
python deploy/huggingface/release.py pull-data --bucket NAMESPACE/sleepops-eeg-data
python deploy/huggingface/release.py pull-model --bucket NAMESPACE/sleepops-eeg-data
python -m sleepops validate --lock configs/dataset.json
```

For direct `dvc pull` or `dvc push` against the S3 gateway, generate Hugging Face S3 credentials in the account’s access-token settings and load them into `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. The credentials are supplied by the shell and kept out of the repository. In PowerShell, use the provided client config.

```powershell
$env:AWS_CONFIG_FILE = (Resolve-Path deploy/huggingface/aws-config).Path
dvc pull sleep_edf_wake_data.npz.dvc best_model.pt.dvc
```

For existing S3 credentials with write access to the bucket, use `dvc push sleep_edf_wake_data.npz.dvc best_model.pt.dvc`. After promoting a new model, run `python deploy/huggingface/release.py pin-model` before deploying it. This records the promoted release ID, selected epoch, checkpoint checksum, and training archive checksum in `configs/best_model.json`. A restored `best_model.pt` includes the model weights and training normalizer. When using a different bucket, run `python deploy/huggingface/release.py configure --bucket NAMESPACE/BUCKET` to update `.dvc/config`.

The archived features are derived from Bob Kemp’s Sleep-EDF Database Expanded v1.0.0 under the Open Data Commons Attribution License v1.0. [Dataset and citations](https://physionet.org/content/sleep-edfx/1.0.0/) · [License](https://opendatacommons.org/licenses/by/1-0/) · [Hugging Face DVC and S3 guide](https://huggingface.co/docs/hub/storage-buckets-s3)
