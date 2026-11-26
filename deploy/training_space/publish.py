"""Provision a separate private Gradio training Space and upload its source."""

import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, SpaceHardware, Volume
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import validate_repo_id

from deploy.huggingface.release import (ARCHIVE, DATASET_PREFIX, PREFIX, ROOT, STAGE_MODEL, STAGE_MODEL_LOCK,
                                        checked_repo_id, publish_space, space_target)


APP = "deploy/training_space/app.py"
SOURCE_FILES = (APP, "deploy/training_space/models.py", "deploy/training_space/datasets.py",
                "deploy/huggingface/cloud_train.py", "deploy/huggingface/release.py",
                "eeg_data.py", "eeg_model.py", "configs/dataset.json", "configs/train.toml",
                "configs/promotion.json", str(STAGE_MODEL_LOCK), f"{ARCHIVE}.dvc", f"{STAGE_MODEL}.dvc")
REACT_FILES = ("deploy/training_space/web.py", "deploy/training_space/promotion.py",
               "frontend/package.json", "frontend/package-lock.json",
               "frontend/vite.admin.config.js", "frontend/admin/index.html", "frontend/admin/main.jsx",
               "frontend/admin/styles.css", "frontend/admin/api.js", "frontend/admin/ui.jsx",
               "frontend/admin/data.jsx", "frontend/admin/data.css", "frontend/admin/notebook.jsx",
               "frontend/admin/notebook.css", "frontend/admin/notebook-format.js", "frontend/admin/notebook.worker.js",
               "frontend/admin/stage-notebook-packages.mjs")


def stage(output, sdk="gradio"):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty directory for training Space source.")
    files = SOURCE_FILES + (REACT_FILES if sdk == "docker" else ())
    for relative in (*files, *(p.relative_to(ROOT) for p in (ROOT / "sleepops").glob("*.py"))):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    entrypoint = "web" if sdk == "docker" else "app"
    (output / "app.py").write_text(f"from deploy.training_space.{entrypoint} import main\n\nmain()\n", encoding="utf-8")
    shutil.copyfile(ROOT / "deploy/training_space/requirements.txt", output / "requirements.txt")
    if sdk == "docker":
        shutil.copyfile(ROOT / "deploy/training_space/Dockerfile", output / "Dockerfile")
        config = "sdk: docker\napp_port: 7860\n"
    else:
        config = "sdk: gradio\nsdk_version: 6.32.0\npython_version: 3.12.12\napp_file: app.py\n"
    (output / "README.md").write_text(f"---\ntitle: SleepOps Admin\n{config}---\n\n"
                                      "Inspect, compare, and train private SleepOps models.\n", encoding="utf-8")


def training_volumes(bucket, prediction_space=None):
    volumes = [Volume(type="bucket", source=bucket, path=PREFIX, mount_path="/sleepops-dvc", read_only=True),
               Volume(type="bucket", source=bucket, path="training-runs", mount_path="/sleepops-candidates",
                      read_only=False),
               Volume(type="bucket", source=bucket, path=DATASET_PREFIX, mount_path="/sleepops-datasets",
                      read_only=False)]
    if prediction_space:
        volumes.extend(Volume(type="space", source=prediction_space, path=path, mount_path=mount, read_only=True)
                       for path, mount in (("registry", "/sleepops-registry"), ("reports", "/sleepops-reports")))
    return volumes


def seed_dataset_storage(api, bucket):
    if any(item.path == DATASET_PREFIX for item in api.list_bucket_tree(bucket, recursive=False)):
        return
    with TemporaryDirectory(prefix="sleepops-dvc-index-") as staging:
        guide = Path(staging) / "README.md"
        guide.write_text("# SleepOps dataset versions\n\nEach release stores a dataset lock and a `.dvc` pointer. "
                         "Use the `dataset-versions` prefix as the DVC remote root to resolve uploaded objects.\n",
                         encoding="utf-8")
        api.batch_bucket_files(bucket, add=[(str(guide), f"{DATASET_PREFIX}/README.md")])


def verify_volumes(info, volumes):
    actual = info.runtime.volumes or []
    for desired in volumes:
        if not any(v.source == desired.source and v.path == desired.path and v.type == desired.type
                   and v.mount_path == desired.mount_path and bool(v.read_only) == desired.read_only for v in actual):
            raise ValueError(f"Check the Space volume settings for {desired.mount_path}.")


def deploy(api, space, bucket, hardware, prediction_space=None):
    space, bucket = checked_repo_id(space), checked_repo_id(bucket)
    if space.split("/")[-1] == "assistance" or not api.bucket_info(bucket).private:
        raise ValueError("Use a separate training Space and a private bucket.")
    volumes = training_volumes(bucket, prediction_space)
    variables = [{"key": "SLEEPOPS_HF_BUCKET", "value": bucket}]
    seed_dataset_storage(api, bucket)
    if prediction_space:
        if api.space_info(prediction_space).sdk != "docker":
            raise ValueError("The prediction release must come from a Docker Space.")
        variables.append({"key": "SLEEPOPS_PREDICTION_SPACE", "value": prediction_space})
    try:
        info = api.space_info(space)
    except RepositoryNotFoundError:
        api.create_repo(space, repo_type="space", private=True, space_sdk="gradio", space_hardware=hardware,
                        space_volumes=volumes, space_variables=variables)
        info = api.space_info(space)
    if not info.private or info.sdk != "gradio":
        raise ValueError("The destination must be a private Gradio Space.")
    actual = api.get_space_runtime(space)
    if actual.hardware not in {None, hardware.value} or actual.requested_hardware not in {None, hardware.value}:
        raise ValueError("The destination uses different hardware than requested.")
    existing = set(api.list_repo_files(space, repo_type="space"))
    if APP not in existing and existing - {".gitattributes", "README.md"}:
        raise ValueError("The destination already contains a different application.")
    verify_volumes(info, volumes)
    with TemporaryDirectory(prefix="sleepops-training-space-") as temp:
        source = Path(temp)
        stage(source)
        api.upload_folder(repo_id=space, repo_type="space", folder_path=source,
                          commit_message="Publish the private SleepOps trainer", parent_commit=info.sha)
    print(f"https://huggingface.co/spaces/{space}")


def reuse_docker(api, space, bucket, prediction_space=None):
    validate_repo_id(space)
    bucket = checked_repo_id(bucket)
    if "/" not in space or space.split("/")[-1].lower() == "assistance" or not api.bucket_info(bucket).private:
        raise ValueError("Use a separate training Space and a private bucket.")
    profile = api.whoami()
    existing = space_target(api, space, profile, reuse_existing=True, make_private=True)
    runtime = api.get_space_runtime(space)
    if runtime.requested_hardware != "cpu-basic" or runtime.hardware not in {None, "cpu-basic"}:
        raise ValueError("The reused Space must already be configured for CPU Basic.")
    if prediction_space and (prediction_space == space or api.space_info(prediction_space).sdk != "docker"):
        raise ValueError("Choose a separate Docker Space for model reports.")
    volumes = training_volumes(bucket, prediction_space)
    mount_paths = {volume.mount_path for volume in volumes}
    if any(volume.mount_path not in mount_paths for volume in existing.runtime.volumes or []):
        raise ValueError("The reused Space has unrelated storage mounts.")
    seed_dataset_storage(api, bucket)
    with TemporaryDirectory(prefix="sleepops-training-space-") as temp:
        source = Path(temp)
        stage(source, sdk="docker")
        publish_space(api, space, source, profile, existing.sha, reuse_existing=True, make_private=True,
                      commit_message="Deploy the SleepOps training dashboard")
    api.add_space_variable(space, "SLEEPOPS_HF_BUCKET", bucket)
    if prediction_space:
        api.add_space_variable(space, "SLEEPOPS_PREDICTION_SPACE", prediction_space)
    api.set_space_volumes(space, volumes)
    verify_volumes(api.space_info(space), volumes)
    print(f"Configured private training volumes for {space}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    source = commands.add_parser("stage")
    source.add_argument("output", type=Path)
    source.add_argument("--sdk", choices=("docker", "gradio"), default="gradio")
    publish = commands.add_parser("publish")
    publish.add_argument("--space", required=True)
    publish.add_argument("--bucket", required=True)
    publish.add_argument("--hardware", choices=("cpu-basic", "zero-a10g"), default="cpu-basic")
    publish.add_argument("--prediction-space")
    reuse = commands.add_parser("reuse-docker")
    reuse.add_argument("--space", required=True)
    reuse.add_argument("--bucket", required=True)
    reuse.add_argument("--prediction-space")
    args = parser.parse_args()
    if args.action == "stage":
        stage(args.output, sdk=args.sdk)
    elif args.action == "reuse-docker":
        reuse_docker(HfApi(), args.space, args.bucket, args.prediction_space)
    else:
        deploy(HfApi(), args.space, args.bucket, SpaceHardware(args.hardware), args.prediction_space)


if __name__ == "__main__":
    main()
