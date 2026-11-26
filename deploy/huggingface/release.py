"""Stage or publish SleepOps and sync its DVC objects to Hugging Face."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
from urllib.parse import urlsplit
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "sleep_edf_wake_data.npz"
MODEL = "best_model.pt"
STAGE_MODEL = "stage_aware_model.pt"
MODEL_LOCK = Path("configs/best_model.json")
STAGE_MODEL_LOCK = Path("configs/stage_aware_model.json")
ASSETS = (ARCHIVE, MODEL)
REMOTE = "hf-bucket"
PREFIX = "dvc-store"
DATASET_PREFIX = "dataset-versions"
SPACE_MARKER = "deploy/huggingface/serve.py"
DEFAULT_SPACE = "sleepops-cs495"
DEFAULT_BUCKET = "sleepops-eeg-data"


def checked_repo_id(repo_id):
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?/[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", repo_id):
        raise ValueError("Use a Hugging Face ID in the form namespace/lowercase-name.")
    return repo_id


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_archive(path):
    lock = json.loads((ROOT / "configs/dataset.json").read_text(encoding="utf-8"))
    if path.stat().st_size != lock["bytes"] or sha256(path) != lock["archive_sha256"]:
        raise ValueError(f"{path.name} does not match the dataset lock.")


def verify_asset(asset, path):
    if asset == ARCHIVE:
        verify_archive(path)
        return
    if asset == STAGE_MODEL:
        manifest = json.loads((ROOT / STAGE_MODEL_LOCK).read_text(encoding="utf-8"))
        if (manifest.get("file") != STAGE_MODEL or path.stat().st_size != manifest.get("bytes")
                or sha256(path) != manifest.get("checkpoint_sha256")):
            raise ValueError(f"{path.name} does not match the stage-aware model lock.")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from eeg_model import load_checkpoint

        _, _, _, payload = load_checkpoint(path)
        if (payload.get("version") != 2
                or payload.get("archive_sha256") != manifest.get("archive_sha256")):
            raise ValueError("The stage-aware checkpoint does not match its model lock.")
        return
    manifest = json.loads((ROOT / MODEL_LOCK).read_text(encoding="utf-8"))
    if path.stat().st_size != manifest["bytes"] or sha256(path) != manifest["checkpoint_sha256"]:
        raise ValueError(f"{path.name} does not match the best model lock.")


def release_checkpoint(release_id):
    if not re.fullmatch(r"[a-f0-9]{32}", release_id):
        raise ValueError("Invalid model release ID.")
    return ROOT / "registry/releases" / release_id / "best_checkpoint.pt"


def pin_model():
    release_id = json.loads((ROOT / "registry/champion.json").read_text(encoding="utf-8"))["release_id"]
    checkpoint = release_checkpoint(release_id)
    metadata = json.loads(checkpoint.with_name("metadata.json").read_text(encoding="utf-8"))
    promoted = json.loads(checkpoint.with_name("promotion.json").read_text(encoding="utf-8"))["passed"]
    validation_path = checkpoint.with_name("validation_metrics.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / "configs/dataset.json").read_text(encoding="utf-8"))
    digest = sha256(checkpoint)
    if (not promoted or metadata["run_id"] != release_id or validation["split"] != "validation"
            or metadata["archive_sha256"] != dataset["archive_sha256"]
            or validation["archive_sha256"] != dataset["archive_sha256"]
            or metadata["file_sha256"][checkpoint.name] != digest
            or metadata["file_sha256"][validation_path.name] != sha256(validation_path)):
        raise ValueError("The active model must pass promotion and match the dataset and release checksums.")
    manifest = {
        "file": MODEL,
        "release_id": release_id,
        "selected_epoch": validation["selected_epoch"],
        "checkpoint_sha256": digest,
        "archive_sha256": dataset["archive_sha256"],
        "bytes": checkpoint.stat().st_size,
    }
    with NamedTemporaryFile(dir=ROOT, delete=False) as pending:
        temp = Path(pending.name)
        with checkpoint.open("rb") as source:
            shutil.copyfileobj(source, pending)
    try:
        temp.replace(ROOT / MODEL)
    finally:
        temp.unlink(missing_ok=True)
    run_dvc(ROOT, "add", MODEL)
    (ROOT / MODEL_LOCK).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_asset(MODEL, ROOT / MODEL)
    print(f"Pinned promoted model {release_id} at selected epoch {validation['selected_epoch']}", flush=True)


def asset_source(asset):
    path = ROOT / asset
    if asset == MODEL and not path.exists():
        manifest = json.loads((ROOT / MODEL_LOCK).read_text(encoding="utf-8"))
        path = release_checkpoint(manifest["release_id"])
    verify_asset(asset, path)
    return path


def build_manifest():
    pinned = json.loads((ROOT / MODEL_LOCK).read_text(encoding="utf-8"))
    active = json.loads((ROOT / "registry/champion.json").read_text(encoding="utf-8"))["release_id"]
    dataset = json.loads((ROOT / "configs/dataset.json").read_text(encoding="utf-8"))
    if pinned["release_id"] != active or pinned["archive_sha256"] != dataset["archive_sha256"]:
        raise ValueError("Run pin-model to match the DVC model lock to the active release and dataset.")
    verify_asset(MODEL, release_checkpoint(active))
    return {"build_id": uuid4().hex, "source_revision": os.getenv("GITHUB_SHA"), "release_id": active,
            "archive_sha256": pinned["archive_sha256"], "checkpoint_sha256": pinned["checkpoint_sha256"]}


def run_dvc(repo, *arguments):
    env = {**os.environ, "DVC_NO_ANALYTICS": "1",
           "DVC_CACHE_DIR": str((repo / ".dvc/cache").resolve()),
           "DVC_SITE_CACHE_DIR": str((repo / ".dvc/site-cache").resolve())}
    executable = shutil.which("dvc")
    command = [executable, *map(str, arguments)] if executable else [sys.executable, "-m", "dvc", *map(str, arguments)]
    subprocess.run(command, cwd=repo, env=env, check=True)


def configure(bucket_id):
    namespace, bucket = checked_repo_id(bucket_id).split("/")
    run_dvc(ROOT, "remote", "add", "--default", "--force", REMOTE, f"s3://{bucket}/{PREFIX}")
    run_dvc(ROOT, "remote", "modify", REMOTE, "endpointurl", f"https://{namespace.lower()}.s3.hf.co")
    run_dvc(ROOT, "remote", "modify", REMOTE, "region", "us-east-1")


def object_key(asset):
    import yaml

    [output] = yaml.safe_load((ROOT / f"{asset}.dvc").read_text(encoding="utf-8"))["outs"]
    digest = output["md5"]
    if output["path"] != asset or not re.fullmatch(r"[a-f0-9]{32}", digest):
        raise ValueError(f"The DVC pointer must identify {asset}.")
    return Path("files", "md5", digest[:2], digest[2:])


def scratch_dvc(repo, remote, assets):
    repo.mkdir(parents=True)
    run_dvc(repo, "init", "--no-scm")
    for asset in assets:
        pointer = f"{asset}.dvc"
        shutil.copyfile(ROOT / pointer, repo / pointer)
    run_dvc(repo, "remote", "add", "--default", "transfer", remote.resolve())


def fetch_objects(api, bucket_id, temp, assets):
    remote = temp / "download-cache"
    files = []
    for asset in assets:
        key = object_key(asset)
        local = remote / key
        local.parent.mkdir(parents=True, exist_ok=True)
        files.append((f"{PREFIX}/{key.as_posix()}", str(local)))
    api.download_bucket_files(bucket_id, files=files, raise_on_missing_files=True)
    repo = temp / "fresh-repo"
    scratch_dvc(repo, remote, assets)
    run_dvc(repo, "pull", *(f"{asset}.dvc" for asset in assets))
    for asset in assets:
        path = repo / asset
        verify_asset(asset, path)
        print(f"Verified {asset} restored from Hugging Face, SHA-256 {sha256(path)}", flush=True)
    return repo


def publish_assets(api, bucket_id, assets):
    sources = {asset: asset_source(asset) for asset in assets}
    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=work, prefix="dvc-hf-") as workspace:
        temp = Path(workspace)
        repo = temp / "upload-repo"
        remote = temp / "upload-cache"
        scratch_dvc(repo, remote, assets)
        for asset, source in sources.items():
            shutil.copyfile(source, repo / asset)
        run_dvc(repo, "add", *assets)
        run_dvc(repo, "push", *(f"{asset}.dvc" for asset in assets))
        files = []
        for asset in assets:
            key = object_key(asset)
            local = remote / key
            verify_asset(asset, local)
            files.append((str(local), f"{PREFIX}/{key.as_posix()}"))
        api.create_bucket(bucket_id, private=True, exist_ok=True)
        if not api.bucket_info(bucket_id).private:
            raise ValueError(f"The bucket {bucket_id} must be private.")
        api.batch_bucket_files(bucket_id, add=files)
        fetch_objects(api, bucket_id, temp, assets)
    configure(bucket_id)
    print(f"Private DVC bucket https://huggingface.co/buckets/{bucket_id}")


def pull_assets(api, bucket_id, assets):
    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    for asset in assets:
        target = ROOT / asset
        if target.exists():
            verify_asset(asset, target)
    with TemporaryDirectory(dir=work, prefix="dvc-hf-") as workspace:
        repo = fetch_objects(api, bucket_id, Path(workspace), assets)
        for asset in assets:
            target = ROOT / asset
            if not target.exists():
                (repo / asset).replace(target)
            print(f"{asset} ready at {target}")


def hydrate_serving():
    """Rebuild serving files in a checkout restored from the private DVC remote."""
    import numpy as np

    from eeg_data import FEATURE_NAMES, load_archive

    lock = json.loads((ROOT / MODEL_LOCK).read_text(encoding="utf-8"))
    active = json.loads((ROOT / "registry/champion.json").read_text(encoding="utf-8"))["release_id"]
    if lock["release_id"] != active:
        raise ValueError("The serving registry and DVC model lock disagree.")
    model = asset_source(MODEL)
    archive = asset_source(ARCHIVE)
    checkpoint = release_checkpoint(active)
    if checkpoint.exists():
        verify_asset(MODEL, checkpoint)
    else:
        shutil.copyfile(model, checkpoint)

    data = load_archive(archive, verbose=False)
    a = data.arrays
    windows = data.state_windows["test"]
    records = a["state_record_id"][windows[:, -1]]
    choices = np.concatenate([windows[records == record][::2] for record in sorted(set(records))[:3]])
    end_rows = choices[:, -1]
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    np.savez_compressed(artifacts / "demo_windows.npz", windows=a["X_state"][choices],
                        record_id=a["state_record_id"][end_rows], epoch_end_sec=a["state_epoch_end_sec"][end_rows],
                        feature_names=np.asarray(FEATURE_NAMES), archive_sha256=np.asarray(sha256(archive)))
    print(f"Restored the serving checkpoint and {len(choices)} held-out windows for {active}.", flush=True)


def stage(output, build=None, *, github_source=False):
    output = Path(output).resolve()
    if output == ROOT or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose an empty directory for the Space staging files.")
    files = [Path(name) for name in (
        "README.md", "MLOPS.md", "HUGGING_FACE.md", "Dockerfile", "compose.yaml", "pyproject.toml",
        "requirements.txt", "eeg_data.py", "eeg_model.py", "demo.py", ".gitignore", ".dockerignore",
        "Sleep_Wake_Training_Colab.ipynb", "Sleep_EDF_Wake_Up_Prediction_Colab.ipynb",
        "Sleep_Stage_Awakening_Training_Colab.ipynb",
        *(f"{asset}.dvc" for asset in ASSETS),
        ".dvc/config", ".dvc/.gitignore", "deploy/huggingface/aws-config",
        f"{STAGE_MODEL}.dvc", str(STAGE_MODEL_LOCK),
    )]
    if (ROOT / STAGE_MODEL).is_file():
        files.append(Path(STAGE_MODEL))
    for directory in (".github", "configs", "deploy", "sleepops", "registry", "reports", "tests", "artifacts"):
        files.extend(path.relative_to(ROOT) for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".yml", ".yaml", ".json", ".toml", ".pt",
                                                          ".npz", ".png", ".txt", ".conf"}
                     and "__pycache__" not in path.parts and path.name not in {"huggingface_deployment.json", "build.json"})
    files.extend(Path("frontend") / name for name in ("package.json", "package-lock.json", "index.html", "vite.config.js"))
    files.extend(path.relative_to(ROOT) for path in (ROOT / "frontend/src").rglob("*")
                 if path.is_file() and path.suffix in {".js", ".jsx", ".css"})
    for relative in sorted(files):
        if github_source and (relative.parts[0] in {"artifacts", "reports"} or relative.suffix == ".pt"):
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if github_source and relative.suffix == ".ipynb":
            notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell.update(outputs=[], execution_count=None)
            target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(ROOT / relative, target)
    if build is not None:
        (output / "deploy/huggingface/build.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
    print(f"Staged {sum(path.is_file() for path in output.rglob('*'))} project files at {output}", flush=True)
    return output


def space_target(api, repo_id, profile, *, reuse_existing=False, make_private=False, allow_public=False):
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        existing = api.space_info(repo_id)
    except RepositoryNotFoundError:
        if reuse_existing:
            raise ValueError(f"The Docker Space {repo_id} does not exist or is inaccessible.") from None
        if repo_id.split("/")[0].lower() == profile["name"].lower() and profile.get("isPro") is False:
            raise ValueError("Hugging Face PRO is required to create this personal Docker Space.") from None
        return None
    files = api.list_repo_files(repo_id, repo_type="space", revision=existing.sha)
    if SPACE_MARKER not in files and set(files) - {".gitattributes"} and not reuse_existing:
        raise ValueError(f"Replacing the existing app in {repo_id} requires --reuse-existing.")
    if existing.sdk != "docker" or (not existing.private and not (make_private or allow_public)):
        raise ValueError(f"The Space {repo_id} must be a private Docker Space unless public access is explicitly allowed.")
    return existing


def publish_space(api, repo_id, staged, profile, expected_head, *, reuse_existing=False, make_private=False,
                  allow_public=False, commit_message="Deploy SleepOps with DVC dataset and model locks"):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    options = {"reuse_existing": reuse_existing, "make_private": make_private, "allow_public": allow_public}
    existing = space_target(api, repo_id, profile, **options)
    current_head = existing.sha if existing is not None else None
    if expected_head != current_head:
        raise RuntimeError("The Space changed during deployment preparation. Review its new revision before retrying.")
    if existing is None:
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=True, space_hardware="cpu-basic")
        current_head = api.space_info(repo_id).sha
    if reuse_existing:
        if not current_head:
            raise ValueError("The existing Space has no revision to preserve.")
        backup = f"before-sleepops-{current_head[:12]}"
        api.create_branch(repo_id, branch=backup, revision=current_head, repo_type="space", exist_ok=True)
        if api.space_info(repo_id, revision=backup).sha != current_head:
            raise RuntimeError("The rollback branch could not be verified.")
        print(f"Preserved previous Space revision on branch {backup}", flush=True)
    if make_private and existing is not None and not existing.private:
        api.update_repo_settings(repo_id, repo_type="space", private=True)
    expected_private = existing.private if existing is not None and not make_private else True
    if api.space_info(repo_id).private != expected_private:
        raise ValueError(f"The Space {repo_id} visibility changed during deployment preparation.")

    if reuse_existing:
        additions = {file.relative_to(staged).as_posix(): file for file in staged.rglob("*") if file.is_file()}
        old_files = set(api.list_repo_files(repo_id, repo_type="space", revision=current_head))
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=file) for name, file in additions.items()]
        operations.extend(CommitOperationDelete(path_in_repo=name, is_folder=False)
                          for name in sorted(old_files - additions.keys() - {".gitattributes"}))
        commit = api.create_commit(repo_id, operations, repo_type="space", parent_commit=current_head,
                                   commit_message=commit_message)
    else:
        commit = api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=staged,
                                   parent_commit=current_head, commit_message=commit_message)
    print(f"Uploaded Space https://huggingface.co/spaces/{repo_id}", flush=True)
    print(f"Space commit {commit.oid}", flush=True)
    return commit.oid


def space_host(info):
    host = info.host or (f"https://{info.subdomain}.hf.space" if info.subdomain else "")
    address = urlsplit(host)
    if (address.scheme != "https" or not address.hostname or not address.hostname.endswith(".hf.space")
            or address.username or address.password or address.port or address.path not in {"", "/"}
            or address.query or address.fragment):
        raise ValueError("The Space must provide a valid hf.space HTTPS address.")
    return host.rstrip("/")


def deployed_build_is_healthy(client, host, build):
    model = client.get(host + "/api/model")
    if model.status_code != 200:
        return False
    details = model.json()
    tracked = ("build_id", "release_id", "archive_sha256", "checkpoint_sha256")
    if any(details.get(key) != build[key] for key in tracked):
        return False
    frontend = client.get(host + "/")
    if frontend.status_code != 200 or 'content="sleepops-react"' not in frontend.text:
        return False
    response = client.post(host + "/api/v1/predict", json={"features": [[0.1] * 11 for _ in range(10)]})
    if response.status_code != 200:
        return False
    prediction = response.json()
    if prediction.get("release_id") != build["release_id"] or len(prediction.get("asleep_probabilities", [])) != 10:
        return False
    forecast = prediction.get("forecast")
    if prediction.get("state") == "awake":
        return forecast is None
    return prediction.get("state") == "asleep" and isinstance(forecast, dict) and len(forecast.get("bins", [])) == 13


def wait_for_deployment(api, repo_id, build, wait_seconds, expected_private=True):
    from huggingface_hub import get_token
    import httpx

    token = get_token()
    if not token:
        raise ValueError("Authentication is required to verify the Space.")
    deadline = time.monotonic() + wait_seconds
    previous = None
    with httpx.Client(headers={"Authorization": f"Bearer {token}"}, follow_redirects=False, timeout=10) as client:
        while time.monotonic() < deadline:
            state = api.get_space_runtime(repo_id).stage
            if state != previous:
                print(f"{repo_id}: {state}", flush=True)
                previous = state
            if state in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "PAUSED", "NO_APP_FILE"}:
                raise RuntimeError(f"Space reports {state}. Review its build or runtime logs.")
            if state == "RUNNING":
                info = api.space_info(repo_id)
                if info.private != expected_private:
                    raise ValueError("The deployed Space visibility changed unexpectedly.")
                if info.host or info.subdomain:
                    try:
                        if deployed_build_is_healthy(client, space_host(info), build):
                            print(f"Verified live app and model {build['release_id']} from build {build['build_id']}", flush=True)
                            return
                    except (httpx.RequestError, json.JSONDecodeError):
                        pass
            time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError("The intended Space build did not pass the live app and model check before the deadline.")


def space_status(api, repo_id, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    previous = None
    while True:
        state = api.get_space_runtime(repo_id).stage
        if state != previous:
            print(f"{repo_id}: {state}", flush=True)
            previous = state
        if state == "RUNNING":
            return
        if state in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "PAUSED", "NO_APP_FILE"}:
            raise RuntimeError(f"Space reports {state}. Review its build or runtime logs.")
        if time.monotonic() >= deadline:
            if wait_seconds and state != "RUNNING":
                raise TimeoutError(f"Space is still {state}.")
            return
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    stage_command = actions.add_parser("stage")
    stage_command.add_argument("--output", type=Path, required=True)
    stage_command.add_argument("--github-source", action="store_true")
    actions.add_parser("configure").add_argument("--bucket", required=True)
    actions.add_parser("pin-model")
    actions.add_parser("hydrate-serving")
    status = actions.add_parser("space-status")
    status.add_argument("--space")
    status.add_argument("--wait-seconds", type=int, default=0)
    for name in ("publish-data", "pull-data", "publish-model", "pull-model", "publish-stage-model",
                 "pull-stage-model", "publish-space", "deploy"):
        action = actions.add_parser(name)
        if name != "publish-space":
            action.add_argument("--bucket")
        if name in {"publish-space", "deploy"}:
            action.add_argument("--space")
            action.add_argument("--wait-seconds", type=int, default=600)
            action.add_argument("--reuse-existing", action="store_true")
            action.add_argument("--make-private", action="store_true")
        if name == "deploy":
            action.add_argument("--allow-public", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "stage":
            stage(args.output, github_source=args.github_source)
        elif args.action == "configure":
            configure(args.bucket)
        elif args.action == "pin-model":
            pin_model()
        elif args.action == "hydrate-serving":
            hydrate_serving()
        else:
            from huggingface_hub import HfApi

            api = HfApi()
            profile = api.whoami()
            owner = profile["name"]
            if args.action == "space-status":
                space_status(api, checked_repo_id(args.space or f"{owner}/{DEFAULT_SPACE}"), args.wait_seconds)
                return
            if args.action != "publish-space":
                bucket_id = checked_repo_id(args.bucket or f"{owner}/{DEFAULT_BUCKET}")
            if args.action in {"publish-space", "deploy"}:
                if args.wait_seconds < 0:
                    raise ValueError("The Space wait time must be nonnegative.")
                if args.reuse_existing and not args.space:
                    raise ValueError("Specify the exact Space ID with --space when reusing a Space.")
                if args.make_private and not args.reuse_existing:
                    raise ValueError("Use --make-private only with --reuse-existing.")
                if args.allow_public and not args.reuse_existing:
                    raise ValueError("Use --allow-public only with --reuse-existing.")
                if args.allow_public and args.make_private:
                    raise ValueError("Choose either --make-private or --allow-public, not both.")
                repo_id = checked_repo_id(args.space or f"{owner}/{DEFAULT_SPACE}")
                build = build_manifest()
                options = {"reuse_existing": args.reuse_existing, "make_private": args.make_private,
                           "allow_public": getattr(args, "allow_public", False)}
                target = space_target(api, repo_id, profile, **options)
                expected_head = target.sha if target is not None else None
                expected_private = target.private if target is not None and not args.make_private else True
            if args.action == "deploy":
                publish_assets(api, bucket_id, ASSETS)
            elif args.action in {"publish-data", "publish-model", "publish-stage-model"}:
                asset = {"publish-data": ARCHIVE, "publish-model": MODEL,
                         "publish-stage-model": STAGE_MODEL}[args.action]
                publish_assets(api, bucket_id, (asset,))
            elif args.action in {"pull-data", "pull-model", "pull-stage-model"}:
                asset = {"pull-data": ARCHIVE, "pull-model": MODEL, "pull-stage-model": STAGE_MODEL}[args.action]
                pull_assets(api, bucket_id, (asset,))
            if args.action in {"publish-space", "deploy"}:
                work = ROOT / ".work"
                work.mkdir(exist_ok=True)
                with TemporaryDirectory(dir=work, prefix="hf-space-") as workspace:
                    publish_space(api, repo_id, stage(workspace, build=build), profile, expected_head, **options)
                if args.wait_seconds:
                    wait_for_deployment(api, repo_id, build, args.wait_seconds, expected_private=expected_private)
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"SleepOps cloud error: {error}\n")


if __name__ == "__main__":
    main()
