"""Run private Admin deployment requests through the production release checks."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi

from deploy.huggingface import release
from deploy.huggingface.cloud_train import candidate_files
from sleepops.files import read_json, utc_now, write_json
from sleepops.registry import CHECKPOINT, RELEASE_FILES, VERSION_PATTERN


QUEUE = "training-runs/deployment"
REQUEST_NAME = re.compile(r"\d{20}-([a-f0-9]{32})\.json")
DATASET_HASH = re.compile(r"[a-f0-9]{64}")


def identifier(value):
    if not VERSION_PATTERN.fullmatch(value):
        raise ValueError("Invalid deployment identifier.")
    return value


def find_request(api, bucket, work):
    requests = f"{QUEUE}/requests/"
    pending = [entry for entry in api.list_bucket_tree(bucket, prefix=QUEUE, recursive=True)
               if entry.path.startswith(requests) and REQUEST_NAME.fullmatch(Path(entry.path).name)]
    if not pending:
        return None
    latest = max(pending, key=lambda entry: entry.path)
    path = work / "request.json"
    api.download_bucket_files(bucket, [(latest, path)], raise_on_missing_files=True)
    request = read_json(path)
    if request["id"] != REQUEST_NAME.fullmatch(Path(latest.path).name)[1]:
        raise ValueError("Deployment request does not match its filename.")
    identifier(request["candidate"])
    identifier(request["expectedRelease"])
    if not DATASET_HASH.fullmatch(request["dataset"]):
        raise ValueError("Invalid dataset version in deployment request.")
    status_key = f"{QUEUE}/status/{request['id']}.json"
    status = api.get_bucket_paths_info(bucket, paths=[status_key])
    if not status:
        return request
    state_path = work / "status.json"
    api.download_bucket_files(bucket, [(status[0], state_path)], raise_on_missing_files=True)
    state = read_json(state_path)
    if state["phase"] == "running" and datetime.fromisoformat(state["updatedAt"]) < (
            datetime.now(timezone.utc) - timedelta(minutes=45)):
        return request
    return None


def set_status(api, bucket, request_id, phase):
    if not request_id:
        return
    identifier(request_id)
    url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    with TemporaryDirectory() as directory:
        file = Path(directory) / "status.json"
        write_json(file, {"phase": phase, "url": url, "updatedAt": utc_now()})
        api.batch_bucket_files(bucket, add=[(str(file), f"{QUEUE}/status/{request_id}.json")])


def restore_dataset(api, bucket, work, candidate, active_lock):
    import numpy as np

    from deploy.training_space.datasets import group
    from sleepops.dataset import load_dataset

    archive = release.ROOT / release.ARCHIVE
    lock = candidate / "dataset.json"
    goal = read_json(lock)
    current = read_json(active_lock)
    if goal == current:
        return
    digest = goal["archive_sha256"]
    if not DATASET_HASH.fullmatch(digest):
        raise ValueError("Invalid candidate dataset fingerprint.")
    prefix = f"{release.DATASET_PREFIX}/releases/{digest}"
    marker, published_lock, pointer = (work / name for name in ("version.json", "dataset.json", f"{release.ARCHIVE}.dvc"))
    api.download_bucket_files(bucket, [(f"{prefix}/{path.name}", path) for path in (marker, published_lock, pointer)],
                              raise_on_missing_files=True)
    info = read_json(marker)
    original = current["archive_sha256"]
    old_version = api.get_bucket_paths_info(bucket, paths=[f"{release.DATASET_PREFIX}/releases/{original}/version.json"])
    if old_version:
        old_marker = work / "active-version.json"
        api.download_bucket_files(bucket, [(old_version[0], old_marker)], raise_on_missing_files=True)
        original = read_json(old_marker)["heldOutAnchor"]
    if info["id"] != digest or info["heldOutAnchor"] != original or read_json(published_lock) != goal:
        raise ValueError("The candidate dataset does not share the frozen evaluation data.")
    import yaml

    [entry] = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"]
    md5 = info["md5"]
    if not re.fullmatch(r"[a-f0-9]{32}", md5) or entry["md5"] != md5 or entry["path"] != release.ARCHIVE:
        raise ValueError("Invalid DVC dataset pointer.")
    target = work / release.ARCHIVE
    api.download_bucket_files(bucket, [(f"{release.DATASET_PREFIX}/files/md5/{md5[:2]}/{md5[2:]}", target)],
                              raise_on_missing_files=True)
    with target.open("rb") as handle:
        if hashlib.file_digest(handle, "md5").hexdigest() != md5:
            raise ValueError("The candidate dataset DVC object does not match its pointer.")
    old, _ = load_dataset(archive, active_lock)
    new, _ = load_dataset(target, lock)
    for key, value in old.arrays.items():
        task = group(key)
        if task:
            before = old.arrays[f"{task}_split"] != "train"
            after = new.arrays[f"{task}_split"] != "train"
            if not np.array_equal(value[before], new.arrays[key][after]):
                raise ValueError("The candidate dataset changes held-out EEG records or labels.")
    shutil.copyfile(target, archive)
    shutil.copyfile(lock, active_lock)
    release.run_dvc(release.ROOT, "add", release.ARCHIVE)


def prepare(api, bucket, space, candidate_id, request_id, work):
    from deploy.training_space.promotion import checkpoint_loss
    from sleepops.registry import promote, validate_bundle

    identifier(candidate_id)
    request = find_request(api, bucket, work) if request_id else None
    if request_id and (not request or request["id"] != request_id or request["candidate"] != candidate_id):
        print("This deployment request is no longer pending.")
        return
    if request_id:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            print("claimed=true", file=output)
        set_status(api, bucket, request_id, "running")
    info = api.space_info(space)
    if not info.private:
        raise ValueError("The production Space must be private.")
    old_head = info.sha
    live = read_json(api.hf_hub_download(space, "registry/champion.json", repo_type="space", revision=old_head))
    baseline = read_json(release.ROOT / "registry/champion.json")["release_id"]
    if baseline != live["release_id"]:
        raise ValueError("The GitHub and live model releases disagree. Review them before deploying.")
    if request and request["expectedRelease"] != baseline:
        raise ValueError("The live model changed. Queue a fresh request.")

    run = work / candidate_id
    run.mkdir()
    prefix = f"training-runs/{candidate_id}"
    api.download_bucket_files(bucket, [(f"{prefix}/{name}", run / name) for name in candidate_files(False)],
                              raise_on_missing_files=True)
    run_info = read_json(run / "cloud_run.json")
    if run_info["test_evaluated"]:
        files = set(candidate_files(True)) - set(candidate_files(False))
        api.download_bucket_files(bucket, [(f"{prefix}/{name}", run / name) for name in files],
                                  raise_on_missing_files=True)
    metadata, validation, _ = validate_bundle(run)
    lock = release.ROOT / "configs/dataset.json"
    active = release.ROOT / "registry/releases" / baseline
    active_validation = read_json(active / "validation_metrics.json")
    if (metadata["run_id"] != candidate_id or run_info["run_id"] != candidate_id
            or request and request["dataset"] != metadata["archive_sha256"]
            or any(validation[key] != active_validation[key] for key in ("subjects", "state_epochs", "forecast_windows"))):
        raise ValueError("Candidate identity and evaluation cohort must match the deployment request and live model.")
    release.pull_assets(api, bucket, release.ASSETS)
    baseline_metadata = read_json(active / "metadata.json")
    active_score = checkpoint_loss(release.ROOT / release.MODEL, baseline_metadata["file_sha256"][CHECKPOINT])
    candidate_score = checkpoint_loss(run / CHECKPOINT, metadata["file_sha256"][CHECKPOINT])
    if candidate_score >= active_score:
        raise ValueError("The candidate does not improve on the live validation loss.")
    restore_dataset(api, bucket, work, run, lock)
    promote(release.ROOT / "registry", run, read_json(release.ROOT / "configs/promotion.json"))
    release.pin_model()
    release.hydrate_serving()
    reports = release.ROOT / "reports"
    reports.mkdir(exist_ok=True)
    for name in ("validation_metrics.json", "training_log.json", "test_metrics.json", "heldout_forecast.png"):
        if (run / name).is_file():
            shutil.copyfile(run / name, reports / name)
        else:
            (reports / name).unlink(missing_ok=True)
    write_json(work / "deployment.json", {"oldHead": old_head, "release": candidate_id})
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        print("ready=true", file=output)
    print(f"Validation loss {active_score:.5f} → {candidate_score:.5f}")


def publish(api, bucket, space, work, trial):
    def git(*args):
        subprocess.run(["git", *args], check=True, cwd=release.ROOT)

    plan = read_json(work / "deployment.json")
    old_head, candidate = plan["oldHead"], plan["release"]
    build = release.build_manifest()
    if build["release_id"] != candidate:
        raise ValueError("The staged model differs from the verified candidate.")
    if trial:
        print(f"Trial verified candidate {candidate}. Production was not changed.")
        return
    profile = api.whoami()
    base_build = read_json(api.hf_hub_download(space, "deploy/huggingface/build.json", repo_type="space", revision=old_head))
    release.publish_assets(api, bucket, release.ASSETS)
    metadata = [f"registry/releases/{candidate}/{name}" for name in (*RELEASE_FILES, "promotion.json") if name != CHECKPOINT]
    git("add", "--", "configs/dataset.json", str(release.MODEL_LOCK), "registry/champion.json",
        *(f"{asset}.dvc" for asset in release.ASSETS), *metadata)
    git("-c", "user.name=SleepOps CI", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit", "-m", f"Promote SleepOps model {candidate[:12]}")
    with TemporaryDirectory() as directory:
        revision = release.publish_space(api, space, release.stage(directory, build), profile, old_head,
                                         reuse_existing=True, commit_message=f"Deploy SleepOps model {candidate[:12]}")
    try:
        release.wait_for_deployment(api, space, build, 720)
        git("push", "origin", f"HEAD:{os.environ['GITHUB_REF']}")
    except Exception:
        with TemporaryDirectory() as directory:
            previous = api.snapshot_download(space, repo_type="space", revision=old_head, local_dir=directory)
            shutil.rmtree(Path(previous) / ".cache", ignore_errors=True)
            release.publish_space(api, space, Path(previous), profile, revision, reuse_existing=True,
                                  commit_message="Restore last verified SleepOps release")
        release.wait_for_deployment(api, space, base_build, 720)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "publish", "finish"))
    parser.add_argument("--trial-candidate", default="")
    parser.add_argument("--job-result", default="")
    args = parser.parse_args()
    api = HfApi()
    bucket = release.checked_repo_id(os.environ["SLEEPOPS_HF_BUCKET"])
    if not api.bucket_info(bucket).private:
        raise ValueError("Deployment storage must be private.")
    if args.action == "plan":
        with TemporaryDirectory() as directory:
            request = None if args.trial_candidate else find_request(api, bucket, Path(directory))
        candidate = identifier(args.trial_candidate) if args.trial_candidate else request["candidate"] if request else ""
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            print(f"candidate={candidate}\nrequest={request['id'] if request else ''}", file=output)
        return
    request_id = os.environ.get("PROMOTION_REQUEST", "")
    if args.action == "finish":
        set_status(api, bucket, request_id, "succeeded" if args.job_result == "success" else "failed")
        return
    space = release.checked_repo_id(os.environ["SLEEPOPS_HF_SPACE"])
    work = release.ROOT / ".work/promotion"
    if args.action == "prepare":
        work.mkdir(parents=True, exist_ok=True)
        prepare(api, bucket, space, os.environ["PROMOTION_CANDIDATE"], request_id, work)
    elif args.action == "publish":
        publish(api, bucket, space, work, trial=not request_id)


if __name__ == "__main__":
    main()
