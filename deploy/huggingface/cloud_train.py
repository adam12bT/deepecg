"""Run the existing trainer on a hosted runner and save private candidates."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from deploy.huggingface.release import ARCHIVE, checked_repo_id
from sleepops.files import read_json, sha256, write_json
from sleepops.registry import RELEASE_FILES, check_quality, validate_bundle


def fit(args):
    from sleepops.lifecycle import evaluate_release, find_run, store_candidate

    run_id = store_candidate(args)
    output = find_run(args.runtime, run_id)
    _, validation, _ = validate_bundle(output)
    gates = check_quality(validation, read_json(Path("configs/promotion.json")))
    if args.evaluate_test:
        args.run = run_id
        evaluate_release(args)
    write_json(output / "cloud_run.json", {
        "run_id": run_id,
        "source_revision": os.getenv("GITHUB_SHA"),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "space_id": os.getenv("SPACE_ID"),
        "test_evaluated": args.evaluate_test,
        "validation_gates": gates,
    })
    if github_output := os.getenv("GITHUB_OUTPUT"):
        with Path(github_output).open("a", encoding="utf-8") as stream:
            stream.write(f"candidate={run_id}\n")
    return run_id


def candidate_files(test_evaluated):
    files = (*RELEASE_FILES, "training_log.json")
    if test_evaluated:
        files += ("test_metrics.json", "heldout_forecast.png")
    return (*files, "cloud_run.json")


def publish(args):
    from huggingface_hub import HfApi, get_token

    from sleepops.lifecycle import find_run

    output = find_run(args.runtime, args.run)
    metadata, validation, _ = validate_bundle(output)
    run = read_json(output / "cloud_run.json")
    if run["run_id"] != metadata["run_id"] or args.run != run["run_id"]:
        raise ValueError("Candidate ID differs across run files.")
    token = get_token()
    if not token:
        raise ValueError("A Hugging Face token with bucket write access is required.")
    api = HfApi(token=token)
    bucket = checked_repo_id(args.bucket)
    if not api.bucket_info(bucket).private:
        raise ValueError(f"The training bucket {bucket} must be private.")

    folder = f"training-runs/{args.run}"
    files = candidate_files(run["test_evaluated"])
    api.batch_bucket_files(bucket, add=[(str(output / name), f"{folder}/{name}") for name in files])
    with TemporaryDirectory() as restored:
        target = Path(restored)
        api.download_bucket_files(bucket, files=[(f"{folder}/{name}", str(target / name)) for name in files],
                                  raise_on_missing_files=True)
        if any(sha256(output / name) != sha256(target / name) for name in files):
            raise ValueError("The uploaded candidate files differ from the training outputs.")

    url = f"https://huggingface.co/buckets/{bucket}/tree/{folder}"
    print(f"Verified private candidate at {url}", flush=True)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        log = read_json(output / "training_log.json")
        gates = "passed" if run["validation_gates"]["passed"] else "did not pass"
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(f"## SleepOps training candidate\n\n"
                         f"- [Private checkpoint and reports]({url})\n"
                         f"- Finished {len(log['history'])} epochs on {log['hardware']}. "
                         f"Selected epoch {validation['selected_epoch']} by validation loss "
                         f"({log['best_validation_mean']:.5f}).\n"
                         f"- Existing validation gates {gates}.\n"
                         f"- Held-out test evaluation {'completed' if run['test_evaluated'] else 'skipped'}.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("runtime"))
    commands = parser.add_subparsers(dest="action", required=True)
    train = commands.add_parser("fit")
    train.add_argument("--epochs", type=int, choices=range(1, 21), metavar="1–20", required=True)
    train.add_argument("--archive", type=Path, default=Path(ARCHIVE))
    train.add_argument("--lock", type=Path, default=Path("configs/dataset.json"))
    train.add_argument("--evaluate-test", action="store_true")
    train.set_defaults(config=Path("configs/train.toml"), device="cpu", promote=False)
    upload = commands.add_parser("publish")
    upload.add_argument("--bucket", required=True)
    upload.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args)
    else:
        publish(args)


if __name__ == "__main__":
    main()
