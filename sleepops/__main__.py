"""Command line for data verification, training, and model release management."""

import argparse
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m sleepops")
    parser.add_argument("--runtime", type=Path, default=Path("runtime"))
    parser.add_argument("--registry", type=Path, default=Path("registry"))
    commands = parser.add_subparsers(dest="action", required=True)

    verify = commands.add_parser("validate", help="Check the archive and print or write a dataset lock.")
    verify.add_argument("--archive", type=Path, default=Path("sleep_edf_wake_data.npz"))
    verify.add_argument("--lock", type=Path, help="An existing lock to verify.")
    verify.add_argument("--write", type=Path, help="Write the resulting dataset lock here.")

    for action, description in (
        ("train", "Train and track a new candidate."),
        ("bootstrap", "Import and track the original project checkpoint."),
        ("evaluate", "Score a completed run on the held-out test split."),
        ("promote", "Apply validation gates and activate a candidate."),
    ):
        sub = commands.add_parser(action, help=description)
        sub.add_argument("--lock", type=Path, default=Path("configs/dataset.json"))
        if action != "promote":
            sub.add_argument("--archive", type=Path, default=Path("sleep_edf_wake_data.npz"))
            sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if action in {"train", "bootstrap"}:
            sub.add_argument("--config", type=Path, default=Path("configs/train.toml"))
            sub.add_argument("--promote", action="store_true")
        if action in {"train", "bootstrap", "promote"}:
            sub.add_argument("--policy", type=Path, default=Path("configs/promotion.json"))
        if action in {"promote", "evaluate"}:
            sub.add_argument("--run", required=True)
        if action == "train":
            sub.add_argument("--epochs", type=int, help="Override the configured maximum epochs.")
        if action == "bootstrap":
            sub.add_argument("--artifacts", type=Path, default=Path("artifacts"))

    activate = commands.add_parser("activate", help="Activate or roll back to a previously promoted release.")
    activate.add_argument("--release", required=True)
    commands.add_parser("status", help="Verify and show the active model release.")
    args = parser.parse_args(argv)

    try:
        if args.action == "validate":
            import json

            from sleepops.dataset import load_dataset
            from sleepops.files import write_json

            _, manifest = load_dataset(args.archive, args.lock)
            if args.write:
                write_json(args.write, manifest)
            print(json.dumps(manifest, indent=2))
        elif args.action in {"train", "bootstrap"}:
            from sleepops.lifecycle import store_candidate

            store_candidate(args, imported=args.action == "bootstrap")
        elif args.action == "evaluate":
            from sleepops.lifecycle import evaluate_release

            evaluate_release(args)
        elif args.action == "promote":
            from sleepops.lifecycle import promote_run

            promote_run(args, args.run)
        elif args.action == "activate":
            from sleepops.registry import activate

            print(f"Active release {activate(args.registry, args.release).name}")
        else:
            from sleepops.registry import champion, validate_bundle

            path = champion(args.registry)
            metadata, validation, _ = validate_bundle(path)
            print(f"Active release {path.name}, selected epoch {validation['selected_epoch']}\n"
                  f"Dataset SHA-256 {metadata['archive_sha256']}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"SleepOps error: {error}\n")


if __name__ == "__main__":
    main()
