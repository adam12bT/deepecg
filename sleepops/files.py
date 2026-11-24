"""Content fingerprints and atomic JSON publication."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as buffer:
        pending = Path(buffer.name)
        try:
            buffer.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        except Exception:
            buffer.close()
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
