from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy.huggingface.release import (STAGE_MODEL, build_manifest, deployed_build_is_healthy, sha256,
                                        space_host, space_target, stage)


class StubClient:
    def __init__(self, details, prediction):
        self.details = details
        self.prediction = prediction
        self.predictions = 0

    def get(self, url):
        data = self.details if url.endswith("/api/model") else None
        return SimpleNamespace(status_code=200, json=lambda: data, text='<meta content="sleepops-react">')

    def post(self, url, *, json):
        assert url.endswith("/api/v1/predict")
        assert len(json["features"]) == 10 and all(len(row) == 11 for row in json["features"])
        self.predictions += 1
        return SimpleNamespace(status_code=200, json=lambda: self.prediction)


@pytest.mark.serving_assets
def test_new_build_manifest_binds_source_revision_to_active_model(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    first = build_manifest()
    second = build_manifest()
    assert first["source_revision"] == "a" * 40
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
    assert first["release_id"] == second["release_id"] and first["build_id"] != second["build_id"]


def test_release_check_rejects_stale_app_or_mismatched_weights():
    expected = {"build_id": "current", "release_id": "model", "archive_sha256": "archive",
                "checkpoint_sha256": "checkpoint"}
    host = "https://owner-sleepops.hf.space"
    prediction = {"release_id": expected["release_id"], "state": "awake", "asleep_probabilities": [0.1] * 10,
                  "forecast": None}
    for field in ("build_id", "release_id", "archive_sha256", "checkpoint_sha256"):
        stale = StubClient({**expected, field: "stale"}, prediction)
        assert not deployed_build_is_healthy(stale, host, expected)
        assert stale.predictions == 0
    current = StubClient(expected, prediction)
    assert deployed_build_is_healthy(current, host, expected)
    assert current.predictions == 1
    prediction.update(state="asleep", forecast={"bins": []})
    assert not deployed_build_is_healthy(current, host, expected)
    prediction["forecast"]["bins"] = [{}] * 13
    assert deployed_build_is_healthy(current, host, expected)


def test_private_space_host_is_restricted_to_https_hf_spaces():
    assert space_host(SimpleNamespace(host=None, subdomain="owner-sleepops")) == "https://owner-sleepops.hf.space"
    for host in ("http://owner.hf.space", "https://hf.space.example.com", "https://owner.hf.space/path"):
        with pytest.raises(ValueError, match="hf.space"):
            space_host(SimpleNamespace(host=host, subdomain=None))


def test_public_production_space_requires_explicit_allowlist():
    class Api:
        def space_info(self, repo_id):
            return SimpleNamespace(sdk="docker", private=False, sha="head")

        def list_repo_files(self, repo_id, *, repo_type, revision):
            return ["deploy/huggingface/serve.py"]

    api = Api()
    with pytest.raises(ValueError, match="private Docker Space"):
        space_target(api, "owner/assistance", {"name": "owner"}, reuse_existing=True)
    result = space_target(api, "owner/assistance", {"name": "owner"}, reuse_existing=True, allow_public=True)
    assert result.private is False and result.sdk == "docker"


@pytest.mark.serving_assets
def test_prediction_space_bundle_contains_the_locked_stage_preview():
    source = Path(STAGE_MODEL)
    assert source.is_file(), "Restore the DVC stage-preview checkpoint before packaging a production release."
    with TemporaryDirectory() as temporary:
        bundle = stage(Path(temporary) / "space")
        bundled = bundle / STAGE_MODEL
        assert bundled.is_file()
        assert sha256(bundled) == sha256(source)
