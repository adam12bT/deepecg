import socket
from pathlib import Path

import httpx
import pytest

from deploy.training_space import datasets


def public_dns(host, port, *, type):
    address = "127.0.0.1" if host == "localhost" else "93.184.216.34"
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]


def test_download_public_prepared_archive_is_streamed_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.socket, "getaddrinfo", public_dns)
    payload = b"prepared-eeg-npz-archive"

    def handle(request):
        return httpx.Response(200, headers={"content-length": str(len(payload))}, content=payload)

    real_client = httpx.Client
    monkeypatch.setattr(datasets.httpx, "Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs))
    destination = tmp_path / "linked.npz"
    size = datasets.download_public_archive("https://data.example.org/eeg/features.npz", destination)
    assert size == len(payload)
    assert destination.read_bytes() == payload


def test_public_archive_links_require_https_public_hosts_and_safe_redirects(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.socket, "getaddrinfo", public_dns)
    for url in ("http://data.example.org/features.npz", "https://user:secret@data.example.org/features.npz",
                "https://data.example.org:8443/features.npz", "https://127.0.0.1/features.npz"):
        with pytest.raises(ValueError):
            datasets.validate_public_archive_url(url)

    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://localhost/private.npz"})

    real_client = httpx.Client
    monkeypatch.setattr(datasets.httpx, "Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(ValueError, match="public Internet address"):
        datasets.download_public_archive("https://data.example.org/shared/features.npz", tmp_path / "bad.npz")
    assert len(calls) == 1


def test_linked_archives_over_the_limit_are_rejected_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.socket, "getaddrinfo", public_dns)

    def handle(request):
        return httpx.Response(200, headers={"content-length": str(datasets.MAX_UPLOAD + 1)}, content=b"")

    real_client = httpx.Client
    monkeypatch.setattr(datasets.httpx, "Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(datasets.DatasetTooLarge, match="24 MiB"):
        datasets.download_public_archive("https://data.example.org/large.npz", tmp_path / "large.npz")
