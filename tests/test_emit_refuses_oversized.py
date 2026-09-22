"""`bench.emit` refuses a body over the API's object limit before any
bytes leave the machine.

The server returns 413 for the same ceiling; this is the version that
says so in one line instead of after five retries of a minute each.
"""

from __future__ import annotations

import pytest

from mechbench_compute import bench


@pytest.fixture
def configured(monkeypatch):
    bench.configure(api_url="https://api.test", api_key="k")
    sent: list[tuple[str, str, int]] = []

    def fake_request(method, url, key, body=None, headers=None, **kw):
        sent.append((method, url, len(body or b"")))
        return {"path": url.split("/objects/", 1)[1], "hash": "sha256:x", "size": 0}

    monkeypatch.setattr(bench, "_request", fake_request)
    return sent


def test_an_oversized_payload_is_refused_locally(configured, monkeypatch):
    monkeypatch.setattr(bench, "MAX_OBJECT_BYTES", 10_000)
    big = {"kind": "records", "records": [{"id": str(i), "text": "x" * 100}
                                            for i in range(500)]}
    with pytest.raises(bench.BenchError) as caught:
        bench.emit("owner/proj/results/big", big)
    msg = str(caught.value)
    assert "10,000-byte" in msg and "bytes, over" in msg
    assert configured == [], "nothing may be sent for a refused body"


def test_a_payload_under_the_limit_is_sent(configured):
    bench.emit("owner/proj/results/small", {"kind": "note", "text": "hi"})
    assert len(configured) == 1
    assert configured[0][0] == "PUT"


def test_the_limit_is_the_servers(monkeypatch):
    """64 MiB, and it must be raised together with body_limit.ts."""
    assert bench.MAX_OBJECT_BYTES == 64 * 1024 * 1024
