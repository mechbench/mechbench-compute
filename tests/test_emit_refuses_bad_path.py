from __future__ import annotations

import pytest
from mechbench_schema import InvalidPathError

from mechbench_compute import bench, tensors

BAD = "owner/proj/results/j_1/nodes/train..checkpoints"


@pytest.fixture
def sent(monkeypatch):
    bench.configure(api_url="https://api.test", api_key="k")
    calls: list[str] = []

    def fake_request(method, url, key, body=None, headers=None, **kw):
        calls.append(url)
        return {"path": url.split("/objects/", 1)[1], "hash": "sha256:x", "size": 0}

    monkeypatch.setattr(bench, "_request", fake_request)
    return calls


def test_an_emit_to_a_path_the_grammar_refuses_is_refused_before_sending(sent):
    with pytest.raises(InvalidPathError, match=r"cannot store at .*train\.\.checkpoints"):
        bench.emit(BAD, {"kind": "note", "text": "hi"})
    assert sent == []


def test_a_tensor_upload_under_a_refused_path_puts_no_shard(tmp_path):
    put: list[str] = []
    collection = {tensors.LOCAL_DIR: str(tmp_path),
                  "shards": [{"name": "s0.safetensors", "sha256": "x"}]}
    with pytest.raises(InvalidPathError, match="cannot store at"):
        tensors.upload(collection, BAD, lambda label, path: put.append(label))
    assert put == []


def test_an_emit_to_a_parsing_path_is_sent(sent):
    bench.emit("owner/proj/results/j_1/nodes/train/checkpoints", {"kind": "note", "text": "hi"})
    assert len(sent) == 1
