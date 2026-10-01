from __future__ import annotations

import pytest
import mechbench_schema
from mechbench_schema.identity import parse_path

from mechbench_compute import bench


def _schema_version() -> tuple[int, ...]:
    return tuple(int(p) for p in mechbench_schema.__version__.split(".")[:3])


def test_the_schema_under_test_admits_dots():
    assert _schema_version() >= (0, 20, 0), (
        f"mechbench-schema {mechbench_schema.__version__} from {mechbench_schema.__file__} "
        "predates the dot rule; compute pins >=0.20.0"
    )


@pytest.mark.parametrize(
    "path",
    [
        "owner/proj/results/j_1/nodes/train.checkpoints",
        "owner/proj/adapters/l23/shards/adapters.safetensors",
        "owner/proj/merged/shards/model-00001-of-00002.safetensors",
    ],
)
def test_a_named_output_and_a_shard_name_parse(path):
    assert parse_path(path).leaf == path.rsplit("/", 1)[1]


def test_an_emit_to_a_dotted_named_output_is_sent(monkeypatch):
    bench.configure(api_url="https://api.test", api_key="k")
    sent: list[str] = []

    def fake_request(method, url, key, body=None, headers=None, **kw):
        sent.append(url)
        return {"path": url.split("/objects/", 1)[1], "hash": "sha256:x", "size": 0}

    monkeypatch.setattr(bench, "_request", fake_request)
    bench.emit("owner/proj/results/j_1/nodes/train.checkpoints", {"kind": "note", "text": "hi"})
    assert [u.split("/objects/", 1)[1] for u in sent] == ["owner/proj/results/j_1/nodes/train.checkpoints"]
