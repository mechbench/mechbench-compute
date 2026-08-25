"""The merge op (000312 Arc C), IO faked at module seams.

The shard arithmetic is test_checkpoint's business; what is worth
asserting here is the op's CONTRACT: destinations validated, files
published under the right prefix, the manifest emitted with adapter
lineage, and the pointer result naming a base the grammar can load.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mechbench_compute import bench, model_ref
from mechbench_compute import protocol as protocol_mod


def _tiny_snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    w = mx.random.normal((4, 4))
    mx.save_safetensors(
        str(snap / "model-00001-of-00001.safetensors"),
        {"model.layers.0.self_attn.q_proj.weight": w},
    )
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors"}})
    )
    (snap / "config.json").write_text("{}")
    return snap


def _adapter_payload():
    import os
    import tempfile

    a = mx.random.normal((2, 4))
    b = mx.random.normal((4, 2))
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    mx.save_safetensors(path, {
        "model.layers.0.self_attn.q_proj.lora_a": a,
        "model.layers.0.self_attn.q_proj.lora_b": b,
    })
    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)
    return {"kind": "adapter", "data": data, "lora": {"rank": 2, "alpha": 4}}


@pytest.fixture()
def executor():
    # The block only touches self._materialize_checkpoint on bench
    # bases; a bare object with the real methods bound is enough.
    class E:
        _block_merge = protocol_mod.ProtocolExecutor._block_merge
        _materialize_checkpoint = protocol_mod.ProtocolExecutor._materialize_checkpoint
    return E()


def _ref(payload):
    ref = model_ref.parse({"base": "org/m@rev", "adapters": [{"bench": "me/p/a1"}]})
    return model_ref.ModelRef(
        base_kind=ref.base_kind, base=ref.base,
        adapter_labels=ref.adapter_labels, adapter_payloads=(payload,))


class TestBenchDestination:
    def test_publishes_files_manifest_and_pointer(self, executor, tmp_path, monkeypatch):
        snap = _tiny_snapshot(tmp_path)
        monkeypatch.setattr(
            "mechbench_compute.hub.ensure_model",
            lambda ref, **k: ("org/m", "rev", snap))
        puts, emits = [], []
        monkeypatch.setattr(bench, "list_prefix_hashes", lambda prefix, **k: {})
        monkeypatch.setattr(bench, "put_file",
                            lambda label, path, **k: puts.append(label) or {"sizeBytes": Path(path).stat().st_size})
        monkeypatch.setattr(bench, "emit",
                            lambda label, payload, **k: emits.append((label, payload, k)) or {})
        out = executor._block_merge(
            {}, {"model": _ref(_adapter_payload()),
                 "to": {"bench": {"name": "fair-v1"}}},
            result_base="benji/training/results/j_x")
        assert out["kind"] == "model_pointer"
        assert out["base"] == {"bench": "benji/training/checkpoints/fair-v1"}
        assert {p.rsplit("/", 1)[-1] for p in puts} == {
            "config.json", "model-00001-of-00001.safetensors",
            "model.safetensors.index.json"}
        (label, payload, kwargs) = emits[0]
        assert label.endswith("/manifest")
        assert payload["kind"] == "checkpoint_manifest"
        assert kwargs["inputs"] == ["me/p/a1"]  # adapter lineage

    def test_a_bare_base_refuses(self, executor):
        bare = model_ref.parse("org/m@rev")
        with pytest.raises(ValueError, match="adapter"):
            executor._block_merge({}, {"model": model_ref.ModelRef(
                base_kind=bare.base_kind, base=bare.base)}, result_base="a/b/results/j")

    def test_destination_is_mandatory_and_explicit(self, executor):
        with pytest.raises(ValueError, match="bench"):
            executor._block_merge(
                {}, {"model": _ref(_adapter_payload())},
                result_base="a/b/results/j")


class TestHfDestination:
    def test_needs_a_token_and_says_where_to_get_one(self, executor, tmp_path, monkeypatch):
        snap = _tiny_snapshot(tmp_path)
        monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                            lambda ref, **k: ("org/m", "rev", snap))
        with pytest.raises(ValueError, match="Integrations"):
            executor._block_merge(
                {}, {"model": _ref(_adapter_payload()),
                     "to": {"hf": {"repo": "me/merged"}}},
                result_base="a/b/results/j", secrets={})


class TestRetryAsResume:
    def test_files_the_server_already_holds_are_skipped(
        self, executor, tmp_path, monkeypatch
    ):
        """A failed upload's intact files stay on the server (torn ones
        were deleted by the hash check); the retry asks first and skips
        byte-identical matches — resume, not restart."""
        import hashlib

        snap = _tiny_snapshot(tmp_path)
        monkeypatch.setattr(
            "mechbench_compute.hub.ensure_model",
            lambda ref, **k: ("org/m", "rev", snap))
        # pretend config.json survived the failed run
        config_sha = hashlib.sha256((snap / "config.json").read_bytes()).hexdigest()
        monkeypatch.setattr(
            bench, "list_prefix_hashes",
            lambda prefix, **k: {"config.json": config_sha})
        puts = []
        monkeypatch.setattr(bench, "put_file",
                            lambda label, path, **k: puts.append(label) or {"sizeBytes": 1})
        monkeypatch.setattr(bench, "emit", lambda *a, **k: {})
        out = executor._block_merge(
            {}, {"model": _ref(_adapter_payload()),
                 "to": {"bench": {"name": "resume-v1"}}},
            result_base="benji/training/results/j_x")
        uploaded = {p.rsplit("/", 1)[-1] for p in puts}
        assert "config.json" not in uploaded
        assert out["skipped_already_stored"] == 1

