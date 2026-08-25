"""The merge is a delta-shard rewrite (000312 Arc C).

Tiny real tensors, real mx arithmetic: the assertions that matter are
W' == W + scale·(B@A) on exactly the targeted tensors, byte-identical
copies of everything else, honest manifests, and a materializer that
refuses corruption rather than caching it.
"""

import hashlib
import json

import mlx.core as mx
import pytest

from mechbench_compute import checkpoint


def _adapter_payload(layer=0, proj="q_proj", rank=2, alpha=4.0, dim=4):
    import os
    import tempfile

    a = mx.random.normal((rank, dim))
    b = mx.random.normal((dim, rank))
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    mx.save_safetensors(
        path,
        {
            f"model.layers.{layer}.self_attn.{proj}.lora_a": a,
            f"model.layers.{layer}.self_attn.{proj}.lora_b": b,
        },
    )
    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)
    return (
        {"kind": "adapter", "data": data, "lora": {"rank": rank, "alpha": alpha}},
        a,
        b,
        alpha / rank,
    )


@pytest.fixture()
def snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    w = mx.random.normal((4, 4))
    other = mx.random.normal((3, 3))
    mx.save_safetensors(
        str(snap / "model-00001-of-00002.safetensors"),
        {"model.layers.0.self_attn.q_proj.weight": w},
    )
    mx.save_safetensors(
        str(snap / "model-00002-of-00002.safetensors"),
        {"model.embed.weight": other},
    )
    (snap / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                    "model.embed.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (snap / "config.json").write_text('{"model_type": "test"}')
    return snap, w


class TestExportMerged:
    def test_targeted_tensor_gets_the_delta_and_nothing_else_changes(
        self, snapshot, tmp_path
    ):
        snap, w = snapshot
        payload, a, b, scale = _adapter_payload()
        out = tmp_path / "out"
        files = checkpoint.export_merged(snap, [payload], out)
        assert set(files) == {
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
        }
        merged = dict(mx.load(str(out / "model-00001-of-00002.safetensors")))
        want = w + (scale * (b @ a)).astype(w.dtype)
        assert bool(mx.allclose(merged["model.layers.0.self_attn.q_proj.weight"], want, atol=1e-5))
        # the untouched shard and config are byte-identical to the base
        for name in ("model-00002-of-00002.safetensors", "config.json"):
            assert (out / name).read_bytes() == (snap / name).read_bytes()

    def test_a_two_round_stack_sums_in_order(self, snapshot, tmp_path):
        snap, w = snapshot
        p1, a1, b1, s1 = _adapter_payload()
        p2, a2, b2, s2 = _adapter_payload(alpha=8.0)
        out = tmp_path / "out"
        checkpoint.export_merged(snap, [p1, p2], out)
        merged = dict(mx.load(str(out / "model-00001-of-00002.safetensors")))
        want = w + (s1 * (b1 @ a1) + s2 * (b2 @ a2)).astype(w.dtype)
        assert bool(mx.allclose(merged["model.layers.0.self_attn.q_proj.weight"], want, atol=1e-5))

    def test_an_unsharded_snapshot_refuses_by_name(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        payload, *_ = _adapter_payload()
        with pytest.raises(ValueError, match="index"):
            checkpoint.export_merged(empty, [payload], tmp_path / "out")


class TestManifest:
    def test_names_every_file_with_its_true_hash(self, snapshot, tmp_path):
        snap, _ = snapshot
        payload, *_ = _adapter_payload()
        out = tmp_path / "out"
        files = checkpoint.export_merged(snap, [payload], out)
        man = checkpoint.build_manifest(
            out, files, {"base": {"hf": "org/m@rev"}, "adapters": []}, "org/m@rev"
        )
        assert man["kind"] == "checkpoint_manifest"
        assert {f["name"] for f in man["files"]} == set(files)
        for f in man["files"]:
            digest = hashlib.sha256((out / f["name"]).read_bytes()).hexdigest()
            assert f["sha256"] == digest


class TestMaterialize:
    def _man_and_store(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_bytes(b'{"a":1}')
        (src / "w.safetensors").write_bytes(b"\x00\x01\x02")
        man = checkpoint.build_manifest(
            src, ["config.json", "w.safetensors"], {"base": {"hf": "x"}, "adapters": []}, "x"
        )
        return man, {n: (src / n).read_bytes() for n in ("config.json", "w.safetensors")}

    def test_fetches_verifies_and_is_idempotent(self, tmp_path):
        man, store = self._man_and_store(tmp_path)
        calls = []

        def fetch(name):
            calls.append(name)
            return store[name]

        d1 = checkpoint.materialize(man, fetch, tmp_path / "cache")
        assert (d1 / "config.json").read_bytes() == b'{"a":1}'
        n_first = len(calls)
        d2 = checkpoint.materialize(man, fetch, tmp_path / "cache")
        assert d2 == d1
        assert len(calls) == n_first  # complete mark short-circuits

    def test_a_corrupt_fetch_caches_nothing(self, tmp_path):
        man, store = self._man_and_store(tmp_path)
        d = None
        with pytest.raises(ValueError, match="wrong hash"):
            d = checkpoint.materialize(
                man, lambda _n: b"not the bytes", tmp_path / "cache"
            )
        assert d is None
        # and a later honest fetch succeeds from scratch
        ok = checkpoint.materialize(man, lambda n: store[n], tmp_path / "cache")
        assert (ok / "w.safetensors").read_bytes() == b"\x00\x01\x02"
