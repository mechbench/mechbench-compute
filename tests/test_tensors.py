"""The tensor store (task 000613): rows in shards beside the object,
read one shard at a time, fetched verified, uploaded once."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mechbench_compute.ops.activations import capture_tokens as capture_tokens_op
from mechbench_compute import tensors
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.direction.regress import fit_regression


def _items(n, d=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append({"id": f"p{i // 3}", "vector": rng.normal(size=d).astype(np.float32), "n_tok": i,
                    "space": {"model": "m", "layer": i % 2, "point": "resid_post", "d": d},
                    "coords": {"position": i, "surprisal": round(float(rng.random()), 4), "topic": "t"},
                    "token": {"id": i, "text": f"w{i}"}})
    return out


class TestWriteAndRead:
    def test_rows_split_into_shards_and_come_back_the_same(self, tmp_path):
        items = _items(10)
        w = tensors.ShardWriter(tmp_path / "s", max_rows=4)
        for it in items:
            w.add(it)
        coll = tensors.collection("activations/vector", w.close(), model="m")
        assert tensors.is_tensor(coll) and coll["items"] == []
        assert [s["rows"] for s in coll["shards"]] == [4, 4, 2] and coll["n_items"] == 10 and coll["d"] == 6
        back = K.items_of(coll)
        assert isinstance(back, tensors.ShardedItems) and len(back) == 10
        for a, b in zip(back, items):
            assert a["id"] == b["id"] and a["space"] == b["space"] and a["token"] == b["token"]
            assert a["coords"] == b["coords"]
            assert np.array_equal(a["vector"], b["vector"])

    def test_indexing_loads_one_shard_at_a_time(self, tmp_path):
        items = _items(10)
        w = tensors.ShardWriter(tmp_path / "s", max_rows=4)
        for it in items:
            w.add(it)
        back = K.items_of(tensors.collection("activations/vector", w.close()))
        assert back[9]["coords"]["position"] == 9 and back._loaded[0] == 2
        assert back[0]["coords"]["position"] == 0 and back._loaded[0] == 0
        assert [it["coords"]["position"] for it in back[3:6]] == [3, 4, 5]
        assert [len(s) for s in back.shards()] == [4, 4, 2]

    def test_a_field_some_items_lack_is_absent_not_nan(self, tmp_path):
        w = tensors.ShardWriter(tmp_path / "s")
        w.add({"id": "a", "vector": np.ones(3, np.float32), "coords": {"surprisal": 1.5}})
        w.add({"id": "b", "vector": np.ones(3, np.float32), "coords": {}})
        back = list(K.items_of(tensors.collection("activations/vector", w.close())))
        assert back[0]["coords"] == {"surprisal": 1.5} and back[1]["coords"] == {}

    def test_one_width_per_collection(self, tmp_path):
        w = tensors.ShardWriter(tmp_path / "s")
        w.add({"id": "a", "vector": np.ones(3, np.float32)})
        with pytest.raises(ValueError, match="one width"):
            w.add({"id": "b", "vector": np.ones(4, np.float32)})

    def test_unmaterialized_shards_are_refused_by_name(self):
        coll = {"kind": "collection", "item_kind": "activations/vector", "storage": "tensor",
                "items": [], "shards": [{"name": "shard-0000.safetensors", "rows": 1}]}
        with pytest.raises(ValueError, match="materialized"):
            list(K.items_of(coll))


class TestMoveBetweenMachines:
    def _written(self, tmp_path):
        w = tensors.ShardWriter(tmp_path / "local", max_rows=4)
        for it in _items(10):
            w.add(it)
        return tensors.collection("activations/vector", w.close())

    def test_upload_sends_each_shard_once_and_strips_the_local_path(self, tmp_path):
        coll = self._written(tmp_path)
        sent = []
        emitted = tensors.upload(coll, "you/lab/big", lambda label, path: sent.append((label, path.name)),
                                 have={"shards/shard-0001.safetensors": coll["shards"][1]["sha256"]})
        assert sent == [("you/lab/big/shards/shard-0000.safetensors", "shard-0000.safetensors"),
                        ("you/lab/big/shards/shard-0002.safetensors", "shard-0002.safetensors")]
        assert "_shard_dir" not in emitted and emitted["shards"] == coll["shards"]

    def test_materialize_fetches_verifies_and_caches(self, tmp_path):
        coll = self._written(tmp_path)
        emitted = tensors.upload(coll, "you/lab/big", lambda label, path: None)
        store = {f"you/lab/big/shards/{s['name']}": (tmp_path / "local" / s["name"]).read_bytes()
                 for s in coll["shards"]}
        fetches = []

        def fetch(label):
            fetches.append(label)
            data = store[label]
            return [data[:100], data[100:]]

        got = tensors.materialize(emitted, "you/lab/big", fetch, tmp_path / "cache")
        assert len(fetches) == 3
        back = list(K.items_of(got))
        assert [it["coords"]["position"] for it in back] == list(range(10))
        # Cached: a second materialize fetches nothing.
        again = tensors.materialize(emitted, "you/lab/big", fetch, tmp_path / "cache")
        assert len(fetches) == 3 and again["_shard_dir"] == got["_shard_dir"]

    def test_a_corrupt_shard_is_refused(self, tmp_path):
        coll = self._written(tmp_path)
        emitted = tensors.upload(coll, "you/lab/big", lambda label, path: None)
        bad = {f"you/lab/big/shards/{s['name']}": b"not the bytes" for s in coll["shards"]}
        with pytest.raises(ValueError, match="wrong hash"):
            tensors.materialize(emitted, "you/lab/big", lambda label: bad[label], tmp_path / "cache2")
        assert not (tmp_path / "cache2").exists() or not any((tmp_path / "cache2").iterdir())


class TestARegressionStreamsTheShards:
    def test_a_planted_direction_is_recovered_from_shards(self, tmp_path):
        """A million-token capture and a thousand-token one cost the same
        memory in `direction/regress`: two passes over the shards, never
        the matrix. Here: a planted axis on 600 rows over five shards."""
        from mechbench_compute import directions as dirs

        rng = np.random.default_rng(3)
        d = 12
        axis = rng.normal(size=d); axis /= np.linalg.norm(axis)
        w = tensors.ShardWriter(tmp_path / "s", max_rows=128)
        for i in range(600):
            v = rng.normal(size=d).astype(np.float32)
            signal = float(v @ axis) * 3.0 + rng.normal() * 0.1
            w.add({"id": f"p{i // 20}", "vector": v,
                   "space": {"model": "m", "layer": 4, "point": "resid_post", "d": d},
                   "coords": {"position": i % 20, "surprisal": round(signal, 5)}})
        coll = tensors.collection("activations/vector", w.close(), model="m", layers=[4])
        assert len(coll["shards"]) == 5
        out = fit_regression(coll, layer=4, target="surprisal", seed=1)
        got = np.asarray(out["vector"], dtype=np.float64)
        cos = float(got @ axis / np.linalg.norm(got))
        assert cos > 0.99, cos
        assert out["derivation"]["r2_test"] > 0.95 and out["derivation"]["n_items"] == 600
        assert out["derivation"]["n_train"] + out["derivation"]["n_test"] == 600


class TestTheExecutorMovesShards:
    """A node that emits a tensor collection: its shards go up as raw
    objects under the result's label and its header is emitted stripped
    of the local path — while a consumer in the same job still reads
    the rows from the local shards. And a $ref to a stored one is
    materialized before its consumer runs."""

    def test_emit_uploads_shards_and_the_consumer_reads_locally(self, tmp_path, monkeypatch):
        from mechbench_compute import bench, blocks, interp
        from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

        put, emitted = [], {}
        monkeypatch.setattr(bench, "put_file", lambda label, path, **kw: put.append((label, path.name)) or {"sizeBytes": 1})
        monkeypatch.setattr(bench, "list_prefix_hashes", lambda prefix: {})
        monkeypatch.setattr(bench, "emit", lambda target, payload, **kw: emitted.__setitem__(target, payload) or {"path": target})

        def fake_capture(model, records, params, on_item=None, on_start=None):
            w = tensors.ShardWriter(tmp_path / "w", max_rows=4)
            for it in _items(10):
                w.add(it)
            return tensors.collection("activations/vector", w.close(), model="m", layers=[0, 1])

        monkeypatch.setattr(capture_tokens_op, "capture_tokens", fake_capture)
        monkeypatch.setattr(ProtocolExecutor, "_model_loaded", lambda self, model_id: object())
        monkeypatch.setattr(ProtocolExecutor, "_run_model_block",
                            lambda self, fn, inputs, params, *a, **k: fn(inputs, params, *a, **k))
        graph = {"dataflow": 2, "nodes": [
            {"id": "cap", "block": "activations/capture-tokens", "params": {"model": "fake/m", "layers": [0, 1]},
             "inputs": {"records": [{"id": "r", "user": "x"}]}},
            {"id": "sum", "block": "records/summarize", "params": {"value": "n_tok", "by": []}, "inputs": {}},
        ], "edges": [{"from": {"node": "cap"}, "to": {"node": "sum", "port": "records"}}]}
        out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                                  extra={"graph": graph, "resultPath": "you/lab/results/j1"}))
        # The consumer read all ten rows from the local shards…
        assert out.payload["outputs"]["sum"]["rows"][0]["n"] == 10
        # …the shards went up under the result's label…
        assert [p[0] for p in put] == [f"you/lab/results/j1/cap/shards/shard-000{k}.safetensors" for k in range(3)]
        # …and the emitted header carries the shards, not the local path.
        header = emitted["you/lab/results/j1/cap"]
        assert header["storage"] == "tensor" and len(header["shards"]) == 3 and "_shard_dir" not in header
        assert header["items"] == []

    def test_a_ref_to_a_stored_tensor_collection_is_materialized(self, tmp_path, monkeypatch):
        from mechbench_compute import bench, blocks
        from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

        w = tensors.ShardWriter(tmp_path / "w", max_rows=4)
        for it in _items(10):
            w.add(it)
        stored = tensors.upload(tensors.collection("activations/vector", w.close()), "you/lab/big",
                                lambda label, path: None)
        files = {f"you/lab/big/shards/{s['name']}": (tmp_path / "w" / s["name"]).read_bytes() for s in stored["shards"]}
        monkeypatch.setattr(bench, "fetch", lambda ref, with_meta=False: ({"payload": stored}, {"content_hash": "sha256:x"}))
        monkeypatch.setattr(bench, "get_file_chunks", lambda label: [files[label]])
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
        graph = {"dataflow": 2, "nodes": [
            {"id": "sum", "block": "records/summarize", "params": {"value": "n_tok", "by": []},
             "inputs": {"records": {"$ref": {"bench": "you/lab/big"}}}}], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
        assert out.payload["outputs"]["sum"]["rows"][0]["n"] == 10
        assert (tmp_path / "home" / ".mechbench" / "tensors").exists()
