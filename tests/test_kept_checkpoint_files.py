from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mechbench_compute import bench, stored_files
from mechbench_compute import resume as rm
from mechbench_compute.protocol import ProtocolExecutor
from mechbench_compute.protocol.resolver import Resolver
from tests.test_keep_checkpoints import (  # noqa: F401
    KEPT,
    MODEL,
    RECORDS,
    RESULTS,
    run_graph,
    tiny_hub,
    train_node,
)


@pytest.fixture(autouse=True)
def store(monkeypatch):
    objects: dict[str, object] = {}
    files: dict[str, bytes] = {}

    def emit(path, payload, **kw):
        objects[path] = payload
        return {"path": path}

    def fetch(path, **kw):
        if path not in objects:
            raise bench.BenchError(f"no object at {path}")
        got = {"payload": objects[path]}
        return (got, {"content_hash": ""}) if kw.get("with_meta") else got

    def put_file(label, path, **kw):
        files[label] = Path(path).read_bytes()
        return {"sizeBytes": len(files[label])}

    def get_file_chunks(label, **kw):
        data = files[label]
        yield data[: len(data) // 2]
        yield data[len(data) // 2:]

    def list_prefix_hashes(prefix, **kw):
        return {k[len(prefix) + 1:]: hashlib.sha256(v).hexdigest()
                for k, v in files.items() if k.startswith(prefix + "/")}

    monkeypatch.setattr(bench, "emit", emit)
    monkeypatch.setattr(bench, "fetch", fetch)
    monkeypatch.setattr(bench, "put_file", put_file)
    monkeypatch.setattr(bench, "get_file_chunks", get_file_chunks)
    monkeypatch.setattr(bench, "list_prefix_hashes", list_prefix_hashes)
    return {"objects": objects, "files": files}


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setattr(bench, "MAX_OBJECT_BYTES", 4096)


def has_bytes(v) -> bool:
    return stored_files.count_bytes(v) > 0


class TestTheLayout:
    def test_under_the_limit_the_object_is_unchanged(self):
        payload = {"kind": "adapter/lora", "data": b"x" * 100}
        assert not stored_files.needs_files(payload, limit=1 << 20)
        assert not stored_files.needs_files({"kind": "x", "values": [1, 2]}, limit=10)

    def test_over_the_limit_each_bytes_value_is_a_file_and_reads_back(self, store):
        a, b = b"a" * 3000, b"b" * 3000
        payload = {"kind": "collection", "items": [{"id": "step-1", "data": a},
                                                   {"id": "step-2", "data": b},
                                                   {"id": "step-3", "data": a}]}
        assert stored_files.needs_files(payload, limit=4096)
        out = stored_files.offload(payload, "u/p/results/j_1/kept", bench.put_file)
        assert out["storage"] == "files" and not has_bytes(out)
        assert len(store["files"]) == 2
        refs = [i["data"]["$file"] for i in out["items"]]
        assert refs[0] == refs[2] and refs[0]["bytes"] == 3000
        assert all(r["path"].startswith("u/p/results/j_1/kept/files/") for r in refs)
        back = stored_files.restore(out, bench.get_file_chunks)
        assert back == payload
        assert rm.content_hash(back) == rm.content_hash(payload)

    def test_files_the_store_already_holds_are_not_sent_again(self, store, monkeypatch):
        payload = {"items": [{"data": b"z" * 5000}]}
        target = "u/p/results/j_1/kept"
        stored_files.offload(payload, target, bench.put_file)
        sent: list[str] = []
        stored_files.offload(payload, target, lambda lab, path: sent.append(lab),
                             have=bench.list_prefix_hashes(target))
        assert sent == []

    def test_a_file_with_the_wrong_hash_is_refused(self, store):
        out = stored_files.offload({"items": [{"data": b"q" * 5000}]},
                                   "u/p/results/j_1/kept", bench.put_file)
        (path,) = store["files"]
        store["files"][path] = b"tampered"
        with pytest.raises(ValueError, match="wrong hash"):
            stored_files.restore(out, bench.get_file_chunks)

    def test_a_file_path_that_does_not_parse_is_refused(self):
        bad = {"storage": "files", "items": [{"data": {"$file": {
            "path": "../../etc/passwd", "bytes": 1, "sha256": "0"}}}]}
        with pytest.raises(Exception, match="cannot store at"):
            stored_files.restore(bad, lambda p: b"")


@pytest.mark.usefixtures("tiny_hub")
class TestKeptCheckpointsPastTheLimit:
    def test_any_number_store_as_files_and_read_back_bit_identical(self, store, monkeypatch):
        reference = run_graph([train_node(keep_checkpoints=True)], outputs=KEPT)
        small = dict(store["objects"])
        assert not store["files"]
        assert "storage" not in small[f"{RESULTS}/kept"]
        assert has_bytes(small[f"{RESULTS}/kept"])
        store["objects"].clear()

        monkeypatch.setattr(bench, "MAX_OBJECT_BYTES", 4096)
        out = run_graph([train_node(keep_checkpoints=True)], outputs=KEPT)
        kept = store["objects"][f"{RESULTS}/kept"]
        assert kept["storage"] == "files" and not has_bytes(kept)
        assert len(kept["items"]) == 3
        assert store["objects"][f"{RESULTS}/adapter"]["storage"] == "files"
        assert len([f for f in store["files"] if f.startswith(f"{RESULTS}/kept/files/")]) == 3
        assert rm.content_hash(out["kept"]) == rm.content_hash(reference["kept"])

        fetched = Resolver(bound_params={}).fetch_object(f"{RESULTS}/kept")
        assert rm.content_hash(fetched) == rm.content_hash(small[f"{RESULTS}/kept"])
        assert [i["data"] for i in fetched["items"]] == [
            i["data"] for i in small[f"{RESULTS}/kept"]["items"]]

    def test_a_ref_to_a_files_collection_feeds_measure(self, store, small_limit):
        run_graph([train_node(keep_checkpoints=True)], outputs=KEPT)
        assert store["objects"][f"{RESULTS}/kept"]["storage"] == "files"
        nodes = [{"id": "wrote", "block": "adapter/measure", "params": {},
                  "inputs": {"adapter": {"$ref": {"bench": f"{RESULTS}/kept"}}}}]
        out = run_graph(nodes, outputs=[{"name": "wrote", "from": {"node": "wrote"}}])
        assert out["wrote"]["measured"]["adapters"] == 3
        assert sorted({i["coords"]["step"] for i in out["wrote"]["items"]}) == [2, 4, 6]

    def test_a_model_reference_to_a_files_adapter_loads_its_bytes(self, store, monkeypatch):
        run_graph([train_node()], outputs=[KEPT[0]])
        plain = store["objects"].pop(f"{RESULTS}/adapter")
        monkeypatch.setattr(bench, "MAX_OBJECT_BYTES", 4096)
        run_graph([train_node()], outputs=[KEPT[0]])
        assert store["objects"][f"{RESULTS}/adapter"]["storage"] == "files"
        resolver = Resolver(bound_params={})
        params = resolver.resolve_node_params(
            {"params": {"model": {"base": MODEL, "adapters": [{"bench": f"{RESULTS}/adapter"}]}}},
            "activations/capture")
        (payload,) = params["model"].adapter_payloads
        assert payload["data"] == plain["data"]
        cap = {"id": "cap", "block": "activations/capture",
               "params": {"model": {"base": MODEL, "adapters": [{"bench": f"{RESULTS}/adapter"}]},
                          "layers": [1], "position": "last"},
               "inputs": {"records": RECORDS}}
        out = run_graph([cap], outputs=[{"name": "cap", "from": {"node": "cap"}}])
        assert len(out["cap"]["items"]) == len(RECORDS)

    def test_a_done_node_restores_its_files_checkpoints(self, store, small_limit, monkeypatch):
        nodes = [train_node(keep_checkpoints=True),
                 {"id": "order", "block": "records/sort", "params": {"by": ["-coords.step"]}}]
        edges = [{"from": {"node": "train", "output": "checkpoints"},
                  "to": {"node": "order", "port": "records"}}]
        outputs = [{"name": "order", "from": {"node": "order"}}]
        done: dict[str, tuple[str, str]] = {}
        executor = ProtocolExecutor(on_node_done=lambda nid, path, fp: done.setdefault(nid, (path, fp)))
        run_graph(nodes, edges, outputs=outputs, executor=executor)
        assert store["objects"][f"{RESULTS}/nodes/train/checkpoints"]["storage"] == "files"
        reference = Resolver(bound_params={}).fetch_object(f"{RESULTS}/order")
        path, fingerprint = done["train"]

        from mechbench_compute.ops.adapter import train

        monkeypatch.setattr(train, "run", lambda *a, **k: pytest.fail("must not train"))
        run_graph(nodes, edges, outputs=outputs,
                  resume={"train": {"fingerprint": fingerprint, "done": path}})
        again = Resolver(bound_params={}).fetch_object(f"{RESULTS}/order")
        assert rm.content_hash(again) == rm.content_hash(reference)
        assert [i["id"] for i in again["items"]] == ["step-6", "step-4", "step-2"]
        assert all(isinstance(i["data"], bytes) for i in again["items"])
