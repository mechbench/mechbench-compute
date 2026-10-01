from __future__ import annotations

import dataclasses
import os
import tempfile

import numpy as np
import pytest

from mechbench_compute import architectures, lexicon
from mechbench_compute import model as model_mod
from mechbench_compute import resume as rm
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from tests.tiny_models import build_tiny_model

MODEL = "tiny/llama"

RECORDS = [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"}]


@pytest.fixture
def tiny_hub(monkeypatch, tmp_path):
    def load(model_id, **_):
        tiny = build_tiny_model("llama")
        return tiny._model, tiny._processor

    architecture = dataclasses.replace(architectures.BY_MODEL_TYPE["llama"], load=load)
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": "llama"})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)


def train_node(**extra):
    return {"id": "train", "block": "adapter/train",
            "params": {"model": MODEL, "target": {"uniform": ["cat", "dog"]},
                       "steps": 6, "lr": 1e-2, "seed": 3, "closer": " mat",
                       "lora": {"rank": 2, "alpha": 4}, "checkpoint_every": 2, **extra},
            "inputs": {"records": RECORDS}}


@pytest.fixture(autouse=True)
def store(monkeypatch):
    from mechbench_compute import bench

    kept: dict[str, object] = {}

    def emit(path, payload, **kw):
        kept[path] = payload
        return {"path": path}

    def fetch(path, **kw):
        if path not in kept:
            raise bench.BenchError(f"no object at {path}")
        return {"payload": kept[path]}

    monkeypatch.setattr(bench, "emit", emit)
    monkeypatch.setattr(bench, "fetch", fetch)
    return kept


RESULTS = "u/p/results/j_1"


def run_graph(nodes, edges=(), outputs=None, executor=None, resume=None):
    extra = {"graph": {"dataflow": 2, "nodes": nodes, "edges": list(edges)},
             "resultPath": RESULTS}
    if outputs is not None:
        extra["outputs"] = outputs
    spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra=extra)
    return (executor or ProtocolExecutor()).run(spec, resume=resume).payload["outputs"]


def stored(store, name):
    return store[f"{RESULTS}/{name}"]


KEPT = [{"name": "adapter", "from": {"node": "train"}},
        {"name": "kept", "from": {"node": "train", "output": "checkpoints"}}]


def read_weights(data: bytes) -> dict[str, np.ndarray]:
    from mechbench_compute.lora import load_adapter

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)
        return {k: np.array(v) for k, v in load_adapter(path).items()}
    finally:
        os.unlink(path)


class TestDeclaration:
    def test_the_param_the_output_and_the_needs(self):
        op = lexicon.BY_NAME["adapter/train"]
        param = next(p for p in op.params if p.name == "keep_checkpoints")
        assert param.type == "bool" and param.default is False
        assert op.output.kind == "adapter/lora" and not op.output.collection
        out = op.outputs["checkpoints"]
        assert out.kind == "adapter/lora" and out.collection
        assert "coords.step" in out.doc and "loss" in out.doc
        assert op.needs == frozenset({"model.backward", "model.forward"})
        assert op.resume.level == "state-restorable" and op.resume.items
        assert rm.resume_level("adapter/train") == "state-restorable"

    def test_the_kind_keys_a_collection_of_adapters_by_step(self):
        kind = lexicon.kinds.BY_KIND["adapter/lora"]
        assert kind.key == ("id", "coords")
        assert {"loss", "coords", "id"} <= set(kind.fields)
        assert "checkpoint_every" in kind.header


class TestKeepCheckpoints:
    def test_one_adapter_per_kept_step_and_the_final_adapter_unchanged(self, tiny_hub, store):
        assert "kept" in run_graph([train_node(keep_checkpoints=True)], outputs=KEPT)
        kept = {"adapter": stored(store, "adapter"), "kept": stored(store, "kept")}
        store.clear()
        assert "kept" not in run_graph([train_node()], outputs=[KEPT[0]])
        plain = {"adapter": stored(store, "adapter")}
        assert f"{RESULTS}/kept" not in store
        assert rm.content_hash(kept["adapter"]) == rm.content_hash(plain["adapter"])
        items = kept["kept"]["items"]
        assert [i["id"] for i in items] == ["step-2", "step-4", "step-6"]
        assert [i["coords"] for i in items] == [{"step": 2}, {"step": 4}, {"step": 6}]
        assert all(isinstance(i["loss"], float) for i in items)
        assert items[-1]["data"] == plain["adapter"]["data"]
        assert round(items[-1]["loss"], 4) == plain["adapter"]["train"]["final_loss"]
        assert items[0]["data"] != items[1]["data"] != items[2]["data"]
        for item in items:
            for field in ("format", "base_model", "trained_on", "lora"):
                assert item[field] == plain["adapter"][field]
            assert item["kind"] == "adapter/lora"
            assert item["train"] == {k: v for k, v in plain["adapter"]["train"].items()
                                     if k != "final_loss"}
        header = kept["kept"]
        assert header["item_kind"] == "adapter/lora" and header["checkpoint_every"] == 2
        assert header["train"] == plain["adapter"]["train"]

    def test_the_last_step_is_kept_when_the_cadence_misses_it(self, tiny_hub):
        node = train_node(keep_checkpoints=True, checkpoint_every=4)
        kept = run_graph([node], outputs=KEPT)
        assert [i["coords"]["step"] for i in kept["kept"]["items"]] == [4, 6]

    def test_the_step_coordinate_reads_through_derive_and_sort(self, tiny_hub):
        nodes = [train_node(keep_checkpoints=True),
                 {"id": "later", "block": "records/derive",
                  "params": {"fields": {"tenfold": "coords.step * 10"}, "drop": ["data"]}},
                 {"id": "order", "block": "records/sort", "params": {"by": ["-coords.step"]}}]
        edges = [{"from": {"node": "train", "output": "checkpoints"},
                  "to": {"node": "later", "port": "records"}},
                 {"from": {"node": "later"}, "to": {"node": "order", "port": "records"}}]
        out = run_graph(nodes, edges, outputs=[{"name": "order", "from": {"node": "order"}}])
        items = out["order"]["items"]
        assert [(i["id"], i["tenfold"], i["rank"]) for i in items] == [
            ("step-6", 60, 1), ("step-4", 40, 2), ("step-2", 20, 3)]

    def test_measure_reads_the_whole_collection_as_a_sweep_over_step(self, tiny_hub):
        nodes = [train_node(keep_checkpoints=True),
                 {"id": "wrote", "block": "adapter/measure", "params": {}}]
        edges = [{"from": {"node": "train", "output": "checkpoints"},
                  "to": {"node": "wrote", "port": "adapter"}}]
        out = run_graph(nodes, edges, outputs=[{"name": "wrote", "from": {"node": "wrote"}}])
        items = out["wrote"]["items"]
        steps = sorted({i["coords"]["step"] for i in items})
        assert steps == [2, 4, 6]
        per_step = len(items) // 3
        assert per_step > 0 and len(items) == 3 * per_step
        for step in steps:
            shares = [i["mass_share"] for i in items if i["coords"]["step"] == step]
            assert abs(sum(shares) - 1.0) < 1e-9
        assert out["wrote"]["measured"]["adapters"] == 3

    def test_a_map_over_the_collection_feeds_each_adapter_to_a_model(self, tiny_hub):
        body = {"nodes": [{"id": "cap", "block": "activations/capture",
                           "params": {"model": MODEL, "layers": [1], "position": "last"},
                           "inputs": {"records": RECORDS}}],
                "edges": [{"from": {"input": "record"}, "to": {"node": "cap", "port": "adapter"}}]}
        nodes = [train_node(keep_checkpoints=True),
                 {"id": "each", "block": "records/map", "params": {"body": body}}]
        edges = [{"from": {"node": "train", "output": "checkpoints"},
                  "to": {"node": "each", "port": "records"}}]
        out = run_graph(nodes, edges, outputs=[{"name": "each", "from": {"node": "each"}}])
        items = out["each"]["items"]
        assert sorted({i["coords"]["step"] for i in items}) == [2, 4, 6]
        assert all(i["id"].startswith(f"step-{i['coords']['step']}:") for i in items)
        vectors = {(i["coords"]["step"], i["id"].split(":", 1)[1]): i["vector"] for i in items}
        shared = sorted(k for s, k in vectors if s == 2)
        assert shared and all(vectors[(2, k)] != vectors[(6, k)] for k in shared)

    def test_several_adapters_on_one_adapter_port_are_refused(self):
        from mechbench_compute.protocol.model import read_one_adapter

        one = {"kind": "adapter/lora", "data": b"x"}
        assert read_one_adapter(lexicon.collection("adapter/lora", [one])) == one
        with pytest.raises(ValueError, match="records/map"):
            read_one_adapter(lexicon.collection("adapter/lora", [one, one]))

    def test_without_the_flag_an_edge_from_the_checkpoints_fails(self, tiny_hub):
        nodes = [train_node(), {"id": "order", "block": "records/sort",
                                "params": {"by": ["coords.step"]}}]
        edges = [{"from": {"node": "train", "output": "checkpoints"},
                  "to": {"node": "order", "port": "records"}}]
        with pytest.raises(ValueError, match="gave no output 'checkpoints'"):
            run_graph(nodes, edges)

    def test_an_output_the_op_does_not_have_is_refused(self, tiny_hub):
        with pytest.raises(ValueError, match="has no output 'snapshots'"):
            run_graph([train_node()], outputs=[
                {"name": "x", "from": {"node": "train", "output": "snapshots"}}])

    def test_a_cadence_of_zero_is_refused(self, tiny_hub):
        with pytest.raises(ValueError, match="above 0"):
            run_graph([train_node(keep_checkpoints=True, checkpoint_every=0)])


class _Interrupted(BaseException):
    pass


class _Spool:
    def __init__(self, stop_at: int | None = None):
        self.stop_at = stop_at
        self.fingerprints: dict[str, str] = {}
        self.items: dict[str, dict] = {}
        self.checkpoints: list[dict] = []

    def executor(self):
        return ProtocolExecutor(
            on_node_start=lambda nid, fp: self.fingerprints.setdefault(nid, fp),
            on_spool_item=lambda nid, key, item: self.items.__setitem__(key, item),
            on_checkpoint=self.checkpoint)

    def checkpoint(self, nid, state):
        self.checkpoints.append(state)
        if self.stop_at is not None and state["step"] >= self.stop_at:
            raise _Interrupted

    def resume_map(self):
        return {"train": {"fingerprint": self.fingerprints["train"],
                          "items": dict(self.items),
                          "checkpoint": self.checkpoints[-1]}}


class TestResume:
    def test_a_resumed_run_keeps_its_items_and_matches_an_unbroken_one(self, tiny_hub, store):
        whole = _Spool()
        run_graph([train_node(keep_checkpoints=True)], outputs=KEPT, executor=whole.executor())
        reference = {name: stored(store, name) for name in ("adapter", "kept")}
        store.clear()
        assert sorted(whole.items) == ["step-2", "step-4"]
        assert [c["step"] for c in whole.checkpoints] == [2, 4]

        broken = _Spool(stop_at=4)
        with pytest.raises(_Interrupted):
            run_graph([train_node(keep_checkpoints=True)], outputs=KEPT,
                      executor=broken.executor())
        assert sorted(broken.items) == ["step-2", "step-4"]
        at_four = broken.checkpoints[-1]
        assert at_four["step"] == 4
        assert read_weights(broken.items["step-4"]["data"]).keys() == at_four["weights"].keys()
        for key, weights in read_weights(broken.items["step-4"]["data"]).items():
            assert weights.tobytes() == np.asarray(at_four["weights"][key]).tobytes()

        resumed = _Spool()
        resumed.fingerprints = dict(broken.fingerprints)
        ticks: list[tuple[int, int]] = []
        executor = resumed.executor()
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"dataflow": 2, "nodes": [train_node(keep_checkpoints=True)], "edges": []},
            "outputs": KEPT, "resultPath": RESULTS})
        executor.run(spec, resume=broken.resume_map(),
                     on_progress=lambda d, t: ticks.append((d, t)))
        out = {name: stored(store, name) for name in ("adapter", "kept")}
        assert resumed.items == {}
        assert resumed.checkpoints == []
        assert ticks[-1][0] == ticks[-1][1]
        assert rm.content_hash(out["kept"]) == rm.content_hash(reference["kept"])
        assert rm.content_hash(out["adapter"]) == rm.content_hash(reference["adapter"])

    def test_a_resume_missing_a_kept_item_is_refused(self, tiny_hub):
        broken = _Spool(stop_at=4)
        with pytest.raises(_Interrupted):
            run_graph([train_node(keep_checkpoints=True)], outputs=KEPT,
                      executor=broken.executor())
        resume = broken.resume_map()
        del resume["train"]["items"]["step-2"]
        with pytest.raises(ValueError, match="kept at step 2"):
            run_graph([train_node(keep_checkpoints=True)], outputs=KEPT, resume=resume)

    def test_a_done_node_restores_its_checkpoints_without_training(self, tiny_hub, store, monkeypatch):
        nodes = [train_node(keep_checkpoints=True),
                 {"id": "order", "block": "records/sort", "params": {"by": ["-coords.step"]}}]
        edges = [{"from": {"node": "train", "output": "checkpoints"},
                  "to": {"node": "order", "port": "records"}}]
        outputs = [{"name": "order", "from": {"node": "order"}}]
        done: dict[str, tuple[str, str]] = {}
        executor = ProtocolExecutor(on_node_done=lambda nid, path, fp: done.setdefault(nid, (path, fp)))
        run_graph(nodes, edges, outputs=outputs, executor=executor)
        reference = stored(store, "order")
        path, fingerprint = done["train"]
        assert f"{RESULTS}/nodes/train/checkpoints" in store

        from mechbench_compute.ops.adapter import train

        monkeypatch.setattr(train, "run", lambda *a, **k: pytest.fail("must not train"))
        run_graph(nodes, edges, outputs=outputs,
                  resume={"train": {"fingerprint": fingerprint, "done": path}})
        assert rm.content_hash(stored(store, "order")) == rm.content_hash(reference)


SORT = {"id": "order", "block": "records/sort", "params": {"by": ["-coords.step"]}}
FROM_CHECKPOINTS = [{"from": {"node": "train", "output": "checkpoints"},
                     "to": {"node": "order", "port": "records"}}]


class TestAddresses:
    @pytest.mark.parametrize("edges, outputs, written", [
        (FROM_CHECKPOINTS, [{"name": "order", "from": {"node": "order"}}],
         ["nodes/train", "nodes/train/checkpoints", "order"]),
        (FROM_CHECKPOINTS, [{"name": "adapter", "from": {"node": "train"}},
                            {"name": "order", "from": {"node": "order"}}],
         ["adapter", "nodes/train/checkpoints", "order"]),
        ((), KEPT, ["adapter", "kept"]),
        (FROM_CHECKPOINTS, None, ["order", "train", "train/checkpoints"]),
    ], ids=["intermediate", "output-node", "declared", "undeclared"])
    def test_every_path_the_executor_writes_parses(self, tiny_hub, store, edges, outputs, written):
        from mechbench_schema import parse_path

        nodes = [train_node(keep_checkpoints=True), *([SORT] if edges else [])]
        run_graph(nodes, edges, outputs=outputs)
        assert sorted(store) == [f"{RESULTS}/{w}" for w in written]
        for path in store:
            parse_path(path)

    def test_the_dotted_spelling_does_not_parse(self):
        from mechbench_schema import InvalidPathError, parse_path

        with pytest.raises(InvalidPathError, match="train.checkpoints"):
            parse_path(f"{RESULTS}/nodes/train.checkpoints")

    def test_a_named_output_at_a_refused_path_fails_its_node(self, tiny_hub, store, monkeypatch):
        from mechbench_schema import InvalidPathError

        from mechbench_compute import bench
        from mechbench_compute.contained import check_target
        from mechbench_compute.protocol import store_result

        def emit_checked(path, payload, **kw):
            check_target(path)
            store[path] = payload
            return {"path": path}

        monkeypatch.setattr(bench, "emit", emit_checked)
        monkeypatch.setattr(store_result, "name_output_targets",
                            lambda state, nid, name: [f"{state.result_base}/nodes/{nid}.{name}"])
        with pytest.raises(InvalidPathError, match=r"cannot store at '.*/nodes/train\.checkpoints'"):
            run_graph([train_node(keep_checkpoints=True), SORT], FROM_CHECKPOINTS,
                      outputs=[{"name": "order", "from": {"node": "order"}}])
        assert not any(p.endswith("train.checkpoints") for p in store)
