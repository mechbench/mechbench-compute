from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from mechbench_compute import architectures, ops, tensors
from mechbench_compute import model as model_mod
from mechbench_compute import shapes as S
from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.capture_rows import CaptureRows
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.activations import capture as capture_op
from mechbench_compute.ops.activations import capture_tokens as capture_tokens_op
from mechbench_compute.ops.activations import differentiate as differentiate_op
from mechbench_compute.ops.records import derive as derive_op
from mechbench_compute.ops.records import filter as filter_op
from mechbench_compute.ops.records import group as group_op
from mechbench_compute.ops.records import join as join_op
from mechbench_compute.ops.records import python as python_op
from mechbench_compute.ops.records import sort as sort_op
from mechbench_compute.ops.records import union as union_op
from mechbench_compute.ops.trajectory import capture as trajectory_op
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from tests.tiny_models import build_tiny_model

RECORDS = [{"id": "a", "text": "the cat sat on a"}, {"id": "b", "text": "a dog ran to the"}]

D = 32
LAYERS = 4


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("llama", BY_MODEL_TYPE["llama"])


def run_capture(model, _stored=True, **params):
    return capture_op.capture_residual_vectors(model, RECORDS, {"layers": "all", **params},
                                               stored=_stored)


def run_tokens(model, _stored=True, **params):
    return capture_tokens_op.capture_tokens(model, RECORDS, {"layers": "all", "positions": "last", **params},
                                            stored=_stored)


def run_differentiate(model, _stored=True, **params):
    records = [{**r, "tracked": {"t": {"id": 12}}} for r in RECORDS]
    return differentiate_op.differentiate_activations(model, records, {"layers": "all", **params},
                                                      stored=_stored)


def run_trajectory(model, _stored=True, **params):
    return trajectory_op.capture(model, RECORDS, {"axis": "layers", **params}, stored=_stored)


OPS = {
    "capture": (run_capture, capture_op, "MAX_VECTOR_FLOATS"),
    "capture-tokens": (run_tokens, capture_tokens_op, "MAX_TOKEN_VECTOR_FLOATS"),
    "differentiate": (run_differentiate, differentiate_op, "MAX_VECTOR_FLOATS"),
    "trajectory": (run_trajectory, None, "MAX_VECTOR_FLOATS"),
}

CAPTURED = len(RECORDS) * LAYERS * D


def lower_ceiling(monkeypatch, which: str, to: int) -> None:
    _, module, name = OPS[which]
    if module is None:
        from mechbench_compute import interp

        monkeypatch.setattr(interp, name, to)
    else:
        monkeypatch.setattr(module, name, to)


@pytest.mark.parametrize("which", sorted(OPS))
class TestEveryCaptureTakesStorage:
    def test_json_under_the_ceiling_is_the_rows_in_the_object(self, tiny, which):
        run = OPS[which][0]
        out = run(tiny, storage="json")
        assert not tensors.is_tensor(out) and len(out["items"]) == CAPTURED // D
        assert out == run(tiny)

    def test_json_past_the_ceiling_refuses_naming_tensor(self, tiny, which, monkeypatch):
        lower_ceiling(monkeypatch, which, CAPTURED - 1)
        with pytest.raises(ValueError, match=r'exceeds the \d+ cap .* `storage: "tensor"`'):
            OPS[which][0](tiny, storage="json")

    def test_tensor_has_no_ceiling_and_holds_the_same_rows(self, tiny, which, monkeypatch):
        run = OPS[which][0]
        plain = run(tiny, storage="json")
        lower_ceiling(monkeypatch, which, 1)
        out = run(tiny, storage="tensor")
        assert tensors.is_tensor(out) and out["items"] == [] and out["d"] == D
        assert out["n_items"] == len(plain["items"])
        rows = list(K.items_of(out))
        for got, want in zip(rows, plain["items"], strict=True):
            assert got["space"] == want["space"] and got["coords"] == want["coords"]
            assert np.allclose(got["vector"], want["vector"], rtol=1e-6, atol=1e-6)

    def test_auto_is_json_under_the_ceiling_and_tensor_above(self, tiny, which, monkeypatch):
        run = OPS[which][0]
        lower_ceiling(monkeypatch, which, CAPTURED)
        assert not tensors.is_tensor(run(tiny))
        lower_ceiling(monkeypatch, which, CAPTURED - 1)
        assert tensors.is_tensor(run(tiny))

    def test_an_unstored_result_has_no_ceiling(self, tiny, which, monkeypatch):
        run = OPS[which][0]
        plain = run(tiny, storage="json")
        lower_ceiling(monkeypatch, which, 1)
        for storage in ("json", "auto"):
            out = run(tiny, storage=storage, _stored=False)
            assert not tensors.is_tensor(out) and out == plain


class TestCaptureRows:
    def test_auto_moves_the_rows_held_so_far_into_shards(self):
        rows = CaptureRows("auto", ceiling=8)
        rows.add(S.vector([0.5, 0.25], S.space(model="m", layer=0, point="resid_post", d=2), id="a"))
        rows.plan(4)
        assert rows.writer is None and len(rows) == 1
        rows.plan(10)
        rows.add(S.vector([0.125, 0.5], S.space(model="m", layer=1, point="resid_post", d=2), id="b"))
        out = rows.finish("activations/vector", model="m")
        assert tensors.is_tensor(out) and [r["id"] for r in K.items_of(out)] == ["a", "b"]

    def test_tensor_storage_of_rows_of_two_widths_says_what_to_do(self):
        rows = CaptureRows("tensor", ceiling=8)
        rows.add(S.vector([0.5, 0.25], S.space(model="m", layer=0, point="attn.weights", d=2)))
        with pytest.raises(ValueError, match="one width .*`storage: \"json\"`"):
            rows.add(S.vector([0.5, 0.25, 0.25], S.space(model="m", layer=0, point="attn.weights", d=3)))

    def test_storage_is_auto_json_or_tensor(self, tiny):
        with pytest.raises(ValueError, match="storage is 'auto', 'json' or 'tensor'"):
            run_capture(tiny, storage="shards")

    def test_projected_trajectories_stay_json_under_tensor(self, tiny):
        direction = {"kind": "direction/vector", "vector": [1.0] + [0.0] * (D - 1),
                     "space": {"layer": 0, "point": "resid_post", "d": D}}
        out = trajectory_op.capture(tiny, RECORDS, {"axis": "layers", "storage": "tensor"},
                                    project=direction)
        assert not tensors.is_tensor(out) and out["projected"]


def build_rows(n: int = 6) -> list[dict]:
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n):
        v = np.round(rng.uniform(-1, 1, 8), 5)
        row = S.vector(v, S.space(model="m", layer=i % 3, point="resid_post", d=8),
                       id=f"r{i // 3}", coords={"position": i, "label": "ab"[i % 2]},
                       token={"id": i, "text": f"t{i}"})
        row["top"] = [{"dim": int(np.argmax(np.abs(v))), "value": float(v[np.argmax(np.abs(v))]),
                       "share": 0.5}]
        rows.append(row)
    return rows


@pytest.fixture
def both(tmp_path):
    rows = build_rows()
    writer = tensors.ShardWriter(tmp_path / "w", max_rows=4)
    for row in rows:
        writer.add(row)
    header = {"model": "m", "point": "resid_post", "layers": [0, 1, 2], "d_model": 8}
    return (K.collection("activations/vector", rows, **header),
            tensors.collection("activations/vector", writer.close(), **header))


def run_op(module, inputs, params):
    return module.run(ops.Context(), inputs, params)


def canon(x):
    return json.loads(json.dumps(x, sort_keys=True))


class TestRecordsReadATensorCollectionAsJson:
    @pytest.mark.parametrize("module, params", [
        (derive_op, {"fields": {"big": "norm > 1"}}),
        (derive_op, {"drop": ["vector"]}),
        (filter_op, {"where": "coords.label == \"a\""}),
        (sort_op, {"by": ["-norm"]}),
        (group_op, {"by": {"layer": "space.layer"}, "aggregates": {"n": "count()"}}),
        (python_op, {"source": "def transform(records, header, params):\n"
                               "    return [{'id': r['id'], 'n': len(r['vector']), 'top': r['top'][0]['dim'],\n"
                               "             'storage': header.get('storage')} for r in records]\n"}),
    ])
    def test_the_op_answers_as_it_does_on_json(self, both, module, params):
        plain, sharded = both
        got = run_op(module, {"records": sharded}, params)
        assert not tensors.is_tensor(got) and tensors.LOCAL_DIR not in json.dumps(got)
        assert canon(got) == canon(run_op(module, {"records": plain}, params))

    def test_a_derive_dropping_the_vector_is_a_json_collection(self, both):
        _, sharded = both
        got = run_op(derive_op, {"records": sharded}, {"drop": ["vector"]})
        assert got["kind"] == K.COLLECTION and "storage" not in got and "shards" not in got
        assert len(got["items"]) == 6 and all("vector" not in r for r in got["items"])
        assert got["items"][0]["top"] and isinstance(got["items"][0]["coords"]["position"], int)

    def test_join_and_union(self, both):
        plain, sharded = both
        right = K.collection("records/record", [{"id": "r0", "flag": 1}, {"id": "r1", "flag": 2}])
        params = {"on": "id"}
        assert canon(run_op(join_op, {"left": sharded, "right": right}, params)) == canon(
            run_op(join_op, {"left": plain, "right": right}, params))
        got = union_op.union({"x": sharded, "y": plain}, {})
        want = union_op.union({"x": plain, "y": plain}, {})
        assert not tensors.is_tensor(got) and canon(got) == canon(want)


@pytest.fixture
def tiny_hub(monkeypatch, tmp_path):
    tiny = build_tiny_model("llama")
    architecture = dataclasses.replace(
        architectures.BY_MODEL_TYPE["llama"],
        load=lambda model_id, **_: (tiny._model, tiny._processor))
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": "llama"})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)
    return tiny


MASSIVE_SOURCE = """\
import statistics


def transform(records, header, params):
    offset = params["gain_offset"]
    caps = [r for r in records if r["coords"].get("source") == "cap"]
    gains = [r for r in records if r["coords"].get("source") == "gains"]
    by_layer = {}
    for r in caps:
        top = r.get("top") or []
        if top:
            by_layer.setdefault(r["space"]["layer"], []).append(top[0])
    layers = sorted(by_layer)
    rows = []
    votes = {}
    for layer in layers:
        tops = by_layer[layer]
        count = {}
        for t in tops:
            count[t["dim"]] = count.get(t["dim"], 0) + 1
            votes[t["dim"]] = votes.get(t["dim"], 0) + 1
        dim, n = max(count.items(), key=lambda kv: (kv[1], -kv[0]))
        rows.append({"id": "L%03d" % layer, "layer": layer, "top_dim": dim, "records": len(tops),
                     "median_share": round(statistics.median(t["share"] for t in tops if t["dim"] == dim), 6)})
    dim, n = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    summary = {"id": "summary", "dim": dim, "votes": n, "layers": len(layers)}
    if gains:
        values = gains[0]["values"]
        summary.update({"d_model": len(values), "final_norm_gain": round(offset + values[dim], 6)})
    rows.append(summary)
    return rows
"""


def build_massive(battery, keep="all"):
    return {
        "params": {"model": "tiny/llama", "top": 4},
        "keep": keep,
        "outputs": [{"name": "massive", "from": {"node": "massive"}},
                    {"name": "by_layer", "from": {"node": "by_layer"}}],
        "graph": {"dataflow": 2, "nodes": [
            {"id": "cap", "block": "activations/capture", "inputs": {"records": battery},
             "params": {"model": {"$param": "model"}, "layers": "all", "point": "resid_post",
                        "position": "last", "top": {"$param": "top"}}},
            {"id": "slim", "block": "records/derive", "inputs": {}, "params": {"drop": ["vector"]}},
            {"id": "gains", "block": "weights/capture", "inputs": {},
             "params": {"model": {"$param": "model"}, "points": ["norm"], "values": True}},
            {"id": "both", "block": "records/union", "inputs": {}, "params": {"batch_axis": "source"}},
            {"id": "rows", "block": "records/python", "inputs": {},
             "params": {"source": MASSIVE_SOURCE, "params": {"gain_offset": 1.0}}},
            {"id": "massive", "block": "records/filter", "inputs": {}, "params": {"where": "id == \"summary\""}},
            {"id": "by_layer", "block": "records/filter", "inputs": {}, "params": {"where": "id != \"summary\""}},
        ], "edges": [
            {"from": {"node": "cap"}, "to": {"node": "slim", "port": "records"}},
            {"from": {"node": "slim"}, "to": {"node": "both", "port": "cap"}},
            {"from": {"node": "gains"}, "to": {"node": "both", "port": "gains"}},
            {"from": {"node": "both"}, "to": {"node": "rows", "port": "records"}},
            {"from": {"node": "rows"}, "to": {"node": "massive", "port": "records"}},
            {"from": {"node": "rows"}, "to": {"node": "by_layer", "port": "records"}},
        ]},
    }


BATTERY = tuple({"id": f"q{i:02d}", "user": f"question {i} the cat sat"} for i in range(13))


class TestTheMassiveGraph:

    def run_massive(self, monkeypatch, keep):
        seen = {}
        real = capture_op.capture_residual_vectors

        def spy(*args, **kwargs):
            seen["out"] = real(*args, **kwargs)
            return seen["out"]

        monkeypatch.setattr(capture_op, "capture_residual_vectors", spy)
        monkeypatch.setattr(capture_op, "MAX_VECTOR_FLOATS", 100)
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                            extra=build_massive(list(BATTERY), keep))
        payload = ProtocolExecutor().run(spec).payload
        assert not payload.get("nodes_missing"), payload.get("nodes_missing")
        return seen["out"], payload["outputs"]

    def test_a_capture_past_the_ceiling_goes_to_tensor_and_the_graph_completes(self, tiny_hub, monkeypatch):
        cap, outputs = self.run_massive(monkeypatch, "all")
        assert tensors.is_tensor(cap) and cap["n_items"] == 13 * LAYERS
        summary = outputs["massive"]["items"]
        assert [r["id"] for r in summary] == ["summary"]
        assert summary[0]["layers"] == LAYERS and summary[0]["d_model"] == D
        by_layer = outputs["by_layer"]["items"]
        assert [r["layer"] for r in by_layer] == list(range(LAYERS))
        assert all(r["records"] == 13 for r in by_layer)

    def test_unstored_the_capture_is_rows_in_memory_and_answers_the_same(self, tiny_hub, monkeypatch):
        _, stored = self.run_massive(monkeypatch, "all")
        cap, held = self.run_massive(monkeypatch, "outputs")
        assert not tensors.is_tensor(cap) and len(cap["items"]) == 13 * LAYERS
        assert canon(held["massive"]["items"]) == canon(stored["massive"]["items"])
        assert canon(held["by_layer"]["items"]) == canon(stored["by_layer"]["items"])
