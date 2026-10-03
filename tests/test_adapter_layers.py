from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mechbench_schema import dump_canonical
from mlx.utils import tree_flatten

from mechbench_compute import bench, checkpoint, lora, model_ref, resume
from mechbench_compute.distill import render
from mechbench_compute.lexicon._base import In, Op
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import merge as merge_op
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec, serialize_params
from tests.test_circuits import build_adapter
from tests.tiny_models import MODEL_TYPES, build_tiny_model

RECORD = {"id": "r", "user": "the cat sat on a mat"}
LABEL = "you/lab/die"


def read_stream(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    names = [f"blocks.{i}.resid_post" for i in range(len(model.lm.model.layers))]
    run = model.run(render(model, RECORD).array, capture=names)
    return {"residuals": [np.array(run.cache[n].astype(mx.float32)).tolist() for n in names],
            "logits": np.array(run.logits[0, -1].astype(mx.float32)).tolist()}


def build_op(*, port: bool):
    inputs = (In("adapter", "adapter/lora", "An adapter.", required=False),) if port else ()
    declared = Op(name="probe/stream", summary="Reads the residual stream.",
                  description="Reads the residual stream.", params=(), inputs=inputs,
                  needs=frozenset({"model.forward"}))
    return SimpleNamespace(op=declared, module=SimpleNamespace(run=read_stream), scope=None)


def build_weights(model, sign=1.0):
    return {k: v * sign if k.endswith("lora_b") else v for k, v in build_adapter(model).items()}


def build_payload(model, tmp_path, name, sign=1.0, alpha=2):
    path = tmp_path / f"{name}.safetensors"
    mx.save_safetensors(str(path), build_weights(model, sign))
    return {"data": path.read_bytes(), "lora": {"rank": 2, "alpha": alpha}}


def adapted(payload, layers=None):
    entry = {"bench": LABEL} if layers is None else {"bench": LABEL, "layers": layers}
    return model_ref.resolve({"base": "tiny", "adapters": [entry]}, fetch=lambda _label: payload)


def read(executor, model, port=None, **params):
    return executor._run_op(build_op(port=port is not None), {} if port is None else {"adapter": port},
                            {"model": model, **params})


def read_by_hand(model, *stack):
    handles = [lora.fuse(model.lm, weights, scale=scale) for weights, scale in stack]
    try:
        return read_stream(Context(loaded=model), {}, {})
    finally:
        for handle in reversed(handles):
            lora.restore(model.lm, handle)


def keep_layers(weights, layers):
    return {k: v for k, v in weights.items() if int(k.split(".")[2]) in layers}


@pytest.fixture(scope="module", params=MODEL_TYPES)
def tiny(request):
    return build_tiny_model(request.param)


def hold(model):
    ex = ProtocolExecutor()
    ex._model, ex._model_id = model, "tiny"
    return ex


@pytest.fixture
def executor(tiny):
    return hold(tiny)


@pytest.fixture
def payload(tiny, tmp_path):
    return build_payload(tiny, tmp_path, "die")


class TestAnAdapterFusedInChosenLayers:
    def test_every_layer_named_is_the_adapter_everywhere(self, executor, tiny, payload):
        everywhere = read(executor, adapted(payload))
        named = read(executor, adapted(payload, list(range(len(tiny.lm.model.layers)))))
        assert named["residuals"] == everywhere["residuals"]
        assert named["logits"] == everywhere["logits"]

    def test_no_layer_named_is_the_bare_model(self, executor, payload):
        bare = read(executor, "tiny")
        none = read(executor, adapted(payload, []))
        assert none["residuals"] == bare["residuals"]
        assert none["logits"] == bare["logits"]

    @pytest.mark.parametrize("layers", [[1, 3], [2]])
    def test_a_subset_moves_the_stream_from_its_first_layer_on(self, executor, tiny, payload, layers):
        bare, everywhere = read(executor, "tiny"), read(executor, adapted(payload))
        subset = read(executor, adapted(payload, layers))
        n = len(tiny.lm.model.layers)
        assert ([subset["residuals"][i] == bare["residuals"][i] for i in range(n)]
                == [i < layers[0] for i in range(n)])
        assert subset["logits"] != bare["logits"]
        assert subset["logits"] != everywhere["logits"]

    def test_a_subset_fuses_those_layers_deltas_and_no_others(self, executor, tiny, payload):
        subset = read(executor, adapted(payload, [1, 3]))
        by_hand = read_by_hand(tiny, (keep_layers(build_weights(tiny), [1, 3]), 1.0))
        assert subset["residuals"] == by_hand["residuals"]
        assert subset["logits"] == by_hand["logits"]

    def test_the_port_fuses_in_every_layer_on_top_of_a_chosen_set(self, executor, tiny, payload, tmp_path):
        extra = build_payload(tiny, tmp_path, "extra", -0.5)
        out = read(executor, adapted(payload, [1]), port=extra, adapter_scale=0.5)
        by_hand = read_by_hand(tiny, (keep_layers(build_weights(tiny), [1]), 1.0),
                               (build_weights(tiny, -0.5), 0.5))
        assert out["residuals"] == by_hand["residuals"]
        assert out["logits"] == by_hand["logits"]
        assert out["fused"] == [{"bench": LABEL, "layers": [1]}, {"port": "adapter", "scale": 0.5}]

    def test_the_model_is_bare_again_after_the_op(self, executor, tiny, payload):
        bare = read(executor, "tiny")
        read(executor, adapted(payload, [2]))
        assert read(executor, "tiny") == bare
        assert tiny.fused_reference is None


class TestALayerTheModelLacks:
    def test_refuses_by_code_and_leaves_the_model_bare(self, executor, tiny, payload):
        n = len(tiny.lm.model.layers)
        bare = read(executor, "tiny")
        stack = model_ref.resolve(
            {"base": "tiny", "adapters": ["you/lab/coin", {"bench": LABEL, "layers": [0, n, n + 2]}]},
            fetch=lambda _label: payload)
        with pytest.raises(ValueError, match=rf"^LAYER_OUT_OF_RANGE: an adapter's `layers` names {n}, "
                                             rf"{n + 2}, and this model's layers are 0 through {n - 1}$"):
            read(executor, stack)
        assert executor._model is None
        assert read(hold(tiny), "tiny") == bare
        assert tiny.fused_reference is None

    def test_refuses_on_a_node_with_the_port_too(self, executor, tiny, payload, tmp_path):
        n = len(tiny.lm.model.layers)
        bare = read(executor, "tiny")
        with pytest.raises(ValueError, match="^LAYER_OUT_OF_RANGE: "):
            read(executor, adapted(payload, [n]), port=build_payload(tiny, tmp_path, "extra"))
        assert executor._model is None
        assert read(hold(tiny), "tiny") == bare
        assert tiny.node_adapter is None


class TestTheFusedHeader:
    def test_an_adapter_without_layers_is_recorded_as_before(self, executor, payload):
        out = read(executor, adapted(payload))
        assert json.dumps(out["fused"]) == '[{"bench": "you/lab/die"}]'
        assert dump_canonical(out["fused"]) == dump_canonical([{"bench": "you/lab/die"}])

    def test_an_adapter_with_layers_records_them(self, executor, payload):
        out = read(executor, adapted(payload, [1, 2]))
        assert json.dumps(out["fused"]) == '[{"bench": "you/lab/die", "layers": [1, 2]}]'
        assert read(executor, adapted(payload, []))["fused"] == [{"bench": LABEL, "layers": []}]


class TestTheWireForm:
    def test_an_adapter_without_layers_serializes_as_before(self):
        ref = model_ref.parse({"base": "org/m", "adapters": [{"bench": "a/b/c"}, "a/b/d"]})
        assert json.dumps(ref.to_wire()) == (
            '{"base": {"hf": "org/m"}, "adapters": [{"bench": "a/b/c"}, {"bench": "a/b/d"}]}')
        assert ref.adapter_layers == (None, None)

    def test_an_adapter_with_layers_round_trips(self):
        wire = {"base": {"hf": "org/m"},
                "adapters": [{"bench": "a/b/c", "layers": [3, 4, 5]}, {"bench": "a/b/d"}]}
        ref = model_ref.parse(wire)
        assert ref.adapter_layers == ((3, 4, 5), None)
        assert ref.to_wire() == wire
        assert model_ref.parse(ref.to_wire()) == ref
        resolved = model_ref.resolve(wire, fetch=lambda _label: {"data": b"\x00"})
        assert resolved.adapter_layers == ((3, 4, 5), None) and resolved == ref

    def test_references_differing_only_in_layers_differ(self):
        def parse(entry):
            return model_ref.parse({"base": "org/m", "adapters": [entry]})

        assert parse({"bench": "a/b/c", "layers": [1]}) != parse({"bench": "a/b/c", "layers": [2]})
        assert parse({"bench": "a/b/c", "layers": [1]}) != parse("a/b/c")
        assert parse({"bench": "a/b/c", "layers": []}) != parse("a/b/c")
        assert parse({"bench": "a/b/c"}) == parse("a/b/c")

    @pytest.mark.parametrize("layers", [[3, 2], [2, 2], [-1], [True], [1.0], [[1]], "3", None])
    def test_layers_out_of_form_refuse_naming_the_adapter(self, layers):
        with pytest.raises(ValueError, match=r"^adapter a/b/c: `layers` is a list of layer indices"):
            model_ref.parse({"base": "org/m", "adapters": [{"bench": "a/b/c", "layers": layers}]})

    def test_another_key_beside_bench_refuses(self):
        with pytest.raises(ValueError, match="each adapter must be"):
            model_ref.parse({"base": "org/m", "adapters": [{"bench": "a/b/c", "layer": [1]}]})


def fingerprint_of(model):
    params = serialize_params({"model": model, "n": 2})
    return resume.node_fingerprint(block="~canonical/ops/text/generate", params=params,
                                   input_hashes=[], core_version="0.0.0", model=str(params["model"]))


class Stop(Exception):
    pass


def read_node_fingerprint(monkeypatch, entry):
    stored = {"payload": {"data": b"\x00", "lora": {"rank": 2, "alpha": 2}}}
    monkeypatch.setattr(bench, "fetch", lambda ref, with_meta=False: (
        (stored, {"content_hash": "sha256:0"}) if with_meta else stored))
    seen = []

    def stop(nid, fingerprint):
        seen.append(fingerprint)
        raise Stop

    graph = {"dataflow": 2, "edges": [], "nodes": [{
        "id": "gen", "block": "text/generate", "params": {"model": {"$param": "model"}, "n": 2},
        "inputs": {"records": [RECORD]}}]}
    spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
        "graph": graph, "params": {"model": {"base": "org/m", "adapters": [entry]}}})
    with pytest.raises(Stop):
        ProtocolExecutor(on_node_start=stop).run(spec)
    return seen[0]


class TestTheFingerprintSeesLayers:
    def test_an_adapter_without_layers_fingerprints_as_before(self):
        before = {"base": {"hf": "org/m"}, "adapters": [{"bench": LABEL}]}
        assert fingerprint_of(model_ref.parse(before)) == fingerprint_of(before)
        assert fingerprint_of(model_ref.parse({"base": "org/m", "adapters": [LABEL]})) == fingerprint_of(before)

    def test_references_differing_only_in_layers_fingerprint_apart(self):
        entries = ({"bench": LABEL}, {"bench": LABEL, "layers": [1]},
                   {"bench": LABEL, "layers": [2]}, {"bench": LABEL, "layers": []})
        prints = {fingerprint_of(model_ref.parse({"base": "org/m", "adapters": [e]})) for e in entries}
        assert len(prints) == len(entries)

    def test_a_run_s_node_fingerprint_sees_layers(self, monkeypatch):
        everywhere = read_node_fingerprint(monkeypatch, {"bench": LABEL})
        assert read_node_fingerprint(monkeypatch, LABEL) == everywhere
        one = read_node_fingerprint(monkeypatch, {"bench": LABEL, "layers": [1]})
        two = read_node_fingerprint(monkeypatch, {"bench": LABEL, "layers": [2]})
        assert len({everywhere, one, two}) == 3


class TestTrainingOnAChosenSet:
    def test_trained_on_records_the_layers(self, tmp_path):
        tiny = build_tiny_model("llama")
        payload = build_payload(tiny, tmp_path, "die")
        ref = model_ref.resolve(
            {"base": "tiny", "adapters": ["you/lab/coin", {"bench": LABEL, "layers": [1]}]},
            fetch=lambda _label: payload)
        out = train_op.run(Context(loaded=tiny),
                           {"records": [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"}]},
                           {"model": ref, "target": {"uniform": ["cat", "dog"]}, "steps": 1, "seed": 3,
                            "closer": " mat", "lora": {"rank": 2, "alpha": 4}})
        assert out["out"]["trained_on"] == {
            "base": "tiny", "adapters": ["you/lab/coin", {"bench": LABEL, "layers": [1]}]}


SHARD = "model-00001-of-00001.safetensors"


def write_snapshot(model, path):
    path.mkdir()
    weights = dict(tree_flatten(model.lm.parameters()))
    mx.save_safetensors(str(path / SHARD), weights)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, SHARD)}))
    shape = {"num_hidden_layers": model.arch.n_layers, "hidden_size": model.arch.d_model,
             "num_attention_heads": model.arch.n_heads, "num_key_value_heads": model.arch.n_kv_heads,
             "vocab_size": model.arch.vocab_size}
    nested = model.architecture.loader == "mlx-vlm"
    (path / "config.json").write_text(json.dumps(
        {"model_type": model.architecture.model_type, **({"text_config": shape} if nested else shape)}))
    return weights


def merge_by_hand(base, weights, layers, scale):
    out = dict(base)
    for name, w in base.items():
        stem = name.removesuffix(".weight")
        if f"{stem}.lora_a" in weights and int(name.split(".")[2]) in layers:
            out[name] = w + (scale * (weights[f"{stem}.lora_b"] @ weights[f"{stem}.lora_a"])).astype(w.dtype)
    return out


def assert_same(got, want):
    assert set(got) == set(want)
    for name in want:
        assert np.array_equal(np.array(got[name]), np.array(want[name])), name


def stub_bench(monkeypatch, snap):
    puts: dict[str, bytes] = {}
    emits: list[dict] = []
    monkeypatch.setattr("mechbench_compute.hub.ensure_model", lambda ref, **k: ("tiny", "rev", snap))
    monkeypatch.setattr(bench, "list_prefix_hashes", lambda prefix, **k: {})
    monkeypatch.setattr(bench, "put_file", lambda label, path, **k: puts.update(
        {label.rsplit("/", 1)[-1]: Path(path).read_bytes()}) or {"sizeBytes": 1})
    monkeypatch.setattr(bench, "emit", lambda label, payload, **k: emits.append(payload) or {})
    return puts, emits


def merge(model, name):
    return merge_op.run(Context(result_base="you/lab/results/j_1"), {},
                        {"model": model, "to": {"bench": {"name": name}}})


class TestMergingAChosenSet:
    def test_the_checkpoint_is_the_base_plus_those_layers_deltas(self, tiny, tmp_path):
        base = write_snapshot(tiny, tmp_path / "snap")
        payload = build_payload(tiny, tmp_path, "die", alpha=3)
        checkpoint.export_merged(tmp_path / "snap", [payload], tmp_path / "chosen", layers=[(1, 3)])
        checkpoint.export_merged(tmp_path / "snap", [payload], tmp_path / "full")
        chosen = dict(mx.load(str(tmp_path / "chosen" / SHARD)))
        full = dict(mx.load(str(tmp_path / "full" / SHARD)))
        assert_same(chosen, merge_by_hand(base, build_weights(tiny), {1, 3}, 1.5))
        assert_same(full, merge_by_hand(base, build_weights(tiny), set(range(tiny.arch.n_layers)), 1.5))
        moved = [name for name in base if not np.array_equal(np.array(chosen[name]), np.array(full[name]))]
        assert sorted({int(name.split(".")[2]) for name in moved}) == [
            i for i in range(tiny.arch.n_layers) if i not in (1, 3)]

    def test_the_checkpoint_holds_the_weights_the_restricted_fuse_runs_on(self, tiny, tmp_path):
        write_snapshot(tiny, tmp_path / "snap")
        payload = build_payload(tiny, tmp_path, "die", alpha=3)
        checkpoint.export_merged(tmp_path / "snap", [payload], tmp_path / "out", layers=[(1, 3)])
        handles = lora.fuse_adapter_stack(tiny.lm, [payload], keys=tiny.architecture.adapter_keys,
                                          layers=[(1, 3)])
        try:
            fused = dict(tree_flatten(tiny.lm.parameters()))
        finally:
            lora.restore_adapter_stack(tiny.lm, handles)
        assert_same(dict(mx.load(str(tmp_path / "out" / SHARD))), fused)

    def test_the_op_merges_those_layers_and_merged_from_records_them(self, tiny, tmp_path, monkeypatch):
        base = write_snapshot(tiny, tmp_path / "snap")
        payload = build_payload(tiny, tmp_path, "die", alpha=3)
        puts, emits = stub_bench(monkeypatch, tmp_path / "snap")
        merge(adapted(payload, [1, 3]), "die-1-3")
        assert emits[0]["merged_from"] == {"base": {"hf": "tiny"},
                                           "adapters": [{"bench": LABEL, "layers": [1, 3]}]}
        (tmp_path / "put.safetensors").write_bytes(puts[SHARD])
        assert_same(dict(mx.load(str(tmp_path / "put.safetensors"))),
                    merge_by_hand(base, build_weights(tiny), {1, 3}, 1.5))

    def test_an_adapter_without_layers_is_recorded_as_before(self, tiny, tmp_path, monkeypatch):
        write_snapshot(tiny, tmp_path / "snap")
        _, emits = stub_bench(monkeypatch, tmp_path / "snap")
        merge(adapted(build_payload(tiny, tmp_path, "die")), "die")
        assert json.dumps(emits[0]["merged_from"]) == (
            '{"base": {"hf": "tiny"}, "adapters": [{"bench": "you/lab/die"}]}')

    def test_a_layer_the_checkpoint_lacks_refuses_before_anything_is_written(self, tiny, tmp_path, monkeypatch):
        write_snapshot(tiny, tmp_path / "snap")
        payload = build_payload(tiny, tmp_path, "die")
        n = tiny.arch.n_layers
        refusal = (rf"^LAYER_OUT_OF_RANGE: an adapter's `layers` names {n}, "
                   rf"and this model's layers are 0 through {n - 1}$")
        with pytest.raises(ValueError, match=refusal):
            checkpoint.export_merged(tmp_path / "snap", [payload], tmp_path / "out", layers=[(1, n)])
        assert not (tmp_path / "out").exists()
        puts, emits = stub_bench(monkeypatch, tmp_path / "snap")
        with pytest.raises(ValueError, match=refusal):
            merge(adapted(payload, [1, n]), "die")
        assert puts == {} and emits == []

    def test_layers_on_a_checkpoint_of_no_architecture_compute_loads_refuse(self, tiny, tmp_path):
        write_snapshot(tiny, tmp_path / "snap")
        (tmp_path / "snap" / "config.json").write_text('{"model_type": "test"}')
        payload = build_payload(tiny, tmp_path, "die")
        with pytest.raises(ValueError, match="names model_type 'test', which compute does not load"):
            checkpoint.export_merged(tmp_path / "snap", [payload], tmp_path / "out", layers=[(1,)])
        assert not (tmp_path / "out").exists()
        assert SHARD in checkpoint.export_merged(tmp_path / "snap", [payload], tmp_path / "out")


def read_twice(ctx, inputs, params):
    ctx.model(params["model"])
    ctx.model(params["other"])
    return {}


class TestTheDescription:
    @pytest.mark.parametrize("adapters,said", [
        ([LABEL], "hf:org/m (+1 adapter)"),
        ([{"bench": LABEL, "layers": [14, 15, 16]}], "hf:org/m (+1 adapter; you/lab/die in layers 14–16)"),
        (["you/lab/coin", {"bench": LABEL, "layers": [0, 1, 2, 5, 6]}],
         "hf:org/m (+2 adapters; you/lab/die in layers 0–2, 5, 6)"),
        ([{"bench": "you/lab/coin", "layers": [1, 3]}, {"bench": LABEL, "layers": [20]}],
         "hf:org/m (+2 adapters; you/lab/coin in layers 1, 3; you/lab/die in layer 20)"),
        ([{"bench": LABEL, "layers": []}], "hf:org/m (+1 adapter; you/lab/die in no layer)"),
    ])
    def test_an_adapter_in_chosen_layers_is_named_with_them(self, adapters, said):
        assert model_ref.parse({"base": "org/m", "adapters": adapters}).describe() == said

    def test_two_sets_of_layers_in_one_node_refuse_naming_both(self, executor, tiny, payload):
        declared = Op(name="probe/twice", summary="Loads the model twice.",
                      description="Loads the model twice.", params=(), inputs=(),
                      needs=frozenset({"model.forward"}))
        op = SimpleNamespace(op=declared, module=SimpleNamespace(run=read_twice), scope=None)
        with pytest.raises(ValueError) as refused:
            executor._run_op(op, {}, {"model": adapted(payload, [1, 3]), "other": adapted(payload, [2])})
        assert str(refused.value) == (
            "probe/twice: the model already carries the adapters of hf:tiny (+1 adapter; you/lab/die "
            "in layers 1, 3), and the operation asked for hf:tiny (+1 adapter; you/lab/die in layer 2); "
            "one node runs one adapted model")
        assert tiny.fused_reference is None
