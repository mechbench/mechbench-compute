from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mechbench_schema import dump_canonical
from mlx.utils import tree_flatten

from mechbench_compute import architectures, bench, lora, model_ref
from mechbench_compute import model as model_mod
from mechbench_compute.adapters.attach_operators import attach_operators
from mechbench_compute.adapters.build_operator_modules import build_operator_modules
from mechbench_compute.adapters.read_operator_spec import read_operator_spec
from mechbench_compute.lexicon import items_of
from mechbench_compute.lexicon._base import Op
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.ops.intervene import apply as apply_op
from mechbench_compute.ops.text import generate as generate_op
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.protocol.model_cache_stale import ModelCacheStale
from mechbench_compute.protocol.model_fingerprint import read_applied
from mechbench_compute.weights import edit_parameters
from tests.test_circuits import build_adapter
from tests.test_dictionary import load as load_dictionary
from tests.test_dictionary import write_dictionary
from tests.tiny_models import MODEL_TYPES, build_tiny_model

BASE = "tiny/m"

RECORDS = [{"id": "a", "user": "the cat sat on a"}, {"id": "b", "user": "a dog ran on the"}]

DOCUMENT = {"id": "d", "text": "the cat sat on a mat and the dog ran"}

GOOD, BROKEN = "you/lab/good", "you/lab/broken"

SFT = {"objective": "sft", "steps": 6, "lr": 0.05, "seed": 3, "lora": {"rank": 2, "alpha": 4},
       "batch": {"record": 1}, "checkpoint_every": 2}

TIMEOUT = "[METAL] Command buffer execution failed: Caused GPU Timeout Error"

GREEDY = {"temperature": 0.0, "max_tokens": 2}


def read_nodes(model=BASE):
    return [
        {"id": "read", "block": "logits/read", "params": {"model": model, "top_k": 5},
         "inputs": {"conditions": RECORDS}},
        {"id": "capture", "block": "activations/capture",
         "params": {"model": model, "layers": "all", "point": "resid_post"},
         "inputs": {"records": RECORDS}},
        {"id": "generate", "block": "text/generate",
         "params": {"model": model, "temperature": 0.0, "max_tokens": 4, "fidelity": "trace"},
         "inputs": {"records": RECORDS}},
        {"id": "chat_base", "block": "text/chat",
         "params": {"model": {"base": model}, "temperature": 0.0, "max_tokens": 4, "seed": 1},
         "inputs": {"records": RECORDS}},
    ]


TRAIN = {"id": "train", "block": "adapter/train", "params": {"model": BASE, **SFT},
         "inputs": {"records": [DOCUMENT]}}


def run_job(executor, nodes):
    spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": {"dataflow": 2, "nodes": nodes, "edges": []}})
    return executor.run(spec).payload["outputs"]


def die(*_):
    raise RuntimeError(TIMEOUT)


class Watchdog:
    def __init__(self, step):
        self.step = step
        self.fired = False

    def __call__(self, *args):
        state = args[-1]
        if not self.fired and state["step"] == self.step:
            self.fired = True
            raise RuntimeError(TIMEOUT)


@pytest.fixture(params=MODEL_TYPES)
def hub(request, monkeypatch, tmp_path):
    arch = request.param
    loads: list[str] = []

    def load(model_id, **_):
        loads.append(model_id)
        tiny = build_tiny_model(arch)
        return tiny._model, tiny._processor

    architecture = dataclasses.replace(architectures.BY_MODEL_TYPE[arch], load=load)
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": arch})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)
    return loads


def save_payload(weights, tmp_path, name):
    path = tmp_path / f"{name}.safetensors"
    mx.save_safetensors(str(path), weights)
    return {"data": path.read_bytes(), "lora": {"rank": 2, "alpha": 2}}


def build_good(model):
    return {k: v for k, v in build_adapter(model).items() if ".o_proj." in k}


def build_broken(model):
    last = len(model.lm.model.layers) - 1
    return {k: v for k, v in build_adapter(model).items()
            if ".q_proj." in k and k != f"model.layers.{last}.self_attn.q_proj.lora_b"}


def read_values(model) -> dict[str, np.ndarray]:
    return {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.lm.parameters())}


def assert_untouched(model, marks, values) -> None:
    assert lora.read_changed(model.lm, marks) == []
    now = read_values(model)
    assert all(np.array_equal(now[k], values[k]) for k in values)


def hold(model, model_id="tiny"):
    ex = ProtocolExecutor()
    ex._model, ex._model_id = model, model_id
    ex._model_loaded(model_id)
    return ex


@pytest.fixture
def held():
    model = build_tiny_model("gemma3")
    return hold(model), model


class TestATrainingThatDiesMidRun:
    def test_the_next_read_of_the_base_is_the_untrained_read_byte_for_byte(self, hub):
        ex = ProtocolExecutor(on_checkpoint=Watchdog(4))
        before = run_job(ex, read_nodes())
        with pytest.raises(RuntimeError, match="GPU Timeout"):
            run_job(ex, [TRAIN])
        assert ex._model is None
        after = run_job(ex, read_nodes())
        assert dump_canonical(after) == dump_canonical(before)
        assert after["chat_base"]["items"][0]["metadata"]["model"] == {
            "base": {"hf": BASE}, "adapters": []}
        assert len(hub) == 2

    def test_the_next_training_trains_as_on_a_fresh_runner(self, hub):
        ex = ProtocolExecutor(on_checkpoint=Watchdog(4))
        with pytest.raises(RuntimeError, match="GPU Timeout"):
            run_job(ex, [TRAIN])
        again = run_job(ex, [TRAIN])["train"]
        fresh = run_job(ProtocolExecutor(on_checkpoint=lambda *_: None), [TRAIN])["train"]
        assert again["data"] == fresh["data"] and again["train"] == fresh["train"]

    def test_the_model_object_itself_comes_back_unwrapped(self):
        model = build_tiny_model("llama")
        marks, values = lora.mark_weights(model.lm), read_values(model)
        with pytest.raises(RuntimeError, match="GPU Timeout"):
            train_op.run(Context(loaded=model, on_checkpoint=Watchdog(4)),
                         {"records": [DOCUMENT]}, {"model": "tiny", **SFT})
        assert_untouched(model, marks, values)

    def test_a_refusal_after_the_lora_is_wrapped_unwraps_it(self):
        model = build_tiny_model("llama")
        marks, values = lora.mark_weights(model.lm), read_values(model)
        with pytest.raises(ValueError, match="kept checkpoints"):
            train_op.run(Context(loaded=model), {"records": [DOCUMENT]},
                         {"model": "tiny", **SFT, "steps": 2_000_000, "checkpoint_every": 1,
                          "keep_checkpoints": True})
        assert_untouched(model, marks, values)

    @pytest.mark.parametrize("when", ["before-the-first-step", "mid-run"])
    def test_an_operator_that_dies_is_detached(self, when):
        model = build_tiny_model("gemma3")
        marks, values = lora.mark_weights(model.lm), read_values(model)
        lent = {"on_start": die} if when == "before-the-first-step" else {"on_checkpoint": Watchdog(2)}
        with pytest.raises(RuntimeError, match="GPU Timeout"):
            train_op.run(Context(loaded=model, **lent), {"records": RECORDS},
                         {"model": "tiny", "target": {"uniform": ["cat", "dog"]}, "closer": " mat",
                          "steps": 6, "lr": 0.05, "seed": 3, "batch": {"target": 2},
                          "checkpoint_every": 2,
                          "operator": {"layers": [1, 3], "mask": {"rank": 2}, "function": "affine"}})
        assert not any("operators" in layer for layer in model.lm.model.layers)
        assert_untouched(model, marks, values)


class TestAFuseThatFailsPartWay:
    def test_the_stack_leaves_every_weight_as_it_was(self, tmp_path):
        model = build_tiny_model("gemma3")
        marks, values = lora.mark_weights(model.lm), read_values(model)
        stack = [save_payload(build_good(model), tmp_path, "good"),
                 save_payload(build_broken(model), tmp_path, "broken")]
        with pytest.raises(ValueError, match="missing lora_a or lora_b for layer 3 self_attn.q_proj"):
            lora.fuse_adapter_stack(model.lm, stack, keys=model.architecture.adapter_keys)
        assert_untouched(model, marks, values)

    def test_the_next_read_of_the_base_is_the_unadapted_read_byte_for_byte(self, hub, monkeypatch, tmp_path):
        ex = ProtocolExecutor()
        before = run_job(ex, read_nodes())
        payloads = {GOOD: save_payload(build_good(ex._model), tmp_path, "good"),
                    BROKEN: save_payload(build_broken(ex._model), tmp_path, "broken")}
        monkeypatch.setattr(bench, "fetch", lambda label, with_meta=False: (
            (payloads[label], {"content_hash": "sha256:0"}) if with_meta else payloads[label]))
        with pytest.raises(ValueError, match="missing lora_a or lora_b"):
            run_job(ex, read_nodes({"base": BASE, "adapters": [GOOD, BROKEN]})[:1])
        assert ex._model is None
        after = run_job(ex, read_nodes())
        assert dump_canonical(after) == dump_canonical(before)

    def test_a_weight_edit_that_fails_part_way_restores_the_ones_it_made(self):
        model = build_tiny_model("gemma3")
        marks, values = lora.mark_weights(model.lm), read_values(model)
        with pytest.raises(ValueError, match="unknown weight op 'melt'"):
            edit_parameters(model.lm, [{"parameter": "layers.1.mlp.down_proj", "op": "zero"},
                                       {"parameter": "layers.2.mlp.down_proj", "op": "melt"}])
        assert_untouched(model, marks, values)

    def test_an_intervention_that_fails_part_way_through_its_cells_restores_its_weight_edits(self):
        model = build_tiny_model("gemma3")
        marks, values = lora.mark_weights(model.lm), read_values(model)
        with pytest.raises(ValueError) as caught:
            apply_op.run(Context(loaded=model), {"records": [RECORDS[0], {"id": "bad"}]},
                         {"spec": [{"parameter": "layers.1.mlp.down_proj", "op": "scale", "strength": 0.5}],
                          "sweep": {"strength": [2.0]}, "tracked": {"c": "cat"}})
        assert caught.value.__traceback__ is not None
        assert_untouched(model, marks, values)

    def test_an_attach_that_fails_part_way_detaches_what_it_attached(self):
        model = build_tiny_model("gemma3")
        d = model.arch.d_model
        spec = read_operator_spec({"layers": [1, 3], "function": "affine"}, n_layers=4, d=d)
        modules = build_operator_modules(spec, d, seed=1)
        classes = [type(layer) for layer in model.lm.model.layers]
        with pytest.raises(IndexError):
            attach_operators(model.lm, {1: modules[1], 9: modules[3]})
        assert not any("operators" in layer for layer in model.lm.model.layers)
        assert [type(layer) for layer in model.lm.model.layers] == classes


class TestTheCacheRefusesWhatANodeDidNotAskFor:
    def refuse(self, ex) -> ModelCacheStale:
        with pytest.raises(ModelCacheStale) as refused:
            ex._run_op("logits/read", {"conditions": RECORDS}, {"model": "tiny"})
        assert ex._model is None
        assert refused.value.issue == {"code": "MODEL_CACHE_STALE", "loaded": refused.value.loaded,
                                       "asked": {"base": "tiny", "adapters": []},
                                       "message": str(refused.value).removeprefix("MODEL_CACHE_STALE: ")}
        assert str(refused.value).endswith(
            "; it does not run on a model other than the one it asked for. The cache is dropped: "
            "the next node that asks for the model loads it afresh.")
        return refused.value

    def test_a_lora_left_in_the_model(self, held):
        ex, model = held
        lora.apply_lora(model.lm, rank=2, alpha=4, seed=1)
        refused = self.refuse(ex)
        assert str(refused).startswith(
            "MODEL_CACHE_STALE: logits/read asked for tiny, and the cached model is tiny with 32 "
            "weights changed since it was loaded (model.layers.0.self_attn.q_proj.base.weight, "
            "model.layers.0.self_attn.q_proj.lora_a, model.layers.0.self_attn.q_proj.lora_b, "
            "and 29 more)")
        assert refused.loaded == {"base": "tiny", "adapters": [], "changed": refused.loaded["changed"]}
        assert "model.layers.3.self_attn.v_proj.weight" in refused.loaded["changed"]

    def test_a_fuse_left_in_the_model(self, held):
        ex, model = held
        lora.fuse(model.lm, build_adapter(model), scale=1.0)
        refused = self.refuse(ex)
        assert refused.loaded["changed"] == sorted(
            f"model.layers.{i}.self_attn.{p}.weight" for i in range(4) for p in ("o_proj", "q_proj"))

    def test_an_operator_left_on_the_model(self, held):
        ex, model = held
        d = model.arch.d_model
        spec = read_operator_spec({"layers": [2], "function": "affine"}, n_layers=4, d=d)
        attach_operators(model.lm, build_operator_modules(spec, d, seed=1))
        assert self.refuse(ex).loaded["changed"] == ["model.layers.2.operators.0.a",
                                                     "model.layers.2.operators.0.b"]

    def test_a_weight_edit_left_in_the_model(self, held):
        ex, model = held
        edit_parameters(model.lm, [{"parameter": "layers.1.mlp.down_proj", "op": "zero"}])
        assert self.refuse(ex).loaded["changed"] == ["model.layers.1.mlp.down_proj.weight"]

    def test_an_adapter_no_running_node_fused(self, held, tmp_path):
        ex, model = held
        ref = model_ref.resolve({"base": "tiny", "adapters": [{"bench": GOOD, "layers": [1, 2, 3]}]},
                                fetch=lambda _label: save_payload(build_adapter(model), tmp_path, "good"))
        model.fingerprint.push(read_applied(ref))
        refused = self.refuse(ex)
        assert str(refused).startswith(
            "MODEL_CACHE_STALE: logits/read asked for tiny, and the cached model is tiny with "
            "you/lab/good in layers 1–3 fused, though no running node fused it;")
        assert refused.loaded == {"base": "tiny", "adapters": [
            {"bench": GOOD, "layers": [1, 2, 3], "sha256": read_applied(ref)[0].sha256}]}

    def test_the_next_node_loads_the_model_afresh(self, held, monkeypatch):
        ex, model = held
        lora.apply_lora(model.lm, rank=2, alpha=4, seed=1)
        self.refuse(ex)
        fresh = build_tiny_model("gemma3")
        monkeypatch.setattr(model_mod.Model, "load", classmethod(lambda cls, model_id, **_: fresh))
        out = ex._run_op("logits/read", {"conditions": RECORDS}, {"model": "tiny"})
        assert ex._model is fresh
        assert out["items"] == hold(build_tiny_model("gemma3"))._run_op(
            "logits/read", {"conditions": RECORDS}, {"model": "tiny"})["items"]


def build_nest(child_model):
    def run(ctx, inputs, params):
        ctx.model(params["model"])
        return ctx.sub("logits/read", {"conditions": RECORDS}, {"model": child_model, "top_k": 5})

    declared = Op(name="probe/nest", summary="Reads inside a node that holds the model.",
                  description="Reads inside a node that holds the model.", params=(), inputs=(),
                  needs=frozenset({"model.forward", "executor.sub"}))
    return SimpleNamespace(op=declared, module=SimpleNamespace(run=run), scope=None)


class TestANodeRunInsideAnother:
    @pytest.fixture
    def adapted(self, held, tmp_path):
        ex, model = held
        payload = save_payload(build_adapter(model), tmp_path, "good")
        return ex, model, model_ref.resolve({"base": "tiny", "adapters": [GOOD]}, fetch=lambda _l: payload)

    def test_asking_for_the_same_reference_runs_on_it_fused_once(self, adapted):
        ex, model, ref = adapted
        inside = ex._run_op(build_nest(ref), {}, {"model": ref})
        alone = hold(model)._run_op("logits/read", {"conditions": RECORDS}, {"model": ref, "top_k": 5})
        assert inside["items"] == alone["items"]
        assert inside["fused"] == alone["fused"] == [{"bench": GOOD}]
        assert model.fingerprint.applied == [] and model.fingerprint.holders == []

    def test_dictionary_encode_reads_the_reference_and_the_adapted_adapter_once_each(self, adapted, tmp_path):
        ex, model, ref = adapted
        write_dictionary(tmp_path, "resid_post/layer_2_width_48_l0_small")
        inputs = {"records": RECORDS, "dictionary": load_dictionary(tmp_path)}
        extra = save_payload({k: v * -0.5 for k, v in build_adapter(model).items()}, tmp_path, "extra")
        both = ex._run_op("dictionary/encode", {**inputs, "adapted": extra}, {"model": ref})
        direct = hold(model)._run_op("dictionary/encode", {**inputs, "adapter": extra}, {"model": ref})

        def strip(items):
            return [{k: v for k, v in i.items() if k != "coords"} for i in items]

        assert strip(i for i in items_of(both) if i["coords"]["model"] == "adapted") == strip(items_of(direct))
        assert both["adapted_fidelity"] == direct["fidelity"]

    def test_asking_for_the_bare_base_is_refused(self, adapted):
        ex, model, ref = adapted
        marks, values = lora.mark_weights(model.lm), read_values(model)
        with pytest.raises(ModelCacheStale, match=(
                "^MODEL_CACHE_STALE: logits/read asked for tiny, and the node it runs inside holds "
                "the model as tiny with you/lab/good fused;")):
            ex._run_op(build_nest("tiny"), {}, {"model": ref})
        assert ex._model is None and model.fused_reference is None
        assert_untouched(model, marks, values)


class TestTheHeaderSaysWhatIsFused:
    def test_a_reference_s_adapters_are_the_fingerprint_s(self, held, tmp_path):
        ex, model = held
        ref = model_ref.resolve({"base": "tiny", "adapters": [{"bench": GOOD, "layers": [2]}]},
                                fetch=lambda _l: save_payload(build_adapter(model), tmp_path, "good"))
        out = ex._run_op("text/generate", {"records": RECORDS[:1]}, {**GREEDY, "model": ref})
        assert out["items"][0]["metadata"]["model"] == ref.to_wire() == {
            "base": {"hf": "tiny"}, "adapters": [{"bench": GOOD, "layers": [2]}]}

    def test_a_model_carrying_an_adapter_the_request_does_not_name_says_so(self, held, tmp_path):
        ex, model = held
        ref = model_ref.resolve({"base": "tiny", "adapters": [GOOD]},
                                fetch=lambda _l: save_payload(build_adapter(model), tmp_path, "good"))
        undo: list = []
        ex._reference_fused(model, ref, undo)
        try:
            out = generate_op.run(Context(loaded=model), {"records": RECORDS[:1]},
                                  {**GREEDY, "model": "tiny"})
            trained = train_op.run(Context(loaded=model), {"records": [DOCUMENT]},
                                   {"model": "tiny", **SFT, "steps": 1})["out"]
        finally:
            ex._reference_restored(undo)
        assert out["items"][0]["metadata"]["model"] == {"base": {"hf": "tiny"},
                                                        "adapters": [{"bench": GOOD}]}
        assert trained["trained_on"] == {"base": "tiny", "adapters": [GOOD]}
        assert model.fingerprint.applied == []
