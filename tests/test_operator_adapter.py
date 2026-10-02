from __future__ import annotations

import contextlib

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mechbench_compute import checkpoint, model_ref
from mechbench_compute import generate as generate_mod
from mechbench_compute.adapters.attach_operators import (
    attach_operators,
    detach_operators,
)
from mechbench_compute.adapters.attach_payload import attach_payload
from mechbench_compute.adapters.build_operator_modules import build_operator_modules
from mechbench_compute.adapters.read_operator_spec import read_operator_spec
from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.distill import render
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.ops.text import generate as generate_op
from mechbench_compute.protocol import ProtocolExecutor
from tests.test_adapter_layers import build_payload, write_snapshot
from tests.tiny_models import MODEL_TYPES, build_tiny_model

RECORDS = [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"},
           {"id": "c", "user": "the dog sat on a mat"}]

WORDS = ["cat", "dog", "mat", "sat", "ran"]

LABEL = "you/lab/temperature"

IDS = mx.array([[1, 5, 9, 2, 7, 3]])


@pytest.fixture(scope="module", params=MODEL_TYPES)
def tiny(request):
    return build_tiny_model(request.param, BY_MODEL_TYPE[request.param])


@pytest.fixture
def executor(tiny):
    ex = ProtocolExecutor()
    ex._model, ex._model_id = tiny, "tiny"
    return ex


def train(model, operator, target, *, steps=60, lr=0.1, **params):
    return train_op.run(Context(loaded=model), {"records": RECORDS},
                        {"model": "tiny", "target": target, "closer": " mat", "steps": steps,
                         "lr": lr, "seed": 3, "batch": {"target": 3}, "operator": operator,
                         **params})["out"]


def plant(d, scale=3.0, shift=4.0, dim=7, layer=3):
    a, b = [1.0] * d, [0.0] * d
    a[dim], b[dim] = scale, shift
    return {"kind": "adapter/operator",
            "operator": {"layers": [layer], "function": "affine", "positions": "last", "d": d},
            "parameters": {str(layer): {"a": a, "b": b}}}


@contextlib.contextmanager
def attached(model, payload):
    handle = attach_payload(model.lm, payload)
    try:
        yield
    finally:
        detach_operators(model.lm, handle)


def read_words(model) -> np.ndarray:
    ids = [model.tokenizer.convert_tokens_to_ids(w) for w in WORDS]
    rows = []
    for record in RECORDS:
        p = np.array(mx.softmax(model.run(render(model, record).array).logits[0, -1].astype(mx.float32)))
        rows.append(p[ids] / p[ids].sum())
    return np.mean(rows, axis=0)


def read_share(model, word, other) -> float:
    p = read_words(model)
    return float(p[WORDS.index(word)] / (p[WORDS.index(word)] + p[WORDS.index(other)]))


def read_weights(model) -> dict[str, np.ndarray]:
    return {k: np.array(v) for k, v in tree_flatten(model.lm.parameters())}


def is_bare(model) -> bool:
    return not any("operators" in layer for layer in model.lm.model.layers)


class TestTraining:
    def test_rung_0_two_parameters_on_one_coordinate_move_the_target(self, tiny):
        before = read_share(tiny, "dog", "cat")
        want, other = ("cat", "dog") if before > 0.5 else ("dog", "cat")
        out = train(tiny, {"layers": [3], "mask": [7], "function": "affine"},
                    {"weights": {want: 0.9, other: 0.1}})
        assert out["kind"] == "adapter/operator"
        assert out["operator"]["params"] == out["operator"]["effective"] == 2
        assert set(out["parameters"]["3"]) == {"a", "b"}
        with attached(tiny, out):
            after = read_share(tiny, want, other)
        assert after > (1 - before if want == "cat" else before) + 0.05
        assert read_share(tiny, "dog", "cat") == before

    def test_the_l1_penalty_prunes_a_diagonal_operator_on_a_planted_sparse_target(self, tiny):
        d = tiny.arch.d_model
        with attached(tiny, plant(d)):
            target = {w: float(p) for w, p in zip(WORDS, read_words(tiny))}
        dense = train(tiny, {"layers": [3], "function": "affine"}, {"weights": target}, steps=150, lr=0.05)
        pruned = train(tiny, {"layers": [3], "function": "affine", "penalty": {"l1": 0.1}},
                       {"weights": target}, steps=150, lr=0.05)
        assert dense["operator"]["params"] == pruned["operator"]["params"] == 2 * d
        assert dense["operator"]["effective"] == 2 * d
        assert 0 < pruned["operator"]["effective"] < d
        a, b = (np.array(pruned["parameters"]["3"][k]) for k in ("a", "b"))
        assert int((a != 1.0).sum() + (b != 0.0).sum()) == pruned["operator"]["effective"]

    def test_a_rank_r_subspace_operator_trains(self, tiny):
        d = tiny.arch.d_model
        operator = {"layers": [2], "mask": {"rank": 2}, "function": "affine"}
        first = train(tiny, operator, {"weights": {"dog": 0.5, "cat": 0.5}}, steps=1)
        out = train(tiny, operator, {"weights": {"dog": 0.5, "cat": 0.5}})
        assert out["train"]["final_loss"] < first["train"]["final_loss"] - 0.05
        values = out["parameters"]["2"]
        assert {k: np.array(v).shape for k, v in values.items()} == {"P": (2, d), "dW": (2, d), "b": (2,)}
        assert out["operator"]["params"] == out["operator"]["effective"] == 2 * 2 * d + 2

    def test_the_gate_trains_and_its_weights_are_recorded(self, tiny):
        d = tiny.arch.d_model
        operator = {"layers": [3], "function": "affine", "gate": True}
        first = train(tiny, operator, {"weights": {"dog": 0.9, "cat": 0.1}}, steps=1)
        out = train(tiny, operator, {"weights": {"dog": 0.9, "cat": 0.1}})
        assert out["train"]["final_loss"] < first["train"]["final_loss"]
        weight, bias = out["parameters"]["3"]["gate.weight"], out["parameters"]["3"]["gate.bias"]
        assert len(weight) == d and np.any(np.array(weight) != 0) and isinstance(bias, float)
        assert out["operator"]["positions"] == "all" and out["operator"]["gate"] is True
        assert out["operator"]["params"] == 2 * d + d + 1

    @pytest.mark.parametrize(("operator", "lr"), [
        ({"layers": [1, 3], "mask": [2, 9], "function": "polynomial", "degree": 3}, 0.003),
        ({"layers": [3], "mask": {"rank": 3}, "function": "mlp", "width": 4}, 0.1),
    ])
    def test_the_other_forms_train(self, tiny, operator, lr):
        first = train(tiny, operator, {"weights": {"dog": 0.9, "cat": 0.1}}, steps=1, lr=lr)
        out = train(tiny, operator, {"weights": {"dog": 0.9, "cat": 0.1}}, lr=lr)
        assert out["train"]["final_loss"] < first["train"]["final_loss"]
        assert sorted(out["parameters"]) == [str(i) for i in operator["layers"]]

    def test_the_frozen_model_comes_back_bit_identical(self, tiny):
        before = read_weights(tiny)
        classes = [type(layer) for layer in tiny.lm.model.layers]
        train(tiny, {"layers": [1, 3], "mask": {"rank": 2}, "function": "affine", "gate": True},
              {"weights": {"dog": 0.9, "cat": 0.1}}, steps=5)
        after = read_weights(tiny)
        assert set(after) == set(before)
        assert all(np.array_equal(after[k], before[k]) for k in before)
        assert is_bare(tiny) and [type(layer) for layer in tiny.lm.model.layers] == classes

    @pytest.mark.parametrize("operator", [
        {"layers": [3], "mask": [4], "function": "affine"},
        {"layers": [2], "mask": {"rank": 2}, "function": "affine", "gate": True},
        {"layers": [1], "function": "polynomial", "degree": 3, "positions": "all"},
        {"layers": [0, 3], "mask": {"rank": 2}, "function": "mlp", "width": 3},
    ])
    def test_every_form_starts_as_the_identity(self, tiny, operator):
        bare = tiny.run(IDS).logits
        spec = read_operator_spec(operator, n_layers=tiny.arch.n_layers, d=tiny.arch.d_model)
        handle = attach_operators(tiny.lm, build_operator_modules(spec, tiny.arch.d_model, seed=1))
        try:
            assert mx.array_equal(tiny.run(IDS).logits, bare)
        finally:
            detach_operators(tiny.lm, handle)

    @pytest.mark.parametrize(("params", "refusal"), [
        ({"operator": {"layers": [3], "function": "affine"}, "lora": {"rank": 2}}, "give one"),
        ({"operator": {"layers": [3], "function": "affine"}, "keep_checkpoints": True}, "keeps none"),
        ({"operator": {"layers": [3], "function": "affine"}, "batch": {"continuation": 2}},
         "one position"),
        ({"operator": {"layers": [3], "function": "affine", "point": "attn_out"}}, "resid_post"),
        ({"operator": {"layers": [3], "function": "affine", "gate": True, "positions": "last"}},
         "a gated one at `all`"),
        ({"operator": {"layers": [9], "function": "affine"}}, "^LAYER_OUT_OF_RANGE: "),
        ({"operator": {"layers": [3], "mask": [99], "function": "affine"}}, "^MASK_OUT_OF_RANGE: "),
        ({"operator": {"layers": [3], "function": "spline"}}, "is one of affine, polynomial, mlp"),
        ({"operator": {"layers": [3], "function": "affine", "degree": 2}}, "belongs to `polynomial`"),
    ])
    def test_refusals(self, tiny, params, refusal):
        with pytest.raises(ValueError, match=refusal):
            train_op.run(Context(loaded=tiny), {"records": RECORDS},
                         {"model": "tiny", "target": {"uniform": ["cat", "dog"]}, "closer": " mat",
                          "steps": 1, **params})
        assert is_bare(tiny)


class TestAttaching:
    def test_the_models_own_forward_and_the_hooked_forward_agree(self, tiny):
        bare = tiny.run(IDS).logits
        with attached(tiny, plant(tiny.arch.d_model)):
            own = tiny.lm(IDS)
            own = own.logits if hasattr(own, "logits") else own
            hooked = tiny.run(IDS).logits
            assert mx.array_equal(own, hooked) and not mx.array_equal(hooked, bare)
        assert mx.array_equal(tiny.run(IDS).logits, bare)

    def test_generation_under_an_operator_is_the_same_function_as_an_intervention(self, tiny, monkeypatch):
        monkeypatch.setattr(generate_mod, "_stop_ids", lambda tokenizer: set())

        def generate(spec=None):
            params = {"model": "tiny", "temperature": 0.0, "max_tokens": 6, "fidelity": "trace"}
            if spec:
                params.update(spec=spec, control=False)
            item = generate_op.run(Context(loaded=tiny), {"records": RECORDS[:1]}, params)["items"][0]
            return item["trace"]["token_ids"]

        bare = generate()
        with attached(tiny, plant(tiny.arch.d_model, scale=6.0, shift=-5.0)):
            carried = generate()
        given = generate([{"point": "resid_post", "layers": [3], "positions": "last", "mask": [7],
                           "f": "6 * x - 5"}])
        assert carried == given and carried != bare

    def test_a_reference_carries_it_into_logits_read_and_fused_names_it(self, tiny, executor):
        payload = plant(tiny.arch.d_model, scale=6.0, shift=-5.0)
        ref = model_ref.resolve({"base": "tiny", "adapters": [LABEL]}, fetch=lambda _label: payload)
        bare = executor._run_op("logits/read", {"conditions": RECORDS}, {"model": "tiny"})
        out = executor._run_op("logits/read", {"conditions": RECORDS}, {"model": ref})
        assert out["fused"] == [{"bench": LABEL}]
        assert [i["top"] for i in out["items"]] != [i["top"] for i in bare["items"]]
        again = executor._run_op("logits/read", {"conditions": RECORDS}, {"model": "tiny"})
        assert [i["top"] for i in again["items"]] == [i["top"] for i in bare["items"]]
        assert is_bare(tiny) and tiny.fused_reference is None

    def test_layers_do_not_apply_and_a_refused_attach_restores_the_model(self, tiny, executor, tmp_path):
        before = read_weights(tiny)
        lora = build_payload(tiny, tmp_path, "die")
        operator = plant(tiny.arch.d_model)
        stack = model_ref.resolve(
            {"base": "tiny", "adapters": ["you/lab/die", {"bench": LABEL, "layers": [1]}]},
            fetch=lambda label: operator if label == LABEL else lora)
        with pytest.raises(ValueError, match="`layers` belongs to a LoRA"):
            executor._run_op("logits/read", {"conditions": RECORDS}, {"model": stack})
        assert is_bare(tiny) and tiny.fused_reference is None
        after = read_weights(tiny)
        assert all(np.array_equal(after[k], before[k]) for k in before)

    def test_an_operator_at_a_layer_the_model_lacks_refuses_and_restores(self, tiny, executor, tmp_path):
        before = read_weights(tiny)
        lora = build_payload(tiny, tmp_path, "die")
        operator = plant(tiny.arch.d_model, layer=9)
        stack = model_ref.resolve({"base": "tiny", "adapters": ["you/lab/die", LABEL]},
                                  fetch=lambda label: operator if label == LABEL else lora)
        with pytest.raises(ValueError, match="^LAYER_OUT_OF_RANGE: "):
            executor._run_op("logits/read", {"conditions": RECORDS}, {"model": stack})
        assert is_bare(tiny)
        after = read_weights(tiny)
        assert all(np.array_equal(after[k], before[k]) for k in before)

    def test_a_width_the_model_does_not_have_refuses(self, tiny):
        with pytest.raises(ValueError, match="^OPERATOR_WIDTH_MISMATCH: "):
            attach_payload(tiny.lm, plant(tiny.arch.d_model + 1))
        assert is_bare(tiny)

    def test_the_port_takes_a_lora_only(self, tiny, executor):
        with pytest.raises(ValueError, match="attaches through the model reference"):
            executor._run_op("logits/read", {"conditions": RECORDS, "adapter": plant(tiny.arch.d_model)},
                             {"model": "tiny"})
        assert is_bare(tiny)

    def test_a_trained_operator_round_trips_through_its_artifact(self, tiny):
        out = train(tiny, {"layers": [3], "mask": {"rank": 2}, "function": "affine"},
                    {"weights": {"dog": 0.9, "cat": 0.1}}, steps=10)
        with attached(tiny, out):
            first = tiny.run(IDS).logits
        with attached(tiny, out):
            assert mx.array_equal(tiny.run(IDS).logits, first)

    def test_an_operator_cannot_be_merged_into_a_checkpoint(self, tiny, tmp_path):
        write_snapshot(tiny, tmp_path / "snap")
        with pytest.raises(ValueError, match="cannot be merged"):
            checkpoint.export_merged(tmp_path / "snap", [plant(tiny.arch.d_model)], tmp_path / "out")
        assert not (tmp_path / "out").exists()
