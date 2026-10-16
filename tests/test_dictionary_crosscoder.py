from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mechbench_compute import lexicon, lora, model_ref
from mechbench_compute.dictionaries.check_sae import check_sae
from mechbench_compute.dictionaries.train_against import read_against
from mechbench_compute.dictionaries.train_corpus import open_stream
from mechbench_compute.dictionaries.train_crosscoder import (
    read_difference,
    read_scales,
    spread_scales,
)
from mechbench_compute.dictionaries.train_dictionary import train_dictionary
from mechbench_compute.dictionaries.train_harvest import harvest
from mechbench_compute.dictionaries.train_sae import Trainer, init_params
from mechbench_compute.dictionaries.train_settings import read_settings
from mechbench_compute.ops import Context
from mechbench_compute.ops.dictionary import encode as encode_op
from mechbench_compute.protocol import ProtocolExecutor
from tests.checkpoint_offers import due_only
from tests.test_adapter_layers import build_payload
from tests.test_dictionary_train import (
    Interrupted,
    draw_planted,
    make_records,
    spool_roundtrip,
)
from tests.tiny_models import build_tiny_model

LAYER = 2

LABEL = "you/lab/planted"

SMALL = {"derivation": "crosscoder", "layer": LAYER, "width": 64, "k": 4, "tokens": 4096, "context": 16,
         "batch_tokens": 128, "buffer_tokens": 512, "heldout_tokens": 60, "checkpoint_tokens": 1024, "seed": 3}

PLANTED = {**SMALL, "tokens": 128 * 400, "buffer_tokens": 1024, "heldout_tokens": 300, "lr": 5e-3,
           "checkpoint_tokens": 10 ** 9}


def plant(model):
    stream = open_stream(None, make_records(), model.tokenizer, seed=0, context=16)
    name = f"blocks.{LAYER}.resid_post"
    x = np.array(harvest(model, name, [stream.next_window() for _ in range(64)], model.tokenizer.bos_token_id),
                 dtype=np.float64)
    _, _, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    gate, planted = vt[-2], vt[-1]
    t = float(np.quantile(x @ gate, 0.85))
    sharp, size = 20.0, float(np.linalg.norm(x, axis=1).mean())
    d = x.shape[1]
    payload = {"kind": "adapter/operator",
               "operator": {"layers": [LAYER], "positions": "all", "function": "mlp", "width": 2},
               "parameters": {str(LAYER): {"W1": [(sharp * gate).tolist()] * 2, "b1": [-sharp * t, -sharp * t - 1],
                                           "W2": [[size * v, -size * v] for v in planted], "b2": [0.0] * d}}}
    return payload, planted


def read_weights(model):
    return {k: np.array(v) for k, v in tree_flatten(model.lm.parameters())}


def is_bare(model):
    return not any("operators" in layer for layer in model.lm.model.layers)


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


@pytest.fixture(scope="module")
def organism(tiny):
    payload, planted = plant(tiny)
    return model_ref.resolve({"base": "tiny", "adapters": [LABEL]}, fetch=lambda _label: payload), planted


@pytest.fixture(scope="module")
def executor(tiny):
    ex = ProtocolExecutor()
    ex._model, ex._model_id = tiny, "tiny"
    return ex


@pytest.fixture(scope="module")
def trained(tiny, organism, executor):
    ref, _ = organism
    return executor._run_op("dictionary/train", {"records": make_records()}, {**PLANTED, "model": ref})


@pytest.fixture
def fused(tiny, organism):
    ref, _ = organism
    handles = lora.fuse_adapter_stack(tiny.lm, list(ref.adapter_payloads))
    try:
        yield ref, [(tiny, handles, ())]
    finally:
        lora.restore_adapter_stack(tiny.lm, handles)


class TestPlantedDictionary:
    def test_the_trainer_separates_shared_and_exclusive_directions(self):
        d, n_atoms, active, batch, steps = 32, 48, 3, 256, 1500
        rng = np.random.default_rng(0)
        atoms = rng.normal(size=(n_atoms, d))
        atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
        mine, theirs = list(range(40, 44)), list(range(44, 48))

        def draw():
            c = draw_planted(rng, np.eye(n_atoms), batch, active)
            first, second = c.copy(), c.copy()
            first[:, theirs] = 0
            second[:, mine] = 0
            return np.concatenate([first @ atoms, second @ atoms], axis=1).astype(np.float32)

        s = read_settings({"derivation": "crosscoder", "layer": 0, "width": 96, "k": active,
                           "tokens": batch * steps, "batch_tokens": batch, "buffer_tokens": batch, "lr": 3e-3})
        first = mx.array(draw())
        scales = mx.array(spread_scales(read_scales(first, 2), 2 * d).astype(np.float32))
        trainer = Trainer(s, init_params(s, first * scales, 0), 2 * d)
        for _ in range(steps):
            metrics = trainer.train(mx.array(draw()) * scales)
        w = np.array(trainer.params["w_dec"], dtype=np.float64)
        assert np.allclose(np.linalg.norm(w, axis=1), 1.0, atol=1e-5)
        difference = read_difference(w)
        rows = [w[:, :d], w[:, d:]]
        rows = [r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12) for r in rows]
        for a in range(n_atoms):
            cos = np.maximum(rows[0] @ atoms[a], rows[1] @ atoms[a])
            best = int(np.argmax(cos))
            assert cos[best] > 0.9, a
            if a in mine:
                assert difference[best] > 0.8, (a, difference[best])
            elif a in theirs:
                assert difference[best] < -0.8, (a, difference[best])
            else:
                assert abs(difference[best]) < 0.1, (a, difference[best])
        assert metrics["variance_explained"] > 0.9


class TestPlantedChange:
    def test_the_planted_direction_is_a_feature_only_the_organism_has(self, trained, organism):
        _, planted = organism
        items = lexicon.items_of(trained)
        difference = np.array([it["difference"] for it in items])
        top = int(np.argmax(difference))
        vector = np.asarray(items[top]["vector"], dtype=np.float64)
        assert difference[top] > 0.75
        assert abs(vector @ planted) / np.linalg.norm(vector) > 0.95
        assert items[top]["norm"] > 5 * items[top]["against_norm"]
        assert 0.05 < items[top]["frequency"] < 0.3 and items[top]["mean_activation"] > 0
        others = [abs(it["difference"]) for i, it in enumerate(items) if i != top and it["frequency"] > 0]
        assert len(others) > 20 and max(others) < 0.15

    def test_the_header_names_both_models(self, trained):
        assert trained["derivation"] == "crosscoder" and trained["fused"] == [{"bench": LABEL}]
        reads = trained["reads"]
        assert [r["adapters"] for r in reads] == [[{"bench": LABEL}], []]
        assert all(r["layer"] == LAYER and r["point"] == "resid_post" and r["d"] == 32 for r in reads)
        assert trained["writes"] == reads and trained["width"] == 64 and trained["d_in"] == 32
        assert len(trained["b_dec"]) == 32 and len(trained["against_b_dec"]) == 32
        t = trained["trained"]
        assert t["against"] == {"base": {"hf": "tiny"}, "adapters": []}
        assert len(t["scale"]) == 2 and all(s > 0 for s in t["scale"])
        assert "derivation" not in t["hyperparameters"]

    def test_fidelity_is_measured_in_each_model(self, trained):
        f = trained["fidelity"]
        assert 0 < f["variance_explained"] < 1 and f["l0"] > 0 and len(f["reads"]) == 2
        for one in f["reads"]:
            assert 0 < one["variance_explained"] < 1
            loss = one["loss"]
            assert loss["delta"] == pytest.approx(loss["spliced"] - loss["clean"], abs=2e-6)
            assert loss["zero_delta"] == pytest.approx(loss["zero"] - loss["clean"], abs=2e-6)
        assert f["reads"][0]["loss"]["clean"] != f["reads"][1]["loss"]["clean"]

    def test_every_item_carries_both_rows(self, trained):
        items = lexicon.items_of(trained)
        assert [it["index"] for it in items] == list(range(64))
        for it in items[:3]:
            assert np.asarray(it["vector"]).shape == np.asarray(it["against_vector"]).shape == (32,)
            assert np.asarray(it["encoder"]).shape == np.asarray(it["against_encoder"]).shape == (32,)
            assert it["norm"] == pytest.approx(float(np.linalg.norm(it["vector"])), rel=1e-5)
            assert -1 <= it["difference"] <= 1

    def test_the_model_is_bare_after_the_node(self, tiny, trained):
        assert is_bare(tiny) and tiny.fused_reference is None


class TestReaders:
    def test_the_feature_readers_refuse_a_crosscoder_by_name(self, tiny, trained):
        with pytest.raises(ValueError, match="the dictionary is a 'crosscoder'"):
            check_sae(trained)
        with pytest.raises(ValueError, match="the dictionary is a 'crosscoder'"):
            encode_op.run(Context(loaded=tiny), {"records": [{"id": "a", "user": "the cat"}],
                                                 "dictionary": trained}, {})


class TestTheSwap:
    def test_against_runs_without_the_top_adapters_and_the_organism_comes_back(self, tiny, tmp_path):
        bare = read_weights(tiny)
        payload, _ = plant(tiny)
        handles = lora.fuse_adapter_stack(tiny.lm, [build_payload(tiny, tmp_path, "die"), payload])
        try:
            fused = read_weights(tiny)
            assert any(not np.array_equal(fused[k], bare[k]) for k in bare)
            with lora.unfused_stack(tiny.lm, handles):
                inside = read_weights(tiny)
                assert all(np.array_equal(inside[k], bare[k]) for k in bare) and is_bare(tiny)
            after = read_weights(tiny)
            assert all(np.array_equal(after[k], fused[k]) for k in fused) and not is_bare(tiny)
        finally:
            lora.restore_adapter_stack(tiny.lm, handles)
        assert is_bare(tiny)


class TestResume:
    def test_an_interrupted_run_resumes_to_the_uninterrupted_result(self, tiny, fused, tmp_path):
        ref, stack = fused
        params = {**SMALL, "model": ref}
        whole = train_dictionary(tiny, make_records(), params, against=read_against(tiny, ref, None, stack))
        saved = []

        def interrupt(state):
            saved.append(spool_roundtrip(state, tmp_path))
            raise Interrupted

        with pytest.raises(Interrupted):
            train_dictionary(tiny, make_records(), params, on_checkpoint=due_only(interrupt),
                             against=read_against(tiny, ref, None, stack))
        assert saved[0]["step"] == 8
        restored = []
        resumed = train_dictionary(tiny, make_records(), params, resume_state=saved[0],
                                   on_item=lambda *a: restored.append(a),
                                   against=read_against(tiny, ref, None, stack))
        assert restored[:24] == [(None, None, True)] * 24 and restored[24] == ()
        a, b = lexicon.items_of(whole), lexicon.items_of(resumed)
        for k in ("vector", "against_vector", "encoder", "against_encoder"):
            assert all(np.array_equal(x[k], y[k]) for x, y in zip(a, b)), k
        assert resumed["fidelity"] == whole["fidelity"] and resumed["trained"] == whole["trained"]

    def test_a_state_from_an_autoencoder_is_refused(self, tiny, fused):
        ref, stack = fused
        saved = []
        train_dictionary(tiny, make_records(), {**SMALL, "derivation": "sae", "tokens": 2048},
                         on_checkpoint=due_only(saved.append))
        with pytest.raises(ValueError, match="its `derivation` differs"):
            train_dictionary(tiny, make_records(), {**SMALL, "model": ref}, resume_state=saved[0],
                             against=read_against(tiny, ref, None, stack))

    def test_progress_counts_both_models(self, tiny, fused):
        ref, stack = fused
        started, items = [], []
        train_dictionary(tiny, make_records(), {**SMALL, "tokens": 1024, "model": ref}, on_start=started.append,
                         on_item=lambda *a: items.append(a), against=read_against(tiny, ref, None, stack))
        assert started == [3 * 8 + 6 * 4] and len(items) == started[0]


class TestRefusals:
    def test_against_on_an_autoencoder(self):
        with pytest.raises(ValueError, match="`against` names a crosscoder's second model"):
            read_settings({**SMALL, "derivation": "sae", "against": "tiny"})

    def test_a_model_without_adapters(self, tiny):
        with pytest.raises(ValueError, match="a crosscoder reads two models"):
            read_against(tiny, "tiny", None, [])

    def test_against_on_another_checkpoint(self, tiny, organism):
        ref, _ = organism
        with pytest.raises(ValueError, match="runs both models on the one loaded checkpoint"):
            read_against(tiny, ref, "other/model", [])

    def test_against_with_adapters_model_lacks(self, tiny, organism):
        ref, _ = organism
        with pytest.raises(ValueError, match="its adapters are the first of `model`'s"):
            read_against(tiny, ref, {"base": "tiny", "adapters": ["you/lab/other"]}, [])

    def test_adapters_this_node_did_not_fuse(self, tiny, organism):
        ref, _ = organism
        with pytest.raises(ValueError, match="this node fused 0 of them"):
            read_against(tiny, ref, None, [])

    def test_the_trainer_wants_the_second_model(self, tiny):
        with pytest.raises(ValueError, match="a 'crosscoder' reads 2 models, and one reached it"):
            train_dictionary(tiny, make_records(), SMALL)
