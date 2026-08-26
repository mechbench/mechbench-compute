"""The interp primitives (mechbench-experiments port), on a stub model.

The stub's residuals are ONE-HOT in the token id (scaled by layer+1),
so every geometric assertion is exact: identical tokens → cosine 1,
different tokens → cosine 0. Ablation is simulated by damping the
favored token's logit by (layer+1), so deltas are computable in the
test with the same softmax arithmetic.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import interp

N_LAYERS = 4
D_MODEL = 8
VOCAB = 12
FAV = 7  # the token the stub's baseline favors


class StubArch:
    n_layers = N_LAYERS
    d_model = D_MODEL


class StubTokenizer:
    all_special_ids = (0,)

    def decode(self, ids):
        return " ".join(f"t{int(i)}" for i in ids)


class StubModel:
    """Word-per-token: each word maps to id 1 + (len(word) % 7)."""

    arch = StubArch()
    tokenizer = StubTokenizer()

    def __init__(self):
        self.runs = 0

    def tokenize(self, prompt: str, chat_template: bool = True):
        ids = [0] + [1 + (len(w) % 7) for w in prompt.split()]
        return mx.array([ids])

    def run(self, ids, interventions=None):
        self.runs += 1
        arr = np.array(ids)[0]
        seq = len(arr)
        # ablation penalty: sum of (layer+1) across ablated layers,
        # detected from the intervention objects' own fields
        penalty = 0.0
        for iv in interventions or []:
            layer_idx = getattr(iv, "layer_idx", None)
            if layer_idx is not None:
                penalty += layer_idx + 1
            for name in getattr(iv, "names", ()) or ():
                if name.endswith(("attn_out", "mlp_out", "gate_out")):
                    penalty += int(name.split(".")[1]) + 1

        logits = np.zeros((1, seq, VOCAB), dtype=np.float32)
        logits[0, -1, FAV] = 5.0 - penalty

        cache = {}
        for layer in range(N_LAYERS):
            resid = np.zeros((1, seq, D_MODEL), dtype=np.float32)
            for pos, tok in enumerate(arr):
                resid[0, pos, int(tok) % D_MODEL] = float(layer + 1)
            cache[f"blocks.{layer}.resid_post"] = mx.array(resid)

        class R:
            pass

        r = R()
        r.logits = mx.array(logits)
        r.cache = cache
        return r


def _expected_delta(layer: int) -> float:
    def logp(fav_logit: float) -> float:
        z = math.log(math.exp(fav_logit) + (VOCAB - 1) * math.exp(0.0))
        return fav_logit - z

    return logp(5.0 - (layer + 1)) - logp(5.0)


class TestAblateLayers:
    def test_deltas_match_the_softmax_arithmetic_exactly(self):
        model = StubModel()
        out = interp.ablate_layers(
            model, [{"id": "c1", "user": "the tower is in"}], {})
        assert out["kind"] == "ablation_sweep"
        assert out["layers"] == [0, 1, 2, 3]
        deltas = {r["layer"]: r["delta_logp"] for r in out["rows"]
                  if r.get("layer") is not None}
        for layer in range(N_LAYERS):
            assert deltas[layer] == pytest.approx(
                _expected_delta(layer), abs=1e-3)
        # deeper stub layers are damped harder — the sweep must say so
        assert deltas[3] < deltas[0] < 0

    def test_progress_covers_every_forward(self):
        model = StubModel()
        ticks = []
        interp.ablate_layers(
            model, [{"id": "a", "user": "x y"}, {"id": "b", "user": "p q"}],
            {"layers": [1, 2]},
            on_item=lambda: ticks.append(1), on_start=lambda n: ticks.append(n))
        assert ticks[0] == 2 * 3  # (baseline + 2 layers) per condition
        assert sum(t for t in ticks[1:]) == 6
        assert model.runs == 6

    def test_a_named_target_wins_over_top1(self):
        model = StubModel()
        out = interp.ablate_layers(
            model, [{"id": "c", "user": "a b", "target": "word"}], {"layers": [0]})
        meta = next(r for r in out["rows"] if r.get("layer") is None)
        assert meta["target_id"] == 1 + (len("word") % 7)

    def test_sublayer_components_route_to_their_hooks(self):
        model = StubModel()
        out = interp.ablate_layers(
            model, [{"id": "c", "user": "a b"}],
            {"component": "mlp", "layers": [2]})
        row = next(r for r in out["rows"] if r.get("layer") == 2)
        assert row["delta_logp"] == pytest.approx(_expected_delta(2), abs=1e-3)

    def test_unknown_component_refuses(self):
        with pytest.raises(ValueError, match="component"):
            interp.ablate_layers(StubModel(), [{"id": "c", "user": "a"}],
                                 {"component": "norm"})


class TestResidualVectors:
    def test_vectors_are_the_positions_residual(self):
        model = StubModel()
        out = interp.residual_vectors(
            model, [{"id": "c", "user": "aa bbb", "label": "en"}],
            {"layers": [1], "position": "final"})
        row = out["rows"][0]
        assert row["label"] == "en"
        v = np.array(row["vector"])
        # final token of "aa bbb" is id 1+(3%7)=4; layer 1 scale = 2
        assert v[4] == pytest.approx(2.0)
        assert np.count_nonzero(v) == 1

    def test_label_coord_pulls_from_coords(self):
        model = StubModel()
        out = interp.residual_vectors(
            model, [{"id": "c", "user": "a", "coords": {"language": "fr"}}],
            {"layers": [0], "label_coord": "language"})
        assert out["rows"][0]["label"] == "fr"

    def test_the_float_cap_refuses_a_runaway_capture(self, monkeypatch):
        monkeypatch.setattr(interp, "MAX_VECTOR_FLOATS", 10)
        with pytest.raises(ValueError, match="cap"):
            interp.residual_vectors(
                StubModel(), [{"id": "c", "user": "a"}], {"layers": "all"})


class TestResidualDivergence:
    def test_identical_pair_diverges_nowhere(self):
        model = StubModel()
        out = interp.residual_divergence(
            model, [{"id": "p", "a": "over the hill", "b": "over the hill"}],
            {"layers": [0, 1]})
        pair = out["pairs"][0]
        flat = [x for row in pair["divergence"] for x in row]
        assert all(x == pytest.approx(0.0, abs=1e-4) for x in flat)

    def test_a_one_word_swap_diverges_exactly_there(self):
        model = StubModel()
        # 'over'(4) vs 'under'(5) -> ids differ at position 1 only
        out = interp.residual_divergence(
            model, [{"id": "p", "a": "go over it", "b": "go under it"}],
            {"layers": [0]})
        div = out["pairs"][0]["divergence"][0]
        assert div[0] == pytest.approx(0.0, abs=1e-4)  # BOS
        assert div[1] == pytest.approx(0.0, abs=1e-4)  # 'go'
        assert div[2] == pytest.approx(1.0, abs=1e-4)  # the swapped word
        assert div[3] == pytest.approx(0.0, abs=1e-4)  # 'it'

    def test_unequal_lengths_report_instead_of_lying(self):
        model = StubModel()
        out = interp.residual_divergence(
            model, [{"id": "p", "a": "one two", "b": "one two three"}],
            {"layers": [0]})
        assert "different lengths" in out["pairs"][0]["error"]


class TestVectorSimilarity:
    def _vectors_record(self):
        # two tight clusters along different axes
        rows = []
        for i, label in enumerate(["cat", "cat", "dog", "dog"]):
            v = [0.0] * 4
            v[0 if label == "cat" else 2] = 1.0
            v[1 if label == "cat" else 3] = 0.1 * i
            rows.append({"id": f"r{i}", "label": label, "layer": 5, "vector": v})
        return {"kind": "residual_vectors", "layers": [5], "position": "final",
                "point": "post", "rows": rows}

    def test_matrix_and_separation(self):
        out = interp.vector_similarity({"vectors": self._vectors_record()}, {})
        assert out["kind"] == "similarity_matrix"
        layer = out["layers"][0]
        assert layer["layer"] == 5
        m = np.array(layer["matrix"])
        assert m.shape == (4, 4)
        assert m[0, 1] > m[0, 2]  # same-label closer than cross-label
        assert layer["separation"]["gap"] > 0
        assert layer["nn_purity"] == pytest.approx(1.0)

    def test_unlabeled_vectors_still_get_a_matrix(self):
        rec = self._vectors_record()
        for r in rec["rows"]:
            r["label"] = None
        out = interp.vector_similarity({"vectors": rec}, {})
        assert "separation" not in out["layers"][0]

    def test_wrong_input_kind_refuses(self):
        with pytest.raises(ValueError, match="residual_vectors"):
            interp.vector_similarity({"vectors": {"kind": "word_list"}}, {})


class TestGateComponent:
    def test_gate_routes_to_the_side_channel_hook(self):
        model = StubModel()
        out = interp.ablate_layers(
            model, [{"id": "c", "user": "a b"}],
            {"component": "gate", "layers": [1]})
        assert out["component"] == "gate"
        row = next(r for r in out["rows"] if r.get("layer") == 1)
        assert row["delta_logp"] < 0  # the stub penalizes any named zero-hook
