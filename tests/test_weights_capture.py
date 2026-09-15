"""`weights/capture` and `weights/decompose` (task 000457).

The parameter half of the points × operations grammar: what the model
IS, rather than what it did on an input. The model here is a stub whose
parameter tree has the shape a real decoder's has, so the selector, the
stats and the residual-side rule are all exercised without 9.5 GB.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mechbench_compute import weights as W


class _Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.q_proj = nn.Linear(d, heads, bias=False)
        self.o_proj = nn.Linear(heads, d, bias=False)


class _MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)


class _Layer(nn.Module):
    def __init__(self, d, heads, hidden):
        super().__init__()
        self.self_attn = _Attn(d, heads)
        self.mlp = _MLP(d, hidden)
        self.input_layernorm = nn.RMSNorm(d)


class _Inner(nn.Module):
    def __init__(self, n, d, heads, hidden):
        super().__init__()
        self.layers = [_Layer(d, heads, hidden) for _ in range(n)]
        self.embed_tokens = nn.Embedding(32, d)
        self.norm = nn.RMSNorm(d)


class _LM(nn.Module):
    def __init__(self, n=3, d=8, heads=4, hidden=16):
        super().__init__()
        self.model = _Inner(n, d, heads, hidden)


@pytest.fixture
def lm():
    mx.random.seed(0)
    return _LM()


class TestNamingAndSelection:
    def test_parameters_are_named_as_the_module_tree_names_them(self, lm):
        names = W.parameter_names(lm)
        assert "layers.0.self_attn.q_proj.weight" in names
        assert "embed_tokens.weight" in names
        assert not any(n.startswith("model.") for n in names)

    def test_a_star_stands_for_one_segment(self, lm):
        names = W.parameter_names(lm)
        got = W.select_points(names, ["layers.*.mlp.down_proj"])
        assert got == [f"layers.{i}.mlp.down_proj.weight" for i in range(3)]

    def test_a_module_takes_its_parameters(self, lm):
        got = W.select_points(W.parameter_names(lm), ["layers.1.self_attn.q_proj"])
        assert got == ["layers.1.self_attn.q_proj.weight"]

    def test_the_scope_is_optional(self, lm):
        names = W.parameter_names(lm)
        assert W.select_points(names, ["model.layers.0.mlp.up_proj"]) == \
            W.select_points(names, ["layers.0.mlp.up_proj"])

    def test_a_point_that_matches_nothing_says_what_there_is(self, lm):
        with pytest.raises(ValueError, match="no parameter matches"):
            W.select_points(W.parameter_names(lm), ["layers.0.self_attn.v_proj"])

    def test_all_is_everything(self, lm):
        names = W.parameter_names(lm)
        assert W.select_points(names, "all") == list(names)


class TestCapture:
    def test_the_cheap_stats_are_always_there(self, lm):
        out = W.capture_weights(lm, {"points": ["layers.*.mlp.down_proj"]})
        assert out["item_kind"] == "weights/parameter"
        assert len(out["items"]) == 3
        it = out["items"][0]
        assert it["id"] == "layers.0.mlp.down_proj.weight"
        assert it["coords"] == {"layer": 0, "container": "mlp",
                                "projection": "down_proj",
                                "module": "layers.0.mlp.down_proj",
                                "parameter": "weight"}
        assert it["shape"] == [8, 16] and it["n"] == 128
        assert it["frobenius"] > 0 and 0.0 <= it["sparsity"] <= 1.0
        assert "singular_values" not in it     # not unless asked
        assert "values" not in it

    def test_the_header_says_what_was_read(self, lm):
        out = W.capture_weights(lm, {"points": ["layers.0.mlp.down_proj"]},
                                model_wire="acme/tiny")
        assert out["model"] == "acme/tiny"
        assert out["captured"]["parameters"] == 1
        assert out["captured"]["values"] == 128
        assert out["captured"]["of"] == len(W.parameter_names(lm))

    def test_the_spectrum_is_asked_for(self, lm):
        out = W.capture_weights(lm, {"points": ["layers.0.mlp.down_proj"],
                                     "spectrum": 3})
        it = out["items"][0]
        assert len(it["singular_values"]) == 3
        assert it["spectral"] == pytest.approx(it["singular_values"][0])
        assert 1.0 <= it["effective_rank"] <= 8.0
        # …and it is the matrix's own spectrum.
        arr = np.array(W.parameter_names(lm)["layers.0.mlp.down_proj.weight"]
                       .astype(mx.float32))
        assert np.allclose(it["singular_values"],
                           np.linalg.svd(arr, compute_uv=False)[:3], atol=1e-5)

    def test_values_are_asked_for_and_capped(self, lm, monkeypatch):
        out = W.capture_weights(lm, {"points": ["layers.0.mlp.down_proj"],
                                     "values": True})
        assert len(out["items"][0]["values"]) == 128
        monkeypatch.setattr(W, "MAX_VALUES", 10)
        with pytest.raises(ValueError, match="past the 10 ceiling"):
            W.capture_weights(lm, {"points": ["layers.0.mlp.down_proj"],
                                   "values": True})

    def test_a_vector_parameter_has_stats_but_no_spectrum(self, lm):
        out = W.capture_weights(lm, {"points": ["layers.0.input_layernorm"],
                                     "spectrum": 4})
        it = out["items"][0]
        assert it["shape"] == [8]
        assert "singular_values" not in it   # an SVD of a vector is nothing


class TestDecompose:
    def test_a_writing_module_gives_directions_in_what_it_writes(self, lm):
        out = W.decompose_weights(lm, {"points": ["layers.1.self_attn.o_proj"],
                                       "top_k": 2}, model_wire="acme/tiny")
        assert out["item_kind"] == "direction/vector"
        assert [it["id"] for it in out["items"]] == [
            "layers.1.self_attn.o_proj.weight#0",
            "layers.1.self_attn.o_proj.weight#1"]
        first = out["items"][0]
        assert first["space"] == {"model": "acme/tiny", "layer": 1,
                                  "point": "attn_out", "d": 8}
        assert np.isclose(np.linalg.norm(first["vector"]), 1.0, atol=1e-6)
        assert first["norm"] >= out["items"][1]["norm"]   # largest first
        assert first["derivation"]["side"] == "out"
        assert first["derivation"]["method"] == "weights/decompose"

    def test_a_reading_module_gives_directions_in_what_it_reads(self, lm):
        out = W.decompose_weights(lm, {"points": ["layers.0.self_attn.q_proj"],
                                       "top_k": 1})
        it = out["items"][0]
        assert it["derivation"]["side"] == "in"
        assert it["space"]["point"] == "attn.in_norm"
        assert it["space"]["d"] == 8           # the residual width, not the head's

    def test_the_directions_are_the_matrix_own_singular_vectors(self, lm):
        arr = np.array(W.parameter_names(lm)["layers.1.self_attn.o_proj.weight"]
                       .astype(mx.float32))
        u, sv, _ = np.linalg.svd(arr, full_matrices=False)
        out = W.decompose_weights(lm, {"points": ["layers.1.self_attn.o_proj"],
                                       "top_k": 1})
        it = out["items"][0]
        assert it["norm"] == pytest.approx(float(sv[0]), rel=1e-5)
        assert abs(float(np.dot(it["vector"], u[:, 0]))) == pytest.approx(1.0, abs=1e-5)

    def test_a_module_with_no_residual_side_is_refused_by_name(self, lm):
        with pytest.raises(ValueError, match="no direction"):
            W.decompose_weights(lm, {"points": ["layers.0.input_layernorm"]})

    def test_naming_a_side_skips_the_modules_whose_side_is_the_other(self, lm):
        out = W.decompose_weights(lm, {"points": ["layers.0.self_attn.q_proj",
                                                  "layers.0.self_attn.o_proj"],
                                       "side": "out", "top_k": 1})
        assert [it["derivation"]["module"] for it in out["items"]] == [
            "layers.0.self_attn.o_proj.weight"]
        assert any("q_proj" in s for s in out["decomposed"]["skipped"])

    def test_points_are_required(self, lm):
        with pytest.raises(ValueError, match="needs `points`"):
            W.decompose_weights(lm, {})
