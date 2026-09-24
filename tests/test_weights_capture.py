from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mechbench_compute.ops.weights import capture as capture_op
from mechbench_compute import weights as W
from mechbench_compute.ops.weights.capture import capture_weights
from mechbench_compute.ops.weights.decompose import decompose_weights


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
        names = W.read_parameters(lm)
        assert "layers.0.self_attn.q_proj.weight" in names
        assert "embed_tokens.weight" in names
        assert not any(n.startswith("model.") for n in names)

    def test_a_star_stands_for_one_segment(self, lm):
        names = W.read_parameters(lm)
        got = W.select_points(names, ["layers.*.mlp.down_proj"])
        assert got == [f"layers.{i}.mlp.down_proj.weight" for i in range(3)]

    def test_a_module_takes_its_parameters(self, lm):
        got = W.select_points(W.read_parameters(lm), ["layers.1.self_attn.q_proj"])
        assert got == ["layers.1.self_attn.q_proj.weight"]

    def test_the_scope_is_optional(self, lm):
        names = W.read_parameters(lm)
        assert W.select_points(names, ["model.layers.0.mlp.up_proj"]) == \
            W.select_points(names, ["layers.0.mlp.up_proj"])

    def test_a_point_that_matches_nothing_says_what_there_is(self, lm):
        with pytest.raises(ValueError, match="no parameter matches"):
            W.select_points(W.read_parameters(lm), ["layers.0.self_attn.v_proj"])

    def test_all_is_everything(self, lm):
        names = W.read_parameters(lm)
        assert W.select_points(names, "all") == list(names)


class TestCapture:
    def test_the_cheap_stats_are_always_there(self, lm):
        out = capture_weights(lm, {"points": ["layers.*.mlp.down_proj"]})
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
        assert "singular_values" not in it
        assert "values" not in it

    def test_the_header_says_what_was_read(self, lm):
        out = capture_weights(lm, {"points": ["layers.0.mlp.down_proj"]},
                                model_wire="acme/tiny")
        assert out["model"] == "acme/tiny"
        assert out["captured"]["parameters"] == 1
        assert out["captured"]["values"] == 128
        assert out["captured"]["of"] == len(W.read_parameters(lm))

    def test_the_spectrum_is_asked_for(self, lm):
        out = capture_weights(lm, {"points": ["layers.0.mlp.down_proj"],
                                     "spectrum": 3})
        it = out["items"][0]
        assert len(it["singular_values"]) == 3
        assert it["spectral"] == pytest.approx(it["singular_values"][0])
        assert 1.0 <= it["effective_rank"] <= 8.0
        arr = np.array(W.read_parameters(lm)["layers.0.mlp.down_proj.weight"]
                       .astype(mx.float32))
        assert np.allclose(it["singular_values"],
                           np.linalg.svd(arr, compute_uv=False)[:3], atol=1e-5)

    def test_values_are_asked_for_and_capped(self, lm, monkeypatch):
        out = capture_weights(lm, {"points": ["layers.0.mlp.down_proj"],
                                     "values": True})
        assert len(out["items"][0]["values"]) == 128
        monkeypatch.setattr(capture_op, "MAX_VALUES", 10)
        with pytest.raises(ValueError, match="past the 10 ceiling"):
            capture_weights(lm, {"points": ["layers.0.mlp.down_proj"],
                                   "values": True})

    def test_a_vector_parameter_has_stats_but_no_spectrum(self, lm):
        out = capture_weights(lm, {"points": ["layers.0.input_layernorm"],
                                     "spectrum": 4})
        it = out["items"][0]
        assert it["shape"] == [8]
        assert "singular_values" not in it


class TestDecompose:
    def test_a_writing_module_gives_directions_in_what_it_writes(self, lm):
        out = decompose_weights(lm, {"points": ["layers.1.self_attn.o_proj"],
                                       "top_k": 2}, model_wire="acme/tiny")
        assert out["item_kind"] == "direction/vector"
        assert [it["id"] for it in out["items"]] == [
            "layers.1.self_attn.o_proj.weight#0",
            "layers.1.self_attn.o_proj.weight#1"]
        first = out["items"][0]
        assert first["space"] == {"model": "acme/tiny", "layer": 1,
                                  "point": "attn_out", "d": 8}
        assert np.isclose(np.linalg.norm(first["vector"]), 1.0, atol=1e-6)
        assert first["norm"] >= out["items"][1]["norm"]
        assert first["derivation"]["side"] == "out"
        assert first["derivation"]["method"] == "weights/decompose"

    def test_a_reading_module_gives_directions_in_what_it_reads(self, lm):
        out = decompose_weights(lm, {"points": ["layers.0.self_attn.q_proj"],
                                       "top_k": 1})
        it = out["items"][0]
        assert it["derivation"]["side"] == "in"
        assert it["space"]["point"] == "attn.in_norm"
        assert it["space"]["d"] == 8

    def test_the_directions_are_the_matrix_own_singular_vectors(self, lm):
        arr = np.array(W.read_parameters(lm)["layers.1.self_attn.o_proj.weight"]
                       .astype(mx.float32))
        u, sv, _ = np.linalg.svd(arr, full_matrices=False)
        out = decompose_weights(lm, {"points": ["layers.1.self_attn.o_proj"],
                                       "top_k": 1})
        it = out["items"][0]
        assert it["norm"] == pytest.approx(float(sv[0]), rel=1e-5)
        assert abs(float(np.dot(it["vector"], u[:, 0]))) == pytest.approx(1.0, abs=1e-5)

    def test_a_module_with_no_residual_side_is_refused_by_name(self, lm):
        with pytest.raises(ValueError, match="no direction"):
            decompose_weights(lm, {"points": ["layers.0.input_layernorm"]})

    def test_naming_a_side_skips_the_modules_whose_side_is_the_other(self, lm):
        out = decompose_weights(lm, {"points": ["layers.0.self_attn.q_proj",
                                                  "layers.0.self_attn.o_proj"],
                                       "side": "out", "top_k": 1})
        assert [it["derivation"]["module"] for it in out["items"]] == [
            "layers.0.self_attn.o_proj.weight"]
        assert any("q_proj" in s for s in out["decomposed"]["skipped"])

    def test_points_are_required(self, lm):
        with pytest.raises(ValueError, match="needs `points`"):
            decompose_weights(lm, {})


class TestParameterIntervention:
    def _w(self, lm, name="layers.0.self_attn.o_proj.weight"):
        return np.array(W.read_parameters(lm)[name].astype(mx.float32))

    def _dir(self, vec):
        v = np.asarray(vec, dtype=np.float32)
        return {"kind": "direction/vector", "id": "d", "vector": [float(x) for x in v],
                "space": {"model": "acme/tiny", "layer": 0, "point": "attn_out",
                          "d": int(v.size)},
                "derivation": {"method": "test"}}

    def test_zero_and_restore_is_exact(self, lm):
        before = self._w(lm).copy()
        handle = W.edit_parameters(
            lm, [{"parameter": "layers.0.self_attn.o_proj", "op": "zero"}])
        assert np.all(self._w(lm) == 0)
        W.restore_parameters(lm, handle)
        assert np.array_equal(self._w(lm), before)

    def test_scale_takes_the_sweep_factor(self, lm):
        before = self._w(lm).copy()
        handle = W.edit_parameters(
            lm, [{"parameter": "layers.0.self_attn.o_proj", "op": "scale",
                  "strength": 2.0}], factor=0.5)
        assert np.allclose(self._w(lm), before, atol=1e-6)
        W.restore_parameters(lm, handle)

    def test_a_star_edits_every_layer(self, lm):
        handle = W.edit_parameters(
            lm, [{"parameter": "layers.*.self_attn.o_proj", "op": "zero"}])
        assert len(handle) == 3
        for i in range(3):
            assert np.all(self._w(lm, f"layers.{i}.self_attn.o_proj.weight") == 0)
        W.restore_parameters(lm, handle)
        assert np.any(self._w(lm) != 0)

    def test_project_out_removes_a_direction_from_what_it_writes(self, lm):
        before = self._w(lm)
        v = before[:, 0] / np.linalg.norm(before[:, 0])
        handle = W.edit_parameters(
            lm, [{"parameter": "layers.0.self_attn.o_proj",
                  "op": "project_out", "direction": self._dir(v)}])
        after = self._w(lm)
        assert np.allclose(v @ after, 0, atol=1e-5)
        assert np.allclose(after, before - np.outer(v, v @ before), atol=1e-5)
        W.restore_parameters(lm, handle)
        assert np.allclose(self._w(lm), before)

    def test_project_out_on_a_reading_module_uses_the_other_side(self, lm):
        name = "layers.0.self_attn.q_proj.weight"
        before = self._w(lm, name)
        v = before[0] / np.linalg.norm(before[0])
        handle = W.edit_parameters(
            lm, [{"parameter": "layers.0.self_attn.q_proj",
                  "op": "project_out", "direction": self._dir(v)}])
        after = self._w(lm, name)
        assert np.allclose(after @ v, 0, atol=1e-5)
        W.restore_parameters(lm, handle)

    def test_a_direction_of_the_wrong_width_is_refused(self, lm):
        with pytest.raises(ValueError, match="only removes from the space"):
            W.edit_parameters(lm, [{"parameter": "layers.0.self_attn.o_proj",
                                    "op": "project_out",
                                    "direction": self._dir([1.0, 0.0])}])

    def test_truncate_keeps_the_top_singular_directions(self, lm):
        name = "layers.0.mlp.down_proj.weight"
        before = self._w(lm, name)
        handle = W.edit_parameters(
            lm, [{"parameter": "layers.0.mlp.down_proj", "op": "truncate",
                  "rank": 2}])
        after = self._w(lm, name)
        sv = np.linalg.svd(after, compute_uv=False)
        assert np.count_nonzero(sv > 1e-4) <= 2
        assert np.linalg.norm(after) < np.linalg.norm(before)
        W.restore_parameters(lm, handle)
        assert np.allclose(self._w(lm, name), before)

    def test_truncate_needs_a_rank_and_zero_needs_nothing(self, lm):
        with pytest.raises(ValueError, match="needs a `rank`"):
            W.edit_parameters(lm, [{"parameter": "layers.0.mlp.down_proj",
                                    "op": "truncate"}])

    def test_an_unknown_weight_op_names_the_ones_there_are(self, lm):
        with pytest.raises(ValueError, match="unknown weight op"):
            W.edit_parameters(lm, [{"parameter": "layers.0.mlp.down_proj",
                                    "op": "resample"}])
