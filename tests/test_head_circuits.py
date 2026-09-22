"""`weights/circuit` (task 000610): what a head does, read from the
weights alone — the OV and QK circuits, and which earlier heads reach
this one."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mechbench_compute import head_weights as hw
from mechbench_compute import weights as weights_mod
from mechbench_compute.ops.weights.circuit import head_circuits


def _spec(w_o, w_v, *, layer=0, head=0, d_model=8, head_dim=2):
    return hw.HeadSpec(
        layer=layer, head=head, kv_group=0, head_dim=head_dim, n_heads=2, n_kv_heads=1,
        W_Q=np.zeros((head_dim, d_model), np.float32), W_K=np.zeros((head_dim, d_model), np.float32),
        W_V=np.asarray(w_v, np.float32), W_O=np.asarray(w_o, np.float32),
        is_global=True, is_kv_shared=False)


class TestCompositionScore:
    """The score is the share of what a head writes that lands in what a
    later head reads — 0 when they never meet."""

    D, H = 8, 2

    def _basis(self, dims):
        """W_O [d_model, head_dim] and W_V [head_dim, d_model] writing
        into exactly the named residual dimensions."""
        w_o = np.zeros((self.D, self.H), np.float32)
        w_v = np.zeros((self.H, self.D), np.float32)
        for i, dim in enumerate(dims):
            w_o[dim, i] = 1.0
            w_v[i, dim] = 1.0
        return w_o, w_v

    def _reader(self, dims):
        r = np.zeros((self.H, self.D), np.float32)
        for i, dim in enumerate(dims):
            r[i, dim] = 1.0
        return r

    def test_writing_where_the_reader_looks_scores_and_elsewhere_does_not(self):
        reader = self._reader([0, 1])
        overlapping = hw.composition_score(reader, _spec(*self._basis([0, 1])))
        disjoint = hw.composition_score(reader, _spec(*self._basis([2, 3])))
        assert disjoint == 0.0
        assert overlapping > 0.7
        half = hw.composition_score(reader, _spec(*self._basis([1, 2])))
        assert 0.0 < half < overlapping

    def test_the_norms_make_it_scale_free(self):
        reader = self._reader([0, 1])
        w_o, w_v = self._basis([0, 1])
        assert hw.composition_score(reader, _spec(w_o * 17.0, w_v)) == pytest.approx(
            hw.composition_score(reader, _spec(w_o, w_v)))

    def test_a_head_composes_only_with_earlier_layers(self):
        class _Model:
            class arch:
                n_layers, n_heads = 4, 2
        with pytest.raises(ValueError, match="only compose with earlier layers"):
            hw.head_composition(_Model(), 2, 0, source_layers=[3])


E2B = "mlx-community/gemma-4-e2b-it-bf16"
_real = pytest.mark.skipif(
    os.environ.get("MECHBENCH_MODEL_TESTS") != "1"
    or not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--" + E2B.replace("/", "--"))),
    reason="set MECHBENCH_MODEL_TESTS=1 with gemma-4-e2b cached",
)


@_real
class TestOnGemma:
    @pytest.fixture(scope="class")
    def model(self):
        from mechbench_compute import Model
        return Model.load(E2B)

    def test_ov_reads_a_head_as_tokens_in_and_tokens_out(self, model):
        out = head_circuits(model, {"circuit": "ov", "head": {"layer": 12, "index": 3},
                                                "components": 2, "top_k": 5})
        assert out["item_kind"] == "records/record" and out["circuit"] == "ov"
        assert [it["coords"]["rank"] for it in out["items"]] == [0, 1]
        first = out["items"][0]
        assert first["coords"] == {"layer": 12, "head": 3, "rank": 0, "circuit": "ov"}
        assert first["strength"] > 0
        assert len(first["left"]) == 5 and len(first["right"]) == 5
        assert all(isinstance(t["token"], str) for t in first["left"])
        # The components are ordered by the gain they carry.
        assert first["strength"] >= out["items"][1]["strength"]

    def test_qk_is_the_other_circuit_and_a_layer_of_heads_is_one_node(self, model):
        out = head_circuits(model, {"circuit": "qk", "layers": [12], "components": 1,
                                                "top_k": 3})
        assert out["circuit"] == "qk"
        assert [it["coords"]["head"] for it in out["items"]] == list(range(model.arch.n_heads))

    def test_composition_scores_every_earlier_head_against_this_one(self, model):
        out = head_circuits(model, {"circuit": "composition",
                                                "head": {"layer": 12, "index": 3},
                                                "layers": [10, 11], "kinds": ["q", "k"]})
        assert out["circuit"] == "composition" and out["into"] == {"layer": 12, "index": 3}
        rows = out["items"]
        assert len(rows) == 2 * model.arch.n_heads * 2
        assert {r["coords"]["kind"] for r in rows} == {"q", "k"}
        assert all(0.0 <= r["score"] <= 1.0 for r in rows)
        # Some head reaches it more than the median one does.
        scores = sorted(r["score"] for r in rows)
        assert scores[-1] > scores[len(scores) // 2] > 0

    def test_composition_needs_the_head_it_reads_into(self, model):
        with pytest.raises(ValueError, match="reads INTO one head"):
            head_circuits(model, {"circuit": "composition"})
