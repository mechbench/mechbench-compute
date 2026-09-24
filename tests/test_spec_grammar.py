from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import intervene as iv
from mechbench_compute import shapes as S
from mechbench_compute.hooks import HookInfo

TOKENS = ["a", "b", "c"]


def _hook(item, *, layer=1, n_layers=6, tokens=TOKENS):
    spec = iv.Spec(item, n_layers=n_layers, seed=0)
    return spec, spec.build(layer, tokens, None)


def _info(point, layer=1):
    return HookInfo(name=f"blocks.{layer}.{point}", layer=layer, point=point)


class TestTheComplement:
    def test_layers_invert_at_parse(self):
        spec, _ = _hook({"point": "resid_post", "layers": [1, 4], "except": True}, n_layers=6)
        assert spec.layers == [0, 2, 3, 5]

    def test_heads_invert_against_the_tensors_own_axis(self):
        act = mx.ones((1, 4, 3, 2))
        _, fn = _hook({"point": "attn.per_head_out", "heads": [1], "positions": "all",
                       "op": "zero", "except": True})
        kept = np.array(fn(act, _info("attn.per_head_out")))[0, :, 0, 0]
        assert list(kept) == [0.0, 1.0, 0.0, 0.0]
        _, plain = _hook({"point": "attn.per_head_out", "heads": [1], "positions": "all", "op": "zero"})
        assert list(np.array(plain(act, _info("attn.per_head_out")))[0, :, 0, 0]) == [1.0, 0.0, 1.0, 1.0]

    def test_neurons_invert_too(self):
        act = mx.ones((1, 3, 4))
        _, fn = _hook({"point": "mlp.act", "neurons": [0, 2], "positions": "all",
                       "op": "zero", "except": True})
        assert list(np.array(fn(act, _info("mlp.act")))[0, 0]) == [1.0, 0.0, 1.0, 0.0]

    def test_it_needs_a_set_to_invert_and_a_point_that_has_one(self):
        with pytest.raises(iv.SpecError, match="needs a set to invert"):
            iv.Spec({"point": "resid_post", "except": True}, n_layers=6, seed=0)
        with pytest.raises(iv.SpecError, match="occurs once per forward pass"):
            iv.Spec({"point": "logits", "neurons": [3], "except": True}, n_layers=6, seed=0)


class TestAnAttentionEdge:
    WEIGHTS = mx.array(np.full((1, 2, 3, 3), 0.25, np.float32))

    def test_zeroing_one_edge_and_renormalising_the_row(self):
        _, fn = _hook({"point": "attn.weights", "op": "zero",
                       "pattern": {"from": {"tokens": ["b"]}, "to": "last"}})
        out = np.array(fn(self.WEIGHTS, _info("attn.weights")))[0, 0]
        assert out[2, 1] == 0.0 and abs(out[2].sum() - 1.0) < 1e-6
        assert abs(out[2, 0] - 0.5) < 1e-6
        assert list(out[0]) == [0.25, 0.25, 0.25]

    def test_renormalize_false_leaves_the_row_short(self):
        _, fn = _hook({"point": "attn.weights", "op": "zero", "renormalize": False,
                       "pattern": {"from": {"tokens": ["b"]}, "to": "last"}})
        out = np.array(fn(self.WEIGHTS, _info("attn.weights")))[0, 0]
        assert out[2, 1] == 0.0 and abs(out[2].sum() - 0.5) < 1e-6

    def test_to_defaults_to_the_items_positions_and_from_is_required(self):
        _, fn = _hook({"point": "attn.weights", "op": "zero", "positions": [0],
                       "pattern": {"from": [2]}})
        out = np.array(fn(self.WEIGHTS, _info("attn.weights")))[0, 0]
        assert out[0, 2] == 0.0 and out[1, 2] == 0.25
        with pytest.raises(iv.SpecError, match="`from` is required"):
            iv.Spec({"point": "attn.weights", "pattern": {"to": "last"}}, n_layers=6, seed=0)

    def test_a_score_is_cut_with_minus_infinity(self):
        act = mx.zeros((1, 2, 3, 3))
        _, fn = _hook({"point": "attn.scores", "op": "zero",
                       "pattern": {"from": [0], "to": "all"}})
        out = np.array(fn(act, _info("attn.scores")))[0, 0]
        assert np.isneginf(out[:, 0]).all() and (out[:, 1] == 0.0).all()
        p = np.array(mx.softmax(mx.array(out), axis=-1))
        assert p[0, 0] == 0.0 and abs(p[0].sum() - 1.0) < 1e-6

    def test_an_edge_needs_a_point_that_has_sources(self):
        with pytest.raises(iv.SpecError, match="no source axis"):
            iv.Spec({"point": "resid_post", "pattern": {"from": [0]}}, n_layers=6, seed=0)


class TestAPatchFromAnotherLayer:
    def _source(self):
        rows = [S.vector(np.full(4, float(layer), np.float32),
                         S.space(model="m", layer=layer, point="resid_post", d=4), id=f"l{layer}")
                for layer in (2, 8)]
        return {"kind": "collection", "item_kind": "activations/vector", "items": rows}

    def test_the_row_is_read_where_from_says(self):
        act = mx.zeros((1, 3, 4))
        item = {"point": "resid_post", "op": "patch", "positions": "last",
                "source": self._source(), "from": {"layer": 8}}
        _, fn = _hook(item, layer=2)
        assert list(np.array(fn(act, _info("resid_post", 2)))[0, 2]) == [8.0] * 4
        _, plain = _hook({k: v for k, v in item.items() if k != "from"}, layer=2)
        assert list(np.array(plain(act, _info("resid_post", 2)))[0, 2]) == [2.0] * 4

    def test_from_belongs_to_a_patch(self):
        with pytest.raises(iv.SpecError, match="this item's op is 'zero'"):
            iv.Spec({"point": "resid_post", "op": "zero", "from": {"layer": 8}}, n_layers=9, seed=0)
