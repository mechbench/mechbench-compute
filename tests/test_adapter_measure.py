"""`adapter/measure`: what training wrote, read from the adapter (000458).

The numbers are checked against the delta formed in full — the point of
the r×r route is that it is exact, so "close to the honest computation"
is the test, not a golden file.
"""

from __future__ import annotations

import os
import tempfile

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import weights as W
from mechbench_compute.blocks import PURE_BLOCKS
from mechbench_compute.ops.adapter.measure import compute_delta_spectrum


def _bytes(flat):
    """`{name: mx.array}` as safetensors bytes."""
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        mx.save_safetensors(path, flat)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def _flat(pairs):
    flat = {}
    for (layer, container, proj), (a, b) in pairs.items():
        key = f"model.layers.{layer}.{container}.{proj}"
        flat[f"{key}.lora_a"] = mx.array(np.asarray(a, dtype=np.float32))
        flat[f"{key}.lora_b"] = mx.array(np.asarray(b, dtype=np.float32))
    return flat


def _adapter(pairs, *, rank=4, alpha=8.0, base="acme/tiny", drop=()):
    """An adapter object as `adapter/train` emits one: safetensors bytes
    keyed by module, plus the shape it was trained at. `drop` leaves keys
    out — the tensors are built here rather than round-tripped through a
    file, because `mx.load` is lazy and rewriting the file its arrays are
    backed by aborts the process."""
    flat = {k: v for k, v in _flat(pairs).items() if k not in set(drop)}
    return {"kind": "adapter/lora", "format": "safetensors", "data": _bytes(flat),
            "base_model": base,
            "lora": {"rank": rank, "alpha": alpha, "scale": alpha / rank,
                     "target_modules": sorted({p for _l, _c, p in pairs})}}


def _random_pairs(seed=0, layers=(0, 1), rank=4, out=6, in_=5):
    rng = np.random.default_rng(seed)
    return {(layer, "self_attn", proj): (rng.normal(size=(rank, in_)),
                                         rng.normal(size=(out, rank)))
            for layer in layers for proj in ("q_proj", "v_proj")}


def _measure(payload, **params):
    return PURE_BLOCKS["adapter/measure"]({"adapter": payload}, params)


class TestTheSpectrumIsExact:
    def test_the_numbers_match_the_delta_formed_in_full(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=(4, 5)), rng.normal(size=(6, 4))
        scale = 2.0
        sv, u = compute_delta_spectrum(a, b, scale)
        full = scale * (b @ a)
        u_full, s_full, _ = np.linalg.svd(full)
        assert np.allclose(sv, s_full[:len(sv)], atol=1e-10)
        assert np.isclose(np.sqrt((sv ** 2).sum()),
                          np.linalg.norm(full, "fro"), atol=1e-10)
        # Left singular vectors up to sign.
        for k in range(len(sv)):
            assert np.isclose(abs(float(u[:, k] @ u_full[:, k])), 1.0, atol=1e-8)

    def test_a_rank_one_write_has_effective_rank_one(self):
        rng = np.random.default_rng(3)
        a = np.zeros((4, 5))
        a[0] = rng.normal(size=5)
        b = np.zeros((6, 4))
        b[:, 0] = rng.normal(size=6)
        sv, _u = compute_delta_spectrum(a, b, 1.0)
        assert W.compute_effective_rank(sv) == pytest.approx(1.0, abs=1e-9)

    def test_an_even_write_has_effective_rank_r(self):
        # Four orthogonal directions of equal size: the spectrum is flat.
        a = np.eye(4, 5)
        b = np.eye(6, 4)
        sv, _u = compute_delta_spectrum(a, b, 1.0)
        assert W.compute_effective_rank(sv) == pytest.approx(4.0, abs=1e-9)

    def test_a_delta_of_nothing_has_no_rank(self):
        assert W.compute_effective_rank([0.0, 0.0]) == 0.0


class TestTheReadout:
    def test_one_item_per_module_with_shares_that_sum_to_one(self):
        out = _measure(_adapter(_random_pairs()))
        items = out["items"]
        assert [it["id"] for it in items] == [
            "layers.0.self_attn.q_proj", "layers.0.self_attn.v_proj",
            "layers.1.self_attn.q_proj", "layers.1.self_attn.v_proj"]
        assert sum(it["mass_share"] for it in items) == pytest.approx(1.0)
        first = items[0]
        assert first["coords"] == {"layer": 0, "module": "layers.0.self_attn.q_proj",
                                   "projection": "q_proj", "container": "self_attn"}
        assert first["shape"] == [6, 5]
        assert first["rank"] == 4
        assert first["spectral"] <= first["frobenius"] + 1e-9
        assert "vector" not in first          # not unless asked for

    def test_the_header_says_what_was_measured(self):
        out = _measure(_adapter(_random_pairs()), layers=[1])
        assert out["item_kind"] == "adapter/delta"
        assert out["measured"]["layers"] == [1]
        assert out["measured"]["modules"] == 2
        assert out["lora"]["rank"] == 4 and out["lora"]["scale"] == 2.0
        assert out["base_model"] == "acme/tiny"
        # …and the shares are shares of what was measured, not of the
        # adapter, which is why the header says which layers those were.
        assert sum(it["mass_share"] for it in out["items"]) == pytest.approx(1.0)

    def test_layers_and_modules_select(self):
        out = _measure(_adapter(_random_pairs()), layers=[0], modules=["v_proj"])
        assert [it["id"] for it in out["items"]] == ["layers.0.self_attn.v_proj"]
        out = _measure(_adapter(_random_pairs()), modules=["self_attn.q_proj"])
        assert all(it["coords"]["projection"] == "q_proj" for it in out["items"])

    def test_a_selection_that_matches_nothing_says_what_there_is(self):
        with pytest.raises(ValueError, match="no module of this adapter"):
            _measure(_adapter(_random_pairs()), modules=["down_proj"])

    def test_vectors_are_unit_and_in_the_output_space(self):
        out = _measure(_adapter(_random_pairs()), vectors=True, source="cats")
        for it in out["items"]:
            assert len(it["vector"]) == 6
            assert np.linalg.norm(it["vector"]) == pytest.approx(1.0, abs=1e-6)
            assert it["basis"] == {"module": it["coords"]["module"],
                                   "side": "out", "d": 6}
            assert it["coords"]["adapter"] == "cats"
        assert out["source"] == "cats"

    def test_top_k_bounds_the_spectrum_recorded(self):
        out = _measure(_adapter(_random_pairs()), top_k=2)
        assert all(len(it["singular_values"]) == 2 for it in out["items"])

    def test_an_adapter_with_no_bytes_is_refused(self):
        with pytest.raises(ValueError, match="carries no `data`"):
            _measure({"kind": "adapter/lora", "lora": {"rank": 4, "alpha": 8}})

    def test_half_a_delta_is_refused_by_name(self):
        payload = _adapter(_random_pairs(layers=(0,)),
                           drop=("model.layers.0.self_attn.q_proj.lora_b",))
        with pytest.raises(ValueError, match="0.self_attn.q_proj"):
            _measure(payload)


class TestComparingTwoAdapters:
    """The alignment question in weight space: two adapters measured
    into one collection, compared at a module by `geometry/compare`."""

    def _union_of_two(self, *, same: bool, vectors: bool = True):
        pairs = _random_pairs(seed=1, layers=(0,))
        other = pairs if same else _random_pairs(seed=2, layers=(0,))
        one = _measure(_adapter(pairs), vectors=vectors, source="a")
        two = _measure(_adapter(other), vectors=vectors, source="b")
        return PURE_BLOCKS["records/union"]({"a": one, "b": two}, {})

    def test_a_union_of_two_measurements_is_still_a_delta_collection(self):
        union = self._union_of_two(same=False)
        assert union["item_kind"] == "adapter/delta"
        assert len(union["items"]) == 4       # two modules, two adapters

    def test_two_adapters_compare_module_by_module(self):
        out = PURE_BLOCKS["geometry/compare"](
            {"items": self._union_of_two(same=False)},
            {"metric": "cosine", "by": "module", "axis": "adapter"})
        groups = out["items"]
        assert sorted(g["group"] for g in groups) == [
            "module=layers.0.self_attn.q_proj",
            "module=layers.0.self_attn.v_proj"]
        for g in groups:
            # One item per adapter in each module's group, and the
            # matrix is the pair of them.
            assert g["labels"] == ["a", "b"]
            assert g["matrix"][0][0] == pytest.approx(1.0)
            assert abs(g["matrix"][0][1]) <= 1.0

    def test_the_same_adapter_twice_is_perfectly_aligned(self):
        out = PURE_BLOCKS["geometry/compare"](
            {"items": self._union_of_two(same=True)},
            {"metric": "cosine", "by": "module", "axis": "adapter"})
        for g in out["items"]:
            assert g["matrix"][0][1] == pytest.approx(1.0, abs=1e-6)

    def test_comparing_across_modules_is_refused_by_name(self):
        with pytest.raises(ValueError, match="own space"):
            PURE_BLOCKS["geometry/compare"](
                {"items": self._union_of_two(same=False)},
                {"metric": "cosine", "by": None})

    def test_without_vectors_the_metric_says_what_is_missing(self):
        with pytest.raises(ValueError, match="vectors: true"):
            PURE_BLOCKS["geometry/compare"](
                {"items": self._union_of_two(same=False, vectors=False)},
                {"metric": "cosine", "by": "module"})


class TestAZeroDelta:
    """A module training never wrote to: `B` is still the zero it was
    initialised at, so ΔW is exactly zero. The eight adapters on the
    bench all carry twenty of these — the v_proj deltas of a KV-shared
    tail, trained but never reached."""

    def _with_a_zero(self):
        pairs = _random_pairs(seed=4, layers=(0,))
        key = (0, "self_attn", "v_proj")
        a, b = pairs[key]
        pairs[key] = (a, np.zeros_like(b))
        return _adapter(pairs)

    def test_it_measures_as_zero_rather_than_failing(self):
        out = _measure(self._with_a_zero(), vectors=True)
        zero = next(it for it in out["items"] if it["id"] == "layers.0.self_attn.v_proj")
        assert zero["frobenius"] == 0.0
        assert zero["mass_share"] == 0.0
        assert zero["effective_rank"] == 0.0

    def test_it_carries_no_direction(self):
        out = _measure(self._with_a_zero(), vectors=True)
        by_id = {it["id"]: it for it in out["items"]}
        assert "vector" not in by_id["layers.0.self_attn.v_proj"]
        assert "vector" in by_id["layers.0.self_attn.q_proj"]

    def test_comparing_it_refuses_by_name_instead_of_comparing_nothing(self):
        one = _measure(self._with_a_zero(), vectors=True, source="a")
        two = _measure(self._with_a_zero(), vectors=True, source="b")
        union = PURE_BLOCKS["records/union"]({"a": one, "b": two}, {})
        with pytest.raises(ValueError, match="carry no vector"):
            PURE_BLOCKS["geometry/compare"](
                {"items": union}, {"metric": "cosine", "by": "module"})
