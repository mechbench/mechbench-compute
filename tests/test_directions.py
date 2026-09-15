"""Directions as first-class objects (task 000367)."""

from __future__ import annotations

import numpy as np
import pytest

from mechbench_compute import directions as d
from mechbench_compute import shapes as S


def _space(layer, d_=3, point="resid_post"):
    return S.space(model="fake/m@r", layer=layer, point=point, d=d_)


def _vectors(layer=3, point="post", n=4, dim=8, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    rows = []
    for i in range(n):
        rows.append({"id": f"p{i}", "label": "pos", "layer": layer,
                     "vector": [float(x) for x in base + rng.normal(scale=0.1, size=dim) + 2.0]})
        rows.append({"id": f"n{i}", "label": "neg", "layer": layer,
                     "vector": [float(x) for x in base + rng.normal(scale=0.1, size=dim) - 2.0]})
    return {"kind": "residual_vectors", "point": point, "model": "fake/m@r", "rows": rows}


def _cos(a, b):
    """The cosine of two directions, through the metric their kind declares."""
    from mechbench_compute import metrics as M

    return float(M.matrix("activations/vector", [a, b], "cosine")[0][0, 1])


class TestMake:
    def test_unit_and_provenance(self):
        x = d.make([3.0, 4.0], _space(2, 2), method="test", sources=["a"])
        assert x["kind"] == "direction/vector" and x["space"]["d"] == 2
        assert x["space"] == {"model": "fake/m@r", "layer": 2, "point": "resid_post", "head": None, "d": 2}
        assert abs(np.linalg.norm(x["vector"]) - 1.0) < 1e-6 and x["norm"] == 5.0
        assert x["derivation"]["method"] == "test" and x["derivation"]["sources"] == ["a"]

    def test_zero_refused(self):
        with pytest.raises(ValueError):
            d.make([0.0, 0.0], _space(0, 2), method="t")


class TestProducers:
    def test_diff_of_means_points_from_neg_to_pos(self):
        v = _vectors()
        x = d.from_vectors(v, layer=3, positive="pos", negative="neg")
        # The fixture is the older flattened spelling: `layer` on the row,
        # `point` and `model` on the header, `label` as a field. The space
        # is assembled from it and the label read as the `label` axis.
        assert x["space"] == {"model": "fake/m@r", "layer": 3, "point": "resid_post", "head": None, "d": 8}
        assert x["derivation"]["axis"] == "label"
        assert (x["derivation"]["positive"], x["derivation"]["negative"]) == ("pos", "neg")
        assert x["derivation"]["n_positive"] == 4
        # the difference is ~ +4 on every coordinate: all-positive unit vector
        assert all(c > 0 for c in x["vector"])

    def test_pca_first_component_is_the_label_axis(self):
        v = _vectors()
        x = d.from_pca(v, layer=3)
        m = d.from_vectors(v, layer=3, positive="pos", negative="neg")
        assert abs(_cos(x, m)) > 0.95
        assert 0.5 < x["derivation"]["explained"] <= 1.0

    def test_missing_layer_refused(self):
        with pytest.raises(ValueError):
            d.from_vectors(_vectors(), layer=9, positive="pos", negative="neg")

    def test_an_axis_fit_across_two_models_lives_in_neither(self):
        # A base capture and an adapted capture in one union: the
        # difference of their means is a direction in the basis they
        # share, not in either model — so two such axes (from two
        # adapters) compare, and the models are on the derivation.
        from mechbench_compute.blocks import PURE_BLOCKS
        from mechbench_compute.lexicon import kinds as K

        def capture(model, seed):
            sp = S.space(model=model, layer=12, point="resid_post", d=4)
            rng = np.random.default_rng(seed)
            return K.collection("activations/vector",
                                [S.vector(list(rng.normal(size=4)), sp, id="p")], model=model)

        base = capture("fake/base", 0)
        axes = {}
        for i, name in enumerate(["die", "letters"]):
            pair = PURE_BLOCKS["records/union"]({"base": base, "adapted": capture(f"fake/{name}", i + 1)}, {})
            axes[name] = d.from_vectors(pair, layer=12, axis="batch", positive="base", negative="adapted")
            assert axes[name]["space"]["model"] is None
            assert axes[name]["derivation"]["models"] == [f"fake/{name}", "fake/base"]
        union = PURE_BLOCKS["records/union"](axes, {})
        sim = PURE_BLOCKS["geometry/compare"]({"items": union}, {"axis": "batch"})
        assert sim["items"][0]["ids"] == ["die", "letters"]
        assert -1.0 <= sim["items"][0]["matrix"][0][1] <= 1.0


class TestArithmetic:
    def test_orthogonalize_removes_the_component(self):
        a = d.make([1.0, 0.0, 0.0], _space(1), method="t")
        b = d.make([1.0, 1.0, 0.0], _space(1), method="t")
        o = d.orthogonalize(b, [a])
        assert abs(_cos(o, a)) < 1e-6
        assert o["derivation"]["method"] == "orthogonalize"

    def test_add_and_average(self):
        a = d.make([1.0, 0.0], _space(1, 2), method="t")
        b = d.make([0.0, 1.0], _space(1, 2), method="t")
        s = d.add([a, b], [1.0, 1.0])
        assert np.allclose(s["vector"], [np.sqrt(0.5)] * 2, atol=1e-6)
        assert d.average([a, b])["derivation"]["method"] == "average"

    def test_spaces_must_match(self):
        a = d.make([1.0, 0.0], _space(1, 2), method="t")
        b = d.make([1.0, 0.0], _space(2, 2), method="t")
        with pytest.raises(ValueError, match="different spaces"):
            d.add([a, b])
        c = d.make([1.0, 0.0], S.space(model="other/m", layer=1, point="resid_post", d=2), method="t")
        with pytest.raises(ValueError, match="model"):
            d.add([a, c])

    def test_project_rows(self):
        v = _vectors()
        m = d.from_vectors(v, layer=3, positive="pos", negative="neg")
        pr = d.project_rows(v, m)
        assert pr["item_kind"] == "activations/coordinate"
        pos = [r["coord"] for r in pr["items"] if r["coords"]["label"] == "pos"]
        neg = [r["coord"] for r in pr["items"] if r["coords"]["label"] == "neg"]
        assert min(pos) > max(neg)
        assert pr["items"][0]["direction"]["method"] == "diff_of_means"
        assert pr["items"][0]["space"]["layer"] == 3


class TestBlocks:
    def test_registered_and_callable(self):
        from mechbench_compute.blocks import PURE_BLOCKS

        v = _vectors()
        fn = PURE_BLOCKS["direction/fit"]
        x = fn({"vectors": v}, {"layer": 3, "positive": "pos", "negative": "neg"})
        assert x["kind"] == "direction/vector"
        pair = PURE_BLOCKS["records/union"]({"a": x, "b": dict(x)}, {})
        sim = PURE_BLOCKS["geometry/compare"]({"items": pair}, {})
        assert abs(sim["items"][0]["matrix"][0][1] - 1.0) < 1e-6
        avg = PURE_BLOCKS["direction/average"]({"d1": x, "d2": x}, {})
        assert avg["derivation"]["method"] == "average"

    def test_levels_declared(self):
        from mechbench_compute import resume as rm

        assert rm.resume_level("direction/add") == "reproducible"
        assert rm.resume_level("intervene/apply") == "reproducible"
        assert rm.item_resumable("intervene/apply")
