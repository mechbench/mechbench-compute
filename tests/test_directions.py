"""Directions as first-class objects (task 000367)."""

from __future__ import annotations

import numpy as np
import pytest

from mechbench_compute import directions as d


def _vectors(layer=3, point="post", n=4, dim=8, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    rows = []
    for i in range(n):
        rows.append({"condition": f"p{i}", "label": "pos", "layer": layer,
                     "vector": [float(x) for x in base + rng.normal(scale=0.1, size=dim) + 2.0]})
        rows.append({"condition": f"n{i}", "label": "neg", "layer": layer,
                     "vector": [float(x) for x in base + rng.normal(scale=0.1, size=dim) - 2.0]})
    return {"kind": "residual_vectors", "point": point, "model": "fake/m@r", "rows": rows}


class TestSimilarityMatrix:
    """Many directions at once (experiment 018's axis geometry): the
    pairwise matrix named by port, norms riding along."""

    def _dir(self, vec, norm_scale=1.0):
        return d.make([x * norm_scale for x in vec], layer=12, point="resid_post",
                      method="t")

    def test_pairwise_cosines_named_by_port(self):
        out = d.block_similarity(
            {"die": self._dir([1, 0, 0], 2.0), "n1000": self._dir([1, 0, 0], 3.0),
             "joint": self._dir([0, 1, 0])}, {})
        assert out["kind"] == "direction_similarity_matrix"
        assert out["names"] == ["die", "joint", "n1000"]  # sorted ports
        i, j = out["names"].index("die"), out["names"].index("n1000")
        assert out["cosines"][i][j] == 1.0
        assert out["cosines"][i][out["names"].index("joint")] == 0.0
        assert out["norms"] == {"die": 2.0, "joint": 1.0, "n1000": 3.0}
        assert out["pairs"][0] == {"a": "die", "b": "n1000", "cosine": 1.0}

    def test_two_named_ports_still_give_one_cosine(self):
        out = d.block_similarity({"a": self._dir([1, 0, 0]),
                                  "b": self._dir([0, 1, 0])}, {})
        assert out["kind"] == "direction_similarity" and out["cosine"] == 0.0

    def test_one_direction_is_refused(self):
        with pytest.raises(ValueError, match="at least two"):
            d.block_similarity({"only": self._dir([1, 0, 0])}, {})


class TestMake:
    def test_unit_and_provenance(self):
        x = d.make([3.0, 4.0], layer=2, point="resid_post", method="test", sources=["a"])
        assert x["kind"] == "direction" and x["d"] == 2
        assert abs(np.linalg.norm(x["vector"]) - 1.0) < 1e-6 and x["norm"] == 5.0
        assert x["derivation"]["method"] == "test" and x["derivation"]["sources"] == ["a"]

    def test_zero_refused(self):
        with pytest.raises(ValueError):
            d.make([0.0, 0.0], layer=0, point="resid_post", method="t")


class TestProducers:
    def test_diff_of_means_points_from_neg_to_pos(self):
        v = _vectors()
        x = d.from_vectors(v, layer=3, positive="pos", negative="neg")
        assert x["layer"] == 3 and x["point"] == "resid_post"
        assert x["derivation"]["labels"] == {"positive": "pos", "negative": "neg"}
        assert x["derivation"]["n_positive"] == 4
        # the difference is ~ +4 on every coordinate: all-positive unit vector
        assert all(c > 0 for c in x["vector"])

    def test_pca_first_component_is_the_label_axis(self):
        v = _vectors()
        x = d.from_pca(v, layer=3)
        m = d.from_vectors(v, layer=3, positive="pos", negative="neg")
        assert abs(d.similarity(x, m)["cosine"]) > 0.95
        assert 0.5 < x["derivation"]["explained"] <= 1.0

    def test_missing_layer_refused(self):
        with pytest.raises(ValueError):
            d.from_vectors(_vectors(), layer=9, positive="pos", negative="neg")


class TestArithmetic:
    def test_orthogonalize_removes_the_component(self):
        a = d.make([1.0, 0.0, 0.0], layer=1, point="resid_post", method="t")
        b = d.make([1.0, 1.0, 0.0], layer=1, point="resid_post", method="t")
        o = d.orthogonalize(b, [a])
        assert abs(d.similarity(o, a)["cosine"]) < 1e-6
        assert o["derivation"]["method"] == "orthogonalize"

    def test_add_and_average(self):
        a = d.make([1.0, 0.0], layer=1, point="resid_post", method="t")
        b = d.make([0.0, 1.0], layer=1, point="resid_post", method="t")
        s = d.add([a, b], [1.0, 1.0])
        assert np.allclose(s["vector"], [np.sqrt(0.5)] * 2, atol=1e-6)
        assert d.average([a, b])["derivation"]["method"] == "average"

    def test_spaces_must_match(self):
        a = d.make([1.0, 0.0], layer=1, point="resid_post", method="t")
        b = d.make([1.0, 0.0], layer=2, point="resid_post", method="t")
        with pytest.raises(ValueError):
            d.add([a, b])

    def test_project_rows(self):
        v = _vectors()
        m = d.from_vectors(v, layer=3, positive="pos", negative="neg")
        pr = d.project_rows(v, m)
        pos = [r["projection"] for r in pr["rows"] if r["label"] == "pos"]
        neg = [r["projection"] for r in pr["rows"] if r["label"] == "neg"]
        assert min(pos) > max(neg)


class TestBlocks:
    def test_registered_and_callable(self):
        from mechbench_compute.blocks import PURE_BLOCKS

        v = _vectors()
        fn = PURE_BLOCKS["~canonical/ops/direction/from-vectors/1"]
        x = fn({"vectors": v}, {"layer": 3, "positive": "pos", "negative": "neg"})
        assert x["kind"] == "direction"
        sim = PURE_BLOCKS["~canonical/ops/direction/similarity/1"]({"a": x, "b": x}, {})
        assert abs(sim["cosine"] - 1.0) < 1e-6
        avg = PURE_BLOCKS["~canonical/ops/direction/average/1"]({"d1": x, "d2": x}, {})
        assert avg["derivation"]["method"] == "average"

    def test_levels_declared(self):
        from mechbench_compute import resume as rm

        assert rm.resume_level("~canonical/ops/direction/add/1") == "reproducible"
        assert rm.resume_level("~canonical/ops/intervene/1") == "reproducible"
        assert rm.item_resumable("~canonical/ops/intervene/1")
