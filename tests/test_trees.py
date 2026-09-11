"""MST-based variety (task 000430).

The measure has to tell three corpus shapes apart: collapsed (every
item the same), clustered (genre attractors with bridges between them),
and evenly varied. Variance alone cannot — collapse and evenness both
give low variance — so the tests pin the JOINT behaviour of mean and
variance, which is the whole point of the instrument.
"""

from __future__ import annotations

import numpy as np
import pytest

from mechbench_compute import trees
from mechbench_compute.blocks import PURE_BLOCKS


def similarity_of(points: np.ndarray) -> dict:
    """A `similarity_matrix` object as `vectors/similarity` emits one."""
    from mechbench_compute import geometry

    unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    return {"kind": "similarity_matrix", "layers": [{
        "layer": 0, "ids": [f"i{i}" for i in range(len(points))],
        "labels": [None] * len(points),
        "matrix": geometry.cosine_matrix(unit.astype(np.float32)).tolist()}]}


def corpus(kind: str, n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "collapsed":                      # one story, told n ways
        return np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.01, (n, 3))
    if kind == "clustered":                      # three genres
        centres = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        return np.concatenate([c + rng.normal(0, 0.03, (n // 3, 3))
                               for c in centres])
    return rng.normal(0, 1.0, (n, 3))            # evenly varied


class TestTheTreeItself:
    def test_prims_finds_the_known_tree(self):
        # A path graph: 0-1-2-3 at distance 1, everything else far.
        d = np.array([[0, 1, 9, 9], [1, 0, 1, 9], [9, 1, 0, 1], [9, 9, 1, 0]],
                     dtype=float)
        edges = trees.minimum_spanning_tree(d)
        assert [(i, j) for i, j, _ in edges] == [(0, 1), (1, 2), (2, 3)]
        assert sum(w for _, _, w in edges) == 3.0

    def test_ties_break_toward_the_lower_index_so_the_tree_reproduces(self):
        d = np.ones((5, 5)) - np.eye(5)          # every distance identical
        once = trees.minimum_spanning_tree(d)
        twice = trees.minimum_spanning_tree(d)
        assert once == twice
        assert [j for _, j, _ in once] == [1, 2, 3, 4]

    def test_a_single_item_has_no_tree(self):
        assert trees.minimum_spanning_tree(np.zeros((1, 1))) == []
        assert trees.tree_stats([]) == {"n_edges": 0}


class TestTheMeasure:
    """The three shapes, and why variance is never reported alone."""

    def stats(self, kind, **params):
        out = PURE_BLOCKS["~canonical/ops/vectors/mst/1"](
            {"matrix": similarity_of(corpus(kind))}, params)
        return out["layers"][0]

    def test_collapse_and_evenness_both_have_low_variance(self):
        collapsed = self.stats("collapsed")
        even = self.stats("even")
        assert collapsed["variance"] < 0.01
        assert even["variance"] < 0.5
        # …and this is exactly why variance alone is not the measure:
        # only the MEAN tells these two apart.
        assert collapsed["mean"] < even["mean"] / 5

    def test_clusters_show_up_as_variance_and_bridges(self):
        clustered = self.stats("clustered")
        even = self.stats("even")
        assert clustered["cv"] > even["cv"]
        # Three genres, two bridges between them — a cluster count
        # nobody had to choose a k for.
        assert clustered["components_after_cut"] == 3

    def test_the_scale_free_number_survives_a_rescale(self):
        # cv is stdev/mean, so it does not move when every distance is
        # multiplied — corpora embedded at different layers stay
        # comparable.
        base = similarity_of(corpus("clustered"))
        cv1 = PURE_BLOCKS["~canonical/ops/vectors/mst/1"](
            {"matrix": base}, {})["layers"][0]["cv"]
        scaled = {**base, "layers": [{**base["layers"][0], "matrix": [
            [1 - (1 - v) * 0.5 for v in row]
            for row in base["layers"][0]["matrix"]]}]}
        cv2 = PURE_BLOCKS["~canonical/ops/vectors/mst/1"](
            {"matrix": scaled}, {})["layers"][0]["cv"]
        assert cv1 == pytest.approx(cv2, abs=0.02)


class TestTheBlock:
    def test_it_takes_vectors_directly_too(self):
        rows = [{"id": f"r{i}", "layer": 3, "head": None, "label": None,
                 "vector": v.tolist()} for i, v in enumerate(corpus("clustered"))]
        out = PURE_BLOCKS["~canonical/ops/vectors/mst/1"](
            {"vectors": {"kind": "residual_vectors", "rows": rows}}, {})
        assert out["layers"][0]["layer"] == 3
        assert out["layers"][0]["n"] == len(rows)

    def test_the_rows_view_renders_without_a_custom_renderer(self):
        out = PURE_BLOCKS["~canonical/ops/vectors/mst/1"](
            {"matrix": similarity_of(corpus("even"))}, {})
        row = out["rows"][0]
        assert {"layer", "n", "mean", "variance", "cv", "bridges"} <= set(row)
        assert "edges" not in row and "ids" not in row

    def test_a_wrong_input_says_what_it_wanted(self):
        with pytest.raises(ValueError, match="similarity_matrix"):
            PURE_BLOCKS["~canonical/ops/vectors/mst/1"]({"matrix": [1, 2]}, {})
