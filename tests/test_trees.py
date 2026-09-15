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
    """A collection of `geometry/similarity` as `geometry/similarity` emits one."""
    from mechbench_compute import geometry
    from mechbench_compute.lexicon import kinds as K

    unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    return K.collection("geometry/similarity", [{
        "layer": 0, "ids": [f"i{i}" for i in range(len(points))],
        "labels": [None] * len(points),
        "matrix": geometry.cosine_matrix(unit.astype(np.float32)).tolist()}],
        metric="cosine", metric_kind="similarity", symmetric=True, options={})


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
        out = PURE_BLOCKS["geometry/span"](
            {"similarity": similarity_of(corpus(kind))}, params)
        return out["items"][0]

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
        cv1 = PURE_BLOCKS["geometry/span"](
            {"similarity": base}, {})["items"][0]["cv"]
        scaled = {**base, "items": [{**base["items"][0], "matrix": [
            [1 - (1 - v) * 0.5 for v in row]
            for row in base["items"][0]["matrix"]]}]}
        cv2 = PURE_BLOCKS["geometry/span"](
            {"similarity": scaled}, {})["items"][0]["cv"]
        assert cv1 == pytest.approx(cv2, abs=0.02)


class TestTheBlock:
    def test_it_follows_a_similarity_over_vectors(self):
        rows = [{"id": f"r{i}", "layer": 3, "head": None, "label": None,
                 "vector": v.tolist()} for i, v in enumerate(corpus("clustered"))]
        sim = PURE_BLOCKS["geometry/compare"](
            {"items": {"kind": "residual_vectors", "rows": rows}}, {})
        out = PURE_BLOCKS["geometry/span"]({"similarity": sim}, {})
        assert out["items"][0]["layer"] == 3 and out["items"][0]["group"] == "layer=3"
        assert out["items"][0]["n"] == len(rows)
        assert out["metric"] == "cosine" and out["over"] == "activations/vector"

    def test_a_table_reads_the_items_directly(self):
        # One item per group, and `records/tabulate` reads them as rows: the
        # flat duplicate the old shape carried is gone.
        out = PURE_BLOCKS["geometry/span"](
            {"similarity": similarity_of(corpus("even"))}, {})
        item = out["items"][0]
        assert {"layer", "n", "mean", "variance", "cv", "bridges"} <= set(item)
        table = PURE_BLOCKS["records/tabulate"]({"records": out}, {})
        assert table["kind"] == "records/table"
        assert [r["n"] for r in table["rows"]] == [item["n"]]

    def test_a_wrong_input_says_what_it_wanted(self):
        with pytest.raises(ValueError, match="geometry/similarity"):
            PURE_BLOCKS["geometry/span"]({"similarity": [1, 2]}, {})


class TestCentering:
    """Anisotropy: transformer vectors sit in a narrow cone, so raw
    cosine mostly measures the cone. Centering removes it (000431
    follow-up, found when 024's pooled re-run disagreed with itself)."""

    def _cone(self, n=40, d=16, spread=0.05, seed=0):
        """Vectors with a large shared component and small differences —
        the shape real residuals have."""
        rng = np.random.default_rng(seed)
        common = np.ones(d, dtype=np.float32) * 10.0
        return common + rng.normal(0, spread, size=(n, d)).astype(np.float32)

    def _mean_edge(self, rows, **options):
        sim = PURE_BLOCKS["geometry/compare"](
            {"items": {"kind": "residual_vectors", "rows": rows}},
            {"metric": "cosine", "options": options})
        out = trees.mst({"similarity": sim}, {})
        return out["items"][0]["mean"], out

    def _rows(self, V):
        return [{"id": f"r{i}", "layer": 23, "vector": v.tolist()}
                for i, v in enumerate(V)]

    def test_centering_expands_a_cone(self):
        rows = self._rows(self._cone())
        raw, _ = self._mean_edge(rows)
        centered, _ = self._mean_edge(rows, center=True)
        # Not a small correction: the raw distances are almost entirely
        # the shared direction.
        assert centered > raw * 10

    def test_the_record_says_it_was_centered(self):
        _, out = self._mean_edge(self._rows(self._cone()), center=True)
        assert out["metric"] == "cosine" and out["options"] == {"center": True}

    def test_uncentered_is_unchanged(self):
        _, out = self._mean_edge(self._rows(self._cone()))
        assert out["metric"] == "cosine" and out["options"] == {"center": False}

    def test_centering_reproduces_the_direct_path_bit_for_bit(self):
        # The variety numbers published from the vectors-straight-to-mst
        # path (float32 cosine over centred rows, 1 − s) must come back
        # exactly from the similarity-then-mst path.
        from mechbench_compute import geometry

        V = self._cone()
        rows = self._rows(V)
        centred = trees.center_rows(np.array(V, dtype=np.float32))
        direct = trees._distance_from_similarity(geometry.cosine_matrix(centred))
        edges = trees.minimum_spanning_tree(direct)
        want = trees.tree_stats(edges)
        _, out = self._mean_edge(rows, center=True)
        got = {k: out["items"][0][k] for k in want}
        assert got == want


class TestParamChecking:
    """000438: a block must refuse a param it cannot honour."""

    def test_an_unknown_param_is_refused_by_name(self):
        from mechbench_compute.block_params import check_params
        with pytest.raises(ValueError, match="does not accept 'centre'"):
            check_params("geometry/span",
                         {"bridge_sigma": 2.0, "centre": True})

    def test_the_message_points_at_the_runner(self):
        from mechbench_compute.block_params import check_params
        with pytest.raises(ValueError, match="predates the parameter"):
            check_params("geometry/span", {"center_rows": True})

    def test_accepted_params_pass(self):
        from mechbench_compute.block_params import check_params
        check_params("geometry/span",
                     {"bridge_sigma": 2.0, "name": "v", "keep_edges": False})

    def test_a_port_given_as_a_param_is_refused_with_directions(self):
        from mechbench_compute.block_params import check_params
        with pytest.raises(ValueError, match="input port"):
            check_params("geometry/span", {"similarity": {}})

    def test_an_unregistered_block_is_unchecked(self):
        # Every CANONICAL op is declared now (000478), so the unchecked
        # case is a block this runner does not know — an extension's op
        # (000410), which is the api's business and not ours. This test
        # used to name `text/measure`, which was merely undeclared.
        from mechbench_compute.block_params import check_params
        check_params("~someone/ops/custom/1", {"anything": 1})

    def test_pooling_params_are_accepted_on_residual_vectors(self):
        from mechbench_compute.block_params import check_params
        check_params("activations/capture",
                     {"layers": [23], "pool": {"reduce": "mean", "over": {"after": 1}},
                      "skip_empty": True})
