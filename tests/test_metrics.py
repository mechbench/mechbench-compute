"""Metrics on kinds: a kind declares how its items compare, and
`geometry/compare` + `geometry/span` stand over anything that does."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mechbench_compute import metrics as M
from mechbench_compute import shapes as S
from mechbench_compute.blocks import PURE_BLOCKS
from mechbench_compute.lexicon import kinds as K

SP = S.space(model="fake/m", layer=12, point="resid_post", d=3)


def vec(id_, v, **coords):
    return S.vector(v, SP, id=id_, coords=coords)


def direction(id_, v):
    from mechbench_compute import directions as d
    out = d.make(v, SP, method="test")
    out["id"] = id_
    return out


def read(id_, masses: dict[str, float], **coords):
    """A decision read carrying `tracked` masses (ids by text hash)."""
    tracked = {name: {"token": {"id": abs(hash(name)) % 1000, "text": name}, "p": p,
                      "logp": math.log(p)} for name, p in masses.items()}
    top = sorted(tracked.values(), key=lambda t: -t["p"])
    return {"id": id_, "coords": coords, "entropy_bits": 1.0, "top": top, "tracked": tracked}


class TestDeclarations:
    def test_subtypes_inherit_their_ancestors_metrics(self):
        assert "cosine" in M.declared("direction/vector")
        assert "cosine" in M.declared("trajectory/point")
        assert "jensen-shannon" in M.declared("logits/decision")
        assert "hamming" in M.declared("text/document")
        assert M.default_metric("activations/vector") == "cosine"
        assert M.default_metric("logits/funnel") == "jensen-shannon"
        assert M.default_metric("records/condition") == "hamming"

    def test_an_unknown_metric_is_refused_with_the_kinds_own(self):
        with pytest.raises(ValueError, match="compares by"):
            M.resolve("activations/vector", "jensen-shannon")
        with pytest.raises(ValueError, match="takes no option"):
            M.options_of(M.resolve("activations/vector", "cosine")[1], {"centre": True})

    def test_every_declared_metric_is_implemented_and_symmetric_as_declared(self):
        for kind in K.KINDS:
            for m in kind.metrics:
                assert (kind.name, m.name) in M.IMPLEMENTATIONS, f"{kind.name}.{m.name} not implemented"
        items = [vec("a", [1, 0, 0]), vec("b", [0.5, 0.5, 0]), vec("c", [0, 0, 1])]
        for name in ("cosine", "euclidean", "dot"):
            m, metric, _ = M.matrix("activations/vector", items, name)
            assert np.allclose(m, m.T) == metric.symmetric
        reads = [read("a", {"x": 0.7, "y": 0.3}), read("b", {"x": 0.5, "y": 0.4})]
        m, metric, _ = M.matrix("logits/distribution", reads, "kl")
        assert not metric.symmetric and m[0, 1] != m[1, 0]


class TestVectorMetrics:
    def test_cosine_euclidean_dot(self):
        items = [vec("a", [1, 0, 0]), vec("b", [0, 1, 0]), vec("c", [2, 0, 0])]
        cos, _, _ = M.matrix("activations/vector", items, "cosine")
        assert cos[0, 2] == pytest.approx(1.0) and cos[0, 1] == pytest.approx(0.0)
        euc, _, _ = M.matrix("activations/vector", items, "euclidean")
        assert euc[0, 1] == pytest.approx(math.sqrt(2)) and euc[0, 2] == pytest.approx(1.0)
        dot, _, _ = M.matrix("activations/vector", items, "dot")
        assert dot[0, 2] == pytest.approx(2.0)

    def test_center_is_an_option_recorded_with_the_result(self):
        items = [vec("a", [10, 0, 0.1]), vec("b", [10, 0.1, 0]), vec("c", [10, -0.1, 0])]
        raw, _, opts = M.matrix("activations/vector", items, "cosine")
        assert opts == {"center": False} and raw[0, 1] > 0.99
        centred, _, opts = M.matrix("activations/vector", items, "cosine", {"center": True})
        assert opts == {"center": True} and centred[1, 2] < 0

    def test_two_spaces_are_refused_with_both_named(self):
        other = S.space(model="fake/m", layer=13, point="resid_post", d=3)
        items = [vec("a", [1, 0, 0]), S.vector([0, 1, 0], other, id="b")]
        with pytest.raises(ValueError, match="different spaces.*layer 12.*13"):
            M.matrix("activations/vector", items, "cosine")


class TestDistributionMetrics:
    def test_identical_reads_are_at_distance_zero_and_opposites_far(self):
        same = [read("a", {"x": 0.6, "y": 0.4}), read("b", {"x": 0.6, "y": 0.4})]
        for name in ("jensen-shannon", "hellinger", "total-variation"):
            m, _, _ = M.matrix("logits/distribution", same, name)
            assert m[0, 1] == pytest.approx(0.0, abs=1e-9)
        far = [read("a", {"x": 1.0}), read("b", {"y": 1.0})]
        js, _, _ = M.matrix("logits/distribution", far, "jensen-shannon")
        tv, _, _ = M.matrix("logits/distribution", far, "total-variation")
        assert js[0, 1] == pytest.approx(1.0) and tv[0, 1] == pytest.approx(1.0)

    def test_the_unnamed_mass_is_one_bucket(self):
        # A read that names half its mass and one that names it all: the
        # rest is compared as one bucket, not ignored.
        a = read("a", {"x": 0.5})
        b = read("b", {"x": 0.5, "y": 0.5})
        tv, _, _ = M.matrix("logits/distribution", [a, b], "total-variation")
        assert tv[0, 1] == pytest.approx(0.5)

    def test_the_retired_read_spelling_compares_too(self):
        old = [{"id": "a", "entropy_bits": 1.0, "outcome_mass": {"x": 0.7, "y": 0.3}},
               {"id": "b", "entropy_bits": 1.0, "outcome_mass": {"x": 0.3, "y": 0.7}}]
        tv, _, _ = M.matrix("logits/distribution", old, "total-variation")
        assert tv[0, 1] == pytest.approx(0.4)


class TestRecordMetric:
    def test_hamming_counts_the_axes_that_differ(self):
        recs = [{"id": "a", "coords": {"g": "noir", "n": 1}},
                {"id": "b", "coords": {"g": "noir", "n": 2}},
                {"id": "c", "coords": {"g": "fable"}}]
        m, _, _ = M.matrix("records/record", recs, "hamming")
        assert m[0, 1] == 1 and m[0, 2] == 2 and m[1, 2] == 2


class TestTheOps:
    def test_a_balanced_design_under_hamming_is_a_flat_lattice(self):
        # A 2×3 design under hamming: same-genre pairs differ on one axis,
        # cross-genre pairs on one or two — so the genre split shows in
        # the separation, while the tree has no bridges, every step being
        # one axis. A sanity test that the record metric reads the design.
        design = PURE_BLOCKS["records/cross"]({}, {"factors": [
            {"name": "genre", "levels": [{"key": "noir"}, {"key": "fable"}]},
            {"name": "seed", "levels": [{"key": "1"}, {"key": "2"}, {"key": "3"}]}]})
        sim = PURE_BLOCKS["geometry/compare"]({"items": design}, {"axis": "genre"})
        assert sim["metric"] == "hamming" and sim["metric_kind"] == "distance"
        item = sim["items"][0]
        assert item["group"] == "all" and len(item["ids"]) == 6
        assert item["separation"]["intra"] == 1.0 and item["separation"]["inter"] > 1.0
        tree = PURE_BLOCKS["geometry/span"]({"similarity": sim}, {"bridge_sigma": 0.5})
        t = tree["items"][0]
        assert tree["metric"] == "hamming" and tree["over"] == "records/record"
        assert t["n"] == 6 and t["max"] == 1.0
        assert t["bridges"] == 0 and t["components_after_cut"] == 1

    def test_eight_axes_align_in_one_node(self):
        # The 018 question: are the adapters' pressure axes the same axis?
        # Six single-domain axes near one direction, two joint axes off
        # it — a union of the directions, one similarity, one tree.
        rng = np.random.default_rng(0)
        base = np.array([1.0, 0.2, 0.0])
        ports = {}
        for i, name in enumerate(["die", "letters", "n1000", "digit6", "animals_u", "animals_s"]):
            ports[name] = direction(None, list(base + rng.normal(0, 0.02, 3)))
        ports["joint4"] = direction(None, [0.0, 0.0, 1.0])
        ports["joint4_s1000"] = direction(None, [0.0, 0.1, 1.0])
        union = PURE_BLOCKS["records/union"](ports, {})
        assert union["item_kind"] == "activations/vector"
        assert sorted(it["id"] for it in union["items"]) == sorted(ports)
        sim = PURE_BLOCKS["geometry/compare"]({"items": union}, {"axis": "batch"})
        item = sim["items"][0]
        assert item["group"] == "layer=12" and len(item["ids"]) == 8
        by = {(p["a"], p["b"]): p["value"] for p in item["pairs"]}
        assert by[("die", "letters")] > 0.99
        assert by[("die", "joint4")] < 0.1
        assert item["pairs"][0]["value"] >= item["pairs"][-1]["value"]
        tree = PURE_BLOCKS["geometry/span"]({"similarity": sim}, {})["items"][0]
        # Six near-identical axes and two strangers: the longest edges
        # are the bridges to the joints.
        assert tree["n"] == 8 and tree["bridges"] >= 1

    def test_a_grid_of_decision_reads_builds_a_tree_of_conditions(self):
        # The 022 matrix: reads under different conditions, compared by
        # what the model says. Two prompt families, two outcome profiles.
        reads = [read(f"a{i}", {"x": 0.8 - i * 0.02, "y": 0.2 + i * 0.02}, family="a") for i in range(4)]
        reads += [read(f"b{i}", {"x": 0.2 + i * 0.02, "y": 0.8 - i * 0.02}, family="b") for i in range(4)]
        coll = K.collection("logits/decision", reads, top_k=2)
        sim = PURE_BLOCKS["geometry/compare"]({"items": coll}, {"axis": "family", "by": None})
        assert sim["metric"] == "jensen-shannon" and sim["over"] == "logits/decision"
        item = sim["items"][0]
        assert item["nn_purity"] == 1.0 and item["separation"]["gap"] > 0
        tree = PURE_BLOCKS["geometry/span"]({"similarity": sim}, {"bridge_sigma": 1.0})["items"][0]
        assert tree["components_after_cut"] == 2

    def test_a_kl_tree_is_refused_naming_the_asymmetry(self):
        reads = [read("a", {"x": 0.7, "y": 0.3}), read("b", {"x": 0.3, "y": 0.7})]
        sim = PURE_BLOCKS["geometry/compare"](
            {"items": K.collection("logits/decision", reads)}, {"metric": "kl", "by": None})
        assert sim["symmetric"] is False
        with pytest.raises(ValueError, match="symmetric"):
            PURE_BLOCKS["geometry/span"]({"similarity": sim}, {})

    def test_grouping_by_a_coordinate(self):
        # A funnel's items are one read per (record, layer): compare the
        # records within each layer.
        items = []
        for layer in (3, 7):
            for i in range(3):
                r = read(f"r{i}", {"x": 0.5 + 0.1 * i, "y": 0.5 - 0.1 * i})
                r["layer"] = layer
                items.append(r)
        coll = K.collection("logits/funnel", items)
        sim = PURE_BLOCKS["geometry/compare"]({"items": coll}, {"by": "layer"})
        assert [it["group"] for it in sim["items"]] == ["layer=3", "layer=7"]
        assert all(len(it["ids"]) == 3 for it in sim["items"])

    def test_the_retired_direction_similarity_is_refused_by_name(self):
        from mechbench_compute import lexicon
        from mechbench_compute.block_params import check_inputs

        # The refusal names what to write instead.
        with pytest.raises(KeyError):
            lexicon.resolve("direction/similarity")
        assert "geometry/compare" in lexicon.explain_unknown("direction/similarity")
        with pytest.raises(ValueError, match="no input port 'a'.*items"):
            check_inputs("geometry/compare", {"a": direction("a", [1, 0, 0])})
