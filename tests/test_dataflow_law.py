"""The map/reduce isomorphism law (tasks 000406 / 000407) and leaf
identity (000402)."""

from __future__ import annotations

import hashlib
import random

import pytest

from mechbench_compute import isomorphism as iso
from mechbench_compute import reduce as rd
from mechbench_compute import seeds
from mechbench_compute.blocks import PURE_BLOCKS


def _leaves(n=60, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append({"id": f"r{i}", "coords": {"g": rng.choice(["a", "b", "c"])},
                    "delta": rng.uniform(-3, 3) * (1 + i % 7) / 3, "score": rng.random()})
    return out


class TestMonoidLaws:
    @pytest.mark.parametrize("block,params", [
        ("~canonical/ops/group-stats/1", {"by": ["g"], "value": "delta"}),
        ("~canonical/ops/reduce/sum/1", {"value": "delta"}),
        ("~canonical/ops/reduce/top-k/1", {"value": "score", "k": 5}),
        ("~canonical/ops/reduce/histogram/1", {"value": "delta", "lo": -5, "hi": 5, "bins": 10}),
    ])
    def test_identity_and_associativity(self, block, params):
        m = rd.monoid_for(block, params)
        leaves = _leaves()
        a, b, c = m.partial(leaves[:20], params), m.partial(leaves[20:45], params), m.partial(leaves[45:], params)
        assert m.merge(m.identity(), a) == a and m.merge(a, m.identity()) == a
        assert m.merge(m.merge(a, b), c) == m.merge(a, m.merge(b, c))
        assert m.merge(a, b) == m.merge(b, a)

    def test_group_stats_monoid_equals_the_flat_block_exactly(self):
        leaves = _leaves(200, seed=3)
        params = {"by": ["g"], "value": "delta"}
        flat = PURE_BLOCKS["~canonical/ops/group-stats/1"]({"records": leaves}, params)
        chunks = [leaves[i:i + 37] for i in range(0, len(leaves), 37)]
        chunked = rd.reduce_chunks("~canonical/ops/group-stats/1", chunks, params)
        # rows compare as sets (the flat block's group order is insertion order)
        def key(r):
            return r["g"]

        assert sorted(flat["rows"], key=key) == sorted(chunked["rows"], key=key)

    def test_merge_tree_shape_depends_only_on_n(self):
        calls = []

        class Spy(rd.Monoid):
            def identity(self):
                return "e"

            def merge(self, a, b):
                calls.append((a, b))
                return f"({a}{b})"

        out = rd.merge_tree(Spy(), list("abcde"))
        assert out == "(((ab)(cd))e)"


class TestHarness:
    @pytest.mark.parametrize("block,params", [
        ("~canonical/ops/group-stats/1", {"by": ["g"], "value": "delta"}),
        ("~canonical/ops/reduce/sum/1", {"value": "delta"}),
        ("~canonical/ops/reduce/top-k/1", {"value": "score", "k": 7}),
        ("~canonical/ops/reduce/histogram/1", {"value": "delta", "lo": -5, "hi": 5, "bins": 8}),
        ("~canonical/ops/select/1", {"where": {"g": "a"}}),
        ("~canonical/ops/table/from-records/1", {"columns": ["id", "delta"]}),
    ])
    def test_random_nested_partitions_reduce_to_the_flat_result(self, block, params):
        report = iso.check(block, _leaves(120, seed=5), params, trials=25, seed=11)
        assert report["exact"] is True

    def test_an_ordered_block_is_refused_from_chunking(self, monkeypatch):
        monkeypatch.setitem(rd.REDUCE_ALGEBRA, "~canonical/ops/select/1", "ordered")
        report = iso.check("~canonical/ops/select/1", _leaves(10), {"where": {}}, trials=1)
        assert report["refused"] is True

    def test_a_broken_monoid_is_caught(self, monkeypatch):
        class Bad(rd.FloatSum):
            def merge(self, a, b):  # drops a value: not a monoid
                return tuple(sorted((a + b)[:-1])) if len(a + b) > 3 else tuple(sorted(a + b))

        monkeypatch.setitem(rd.MONOIDS, "~canonical/ops/reduce/sum/1", Bad)
        with pytest.raises(AssertionError):
            iso.check("~canonical/ops/reduce/sum/1", _leaves(40), {"value": "delta"}, trials=10)

    def test_fsum_makes_float_sums_partition_independent(self):
        leaves = [{"id": str(i), "v": (0.1 * i) ** 3 * (-1) ** i} for i in range(500)]
        report = iso.check("~canonical/ops/reduce/sum/1", leaves, {"value": "v"}, trials=30)
        assert report["exact"] is True


def _text_leaves(n=40, seed=2):
    rng = random.Random(seed)
    words = ["dust", "kettle", "harbor", "glass", "moth", "ledger", "salt", "rope"]
    return [{"id": f"t{i}", "coords": {"g": rng.choice(["a", "b"])},
             "text": " ".join(rng.choice(words) for _ in range(rng.randint(5, 20)))}
            for i in range(n)]


def _paired_leaves(n=40, seed=4):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        for arm in ("base", "test"):
            out.append({"id": f"p{i}-{arm}", "coords": {"item": str(i), "arm": arm},
                        "delta": rng.uniform(-2, 2)})
    return out


def _decision_leaves(n=30, seed=6):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = rng.random()
        out.append({"id": f"d{i}", "entropy_bits": round(rng.uniform(0.2, 1.0), 4),
                    "top_tokens": [{"token": "left", "p": round(p, 4)},
                                   {"token": "right", "p": round(1 - p, 4)}]})
    return out


def _template_leaves(n=25, seed=8):
    rng = random.Random(seed)
    return [{"id": f"v{i}", "coords": {"g": rng.choice(["a", "b"])},
             "values": {"g": rng.choice(["dusk", "dawn"]), "n": i}}
            for i in range(n)]


#: Every pure block whose input IS a leaf stream, with the fixture the
#: law runs on. A new block must land here (or in NOT_LEAF_STREAM) or
#: `test_every_pure_block_is_classified` fails: the catalog stays
#: covered by construction (task 000407, "CI over the whole catalog").
CATALOG: dict[str, dict] = {
    "~canonical/ops/group-stats/1": {
        "leaves": _leaves(120, seed=5), "params": {"by": ["g"], "value": "delta"}},
    "~canonical/ops/reduce/sum/1": {
        "leaves": _leaves(120, seed=5), "params": {"value": "delta"}},
    "~canonical/ops/reduce/top-k/1": {
        "leaves": _leaves(120, seed=5), "params": {"value": "score", "k": 7}},
    "~canonical/ops/reduce/histogram/1": {
        "leaves": _leaves(120, seed=5),
        "params": {"value": "delta", "lo": -5, "hi": 5, "bins": 8}},
    "~canonical/ops/select/1": {
        "leaves": _leaves(120, seed=5), "params": {"where": {"g": ["a", "b"]}}},
    "~canonical/ops/table/from-records/1": {
        "leaves": _leaves(120, seed=5), "params": {"name": "leaves"}},
    "~canonical/ops/template/1": {
        "leaves": _template_leaves(),
        "params": {"templates": {"prompt": "a {g} of {n}"}}},
    "~canonical/ops/paired-delta/1": {
        "leaves": _paired_leaves(),
        "params": {"match_on": ["item"], "baseline_where": {"arm": "base"},
                   "value": "delta"}},
    "~canonical/ops/text/stats/1": {
        "leaves": _text_leaves(),
        "params": {"field": "text", "mode": "corpus", "measures": [
            {"kind": "lexical", "name": "lex"},
            {"kind": "pattern", "name": "salt", "patterns": ["salt"]}]}},
    "~canonical/ops/union/1": {
        "leaves": _leaves(60, seed=9), "port": "a",
        "inputs": {"b": _leaves(20, seed=10)}, "params": {}},
    "~canonical/ops/eval/expectation/1": {
        "leaves": _decision_leaves(), "port": "results",
        "inputs": {"expectations": [
            {"id": f"d{i}", "expect": {"kind": "answer", "value": "left",
                                        "min_p": 0.4}}
            for i in range(30)]},
        "params": {}},
}

#: Pure blocks whose input is NOT a stream of leaf records — the law
#: does not apply to them as written, and why.
NOT_LEAF_STREAM: dict[str, str] = {
    "~canonical/ops/factor-cross/1": "generator: builds leaves from params, consumes none",
    "~canonical/ops/grid/1": "generator (factor-cross alias)",
    "~canonical/ops/vectors/similarity/1":
        "one residual_vectors record; its rows are not bench leaves",
    "~canonical/ops/vectors/mst/1":
        "one similarity matrix or vector set, not a leaf stream",
    "~canonical/ops/tools/calc/1":
        "a tool handler: its input is one call's arguments, not a leaf stream",
    "~canonical/ops/tools/bench-lookup/1":
        "a tool handler: its input is one call's arguments, not a leaf stream",
    **{f"~canonical/ops/direction/{n}/1":
       "residual_vectors / direction records, not a leaf stream"
       for n in ("from-vectors", "from-pca", "add", "average", "orthogonalize",
                 "normalize", "similarity", "project")},
    # Trajectory readouts (task 000368) read ONE trajectory record —
    # rows are (item, step) points along an axis, not bench leaves —
    # the same footing as the direction algebra above.
    **{f"~canonical/ops/trajectory/{n}/1":
       "one trajectory record; its rows are (item, step) points, not leaves"
       for n in ("project", "compare", "aggregate")},
}


class TestCatalog:
    def test_every_pure_block_is_classified(self):
        assert set(PURE_BLOCKS) == set(CATALOG) | set(NOT_LEAF_STREAM)

    @pytest.mark.parametrize("block", sorted(CATALOG))
    def test_the_law_holds_across_the_catalog(self, block):
        f = CATALOG[block]
        report = iso.check(block, f["leaves"], f["params"], trials=15, seed=3,
                           port=f.get("port", "records"), inputs=f.get("inputs"))
        assert report["exact"] is True


class TestSeeds:
    def test_item_seed_is_generate_s_rule_verbatim(self):
        seed, rid, k = 7, "flash", 42
        digest = hashlib.sha256(f"{seed}:{rid}:{k}".encode()).digest()
        assert seeds.item_seed(seed, rid, k) == int.from_bytes(digest[:8], "little")

    def test_member_and_derived_seeds_are_stable_and_distinct(self):
        assert seeds.member_seed(7, 3) == seeds.member_seed(7, 3)
        assert seeds.member_seed(7, 3) != seeds.member_seed(7, 4)
        assert seeds.derive(7, "member", 3, "record", "x") != seeds.derive(7, "member", 3, "record", "y")

    def test_hardware_class_has_the_lineage_fields(self):
        h = seeds.hardware_class()
        assert {"platform", "python", "mlx"} <= set(h)
        assert "|mlx " in seeds.hardware_class_id(h)
