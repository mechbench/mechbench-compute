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
from mechbench_compute.ops.records.total import FloatSum


def _leaves(n=60, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append({"id": f"r{i}", "coords": {"g": rng.choice(["a", "b", "c"])},
                    "delta": rng.uniform(-3, 3) * (1 + i % 7) / 3, "score": rng.random()})
    return out


class TestMonoidLaws:
    @pytest.mark.parametrize("block,params", [
        ("records/summarize", {"by": ["g"], "value": "delta"}),
        ("records/summarize", {"by": ["g"], "value": "delta", "interval": 0.9, "resamples": 300}),
        ("records/total", {"value": "delta"}),
        ("records/rank", {"value": "score", "k": 5}),
        ("records/bin", {"value": "delta", "lo": -5, "hi": 5, "bins": 10}),
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
        flat = PURE_BLOCKS["records/summarize"]({"records": leaves}, params)
        chunks = [leaves[i:i + 37] for i in range(0, len(leaves), 37)]
        chunked = rd.reduce_chunks("records/summarize", chunks, params)
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
        ("records/summarize", {"by": ["g"], "value": "delta"}),
        ("records/total", {"value": "delta"}),
        ("records/rank", {"value": "score", "k": 7}),
        ("records/bin", {"value": "delta", "lo": -5, "hi": 5, "bins": 8}),
        ("records/select", {"where": {"g": "a"}}),
        ("records/tabulate", {"columns": ["id", "delta"]}),
    ])
    def test_random_nested_partitions_reduce_to_the_flat_result(self, block, params):
        report = iso.check(block, _leaves(120, seed=5), params, trials=25, seed=11)
        assert report["exact"] is True

    def test_an_ordered_block_is_refused_from_chunking(self, monkeypatch):
        monkeypatch.setitem(rd.REDUCE_ALGEBRA, "records/select", "ordered")
        report = iso.check("records/select", _leaves(10), {"where": {}}, trials=1)
        assert report["refused"] is True

    def test_a_broken_monoid_is_caught(self, monkeypatch):
        class Bad(FloatSum):
            def merge(self, a, b):  # drops a value: not a monoid
                return tuple(sorted((a + b)[:-1])) if len(a + b) > 3 else tuple(sorted(a + b))

        from mechbench_compute.ops.records import total

        monkeypatch.setattr(total, "MONOID", Bad)
        with pytest.raises(AssertionError):
            iso.check("records/total", _leaves(40), {"value": "delta"}, trials=10)

    def test_fsum_makes_float_sums_partition_independent(self):
        leaves = [{"id": str(i), "v": (0.1 * i) ** 3 * (-1) ** i} for i in range(500)]
        report = iso.check("records/total", leaves, {"value": "v"}, trials=30)
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
    "records/summarize": {
        "leaves": _leaves(120, seed=5), "params": {"by": ["g"], "value": "delta"}},
    "records/total": {
        "leaves": _leaves(120, seed=5), "params": {"value": "delta"}},
    "records/rank": {
        "leaves": _leaves(120, seed=5), "params": {"value": "score", "k": 7}},
    "records/bin": {
        "leaves": _leaves(120, seed=5),
        "params": {"value": "delta", "lo": -5, "hi": 5, "bins": 8}},
    "records/select": {
        "leaves": _leaves(120, seed=5), "params": {"where": {"g": ["a", "b"]}}},
    "records/rename": {
        "leaves": _leaves(120, seed=5), "params": {"fields": {"v": "value"}}},
    "records/tabulate": {
        "leaves": _leaves(120, seed=5), "params": {"name": "leaves"}},
    "records/fill": {
        "leaves": _template_leaves(),
        "params": {"templates": {"prompt": "a {g} of {n}"}}},
    "records/subtract": {
        "leaves": _paired_leaves(),
        "params": {"match_on": ["item"], "baseline_where": {"arm": "base"},
                   "value": "delta"}},
    "records/contrast": {
        "leaves": _paired_leaves(),
        "params": {"value": "delta", "on": "arm", "a": "test", "b": "base",
                   "paired": "item", "resamples": 300}},
    "text/measure": {
        "leaves": _text_leaves(),
        "params": {"field": "text", "mode": "corpus", "measures": [
            {"kind": "lexical", "name": "lex"},
            {"kind": "pattern", "name": "salt", "patterns": ["salt"]}]}},
    "records/union": {
        "leaves": _leaves(60, seed=9), "port": "a",
        "inputs": {"b": _leaves(20, seed=10)}, "params": {}},
    "eval/expect": {
        "leaves": _decision_leaves(), "port": "results",
        "inputs": {"expectations": [
            {"id": f"d{i}", "expect": {"kind": "answer", "value": "left",
                                        "min_p": 0.4}}
            for i in range(30)]},
        "params": {}},
}

def _transcript_leaves(n=12, seed=11):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        msgs = [{"index": 0, "participant": "user", "role_as_seen": "assistant", "text": f"opening {i}"}]
        for k in range(1, 1 + rng.randint(1, 4)):
            who = "ana" if k % 2 else "bo"
            msgs.append({"index": k, "participant": who, "role_as_seen": "assistant",
                         "text": f"{who} says {k}", **({"thinking": f"hm {k}"} if rng.random() < 0.5 else {})})
        out.append({"id": f"c{i}", "kind": "text/transcript", "participants": ["ana", "bo"],
                    "messages": msgs, "stopped": "", "coords": {"g": rng.choice(["a", "b"])}})
    return out


CATALOG["text/render"] = {
    "leaves": _transcript_leaves(), "port": "transcripts",
    "params": {"participant": "ana", "sees": {"own_thinking": {"last_turns": 1}}}}

#: Pure blocks whose input is NOT a stream of leaf records — the law
#: does not apply to them as written, and why.
NOT_LEAF_STREAM: dict[str, str] = {
    "text/extend":
        "two streams aligned by conversation — transcripts and the replies "
        "to them — the way records/zip aligns branches",
    "records/cross": "generator: builds leaves from params, consumes none",
    "records/cross": "generator (factor-cross alias)",
    "geometry/compare":
        "one collection compared under a metric; its items are not bench leaves",
    "records/zip":
        "several branches aligned by key; its input is a set of streams, not one",
    "geometry/span":
        "one similarity collection, not a leaf stream",
    "adapter/measure":
        "one adapter object, read module by module; its input is a set of "
        "weights, not a stream of leaves that could be chunked",
    "tools/calc":
        "a tool handler: its input is one call's arguments, not a leaf stream",
    "tools/lookup":
        "a tool handler: its input is one call's arguments, not a leaf stream",
    **{f"direction/{n}":
       "residual_vectors / direction records, not a leaf stream"
       for n in ("fit", "classify", "regress", "decompose", "add", "average",
                 "orthogonalize", "normalize", "project")},
    # Trajectory readouts (task 000368) read ONE trajectory record —
    # rows are (item, step) points along an axis, not bench leaves —
    # the same footing as the direction algebra above.
    **{f"trajectory/{n}":
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
