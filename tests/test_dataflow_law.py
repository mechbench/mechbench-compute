from __future__ import annotations

import hashlib
import random

import pytest

from mechbench_compute import isomorphism as iso
from mechbench_compute import ops, seeds
from mechbench_compute import reduce as rd


def _leaves(n=60, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append({"id": f"r{i}", "coords": {"g": rng.choice(["a", "b", "c"])},
                    "delta": rng.uniform(-3, 3) * (1 + i % 7) / 3, "score": rng.random()})
    return out


class TestMergeTree:
    def test_merge_tree_shape_depends_only_on_n(self):
        class Spy(rd.Monoid):
            def identity(self):
                return "e"

            def merge(self, a, b):
                return f"({a}{b})"

        assert rd.merge_tree(Spy(), list("abcde")) == "(((ab)(cd))e)"


class TestHarness:
    @pytest.mark.parametrize("block,params,leaves,extra", [
        ("records/filter", {"where": 'coords.g == "a"'}, None, {}),
        ("records/derive", {"fields": {"half": "delta / 2"}, "templates": {"g": "{coords.g}!"}}, None, {}),
        ("records/join", {"on": "coords.item", "on_right": "item", "as": "w"}, "paired",
         {"port": "left", "inputs": {"right": [{"id": f"i{i}", "item": str(i), "weight": i} for i in range(40)]}}),
        ("records/tabulate", {"columns": ["id", "delta"]}, None, {}),
    ])
    def test_random_nested_partitions_reduce_to_the_flat_result(self, block, params, leaves, extra):
        data = _paired_leaves() if leaves == "paired" else _leaves(120, seed=5)
        report = iso.check(block, data, params, trials=25, seed=11, **extra)
        assert report["exact"] is True

    @pytest.mark.parametrize("block,params", [
        ("records/group", {"by": {"g": "coords.g"}, "aggregates": {"n": "count()"}}),
        ("records/sort", {"by": ["-score"], "limit": 5}),
    ])
    def test_an_ordered_block_is_refused_from_chunking(self, block, params):
        assert iso.check(block, _leaves(10), params, trials=1)["refused"] is True


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


def _nested_leaves(n=30, seed=12):
    rng = random.Random(seed)
    return [{"id": f"n{i}", "coords": {"g": rng.choice(["a", "b"])},
             "votes": [{"winner": rng.choice("AB")} for _ in range(rng.randint(0, 3))]}
            for i in range(n)]


CATALOG: dict[str, dict] = {
    "records/tabulate": {
        "leaves": _leaves(120, seed=5), "params": {"name": "leaves"}},
    "records/unnest": {
        "leaves": _nested_leaves(), "params": {"field": "votes", "index": "vote"}},
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
    "records/diff":
        "two collections matched by key and compared; a pair of streams, not one",
    "geometry/span":
        "one similarity collection, not a leaf stream",
    "adapter/measure":
        "one adapter object, read module by module; its input is a set of "
        "weights, not a stream of leaves that could be chunked",
    "eval/score":
        "one collection scored against a reference by a metric; it became "
        "callable by name when its file made it standalone (docs/OPS_LAYOUT.md)",
    "tools/calc":
        "a tool handler: its input is one call's arguments, not a leaf stream",
    "tools/lookup":
        "a tool handler: its input is one call's arguments, not a leaf stream",
    **{f"direction/{n}":
       "residual_vectors / direction records, not a leaf stream"
       for n in ("fit", "classify", "regress", "decompose", "add", "average",
                 "orthogonalize", "normalize", "project")},
    **{f"trajectory/{n}":
       "one trajectory record; its rows are (item, step) points, not leaves"
       for n in ("project", "compare", "aggregate")},
}


class TestCatalog:
    def test_every_pure_block_is_classified(self):
        assert set(ops.find_standalone()) == set(CATALOG) | set(NOT_LEAF_STREAM)

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
