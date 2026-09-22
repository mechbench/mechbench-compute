"""Variadic ports and `records/zip`.

With nothing saying how many edges a port takes, two edges into one
port silently keep whichever comes LAST in the graph's edge list — a
winner decided by the order the author wrote the lines in. A port says
how many edges it takes: one, or several in a declared order.
`records/zip` is the op that wants several, and is what proves the
ordering is real.
"""

from __future__ import annotations

import pytest

from mechbench_compute.blocks import PURE_BLOCKS
from mechbench_compute.lexicon import BY_NAME
from mechbench_compute.lexicon._base import In
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec, sort_edges

CROSS = {"factors": [{"name": "x", "levels": [{"key": "a"}, {"key": "b"}]}]}


def _branch(node, ids, **fields):
    return {"node": node, "value": [
        {"id": i, "coords": {"prompt": i}, **{k: f"{v}-{i}" for k, v in fields.items()}}
        for i in ids]}


def _zip(branches, **params):
    return PURE_BLOCKS["records/zip"]({"branches": branches}, params)


class TestPortArity:
    def test_a_plain_port_takes_one_edge(self):
        p = In("records", "records/record", "…")
        assert p.arity_error(1) is None
        assert "takes one edge" in p.arity_error(2)

    def test_a_variadic_port_takes_several_within_its_bounds(self):
        p = In("branches", "collection", "…", variadic=True, min_edges=2, max_edges=3)
        assert p.arity_error(2) is None and p.arity_error(3) is None
        assert "at least 2" in p.arity_error(1)
        assert "at most 3" in p.arity_error(4)

    def test_the_declaration_travels_to_the_docs(self):
        d = BY_NAME["records/zip"].port("branches").to_dict()
        assert d["variadic"] is True and d["min_edges"] == 2
        assert BY_NAME["records/select"].port("records").to_dict()["variadic"] is False


class TestEdgeOrder:
    def _edges(self, *specs):
        return [{"from": {"node": n, "port": "out"},
                 "to": {"node": "z", "port": p},
                 **({"index": i} if i is not None else {})}
                for n, p, i in specs]

    def test_ordered_by_port_then_index_then_source(self):
        edges = self._edges(("c", "branches", 2), ("a", "branches", 0),
                            ("b", "branches", 1), ("d", "other", None))
        got = [(e["from"]["node"], e["to"]["port"])
               for e in sort_edges(edges, "z")]
        assert got == [("a", "branches"), ("b", "branches"), ("c", "branches"),
                       ("d", "other")]

    def test_without_an_index_the_source_id_orders_them(self):
        edges = self._edges(("zeta", "branches", None), ("alpha", "branches", None))
        assert [e["from"]["node"] for e in sort_edges(edges, "z")] == \
            ["alpha", "zeta"]

    def test_edges_to_other_nodes_are_not_this_node_s(self):
        edges = [{"from": {"node": "a", "port": "out"}, "to": {"node": "y", "port": "p"}}]
        assert sort_edges(edges, "z") == []


class TestTwoEdgesIntoOnePort:
    """The silent loss this began with."""

    def _spec(self, port="records"):
        return ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [
                {"id": "one", "block": "records/cross", "params": dict(CROSS)},
                {"id": "two", "block": "records/cross", "params": dict(CROSS)},
                {"id": "pick", "block": "records/select", "params": {}},
            ], "edges": [
                {"from": {"node": "one", "port": "records"},
                 "to": {"node": "pick", "port": port}, "kind": "records"},
                {"from": {"node": "two", "port": "records"},
                 "to": {"node": "pick", "port": port}, "kind": "records"},
            ]}})

    def test_it_is_refused_at_load_now(self):
        with pytest.raises(ValueError) as e:
            ProtocolExecutor().run(self._spec())
        msg = str(e.value)
        assert "port 'records' takes one edge; 2 arrive" in msg
        assert "silently won" in msg          # the message names the failure


class TestZip:
    def test_two_branches_pair_by_id(self):
        out = _zip([_branch("base", ["p1", "p2"], text="b"),
                    _branch("adapted", ["p1", "p2"], text="a")])
        assert out["item_kind"] == "records/record"
        assert [r["id"] for r in out["items"]] == ["p1", "p2"]
        first = out["items"][0]
        assert set(first["branches"]) == {"base", "adapted"}
        assert first["branches"]["base"]["text"] == "b-p1"
        assert first["branches"]["adapted"]["text"] == "a-p1"
        assert out["zipped"]["keys"] == 2
        assert [b["name"] for b in out["branches"]] == ["base", "adapted"]

    def test_the_branches_are_named_by_their_source_node(self):
        out = _zip([_branch("cap_die", ["p1"]), _branch("cap_names", ["p1"])])
        assert set(out["items"][0]["branches"]) == {"cap_die", "cap_names"}

    def test_names_override_in_edge_order(self):
        out = _zip([_branch("n1", ["p1"]), _branch("n2", ["p1"])],
                   names=["before", "after"])
        assert list(out["items"][0]["branches"]) == ["before", "after"]

    def test_by_coordinates_when_ids_disagree(self):
        left = {"node": "l", "value": [{"id": "x1", "coords": {"prompt": "p", "seed": 1}}]}
        right = {"node": "r", "value": [{"id": "y9", "coords": {"prompt": "p", "seed": 1}}]}
        out = _zip([left, right], by=["prompt", "seed"])
        assert len(out["items"]) == 1
        assert out["items"][0]["coords"] == {"prompt": "p", "seed": 1}

    def test_flatten_lifts_each_branch_s_fields(self):
        out = _zip([_branch("base", ["p1"], text="b"),
                    _branch("adapted", ["p1"], text="a")], flatten=True)
        rec = out["items"][0]
        assert rec["base_text"] == "b-p1" and rec["adapted_text"] == "a-p1"
        assert "branches" not in rec

    def test_a_key_missing_from_one_branch_fails_by_default(self):
        with pytest.raises(ValueError, match="not in every branch"):
            _zip([_branch("base", ["p1", "p2"]), _branch("adapted", ["p1"])])

    def test_drop_keeps_what_every_branch_has(self):
        out = _zip([_branch("base", ["p1", "p2"]), _branch("adapted", ["p1"])],
                   on_mismatch="drop")
        assert [r["id"] for r in out["items"]] == ["p1"]
        assert out["zipped"]["dropped"] == ["p2"]

    def test_placeholder_keeps_them_and_marks_what_is_absent(self):
        out = _zip([_branch("base", ["p1", "p2"]), _branch("adapted", ["p1"])],
                   on_mismatch="placeholder")
        assert [r["id"] for r in out["items"]] == ["p1", "p2"]
        assert out["items"][1]["branches"]["adapted"] == {"missing": True}
        assert out["items"][1]["branches"]["base"]["id"] == "p2"

    def test_three_branches(self):
        out = _zip([_branch("a", ["p1"]), _branch("b", ["p1"]), _branch("c", ["p1"])])
        assert len(out["items"][0]["branches"]) == 3

    def test_one_branch_is_refused(self):
        with pytest.raises(ValueError, match="at least two branches"):
            _zip([_branch("only", ["p1"])])

    def test_a_duplicate_key_within_a_branch_is_refused(self):
        dupe = {"node": "d", "value": [{"id": "p1", "coords": {"prompt": "p"}},
                                       {"id": "p2", "coords": {"prompt": "p"}}]}
        with pytest.raises(ValueError, match="two records keyed"):
            _zip([dupe, _branch("other", ["p1"])], by=["prompt"])

    def test_an_unknown_policy_and_a_bad_name_count_are_refused(self):
        with pytest.raises(ValueError, match="on_mismatch"):
            _zip([_branch("a", ["p1"]), _branch("b", ["p1"])], on_mismatch="shrug")
        with pytest.raises(ValueError, match="entries for 2 branches"):
            _zip([_branch("a", ["p1"]), _branch("b", ["p1"])], names=["only"])


class TestZipInAGraph:
    def test_two_branches_through_the_executor(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [
                {"id": "design", "block": "records/cross", "params": dict(CROSS)},
                {"id": "left", "block": "records/fill",
                 "params": {"templates": {"user": "left {x}"}}},
                {"id": "right", "block": "records/fill",
                 "params": {"templates": {"user": "right {x}"}}},
                {"id": "pairs", "block": "records/zip",
                 "params": {"names": ["left", "right"]}},
            ], "edges": [
                {"from": {"node": "design", "port": "records"},
                 "to": {"node": "left", "port": "records"}, "kind": "records"},
                {"from": {"node": "design", "port": "records"},
                 "to": {"node": "right", "port": "records"}, "kind": "records"},
                {"from": {"node": "left", "port": "records"},
                 "to": {"node": "pairs", "port": "branches"},
                 "kind": "records", "index": 0},
                {"from": {"node": "right", "port": "records"},
                 "to": {"node": "pairs", "port": "branches"},
                 "kind": "records", "index": 1},
            ]}})
        out = ProtocolExecutor().run(spec)
        pairs = out.payload["outputs"]["pairs"]
        assert len(pairs["items"]) == 2
        rec = pairs["items"][0]
        assert rec["branches"]["left"]["user"].startswith("left")
        assert rec["branches"]["right"]["user"].startswith("right")

    def test_a_variadic_port_with_too_few_edges_is_refused_at_load(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [
                {"id": "design", "block": "records/cross", "params": dict(CROSS)},
                {"id": "pairs", "block": "records/zip", "params": {}},
            ], "edges": [
                {"from": {"node": "design", "port": "records"},
                 "to": {"node": "pairs", "port": "branches"}, "kind": "records"},
            ]}})
        with pytest.raises(ValueError, match="at least 2 edges"):
            ProtocolExecutor().run(spec)
