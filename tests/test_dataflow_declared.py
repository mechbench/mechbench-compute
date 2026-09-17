"""The declared dataflow form through the executor (epic 000553, task
000557): `{"$param"}` and `{"$ref"}` resolved as values, protocol inputs
as edge sources, a stored object named as a lineage input wherever it sat
— and the property the migration depends on: a protocol rewritten from the
legacy form computes the same bytes under the same node fingerprints.

Pure blocks only, with the bench faked, so nothing here needs a model.
"""

from __future__ import annotations

import hashlib

import pytest
from mechbench_schema import dump_canonical

from mechbench_compute import bench, dataflow
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

FREQS = {"kind": "text/word-list", "weights": {"Mystery": 3.0, "Humor": 1.0, "Witches": 1.0}}
DRAWS = {
    "kind": "collection", "item_kind": "text/document", "key": ["id"],
    "items": [
        {"id": "a", "kind": "text/document", "text": '{ "genres": "Mystery, Humor'},
        {"id": "b", "kind": "text/document", "text": '{ "genres": "Steampunk Fantasy, Mystery'},
    ],
}
STORE = {"lab/p/freqs": FREQS, "lab/p/draws": DRAWS}


@pytest.fixture
def fake_bench(monkeypatch):
    """A bench that holds two objects, and records what was emitted."""
    emitted: dict[str, dict] = {}

    def fetch(ref, with_meta=False):
        obj = {"payload": STORE[str(ref)]}
        meta = {"content_hash": "sha256:" + hashlib.sha256(dump_canonical(STORE[str(ref)])).hexdigest()}
        return (obj, meta) if with_meta else obj

    def emit(target, payload, *, inputs=(), operation=None, params=None, **kw):
        emitted[target] = {"inputs": list(inputs), "operation": operation}
        return {"path": target}

    monkeypatch.setattr(bench, "fetch", fetch)
    monkeypatch.setattr(bench, "emit", emit)
    return emitted


MEASURE = {"type": "list", "name": "genres", "separator": ", ",
           "extract": r'"genres"\s*:\s*"([^"]*)'}

LEGACY = {
    "nodes": [{
        "id": "said", "block": "text/measure",
        "params": {"mode": "$mode", "measures": [{**MEASURE, "items": {"$fetch": "lab/p/freqs"}}]},
        "inputs": {"documents": {"$fetch": "$draws"}},
    }],
    "edges": [],
}
DECLARED = {
    "dataflow": 2,
    "nodes": [{
        "id": "said", "block": "text/measure",
        "params": {"mode": {"$param": "mode"},
                   "measures": [{**MEASURE, "items": {"$ref": {"bench": "lab/p/freqs"}}}]},
        "inputs": {},
    }],
    "edges": [{"from": {"input": "draws"}, "to": {"node": "said", "port": "documents"}}],
}


class _Hooks:
    def __init__(self):
        self.fingerprints: dict[str, str] = {}

    def executor(self):
        return ProtocolExecutor(on_node_start=lambda nid, fp: self.fingerprints.__setitem__(nid, fp))


def _run(extra, hooks=None):
    hooks = hooks or _Hooks()
    out = hooks.executor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra=extra))
    payload = out.payload if hasattr(out, "payload") else out
    return payload, hooks


def test_a_migrated_protocol_computes_the_same_bytes_under_the_same_fingerprints(fake_bench):
    """What lets stored protocols be rewritten without invalidating a
    cache or a resume: a fingerprint is over what a node resolved TO, and
    both forms resolve to the same values in the same places."""
    old, old_hooks = _run({"graph": LEGACY, "bindings": {"mode": "items", "draws": "lab/p/draws"},
                           "resultPath": "lab/p/results/j_old"})
    new, new_hooks = _run({"graph": DECLARED, "params": {"mode": "items"},
                           "inputs": {"draws": {"$ref": {"bench": "lab/p/draws"}}},
                           "resultPath": "lab/p/results/j_new"})
    assert dump_canonical(old["outputs"]["said"]) == dump_canonical(new["outputs"]["said"])
    assert old_hooks.fingerprints == new_hooks.fingerprints
    said = {r["item"]: r for r in new["outputs"]["said"]["items"]}
    assert said["Mystery"]["count"] == 2 and said["Steampunk Fantasy"]["in_vocabulary"] is False


def test_every_stored_object_a_node_reads_is_a_lineage_input(fake_bench):
    """A frequency table fetched into a param is an input of the node that
    used it. Lineage used to name only what arrived by edge."""
    _run({"graph": DECLARED, "params": {"mode": "items"},
          "inputs": {"draws": {"$ref": {"bench": "lab/p/draws"}}},
          "resultPath": "lab/p/results/j_new"})
    assert fake_bench["lab/p/results/j_new/said"]["inputs"] == ["lab/p/draws", "lab/p/freqs"]


def test_a_dollar_string_is_a_string_in_a_declared_graph(fake_bench):
    graph = {"dataflow": 2, "edges": [], "nodes": [{
        "id": "grid", "block": "records/cross",
        "params": {"factors": [{"name": "price", "levels": [{"key": "$5 a bag"}]}]}}]}
    payload, _ = _run({"graph": graph, "params": {}, "inputs": {}})
    assert payload["outputs"]["grid"]["items"][0]["coords"] == {"price": "$5 a bag"}


def test_an_unbound_param_or_input_is_refused_before_anything_runs(fake_bench):
    with pytest.raises(ValueError, match="unbound param 'mode'"):
        _run({"graph": DECLARED, "params": {}, "inputs": {"draws": {"$ref": {"bench": "lab/p/draws"}}}})
    with pytest.raises(ValueError, match="unbound input: 'draws'"):
        _run({"graph": DECLARED, "params": {"mode": "items"}, "inputs": {}})


def test_a_ref_where_no_declaration_admits_one_is_refused_by_name(fake_bench):
    graph = {"dataflow": 2, "edges": [], "nodes": [{
        "id": "said", "block": "text/measure",
        "params": {"mode": "items", "measures": {"$ref": {"bench": "lab/p/freqs"}}},
        "inputs": {"documents": {"$ref": {"bench": "lab/p/draws"}}}}]}
    with pytest.raises(ValueError, match=r"said\.measures: a \$ref sits where text/measure declares no"):
        _run({"graph": graph, "params": {}, "inputs": {}})


def test_the_legacy_macro_is_named_as_such_in_a_declared_graph(fake_bench):
    graph = {"dataflow": 2, "edges": [], "nodes": [{
        "id": "said", "block": "text/measure",
        "params": {"mode": "items", "measures": [{**MEASURE, "items": {"$fetch": "lab/p/freqs"}}]},
        "inputs": {"documents": {"$ref": {"bench": "lab/p/draws"}}}}]}
    with pytest.raises(ValueError, match=r"\$fetch is the legacy form"):
        _run({"graph": graph, "params": {}, "inputs": {}})


def test_a_param_may_be_bound_to_a_reference(fake_bench):
    """References are values: a run binds the frequency table itself."""
    graph = {"dataflow": 2, "edges": [], "nodes": [{
        "id": "said", "block": "text/measure",
        "params": {"mode": "items", "measures": [{**MEASURE, "items": {"$param": "vocabulary"}}]},
        "inputs": {"documents": {"$ref": {"bench": "lab/p/draws"}}}}]}
    payload, _ = _run({"graph": graph, "inputs": {},
                       "params": {"vocabulary": {"$ref": {"bench": "lab/p/freqs"}}},
                       "resultPath": "lab/p/results/j_ref"})
    assert {r["item"] for r in payload["outputs"]["said"]["items"] if r["in_vocabulary"]} == {"Mystery", "Humor"}
    assert fake_bench["lab/p/results/j_ref/said"]["inputs"] == ["lab/p/draws", "lab/p/freqs"]


def test_lower_turns_an_input_edge_into_the_ports_value_and_keeps_node_edges():
    graph = {"dataflow": 2,
             "nodes": [{"id": "a", "block": "x"}, {"id": "b", "block": "y"}],
             "edges": [{"from": {"input": "recs"}, "to": {"node": "a", "port": "records"}},
                       {"from": {"node": "a"}, "to": {"node": "b", "port": "records"}},
                       {"from": {"node": "a", "output": "fit"}, "to": {"node": "b", "port": "other"}}]}
    low = dataflow.lower(graph, {"recs": [{"id": "r1"}]})
    assert low["nodes"][0]["inputs"] == {"records": [{"id": "r1"}]}
    assert [e["from"] for e in low["edges"]] == [{"node": "a", "port": "out"}, {"node": "a", "port": "fit"}]
    with pytest.raises(ValueError, match="one or the other"):
        dataflow.lower({**graph, "nodes": [{"id": "a", "block": "x", "inputs": {"records": []}},
                                           {"id": "b", "block": "y"}]}, {"recs": []})


def test_an_op_may_declare_that_it_wants_the_reference_itself(monkeypatch):
    """To stream from an address lazily, or to publish to one. No op asks
    yet; the declaration is what lets one."""
    from mechbench_compute import lexicon
    from mechbench_compute.lexicon._base import P

    class FakeOp:
        name = "test/publish"
        params = (P("target", "ref", "Where to publish.", reference=True),
                  P("label", "string", "A label.", None))

    monkeypatch.setitem(lexicon.BY_NAME, "test/publish", FakeOp())
    assert dataflow.wants_reference("test/publish", "target") is True
    assert dataflow.wants_reference("test/publish", "label") is False
    assert dataflow.wants_reference("test/publish", "absent") is False
    # …and such a position admits a $ref, where an undeclared one does not.
    monkeypatch.setattr(lexicon, "resolve", lambda block: block)
    nodes = {"pub": {"block": "test/publish", "params": {"target": {"$ref": {"bench": "lab/p/out"}}}}}
    dataflow.check_refs(nodes, {})
    nodes["pub"]["params"]["label"] = {"$ref": {"bench": "lab/p/out"}}
    with pytest.raises(ValueError, match=r"pub\.label: a \$ref sits where"):
        dataflow.check_refs(nodes, {})


# --- declared outputs are the run's results (000558) -----------------------

TWO_NODES = {
    "dataflow": 2,
    "nodes": [
        {"id": "grid", "block": "records/cross",
         "params": {"factors": [{"name": "x", "levels": [{"key": "1"}, {"key": "2"}]}]}},
        {"id": "picked", "block": "records/select", "params": {"where": {"x": "1"}}},
    ],
    "edges": [{"from": {"node": "grid"}, "to": {"node": "picked", "port": "records"}}],
}


def test_a_declared_output_is_stored_under_its_name_and_the_rest_apart(fake_bench):
    payload, _ = _run({"graph": TWO_NODES, "params": {}, "inputs": {},
                       "outputs": [{"name": "kept", "from": {"node": "picked"}}],
                       "resultPath": "lab/p/results/j1"})
    assert sorted(fake_bench) == ["lab/p/results/j1/kept", "lab/p/results/j1/nodes/grid"]
    # The manifest speaks in output names, and says which node each is.
    assert list(payload["outputs"]) == ["kept"]
    assert payload["output_nodes"] == {"kept": "picked"}
    # Lineage follows the stored paths, wherever they now are.
    assert fake_bench["lab/p/results/j1/kept"]["inputs"] == ["lab/p/results/j1/nodes/grid"]


def test_renaming_a_node_moves_no_result(fake_bench):
    renamed = {**TWO_NODES,
               "nodes": [TWO_NODES["nodes"][0], {**TWO_NODES["nodes"][1], "id": "chosen"}],
               "edges": [{"from": {"node": "grid"}, "to": {"node": "chosen", "port": "records"}}]}
    _run({"graph": renamed, "params": {}, "inputs": {},
          "outputs": [{"name": "kept", "from": {"node": "chosen"}}], "resultPath": "lab/p/results/j2"})
    assert "lab/p/results/j2/kept" in fake_bench


def test_an_intermediate_can_be_an_output_too_and_a_node_can_have_two_names(fake_bench):
    payload, _ = _run({"graph": TWO_NODES, "params": {}, "inputs": {},
                       "outputs": [{"name": "all", "from": {"node": "grid"}},
                                   {"name": "kept", "from": {"node": "picked"}},
                                   {"name": "also", "from": {"node": "picked"}}],
                       "resultPath": "lab/p/results/j3"})
    assert sorted(fake_bench) == ["lab/p/results/j3/all", "lab/p/results/j3/also", "lab/p/results/j3/kept"]
    assert sorted(payload["outputs"]) == ["all", "also", "kept"]


def test_an_output_must_name_a_node_and_not_the_intermediates_folder(fake_bench):
    with pytest.raises(ValueError, match="output 'kept' comes from 'gone'"):
        _run({"graph": TWO_NODES, "params": {}, "inputs": {},
              "outputs": [{"name": "kept", "from": {"node": "gone"}}]})
    with pytest.raises(ValueError, match="cannot be named 'nodes'"):
        _run({"graph": TWO_NODES, "params": {}, "inputs": {},
              "outputs": [{"name": "nodes", "from": {"node": "picked"}}]})


def test_a_legacy_protocol_keeps_its_terminals_under_their_ids(fake_bench):
    legacy = {"nodes": TWO_NODES["nodes"],
              "edges": [{"from": {"node": "grid", "port": "out"}, "to": {"node": "picked", "port": "records"}}]}
    payload, _ = _run({"graph": legacy, "bindings": {}, "resultPath": "lab/p/results/j4"})
    assert sorted(fake_bench) == ["lab/p/results/j4/grid", "lab/p/results/j4/picked"]
    assert list(payload["outputs"]) == ["picked"] and "output_nodes" not in payload
