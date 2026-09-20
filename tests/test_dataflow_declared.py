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


def _run(extra, hooks=None, resume=None):
    hooks = hooks or _Hooks()
    out = hooks.executor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra=extra),
                               resume=resume)
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


def test_a_ref_inside_a_map_body_is_judged_by_the_body_nodes_op(fake_bench):
    """A map's body is a graph; its nodes' $refs sit on THEIR declarations,
    not on records/map's (000598). On a body node's inputs a $ref is a
    port, and a port always takes one."""
    body = {"nodes": [
        {"id": "say", "block": "text/measure",
         "params": {"mode": "items", "measures": {"$ref": {"bench": "lab/p/freqs"}}},
         "inputs": {"documents": {"$ref": {"bench": "lab/p/draws"}}}}]}
    nodes = {"each": {"block": "records/map", "params": {"body": body, "bind": {}}}}
    with pytest.raises(ValueError, match=r"each\.body\.nodes\.0\.params\.measures: a \$ref sits where text/measure declares no") as err:
        dataflow.check_refs(nodes, {})
    # The port alone is never a problem: one refusal, and it names measures.
    assert "documents" not in str(err.value)
    del body["nodes"][0]["params"]["measures"]
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


# --- eager discard: keep: outputs (000561) --------------------------------

class _Keeping(_Hooks):
    """Hooks that also spool held results, as the runner does."""

    def __init__(self):
        super().__init__()
        self.held: dict[str, tuple[str, object]] = {}
        self.done: dict[str, object] = {}

    def executor(self):
        return ProtocolExecutor(
            on_node_start=lambda nid, fp: self.fingerprints.__setitem__(nid, fp),
            on_node_kept=lambda nid, fp, result: self.held.__setitem__(nid, (fp, result)),
            on_node_done=lambda nid, path, fp: self.done.__setitem__(nid, path),
        )


def test_keep_outputs_emits_only_the_declared_outputs_and_cites_the_rest_by_hash(fake_bench):
    payload, hooks = _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "outputs",
                           "outputs": [{"name": "kept", "from": {"node": "picked"}}],
                           "resultPath": "lab/p/results/j5"}, _Keeping())
    # The API received the output and nothing else.
    assert sorted(fake_bench) == ["lab/p/results/j5/kept"]
    # The intermediate went to the device's spool, under its fingerprint.
    assert set(hooks.held) == {"grid"} and hooks.held["grid"][0] == hooks.fingerprints["grid"]
    assert "grid" not in hooks.done
    # Its consumer's lineage cites it by content hash — a path form of
    # its own — and the manifest records every node's hash and inputs, so
    # the result verifies without the bytes.
    grid_hash = payload["node_hashes"]["grid"]
    assert fake_bench["lab/p/results/j5/kept"]["inputs"] == [f"~hash/sha256:{grid_hash}"]
    assert payload["node_inputs"] == {"grid": [], "picked": ["grid"]}
    assert payload["nodes_held"] == ["grid"] and payload["keep"] == "outputs"
    assert "grid" not in payload["node_paths"]
    # Inspection keeps the summaries either way.
    assert set(payload["node_summaries"]) == {"grid", "picked"}


def test_a_discard_mode_run_resumes_from_the_held_result_to_the_same_bytes(fake_bench):
    first, hooks = _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "outputs",
                         "outputs": [{"name": "kept", "from": {"node": "picked"}}],
                         "resultPath": "lab/p/results/j6"}, _Keeping())
    fp, result = hooks.held["grid"]
    # Interrupted after grid, resumed on the same device: grid is not run
    # again (no second spool write), and the output is byte-identical.
    again = _Keeping()
    second, again = _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "outputs",
                          "outputs": [{"name": "kept", "from": {"node": "picked"}}],
                          "resultPath": "lab/p/results/j6"}, again,
                         resume={"grid": {"fingerprint": fp, "held": result}})
    assert again.held == {}
    assert dump_canonical(second["outputs"]) == dump_canonical(first["outputs"])
    assert second["node_hashes"] == first["node_hashes"]
    # A held result under a changed fingerprint is not trusted: the node
    # runs again and is spooled again.
    third = _Keeping()
    _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "outputs",
          "outputs": [{"name": "kept", "from": {"node": "picked"}}],
          "resultPath": "lab/p/results/j6"}, third,
         resume={"grid": {"fingerprint": "stale", "held": result}})
    assert set(third.held) == {"grid"}


def test_a_failed_discard_mode_run_stores_its_held_intermediates(fake_bench):
    failing = {**TWO_NODES,
               "nodes": [TWO_NODES["nodes"][0],
                         {"id": "picked", "block": "records/select", "params": {"where": "not a mapping"}}]}
    with pytest.raises(Exception):
        _run({"graph": failing, "params": {}, "inputs": {}, "keep": "outputs",
              "outputs": [{"name": "kept", "from": {"node": "picked"}}],
              "resultPath": "lab/p/results/j7"}, _Keeping())
    # The evidence is where a kept run would have stored it.
    assert sorted(fake_bench) == ["lab/p/results/j7/nodes/grid"]


def test_keep_takes_two_words(fake_bench):
    with pytest.raises(ValueError, match="keep must be 'all' or 'outputs'"):
        _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "some",
              "outputs": [{"name": "kept", "from": {"node": "picked"}}]})


def test_a_map_over_plain_values_needs_no_corpus(fake_bench):
    """`over` is the other way to give a map its stream (000603): the
    values become one-field records, and the name `as` gives them is the
    body's own `$param`."""
    # The body's own records are literal; what varies is the `$param`
    # the map binds per value.
    body = {"nodes": [{"id": "say", "block": "records/fill",
                       "params": {"templates": {"note": "layer {layer}"}},
                       "inputs": {"records": [{"id": "x", "coords": {}, "values": {}}]}}],
            "edges": []}
    graph = {"dataflow": 2, "nodes": [
        {"id": "sweep", "block": "records/map",
         "params": {"over": {"range": [0, 3]}, "as": "layer", "body": body},
         "inputs": {}}], "edges": []}
    payload, _ = _run({"graph": graph, "params": {}, "inputs": {}})
    items = payload["outputs"]["sweep"]["items"]
    assert [it["coords"]["layer"] for it in items] == [0, 1, 2]
    assert [it["coords"]["mapped"] for it in items] == ["0", "1", "2"]


def test_a_map_takes_one_stream_or_the_other(fake_bench):
    body = {"nodes": [{"id": "n", "block": "records/rename", "params": {"fields": {}}}],
            "edges": []}
    graph = {"dataflow": 2, "nodes": [
        {"id": "sweep", "block": "records/map",
         "params": {"over": [1, 2], "body": body},
         "inputs": {"records": [{"id": "r"}]}}], "edges": []}
    with pytest.raises(ValueError, match="not both"):
        _run({"graph": graph, "params": {}, "inputs": {}})
