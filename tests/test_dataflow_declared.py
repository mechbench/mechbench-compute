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


def test_a_run_binds_params_and_inputs_by_name(fake_bench):
    out, _ = _run({"graph": DECLARED, "params": {"mode": "items"},
                   "inputs": {"draws": {"$ref": {"bench": "lab/p/draws"}}},
                   "resultPath": "lab/p/results/j_new"})
    said = {r["item"]: r for r in out["outputs"]["said"]["items"]}
    assert said["Mystery"]["count"] == 2 and said["Steampunk Fantasy"]["in_vocabulary"] is False


_NODE = {"id": "said", "block": "text/measure", "params": {"mode": "items"}}


@pytest.mark.parametrize("extra, found", [
    ({"graph": {"nodes": [_NODE], "edges": []}},
     'no "dataflow": 2 marker'),
    ({"graph": {"nodes": [{**_NODE, "params": {"mode": "$mode"}}], "edges": []}},
     'a "$mode" string hole at said.params.mode'),
    ({"graph": {"nodes": [{**_NODE, "inputs": {"documents": {"$fetch": "lab/p/draws"}}}], "edges": []}},
     "$fetch at said.inputs.documents"),
    ({"graph": {"nodes": [{**_NODE, "inputs": {"documents": {"$hf_dataset": {"repo": "r"}}}}], "edges": []}},
     "$hf_dataset at said.inputs.documents"),
    ({"graph": {"nodes": [{"id": "corpus", "block": "protocol-input"}, _NODE], "edges": []}},
     "a protocol-input node 'corpus'"),
    ({"graph": {"nodes": [_NODE, {**_NODE, "id": "again"}],
                "edges": [{"from": {"node": "said", "port": "out"}, "to": {"node": "again", "port": "documents"}}]}},
     "an edge from {node, port} (said.out)"),
    ({"graph": DECLARED, "bindings": {"mode": "items"}},
     "a run bound by `bindings` rather than `params` and `inputs`"),
])
def test_a_graph_in_the_legacy_form_is_refused_by_what_it_carries(fake_bench, extra, found):
    ran = []
    hooks = _Hooks()
    hooks.executor = lambda: ProtocolExecutor(on_node_start=lambda nid, fp: ran.append(nid))
    with pytest.raises(ValueError) as refused:
        _run(extra, hooks)
    assert str(refused.value) == (
        f"this protocol is in the legacy dataflow form ({found}), which is no longer "
        f'read. Write it in the declared form, marked "dataflow": 2: '
        f"see https://docs.mechbench.ai/dataflow/")
    assert ran == [] and fake_bench == {}


def test_every_stored_object_a_node_reads_is_a_lineage_input(fake_bench):
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
    monkeypatch.setattr(lexicon, "resolve", lambda block: block)
    nodes = {"pub": {"block": "test/publish", "params": {"target": {"$ref": {"bench": "lab/p/out"}}}}}
    dataflow.check_refs(nodes, {})
    nodes["pub"]["params"]["label"] = {"$ref": {"bench": "lab/p/out"}}
    with pytest.raises(ValueError, match=r"pub\.label: a \$ref sits where"):
        dataflow.check_refs(nodes, {})


def test_a_ref_inside_a_map_body_is_judged_by_the_body_nodes_op(fake_bench):
    body = {"nodes": [
        {"id": "say", "block": "text/measure",
         "params": {"mode": "items", "measures": {"$ref": {"bench": "lab/p/freqs"}}},
         "inputs": {"documents": {"$ref": {"bench": "lab/p/draws"}}}}]}
    nodes = {"each": {"block": "records/map", "params": {"body": body, "bind": {}}}}
    with pytest.raises(ValueError, match=r"each\.body\.nodes\.0\.params\.measures: a \$ref sits where text/measure declares no") as err:
        dataflow.check_refs(nodes, {})
    assert "documents" not in str(err.value)
    del body["nodes"][0]["params"]["measures"]
    dataflow.check_refs(nodes, {})


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
    assert list(payload["outputs"]) == ["kept"]
    assert payload["output_nodes"] == {"kept": "picked"}
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


def test_a_run_with_no_declared_outputs_keeps_its_terminals_under_their_ids(fake_bench):
    payload, _ = _run({"graph": TWO_NODES, "params": {}, "resultPath": "lab/p/results/j4"})
    assert sorted(fake_bench) == ["lab/p/results/j4/grid", "lab/p/results/j4/picked"]
    assert list(payload["outputs"]) == ["picked"] and "output_nodes" not in payload


class _Keeping(_Hooks):
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
    assert sorted(fake_bench) == ["lab/p/results/j5/kept"]
    assert set(hooks.held) == {"grid"} and hooks.held["grid"][0] == hooks.fingerprints["grid"]
    assert "grid" not in hooks.done
    grid_hash = payload["node_hashes"]["grid"]
    assert fake_bench["lab/p/results/j5/kept"]["inputs"] == [f"~hash/sha256:{grid_hash}"]
    assert payload["node_inputs"] == {"grid": [], "picked": ["grid"]}
    assert payload["nodes_held"] == ["grid"] and payload["keep"] == "outputs"
    assert "grid" not in payload["node_paths"]
    assert set(payload["node_summaries"]) == {"grid", "picked"}


def test_a_discard_mode_run_resumes_from_the_held_result_to_the_same_bytes(fake_bench):
    first, hooks = _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "outputs",
                         "outputs": [{"name": "kept", "from": {"node": "picked"}}],
                         "resultPath": "lab/p/results/j6"}, _Keeping())
    fp, result = hooks.held["grid"]
    again = _Keeping()
    second, again = _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "outputs",
                          "outputs": [{"name": "kept", "from": {"node": "picked"}}],
                          "resultPath": "lab/p/results/j6"}, again,
                         resume={"grid": {"fingerprint": fp, "held": result}})
    assert again.held == {}
    assert dump_canonical(second["outputs"]) == dump_canonical(first["outputs"])
    assert second["node_hashes"] == first["node_hashes"]
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
    assert sorted(fake_bench) == ["lab/p/results/j7/nodes/grid"]


def test_keep_takes_two_words(fake_bench):
    with pytest.raises(ValueError, match="keep must be 'all' or 'outputs'"):
        _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "some",
              "outputs": [{"name": "kept", "from": {"node": "picked"}}]})


def test_a_map_over_plain_values_needs_no_corpus(fake_bench):
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


XS = {
    "kind": "collection", "item_kind": "records/record", "key": ["id"],
    "items": [{"id": f"s{x}", "kind": "records/record", "coords": {"x": x}, "values": {"v": x}}
              for x in ("1", "2")],
}

ZIP_INPUT_AND_EDGE = {
    "dataflow": 2,
    "nodes": [
        TWO_NODES["nodes"][0],
        {"id": "paired", "block": "records/zip", "params": {"by": ["x"]}},
    ],
    "edges": [
        {"from": {"input": "stored"}, "to": {"node": "paired", "port": "branches"}, "index": 0},
        {"from": {"node": "grid"}, "to": {"node": "paired", "port": "branches"}, "index": 1},
    ],
}


def test_a_variadic_port_takes_a_protocol_input_beside_a_node_edge(fake_bench, monkeypatch):
    monkeypatch.setitem(STORE, "lab/p/xs", XS)
    payload, _ = _run({"graph": ZIP_INPUT_AND_EDGE, "params": {},
                       "inputs": {"stored": {"$ref": {"bench": "lab/p/xs"}}},
                       "outputs": [{"name": "paired", "from": {"node": "paired"}}],
                       "resultPath": "lab/p/results/j8"})
    items = payload["outputs"]["paired"]["items"]
    assert [it["coords"]["x"] for it in items] == ["1", "2"]
    assert list(items[0]["branches"]) == ["stored", "grid"]
    assert items[0]["branches"]["stored"]["values"] == {"v": "1"}
    assert fake_bench["lab/p/results/j8/paired"]["inputs"] == [
        "lab/p/results/j8/nodes/grid", "lab/p/xs"]


def test_the_edge_index_orders_inputs_and_node_edges_together(fake_bench, monkeypatch):
    monkeypatch.setitem(STORE, "lab/p/xs", XS)
    swapped = {**ZIP_INPUT_AND_EDGE, "edges": [
        {**ZIP_INPUT_AND_EDGE["edges"][0], "index": 1},
        {**ZIP_INPUT_AND_EDGE["edges"][1], "index": 0}]}
    _first, hooks = _run({"graph": ZIP_INPUT_AND_EDGE, "params": {},
                          "inputs": {"stored": {"$ref": {"bench": "lab/p/xs"}}}})
    second, again = _run({"graph": swapped, "params": {},
                          "inputs": {"stored": {"$ref": {"bench": "lab/p/xs"}}}})
    assert list(second["outputs"]["paired"]["items"][0]["branches"]) == ["grid", "stored"]
    assert hooks.fingerprints["paired"] != again.fingerprints["paired"]


def test_a_variadic_port_takes_two_inputs_and_counts_them_as_edges(fake_bench, monkeypatch):
    monkeypatch.setitem(STORE, "lab/p/xs", XS)
    two_inputs = {**ZIP_INPUT_AND_EDGE, "edges": [
        {"from": {"input": "a"}, "to": {"node": "paired", "port": "branches"}, "index": 0},
        {"from": {"input": "b"}, "to": {"node": "paired", "port": "branches"}, "index": 1}]}
    payload, _ = _run({"graph": two_inputs, "params": {},
                       "inputs": {"a": {"$ref": {"bench": "lab/p/xs"}},
                                  "b": {"$ref": {"bench": "lab/p/xs"}}}})
    assert list(payload["outputs"]["paired"]["items"][0]["branches"]) == ["a", "b"]
    one_input = {**ZIP_INPUT_AND_EDGE, "edges": two_inputs["edges"][:1]}
    with pytest.raises(ValueError, match="takes at least 2 edges; 1 arrive"):
        _run({"graph": one_input, "params": {}, "inputs": {"a": {"$ref": {"bench": "lab/p/xs"}}}})


def test_a_port_that_takes_one_source_still_refuses_an_input_and_an_edge():
    graph = {"dataflow": 2,
             "nodes": [{"id": "a", "block": "records/cross"},
                       {"id": "b", "block": "records/select"}],
             "edges": [{"from": {"input": "recs"}, "to": {"node": "b", "port": "records"}},
                       {"from": {"node": "a"}, "to": {"node": "b", "port": "records"}}]}
    with pytest.raises(ValueError, match="one or the other"):
        dataflow.lower(graph, {"recs": []})


def test_keep_outputs_stores_an_output_that_reads_a_held_intermediate(monkeypatch):
    sent: dict[str, bytes] = {}

    def request(method, url, key, body=None, headers=None, **kw):
        target = url.split("/objects/", 1)[1]
        sent[target] = body
        return {"path": target}

    monkeypatch.setattr(bench, "_config", lambda url, key: ("http://bench.test", "k"))
    monkeypatch.setattr(bench, "_request", request)
    payload, _ = _run({"graph": TWO_NODES, "params": {}, "inputs": {}, "keep": "outputs",
                       "outputs": [{"name": "kept", "from": {"node": "picked"}}],
                       "resultPath": "lab/p/results/j9"}, _Keeping())
    assert sorted(sent) == ["lab/p/results/j9/kept"]
    assert payload["nodes_held"] == ["grid"]
