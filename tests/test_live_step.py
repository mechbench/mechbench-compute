from __future__ import annotations

import pytest

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.live.run_step import run_step
from mechbench_compute.protocol import ProtocolExecutor

GRAPH = {
    "nodes": [{"id": "said", "block": "records/union", "params": {}, "inputs": {}}],
    "edges": [
        {"from": {"input": "state"}, "to": {"node": "said", "port": "before"}},
        {"from": {"input": "event"}, "to": {"node": "said", "port": "now"}},
    ],
}
OUTPUTS = [{"name": "state", "from": {"node": "said"}}]


def _step(executor, state, text, seq):
    return run_step(executor, graph=GRAPH, params={}, outputs=OUTPUTS,
                    event={"id": f"e{seq}", "type": "message", "text": text}, state=state)


def test_each_event_is_one_run_of_the_handler_and_its_state_carries_on():
    ex = ProtocolExecutor()
    state = K.collection("records/record", [])
    first = _step(ex, state, "hello", 1)
    second = _step(ex, first["state"], "again", 2)
    texts = [it.get("text") for it in K.items_of(second["state"])]
    assert sorted(texts) == ["again", "hello"]
    assert second["outputs"]["state"] is second["state"]


def test_a_handler_without_a_state_output_is_refused():
    with pytest.raises(ValueError, match="state"):
        run_step(ProtocolExecutor(), graph=GRAPH, params={}, outputs=[{"name": "said", "from": {"node": "said"}}],
                 event={"id": "e1", "type": "message"}, state=K.collection("records/record", []))


def test_an_event_without_a_type_is_refused():
    with pytest.raises(ValueError, match="type"):
        run_step(ProtocolExecutor(), graph=GRAPH, params={}, outputs=OUTPUTS,
                 event={"id": "e1"}, state=K.collection("records/record", []))


def test_a_replay_folds_the_handler_over_the_events_and_applies_param_changes_between():
    from mechbench_compute.live.replay import replay

    events = [{"id": "e1", "type": "message", "text": "hello"},
              {"id": "e2", "type": "param", "name": "strength", "value": 2.0},
              {"id": "e3", "type": "message", "text": "again"}]
    seen = []
    out = replay(ProtocolExecutor(), graph=GRAPH, params={"strength": 0.0}, outputs=OUTPUTS,
                 events=events, state=K.collection("records/record", []), on_step=lambda i, s: seen.append(i))
    assert seen == [0, 1, 2]
    assert out["params"] == {"strength": 2.0}
    assert out["steps"][1]["params"] == {"strength": 2.0}
    assert sorted(it.get("text") for it in K.items_of(out["state"])) == ["again", "hello"]


def test_a_param_change_to_a_param_the_run_does_not_bind_is_refused():
    from mechbench_compute.live.replay import replay

    with pytest.raises(ValueError, match="does not bind"):
        replay(ProtocolExecutor(), graph=GRAPH, params={}, outputs=OUTPUTS,
               events=[{"id": "e1", "type": "param", "name": "nope", "value": 1}],
               state=K.collection("records/record", []))


def test_a_replay_is_a_run_kind_the_runner_can_claim():
    from mechbench_compute.protocol import ProtocolSpec

    progress = []
    out = ProtocolExecutor().run(
        ProtocolSpec(kind="replay", prompt="", model_id=None, extra={
            "graph": GRAPH, "params": {}, "outputs": OUTPUTS,
            "events": [{"id": "e1", "type": "message", "text": "hello"}],
            "state": K.collection("records/record", [])}),
        on_progress=lambda done, total: progress.append((done, total)))
    assert out["kind"] == "run/replay" and progress == [(1, 1)]


def test_a_step_is_given_the_live_runs_other_inputs_beside_its_event_and_state():
    graph = {
        "nodes": [{"id": "said", "block": "records/union", "params": {}, "inputs": {}}],
        "edges": [
            {"from": {"input": "state"}, "to": {"node": "said", "port": "before"}},
            {"from": {"input": "event"}, "to": {"node": "said", "port": "now"}},
            {"from": {"input": "notes"}, "to": {"node": "said", "port": "notes"}},
        ],
    }
    got = run_step(ProtocolExecutor(), graph=graph, params={}, outputs=OUTPUTS,
                   event={"id": "e1", "type": "message", "text": "hi"}, state=K.collection("records/record", []),
                   inputs={"notes": [{"id": "n1", "text": "a note"}]})
    assert sorted(it.get("text") for it in K.items_of(got["state"])) == ["a note", "hi"]
