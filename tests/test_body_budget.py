from __future__ import annotations

import pytest

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.providers.errors import BudgetExceeded

RECORDS = [{"id": f"r{i}", "user": word} for i, word in enumerate(["dusk", "kettle", "harbor", "lantern", "salt"])]

MOCK = {"provider": "mock", "model": "mock-large"}


def chat(nid):
    return {"id": nid, "block": "text/chat", "params": {"model": MOCK, "budget_usd": 1.0, "max_tokens": 4}}


BODY = {"nodes": [
    {"id": "ask", "block": "records/cross",
     "params": {"factors": [{"name": "user", "levels": [{"key": {"$param": "topic"}}]}]}},
    chat("first"), chat("second"),
], "edges": [
    {"from": {"node": "ask"}, "to": {"node": "first", "port": "records"}},
    {"from": {"node": "ask"}, "to": {"node": "second", "port": "records"}},
]}

FREE_BODY = {"nodes": [
    {"id": "write", "block": "records/derive", "params": {"templates": {"text": "{user}!"}}},
], "edges": [{"from": {"input": "record"}, "to": {"node": "write", "port": "records"}}]}


def run_map(cap, body=BODY):
    params = {"body": body, **({"budget_usd": cap} if cap is not None else {})}
    if body is BODY:
        params.update(bind={"topic": "user"}, output="first")
    graph = {"dataflow": 2, "nodes": [
        {"id": "each", "block": "records/map", "params": params, "inputs": {"records": RECORDS}}], "edges": []}
    return ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                               extra={"graph": graph})).payload["outputs"]["each"]


@pytest.fixture(scope="module")
def per_call():
    out = run_map(100.0)
    assert out["budget_usd"] == 100.0
    return out["spent_usd"] / (2 * len(RECORDS))


class TestTheMapsCap:
    def test_the_header_carries_the_spend_and_the_cap(self, per_call):
        out = run_map(100.0)
        assert out["spent_usd"] == pytest.approx(2 * len(RECORDS) * per_call)
        assert per_call > 0 and len(K.items_of(out)) == len(RECORDS)

    def test_a_cap_that_admits_three_invocations_refuses_the_fourth_by_record_id(self, per_call):
        cap = 6.5 * per_call
        with pytest.raises(BudgetExceeded) as caught:
            run_map(cap)
        said = str(caught.value)
        assert said.startswith(f"records/map 'each' stopped at record 'r3': budget cap ${cap:.4f} would be "
                               f"exceeded (mock/mock-large): ${6 * per_call:.4f} already spent across the body's "
                               f"calls and this call's worst case is ${per_call:.4f}.")
        assert said.endswith("Raise budget_usd on the map, or lower the body's max_tokens.")

    def test_a_body_node_s_own_cap_still_bounds_its_calls(self, per_call):
        body = {**BODY, "nodes": [BODY["nodes"][0], chat("first"),
                                  {**chat("second"), "params": {**chat("second")["params"],
                                                                "budget_usd": per_call / 2}}]}
        graph = {"dataflow": 2, "nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": body, "bind": {"topic": "user"}, "output": "first", "budget_usd": 100.0},
             "inputs": {"records": RECORDS}}], "edges": []}
        with pytest.raises(BudgetExceeded, match=r"^budget cap .* Raise budget_usd on the node"):
            ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))

    def test_a_body_with_no_provider_call_ignores_the_cap(self):
        out = run_map(1e-9, body=FREE_BODY)
        assert [i["text"] for i in K.items_of(out)] == [f"{r['user']}!" for r in RECORDS]
        assert out["spent_usd"] == 0.0 and out["budget_usd"] == 1e-9

    def test_no_cap_leaves_the_header_as_it_was(self):
        out = run_map(None)
        assert "spent_usd" not in out and "budget_usd" not in out


class TestTheFoldsCap:
    def fold(self, cap):
        body = {"nodes": [
            {"id": "pass", "block": "records/derive", "params": {"templates": {"user": "{user}"}}},
            chat("say"),
        ], "edges": [
            {"from": {"input": "state"}, "to": {"node": "pass", "port": "records"}},
            {"from": {"input": "state"}, "to": {"node": "say", "port": "records"}},
        ]}
        graph = {"dataflow": 2, "nodes": [
            {"id": "loop", "block": "records/fold",
             "params": {"body": body, "steps": 4, "output": "pass", "budget_usd": cap},
             "inputs": {"state": K.collection("records/record", RECORDS[:1])}}], "edges": []}
        return ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                                   extra={"graph": graph})).payload["outputs"]["loop"]

    def test_the_fold_refuses_the_step_past_its_cap(self):
        out = self.fold(100.0)
        step = out["spent_usd"] / 4
        assert out["budget_usd"] == 100.0 and out["folded"]["steps"] == 4
        with pytest.raises(BudgetExceeded, match=r"^records/fold 'loop' stopped at step 2: budget cap"):
            self.fold(2.5 * step)
