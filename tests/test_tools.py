"""Tools as blocks (task 000340): the toolbox, the two first tools,
local parsing, and the tool loop in chat and conversation nodes.

Nothing here spends: remote tool calls come from the mock, which emits
a tool call on demand.
"""

from __future__ import annotations

import pytest

from mechbench_compute import chat as chat_mod
from mechbench_compute import conversation as cv
from mechbench_compute import model_ref as mr
from mechbench_compute import tools as T
from mechbench_compute.providers import messages as pm


def call(name, **arguments):
    return pm.ToolCallPart(id="c1", name=name, arguments=arguments)


class TestTheToolbox:
    def test_a_handler_is_an_ordinary_block(self):
        box = T.toolbox_from(["calc"])
        out = box.call(call("calc", expression="2*(3+4)"))
        assert out.content == '{"expression": "2*(3+4)", "result": 14}'
        assert out.is_error is False
        run = box.runs[0]
        assert run.tool == "calc"
        assert run.handler["block"] == "~canonical/ops/tools/calc/1"

    def test_calc_evaluates_arithmetic_and_refuses_code(self):
        box = T.toolbox_from(["calc"])
        bad = box.call(call("calc", expression="__import__('os').system('ls')"))
        assert bad.is_error is True
        assert "refuses Call" in bad.content
        # …and the refusal is recorded, because what a model tried is
        # part of what happened.
        assert box.runs[0].error.startswith("CalcRefused")

    def test_an_unknown_tool_is_an_answer_the_model_can_read(self):
        box = T.toolbox_from(["calc"])
        out = box.call(call("rm_rf", path="/"))
        assert out.is_error is True
        assert "no such tool" in out.content and "calc" in out.content

    def test_a_handler_that_raises_becomes_an_error_result(self):
        box = T.toolbox_from(["calc"])
        out = box.call(call("calc"))          # no expression
        assert out.is_error and "expression" in out.content

    def test_bench_lookup_consults_the_bench_through_an_injected_fetch(self):
        box = T.Toolbox([{
            "name": "bench.lookup",
            "schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "handler": {"block": "~canonical/ops/tools/bench-lookup/1",
                        "params": {"fetch": lambda path: {
                            "payload": {"path": path, "kind": "metric_table",
                                        "rows": [{"n": 3}]}}}},
        }])
        out = box.call(call("bench.lookup", path="benji/lab/results/j_1/stats"))
        assert '"kind": "metric_table"' in out.content
        assert box.runs[0].error == ""

    def test_a_non_pure_handler_says_it_needs_the_executor(self):
        box = T.Toolbox([{"name": "read", "handler": {
            "block": "~canonical/ops/decision-read/1"}}])
        out = box.call(call("read"))
        assert out.is_error and "executor's runner" in out.content

    def test_the_executors_runner_takes_the_handlers_it_owns(self):
        seen = {}

        def runner(ref, inputs, params):
            seen.update(ref=ref, inputs=inputs, params=params)
            return {"conditions": [{"id": "a"}]}

        box = T.Toolbox([{"name": "read", "handler": {
            "block": "~canonical/ops/decision-read/1", "params": {"model": "$model"}}}],
            block_runner=runner)
        out = box.call(call("read", prompt="hi"))
        assert not out.is_error
        assert seen["ref"] == "~canonical/ops/decision-read/1"
        # The arguments arrive on their own port AND as one record, so
        # an ordinary record block works as a tool unmodified.
        assert seen["inputs"]["arguments"] == {"prompt": "hi"}
        assert seen["inputs"]["records"] == [{"prompt": "hi"}]
        assert seen["params"] == {"model": "$model"}

    def test_a_bad_definition_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="needs a name"):
            T.Toolbox([{"description": "nameless"}])
        with pytest.raises(ValueError, match="handler is"):
            T.Toolbox([{"name": "x", "handler": {"lambda": "nope"}}])
        with pytest.raises(ValueError, match="unique"):
            T.Toolbox([{"name": "x"}, {"name": "x"}])
        with pytest.raises(ValueError, match="unknown built-in tool"):
            T.toolbox_from(["telepathy"])


class TestLocalParsing:
    def test_a_fenced_call_is_read_and_removed(self):
        box = T.toolbox_from(["calc"])
        text, calls = T.parse_tool_calls(
            'Working.\n```tool_code\n{"name": "calc", "arguments": '
            '{"expression": "6*7"}}\n```\nOne moment.', tools=box.tools)
        assert text == "Working.\n\nOne moment."
        assert calls[0].name == "calc" and calls[0].arguments == {"expression": "6*7"}

    def test_single_quotes_still_meant_a_call(self):
        box = T.toolbox_from(["calc"])
        _, calls = T.parse_tool_calls(
            "```json\n{'name': 'calc', 'arguments': {'expression': '1+1'}}\n```",
            tools=box.tools)
        assert calls[0].arguments == {"expression": "1+1"}

    def test_a_call_to_a_tool_that_does_not_exist_is_left_as_prose(self):
        box = T.toolbox_from(["calc"])
        text, calls = T.parse_tool_calls(
            '```tool_code\n{"name": "launch_missiles", "arguments": {}}\n```',
            tools=box.tools)
        assert calls == [] and "launch_missiles" in text

    def test_arguments_may_be_written_flat(self):
        box = T.toolbox_from(["calc"])
        _, calls = T.parse_tool_calls(
            '{"name": "calc", "expression": "8/2"}', tools=box.tools)
        assert calls[0].arguments == {"expression": "8/2"}

    def test_the_tools_are_described_where_a_local_model_can_see_them(self):
        box = T.toolbox_from(["calc", "bench.lookup"])
        text = T.render_tools(box.tools)
        assert "calc(expression)" in text
        assert "bench.lookup(path, field)" in text
        assert "```tool_code" in text

    def test_an_unknown_family_is_refused(self):
        with pytest.raises(ValueError, match="unknown tool family"):
            T.parse_tool_calls("hi", family="smoke-signals")


class TestTheRemoteToolLoop:
    def _params(self, **kw):
        base = {
            "model": {"provider": "mock", "model": "mock-large"},
            "budget_usd": 1.0,
            "tools": ["calc"],
            "records": [{"id": "r0", "user": "what is 6*7?"}],
            # The mock emits a tool call when asked to.
            "provider_options": {"mock": {"tool_call": "calc"}},
        }
        base.update(kw)
        return base

    def test_a_model_that_calls_a_tool_gets_its_result_and_answers(self):
        # A tool that succeeds whatever the mock fabricates, so this
        # tests the LOOP rather than the mock's arithmetic.
        lookup = {
            "name": "bench.lookup",
            "schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "handler": {"block": "~canonical/ops/tools/bench-lookup/1",
                        "params": {"fetch": lambda path: {"payload": {"rows": 3}}}},
        }
        params = self._params(max_tool_rounds=1, tools=[lookup],
                              provider_options={"mock": {"tool_call": "bench.lookup"}})
        out = chat_mod.run_remote(mr.parse(params["model"]),
                                  params["records"], params)
        runs = out["items"][0]["metadata"]["tool_runs"]
        assert len(runs) == 1 and runs[0]["tool"] == "bench.lookup"
        assert "error" not in runs[0]        # the handler really ran
        # Two calls: the one that asked for the tool, and the one after
        # the result went back. Both metered.
        assert out["spend"]["calls"] == 2

    def test_a_tool_that_fails_still_comes_back_as_an_answer(self):
        # calc gets whatever the mock invents, which is not arithmetic:
        # the model is told so, and the run is recorded as an error.
        params = self._params(max_tool_rounds=1)
        out = chat_mod.run_remote(mr.parse(params["model"]),
                                  params["records"], params)
        run = out["items"][0]["metadata"]["tool_runs"][0]
        assert run["tool"] == "calc" and run["error"].startswith("CalcRefused")
        assert out["items"][0]["text"]      # the run continued regardless

    def test_the_loop_is_bounded(self):
        # The mock asks for a tool every time; max_tool_rounds is what
        # stops a model and its tools talking forever at your expense.
        out = chat_mod.run_remote(
            mr.parse({"provider": "mock", "model": "mock-large"}),
            [{"id": "r0", "user": "loop"}],
            self._params(max_tool_rounds=1))
        assert out["spend"]["calls"] == 2
        out2 = chat_mod.run_remote(
            mr.parse({"provider": "mock", "model": "mock-large"}),
            [{"id": "r0", "user": "loop"}],
            self._params(max_tool_rounds=3))
        assert out2["spend"]["calls"] == 4

    def test_without_tools_nothing_changes(self):
        out = chat_mod.run_remote(
            mr.parse({"provider": "mock", "model": "mock-large"}),
            [{"id": "r0", "user": "hi"}],
            {"model": {"provider": "mock", "model": "mock-large"},
             "budget_usd": 1.0})
        assert out["spend"]["calls"] == 1
        assert "tool_runs" not in out["items"][0]["metadata"]


class TestToolsInAConversation:
    def test_a_participant_may_carry_tools_and_the_transcript_records_them(self):
        out = cv.run({
            "participants": [
                {"name": "asker", "model": {"provider": "mock", "model": "m"},
                 "tools": ["calc"],
                 "provider_options": {"mock": {"tool_call": "calc"}}},
                {"name": "other", "model": {"provider": "mock", "model": "m"}},
            ],
            "opening": ["What is 6*7?"],
            "turns": {"policy": "round_robin", "max_turns": 2},
            "budget_usd": 1.0,
        })
        messages = out["items"][0]["metadata"]["transcript"]["messages"]
        asker = next(m for m in messages if m["participant"] == "asker")
        assert asker["call"]["tool_runs"][0]["tool"] == "calc"
        # The room saw an answer, not the plumbing.
        assert all("tool_code" not in t["text"] for t in out["items"][0]["turns"])


class TestThroughTheExecutor:
    def test_decision_read_is_available_as_a_tool(self, monkeypatch):
        """A model that can consult another model mid-turn (task
        000340's third first-tool): the handler is a MODEL block, which
        only the executor can run."""
        from mechbench_compute.protocol import ProtocolExecutor

        ex = ProtocolExecutor()
        seen = {}

        def fake_model_block(fn, inputs, params, *a, **kw):
            seen.update(inputs=inputs, params=params)
            return {"conditions": [{"id": "q", "entropy_bits": 0.9}]}

        monkeypatch.setattr(ex, "_run_model_block", fake_model_block)
        runner = ex._tool_block_runner()
        out = runner("~canonical/ops/decision-read/1",
                     {"arguments": {"prompt": "left or right?"},
                      "records": [{"prompt": "left or right?"}]},
                     {"model": "google/gemma-3-4b-it"})
        assert out["conditions"][0]["entropy_bits"] == 0.9
        assert seen["params"]["model"] == "google/gemma-3-4b-it"

    def test_a_block_that_is_not_a_tool_handler_says_so(self):
        from mechbench_compute.protocol import ProtocolExecutor

        runner = ProtocolExecutor()._tool_block_runner()
        with pytest.raises(ValueError, match="not available as a tool handler"):
            runner("~canonical/ops/finetune/lora/1", {}, {})

    def test_a_chat_node_with_tools_runs_end_to_end(self):
        from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

        graph = {"nodes": [{
            "id": "ask", "block": "~canonical/ops/chat/1",
            "params": {
                "model": {"provider": "mock", "model": "mock-large"},
                "budget_usd": 1.0,
                "tools": ["calc"],
                "max_tool_rounds": 1,
                "provider_options": {"mock": {"tool_call": "calc"}},
                "records": [{"id": "r0", "user": "what is 6*7?"}],
            }}], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
        item = out.payload["outputs"]["ask"]["items"][0]
        assert item["metadata"]["tool_runs"][0]["tool"] == "calc"
        # The injected runner never reaches the node's recorded params.
        assert "_block_runner" not in str(out.payload["nodes_executed"])
