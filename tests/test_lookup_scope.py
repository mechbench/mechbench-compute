from __future__ import annotations

import pytest

from mechbench_compute import tools as T
from mechbench_compute.ops.tools.lookup import (
    LookupRefused,
    check_lookup_path,
    fetch_bench_object,
    read_run_project,
)
from mechbench_compute.providers import messages as pm
from mechbench_compute.sandbox_session import SandboxSession


def call(name, **arguments):
    return pm.ToolCallPart(id="c1", name=name, arguments=arguments)


def lookup_box(project=None, **params):
    seen: list[str] = []
    box = T.Toolbox([{
        "name": "bench.lookup",
        "handler": {"block": "tools/lookup",
                    "params": {"fetch": lambda path: seen.append(path) or {"payload": {"p": path}},
                               **params}},
    }], project=project)
    return box, seen


class TestAModelReadsOnlyWhereTheRunMayRead:
    def test_the_runs_own_project_is_open(self):
        box, seen = lookup_box("benji/lab")
        out = box.call(call("bench.lookup", path="benji/lab/results/j_1/stats"))
        assert not out.is_error and seen == ["benji/lab/results/j_1/stats"]

    @pytest.mark.parametrize("path", [
        "alice/private/notes", "benji/lab-other/x", "benji/labx", "benji",
    ])
    def test_another_project_is_refused_before_any_fetch(self, path):
        box, seen = lookup_box("benji/lab")
        out = box.call(call("bench.lookup", path=path))
        assert out.is_error and "LookupRefused" in out.content and seen == []

    def test_a_granted_prefix_opens_it(self):
        box, seen = lookup_box("benji/lab", prefixes=["benji/stories/word-lists"])
        assert not box.call(call("bench.lookup", path="benji/stories/word-lists/a")).is_error
        assert box.call(call("bench.lookup", path="benji/stories/other")).is_error
        assert seen == ["benji/stories/word-lists/a"]

    def test_without_a_project_or_a_grant_nothing_is_open(self):
        box, seen = lookup_box(None)
        out = box.call(call("bench.lookup", path="benji/lab/x"))
        assert out.is_error and "this run has none" in out.content and seen == []

    def test_a_tool_definition_cannot_name_its_own_project(self):
        box, seen = lookup_box("benji/lab", _project="alice/private")
        assert box.call(call("bench.lookup", path="alice/private/notes")).is_error
        assert seen == []

    @pytest.mark.parametrize("path", [
        "benji/lab/../../alice/private/x", "benji/lab/./x", "benji/lab/~lineage",
        "benji/lab/x?path=alice/private", "benji/lab/x#y", "benji/lab/%2e%2e/x",
        "benji/lab//x", "benji/lab/x y", "benji/lab/x\\..\\y",
    ])
    def test_a_path_that_could_leave_the_prefix_is_refused(self, path):
        with pytest.raises(LookupRefused):
            check_lookup_path(path, ["benji/lab"])

    def test_the_protocols_own_path_needs_no_grant_but_must_be_plain(self):
        got = fetch_bench_object({}, {"path": "alice/x/y", "fetch": lambda p: {"payload": p}})
        assert got == "alice/x/y"
        with pytest.raises(LookupRefused):
            fetch_bench_object({}, {"path": "alice/../y", "fetch": lambda p: p})

    @pytest.mark.parametrize("base, project", [
        ("benji/lab/results/j_1", "benji/lab"), (None, None), ("benji", None), ("/x", None),
    ])
    def test_the_run_project_is_the_result_paths_first_two_segments(self, base, project):
        assert read_run_project(base) == project


class TestASandboxToolDispatchesOnlyItsTools:
    def test_a_handler_naming_another_session_attribute_is_refused(self):
        box = T.Toolbox([{"name": "sneak", "handler": {"sandbox": "_run"}}],
                        session=SandboxSession())
        out = box.call(call("sneak", argv=["sh"], tool="bash"))
        assert out.is_error and "no method '_run'" in out.content


class TestTheChatLoopCarriesTheRunProject:
    def test_the_remote_loop_hands_its_project_to_the_toolbox(self):
        from mechbench_compute import chat as chat_mod
        from mechbench_compute import model_ref as mr

        lookup = {"name": "bench.lookup",
                  "schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                  "handler": {"block": "tools/lookup",
                              "params": {"fetch": lambda path: {"payload": path}}}}
        params = {"model": {"provider": "mock", "model": "mock-large"}, "budget_usd": 1.0,
                  "tools": [lookup], "max_tool_rounds": 1, "_project": "benji/lab",
                  "records": [{"id": "r0", "user": "look it up"}],
                  "provider_options": {"mock": {"tool_call": "bench.lookup"}}}
        out = chat_mod.run_remote(mr.parse(params["model"]), params["records"], params)
        run = out["items"][0]["metadata"]["tool_runs"][0]
        assert "reads only under benji/lab" in run["error"]
