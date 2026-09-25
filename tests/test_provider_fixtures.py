from __future__ import annotations

import json
import pathlib
import runpy

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "dump_provider_fixtures.py"
FIXTURES = HERE / "fixtures" / "providers"

BUILT = runpy.run_path(str(SCRIPT))["build"]()


def test_every_case_is_committed_and_nothing_else_is() -> None:
    committed = sorted(p.stem for p in FIXTURES.glob("*.json"))
    assert committed == sorted(BUILT), (
        "tests/fixtures/providers is stale: run\n"
        "  python scripts/dump_provider_fixtures.py")


@pytest.mark.parametrize("name", sorted(BUILT))
def test_the_adapters_reproduce_the_committed_fixture(name: str) -> None:
    committed = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(BUILT[name])), (
        f"{name}.json no longer matches what the adapters send and read: run\n"
        "  python scripts/dump_provider_fixtures.py\n"
        "and make mechbench-api's thread providers agree with the new file")


def test_the_fixtures_cover_each_wire_protocol_and_each_ending() -> None:
    apis = {(f["provider"], f["api"]) for f in BUILT.values()}
    assert {("anthropic", None), ("gemini", None), ("deepseek", None),
            ("openai", "responses"), ("xai", "responses")} <= apis
    endings = {c["expected"]["ending"] for f in BUILT.values() for c in f["calls"]}
    assert {"end", "max_tokens", "tool_call", "empty"} <= endings
    kinds = {p["type"] for f in BUILT.values() for c in f["calls"]
             for p in c["expected"]["parts"]}
    assert kinds == {"text", "reasoning", "tool_call"}
