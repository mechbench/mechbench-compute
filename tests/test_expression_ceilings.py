from __future__ import annotations

import pytest

from mechbench_compute.expr.engine import (
    ExprError,
    describe_trap,
    load_engine,
    read_ceiling,
)


def grown(times: int) -> str:
    expr = "params.s"
    for _ in range(times):
        expr = f'({expr}).replace("x", "xxxxxxxxxx")'
    return expr


class TestTheRunnerBoundsAnExpression:
    def test_a_string_that_outgrows_the_memory_ceiling_is_refused(self, monkeypatch):
        monkeypatch.setenv("MECHBENCH_EXPR_CEILING_MEMORY_MB", "64")
        with pytest.raises(ExprError, match="64 MB of memory") as caught:
            load_engine().evaluate(grown(10), params={"s": "x"})
        assert caught.value.kind == "limit"

    def test_within_the_ceiling_the_answer_is_unchanged(self, monkeypatch):
        monkeypatch.setenv("MECHBENCH_EXPR_CEILING_MEMORY_MB", "64")
        got = load_engine().evaluate(grown(3), params={"s": "ax"})
        assert got.values == ["a" + "x" * 1000]

    def test_the_engine_still_answers_after_a_refusal(self, monkeypatch):
        monkeypatch.setenv("MECHBENCH_EXPR_CEILING_MEMORY_MB", "64")
        with pytest.raises(ExprError):
            load_engine().evaluate(grown(10), params={"s": "x"})
        assert load_engine().evaluate("1 + 2").values == [3]

    def test_an_interrupt_is_named_as_time(self, monkeypatch):
        monkeypatch.setenv("MECHBENCH_EXPR_CEILING_SECONDS", "5")
        said = describe_trap(RuntimeError("wasm trap: interrupt"))
        assert said == "the expression ran past this runner's 5 seconds"

    @pytest.mark.parametrize("raw", ["0", "-3", "inf", "nan", "lots"])
    def test_a_bad_ceiling_is_named(self, monkeypatch, raw):
        monkeypatch.setenv("MECHBENCH_EXPR_CEILING_SECONDS", raw)
        with pytest.raises(ValueError, match="MECHBENCH_EXPR_CEILING_SECONDS"):
            read_ceiling("SECONDS", 1.0)
