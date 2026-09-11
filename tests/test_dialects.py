"""Tool dialects, checked against what each model's own template renders
(epic 000439).

Two layers:

- **fixture tests**, which run everywhere: the renderings in
  `fixtures/chat_templates.json` were captured from the real
  tokenizers, and the parsers must read them.
- **the live round trip**, gated on MECHBENCH_MODEL_TESTS=1: render a
  canonical call through the model's own chat template TODAY and parse
  it back. When a model publishes a new template revision, that test
  fails instead of an experiment.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from mechbench_compute import dialects as dl
from mechbench_compute import tools as T

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "chat_templates.json"
CAPTURED = json.loads(FIXTURES.read_text())
CALC = T.toolbox_from(["calc"]).tools

#: repo -> the dialect its rendering should be identified as.
EXPECTED = {
    "mlx-community/gemma-4-e2b-it-bf16": "gemma-4",
    "mlx-community/Qwen2.5-3B-Instruct-bf16": "qwen-2.5",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16": "llama-3",
    "mlx-community/Llama-3.2-3B-Instruct-bf16": "llama-3",
}


def _probe(repo: str) -> dl.TemplateProbe:
    rec = CAPTURED[repo]
    return dl.TemplateProbe(
        supports_tools=rec["tools_change_the_prompt"],
        rendered=rec.get("round_trip"),
    )


class TestIdentification:
    @pytest.mark.parametrize("repo,name", sorted(EXPECTED.items()))
    def test_a_model_is_identified_from_its_own_rendering(self, repo, name):
        found = dl.identify(_probe(repo))
        assert found is not None, f"{repo} matched no dialect"
        assert found.name == name

    def test_gemma_3_has_no_tool_protocol(self):
        # Its template accepts `tools=` and renders the same prompt
        # either way. That is not support; it is a silent no-op, and
        # the probe decides by DIFFERENCE for exactly this reason.
        repo = "mlx-community/gemma-3-4b-it-bf16"
        assert CAPTURED[repo]["tools_change_the_prompt"] is False
        assert dl.identify(_probe(repo)) is None


class TestParsingWhatTheTemplatesRender:
    """The point of the epic: read back what the model's own template
    writes, rather than what we wish it wrote."""

    @pytest.mark.parametrize("repo,name", sorted(EXPECTED.items()))
    def test_the_rendered_call_parses_back_exactly(self, repo, name):
        rendered = CAPTURED[repo]["round_trip"]
        dialect = dl.identify(_probe(repo))
        assert dialect is not None
        _, calls = dialect.parse(rendered, CALC)
        assert len(calls) == 1, f"{repo}: expected one call, got {len(calls)}"
        assert calls[0].name == "calc"
        assert calls[0].arguments == dl.PROBE_ARGS

    def test_gemma_4_reads_its_own_quoting(self):
        text = '<|tool_call>call:calc{expression:<|"|>37 + 18<|"|>}<tool_call|>'
        _, calls = dl._gemma4(text, CALC)
        assert [(c.name, c.arguments) for c in calls] == [
            ("calc", {"expression": "37 + 18"})]

    def test_gemma_4_reads_an_unquoted_scalar(self):
        seek = [T.ToolDef(name="seek", description="",
                          schema={"type": "object",
                                  "properties": {"depth": {"type": "integer"}}})]
        text = '<|tool_call>call:seek{depth:3}<tool_call|>'
        assert dl._gemma4(text, seek)[1][0].arguments == {"depth": 3}

    def test_a_call_to_an_unoffered_tool_stays_in_the_text(self):
        # So the error can quote it. Stripping it would erase the only
        # evidence of what the model tried to do.
        text = '<|tool_call>call:wget{url:<|"|>x<|"|>}<tool_call|>'
        rest, calls = dl._gemma4(text, CALC)
        assert calls == []
        assert "wget" in rest

    def test_qwen_reads_its_envelope(self):
        text = '<tool_call>\n{"name": "calc", "arguments": {"expression": "2+2"}}\n</tool_call>'
        assert dl._qwen(text, CALC)[1][0].arguments == {"expression": "2+2"}

    def test_llama_uses_parameters_not_arguments(self):
        text = '{"name": "calc", "parameters": {"expression": "2+2"}}'
        _, calls = dl._llama(text, CALC)
        assert calls[0].arguments == {"expression": "2+2"}

    def test_prose_is_never_a_call(self):
        for parse in (dl._gemma4, dl._qwen, dl._llama):
            assert parse("The stall has 55 apples.", CALC)[1] == []


class TestRefusal:
    def test_no_tool_protocol_refuses_rather_than_inventing_one(self):
        class Tok:
            def apply_chat_template(self, msgs, tools=None, **kw):
                return "same either way"

        with pytest.raises(dl.NoToolDialect, match="no tool protocol"):
            dl.dialect_for(Tok(), model="pretend/model")

    def test_an_unknown_dialect_refuses_and_shows_the_rendering(self):
        class Tok:
            def apply_chat_template(self, msgs, tools=None, **kw):
                return "TOOLS!! <fancy_call/>" if tools else "plain"

        with pytest.raises(dl.NoToolDialect, match="matches no known dialect"):
            dl.dialect_for(Tok(), model="pretend/model")

    def test_a_template_that_cannot_be_driven_refuses(self):
        class Tok:
            def apply_chat_template(self, msgs, tools=None, **kw):
                raise ValueError("nope")

        with pytest.raises(dl.NoToolDialect, match="no usable chat template"):
            dl.dialect_for(Tok())


@pytest.mark.skipif(os.environ.get("MECHBENCH_MODEL_TESTS") != "1",
                    reason="needs the real tokenizers (MECHBENCH_MODEL_TESTS=1)")
class TestLiveRoundTrip:
    """The conformance test the epic is built around. The fixtures can
    go stale; this cannot."""

    @pytest.mark.parametrize("repo,name", sorted(EXPECTED.items()))
    def test_render_then_parse_recovers_the_call(self, repo, name):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(repo)
        probe = dl.probe_template(tok)
        assert probe.supports_tools, f"{repo} lost its tool support"
        dialect = dl.identify(probe)
        assert dialect is not None and dialect.name == name
        _, calls = dialect.parse(probe.rendered or "", CALC)
        assert len(calls) == 1
        assert calls[0].name == "calc"
        assert calls[0].arguments == dl.PROBE_ARGS


class TestDescribe:
    """`doctor`'s answer, derived from the template rather than from a
    list of model names that would go stale."""

    def test_a_known_dialect_is_named_with_its_result_role(self):
        class Tok:
            def apply_chat_template(self, msgs, tools=None, **kw):
                if not tools:
                    return "plain"
                return ('TOOLS <|start_header_id|>ipython<|end_header_id|>'
                        '{"name": "calc", "parameters": {"expression": "37 + 18"}}')

        r = dl.describe(Tok(), "some/llama")
        assert r.supports_tools and r.dialect == "llama-3"
        assert "ipython" in r.detail

    def test_a_model_without_tools_says_why(self):
        class Tok:
            def apply_chat_template(self, msgs, tools=None, **kw):
                return "same either way"

        r = dl.describe(Tok(), "google/gemma-3-4b-it")
        assert not r.supports_tools and r.dialect is None
        assert "not trained to receive" in r.detail

    def test_an_unknown_dialect_is_reported_as_such_not_as_absent(self):
        class Tok:
            def apply_chat_template(self, msgs, tools=None, **kw):
                return "TOOLS <fancy/>" if tools else "plain"

        r = dl.describe(Tok(), "new/model")
        assert r.supports_tools and r.dialect is None
        assert "no known dialect" in r.detail
