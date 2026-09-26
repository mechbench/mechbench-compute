#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mechbench_compute.chat.read_ending import read_ending
from mechbench_compute.providers import http, make_transport
from mechbench_compute.providers import messages as m
from mechbench_compute.providers.openai_responses import default_api

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "providers"

SIG_A = "EqQBCkYIBxgCKkDz+/0==" + "x" * 300
SIG_B = "Es8CCkYICxIM+sig/B=="
ENC_A = "gAAAAABo" + "A" * 400 + "=="
ENC_B = "gAAAAABo" + "B" * 400 + "=="

CALC = {"name": "calc", "description": "Evaluate an arithmetic expression.",
        "input_schema": {"type": "object",
                         "properties": {"expression": {"type": "string"}},
                         "required": ["expression"]}}

ANTHROPIC_THINK = {"type": "thinking", "thinking": "", "signature": SIG_A}
ANTHROPIC_REDACTED = {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4a"}
ANTHROPIC_FINAL_THINK = {"type": "thinking", "thinking": "The tool said 4.", "signature": SIG_B}


def anthropic(content, *, stop="end_turn", usage=None, rid="msg_1",
              model="claude-opus-5") -> dict[str, Any]:
    return {"id": rid, "type": "message", "role": "assistant", "model": model,
            "stop_reason": stop, "stop_sequence": None,
            "usage": usage or {"input_tokens": 10, "output_tokens": 30},
            "content": content}


def chat(message, *, finish="stop", usage=None, rid="chatcmpl-1", model="x") -> dict[str, Any]:
    return {"id": rid, "object": "chat.completion", "model": model,
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 30,
                               "completion_tokens_details": {"reasoning_tokens": 20}},
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", **message}}]}


def calc_call(call_id="call_1", expression="2+2") -> dict[str, Any]:
    return {"id": call_id, "type": "function",
            "function": {"name": "calc",
                         "arguments": json.dumps({"expression": expression})}}


GEMINI_THOUGHT = {"text": "I should use the calculator.", "thought": True}
GEMINI_USAGE = {"promptTokenCount": 10, "candidatesTokenCount": 5,
                "thoughtsTokenCount": 20, "totalTokenCount": 35}


def gemini(parts, *, finish="STOP", usage=None, model="gemini-2.5-pro",
           rid="resp-1") -> dict[str, Any]:
    return {"responseId": rid, "modelVersion": model,
            "usageMetadata": usage or GEMINI_USAGE,
            "candidates": [{"index": 0, "finishReason": finish,
                            "content": {"role": "model", "parts": parts}}]}


RS1 = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": ENC_A}
FC1 = {"id": "fc_1", "type": "function_call", "status": "completed", "call_id": "call_1",
       "name": "calc", "arguments": '{"expression": "2+2"}'}
RS2 = {"id": "rs_2", "type": "reasoning",
       "summary": [{"type": "summary_text", "text": "The tool said 4."}],
       "encrypted_content": ENC_B}
MSG = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
       "content": [{"type": "output_text", "text": "Four.", "annotations": []}]}
XRS1 = {"id": "rs_x1", "type": "reasoning", "status": "completed", "summary": [],
        "content": [{"type": "reasoning_text", "text": "Use the tool."}],
        "encrypted_content": ENC_A}
XRS2 = {"id": "rs_x2", "type": "reasoning", "status": "completed", "summary": [],
        "content": [{"type": "reasoning_text", "text": "It said 4."}],
        "encrypted_content": ENC_B}
RESPONSES_USAGE = {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4},
                   "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 20},
                   "total_tokens": 40}

XAI_USAGE = {"input_tokens": 20, "input_tokens_details": {"cached_tokens": 5},
             "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 20},
             "total_tokens": 50}


def responses(output, *, model="gpt-6-astra-2026-08-01", status="completed",
              incomplete=None, rid="resp_1", usage=None) -> dict[str, Any]:
    return {"id": rid, "object": "response", "model": model, "status": status,
            "incomplete_details": incomplete, "output": output,
            "usage": usage or RESPONSES_USAGE}


OPENROUTER_DETAILS = [
    {"type": "reasoning.text", "text": "Use the tool.", "signature": SIG_A,
     "id": "r1", "format": "anthropic-claude-v1", "index": 0},
    {"type": "reasoning.encrypted", "data": "eyJlbmNyeXB0ZWQiOnRydWV9",
     "id": "r2", "format": "anthropic-claude-v1", "index": 1},
]

CASES: list[dict[str, Any]] = [
    {"name": "anthropic_text", "provider": "anthropic", "model": "claude-sonnet-5",
     "system": "Be brief.", "tools": [],
     "bodies": [anthropic([{"type": "text", "text": "Four."}], model="claude-sonnet-5",
                          usage={"input_tokens": 12, "output_tokens": 5,
                                 "cache_read_input_tokens": 100,
                                 "cache_creation_input_tokens": 50})]},
    {"name": "anthropic_one_hour_cache", "provider": "anthropic", "model": "claude-opus-5-5",
     "system": "Be brief.", "tools": [],
     "bodies": [anthropic([{"type": "text", "text": "Four."}], model="claude-opus-5-5",
                          usage={"input_tokens": 12, "output_tokens": 30,
                                 "cache_read_input_tokens": 100,
                                 "cache_creation_input_tokens": 50,
                                 "cache_creation": {"ephemeral_5m_input_tokens": 20,
                                                    "ephemeral_1h_input_tokens": 30}})]},
    {"name": "anthropic_effort_updates_hour_cache", "provider": "anthropic",
     "model": "claude-opus-5-5", "system": "Be brief.", "tools": [],
     "effort": "high", "reasoning_display": "updates", "prompt_cache": "1h",
     "bodies": [anthropic([{"type": "thinking", "thinking": "Adding the numbers.",
                            "signature": "sig-u"},
                           {"type": "text", "text": "Four."}], model="claude-opus-5-5")]},
    {"name": "anthropic_thinking_tool_round", "provider": "anthropic",
     "model": "claude-opus-5",
     "provider_options": {"anthropic": {"thinking": {"type": "enabled",
                                                      "budget_tokens": 1024}}},
     "bodies": [
         anthropic([ANTHROPIC_THINK, ANTHROPIC_REDACTED,
                    {"type": "tool_use", "id": "toolu_1", "name": "calc",
                     "input": {"expression": "2+2"}}], stop="tool_use"),
         anthropic([ANTHROPIC_FINAL_THINK, {"type": "text", "text": "Four."}], rid="msg_2"),
     ]},
    {"name": "anthropic_reasoning_only", "provider": "anthropic", "model": "claude-opus-5",
     "tools": [], "max_tokens": 100,
     "bodies": [anthropic([{"type": "thinking", "thinking": "Score: 5", "signature": SIG_B}],
                          stop="max_tokens",
                          usage={"input_tokens": 5, "output_tokens": 100})]},
    {"name": "anthropic_max_tokens", "provider": "anthropic", "model": "claude-haiku-4-5",
     "tools": [], "max_tokens": 4,
     "bodies": [anthropic([{"type": "text", "text": "Four, and"}], stop="max_tokens",
                          model="claude-haiku-4-5",
                          usage={"input_tokens": 9, "output_tokens": 4})]},
    {"name": "deepseek_reasoning_tool_round", "provider": "deepseek",
     "model": "deepseek-flash",
     "bodies": [
         chat({"content": None, "reasoning_content": "Use the calculator.",
               "tool_calls": [calc_call()]}, finish="tool_calls"),
         chat({"content": "Four.", "reasoning_content": "It said 4."}, rid="chatcmpl-2"),
     ]},
    {"name": "xai_chat_reasoning_kept_home", "provider": "xai", "model": "grok-4.7",
     "bodies": [
         chat({"content": None, "reasoning_content": "Use the tool.",
               "tool_calls": [calc_call()]}, finish="tool_calls"),
         chat({"content": "Four.", "reasoning_content": "Done."}, rid="chatcmpl-2"),
     ]},
    {"name": "openrouter_reasoning_details", "provider": "openai-compatible",
     "model": "anthropic/claude-opus-5", "base_url": "https://openrouter.ai/api/v1",
     "bodies": [
         chat({"content": None, "reasoning": "Use the tool.",
               "reasoning_details": OPENROUTER_DETAILS,
               "tool_calls": [calc_call()]}, finish="tool_calls"),
         chat({"content": "Four."}, rid="chatcmpl-2"),
     ]},
    {"name": "fireworks_inline_think", "provider": "fireworks",
     "model": "accounts/fireworks/models/qwen3-235b", "tools": [],
     "bodies": [chat({"content": "<think>Score: 5</think>\n\nScore: 2"},
                     usage={"prompt_tokens": 20, "completion_tokens": 12})]},
    {"name": "openai_chat_max_tokens", "provider": "openai", "model": "gpt-5",
     "tools": [], "max_tokens": 4,
     "bodies": [chat({"content": "Four, and"}, finish="length",
                     usage={"prompt_tokens": 1000, "completion_tokens": 4,
                            "prompt_tokens_details": {"cached_tokens": 800}})]},
    {"name": "gemini_thought_signature_tool_round", "provider": "gemini",
     "model": "gemini-2.5-pro",
     "bodies": [
         gemini([GEMINI_THOUGHT,
                 {"functionCall": {"name": "calc", "args": {"expression": "2+2"}},
                  "thoughtSignature": SIG_A},
                 {"functionCall": {"name": "calc", "args": {"expression": "3+3"}}}]),
         gemini([{"text": "Four, and six.", "thoughtSignature": SIG_B}], rid="resp-2"),
     ]},
    {"name": "gemini_thoughts_only", "provider": "gemini", "model": "gemini-2.5-flash",
     "tools": [], "max_tokens": 50,
     "bodies": [gemini([{"text": "Weighing it.", "thought": True}], finish="MAX_TOKENS",
                       model="gemini-2.5-flash",
                       usage={"promptTokenCount": 8, "thoughtsTokenCount": 50,
                              "totalTokenCount": 58})]},
    {"name": "openai_responses_reasoning_tool_round", "provider": "openai",
     "model": "gpt-6-astra", "system": "Be brief.",
     "bodies": [responses([RS1, FC1]), responses([RS2, MSG], rid="resp_2")]},
    {"name": "openai_responses_effort", "provider": "openai", "model": "gpt-5",
     "api": "responses", "system": "Be brief.", "tools": [], "effort": "low",
     "bodies": [responses([MSG], model="gpt-5-2026-08-07",
                          usage={"input_tokens": 40, "output_tokens": 150,
                                 "input_tokens_details": {"cached_tokens": 0},
                                 "output_tokens_details": {"reasoning_tokens": 120}})]},
    {"name": "openai_chat_effort", "provider": "openai", "model": "gpt-5",
     "api": "chat_completions", "system": "Be brief.", "tools": [], "effort": "minimal",
     "bodies": [chat({"role": "assistant", "content": "Four."}, model="gpt-5")]},
    {"name": "xai_responses_reasoning_tool_round", "provider": "xai", "model": "grok-4.7",
     "api": "responses",
     "bodies": [responses([XRS1, FC1], model="grok-4.7", usage=XAI_USAGE),
                responses([XRS2, MSG], model="grok-4.7", rid="resp_2", usage=XAI_USAGE)]},
    {"name": "openai_responses_reasoning_only", "provider": "openai", "model": "gpt-6-astra",
     "tools": [],
     "bodies": [responses([RS2], status="incomplete",
                          incomplete={"reason": "max_output_tokens"})]},
]

TOOL_ANSWER = "4"


@contextmanager
def scripted(bodies: list[dict[str, Any]]) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    sent: list[tuple[str, dict[str, Any]]] = []
    queue = [copy.deepcopy(b) for b in bodies]

    def post(url, *, headers, payload, timeout, secrets=()):
        if url.endswith(("count_tokens", ":countTokens")):
            return http.HttpResponse(status=200, headers={},
                                     body={"input_tokens": 10, "totalTokens": 10})
        beta = {k: v for k, v in headers.items() if k == "anthropic-beta"}
        sent.append((url, copy.deepcopy(payload), beta))
        return http.HttpResponse(status=200, headers={}, body=queue.pop(0))

    original = http.post_json
    http.post_json = post
    try:
        yield sent
    finally:
        http.post_json = original


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    provider, model = case["provider"], case["model"]
    api = case.get("api") or default_api(provider, model)
    credential = {"token": "k", **({"base_url": case["base_url"]} if "base_url" in case else {})}
    tools = case.get("tools", [CALC])
    req = m.request({
        "model": model, "system": case.get("system", ""),
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": case.get("max_tokens", 1024), "tools": tools,
        "provider_options": case.get("provider_options", {}), "api": api,
        **{k: case[k] for k in ("effort", "reasoning_display", "prompt_cache") if k in case},
    })
    calls: list[dict[str, Any]] = []
    with scripted(case["bodies"]) as sent:
        transport = make_transport(provider, credential)
        for i, body in enumerate(case["bodies"]):
            out = transport.chat(req)
            url, wire, beta = sent[-1]
            calls.append({
                **({"headers": beta} if beta else {}),
                "request": m.canonical(req, provider=provider),
                "url": url,
                "body": wire,
                "response": body,
                "expected": {
                    "parts": [p.to_wire() for p in out.parts],
                    "text": out.text,
                    "stop_reason": out.stop_reason,
                    "ending": read_ending(provider, out.stop_reason,
                                          tool_call=bool(out.tool_calls),
                                          empty=out.empty is not None),
                    "usage": out.usage.to_wire(),
                    "empty": out.empty.to_wire() if out.empty is not None else None,
                    "cost_usd": out.call.cost_usd,
                    "priced": out.call.priced,
                },
            })
            if i + 1 < len(case["bodies"]):
                results = tuple(m.ToolResultPart(c.id, TOOL_ANSWER) for c in out.tool_calls)
                req = req.with_messages([*req.messages, out.as_message(),
                                         m.Message(role="user", content=results)])
    return {"name": case["name"], "provider": provider, "model": model, "api": api,
            "credential": {k: v for k, v in credential.items() if k != "token"},
            "tool_answer": TOOL_ANSWER, "calls": calls}


def build() -> dict[str, dict[str, Any]]:
    return {case["name"]: run_case(case) for case in CASES}


def dump(fixture: dict[str, Any]) -> str:
    return json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    built = build()
    for stale in OUT.glob("*.json"):
        if stale.stem not in built:
            stale.unlink()
    for name, fixture in built.items():
        (OUT / f"{name}.json").write_text(dump(fixture), encoding="utf-8")
    print(f"wrote {len(built)} provider fixtures to {OUT}")


if __name__ == "__main__":
    main()
