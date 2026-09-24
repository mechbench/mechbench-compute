from __future__ import annotations

import json
import os

import pytest

from mechbench_compute import distill, generate
from mechbench_compute.chat.split_reasoning import DELIMITERS, split_reasoning
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

GEMMA4 = DELIMITERS[0]
MODEL = "mlx-community/gemma-4-e2b-it-bf16"

PLAN = ("Thinking Process:\n\n1.  **Analyze the Request:** The user wants a "
        "\"100 word piece of flash fiction.\"\n\n"
        "6.  **Final Output Generation.** (This leads to the final output.)")
STORY = ("The silence was geological. It moved through the cheap carpet.\n\n"
         "It was a door to a place where no one was supposed to wait.")
FLASH_48 = f"<|channel>thought\n{PLAN}<channel|>{STORY}"


class Tok:
    def __init__(self, vocab=()):
        self.vocab = {t: i for i, t in enumerate(vocab, start=10)}
        self.unk_token_id = 3

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def run_generate(monkeypatch, tok, replies, **params):
    class Model:
        tokenizer = tok

    script = iter(replies)
    monkeypatch.setattr(ProtocolExecutor, "_model_loaded", lambda self, model_id: Model())
    monkeypatch.setattr(ProtocolExecutor, "_run_model_block",
                        lambda self, fn, inputs, params, *a, **k: fn(inputs, params, *a, **k))
    monkeypatch.setattr(distill, "render_chat", lambda tok, s, u, p: f"<{s}|{u}>{p}")
    monkeypatch.setattr(distill, "encode", lambda tok, text: [ord(c) for c in text])
    monkeypatch.setattr(distill, "prefill_decision", lambda model, ids: ("cache", ids))

    def sample(model, ids, *, max_tokens, temperature, top_p, rng, prefill,
               return_ids=False, stop_strings=()):
        raw = next(script)
        return raw, list(range(min(len(raw), max_tokens)))

    monkeypatch.setattr(generate, "sample_completion_cached", sample)
    graph = {"dataflow": 2, "nodes": [{"id": "gen", "block": "text/generate",
                        "params": {"model": MODEL, "n": len(replies), "seed": 7,
                                   "max_tokens": 4096, **params},
                        "inputs": {"records": [{"id": "flash", "user": "A story."}]}}],
             "edges": []}
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                              extra={"graph": graph}))
    payload = out.payload if hasattr(out, "payload") else out
    return payload["outputs"]["gen"]


def test_flash_48_splits_into_the_plan_as_reasoning_and_the_story_as_text():
    assert split_reasoning(FLASH_48, GEMMA4) == ([PLAN], STORY)


def test_a_thought_is_the_items_reasoning_and_never_its_text(monkeypatch):
    out = run_generate(monkeypatch, Tok(("<|channel>", "<channel|>")), [FLASH_48])
    item = out["items"][0]
    assert item["text"] == STORY
    assert item["reasoning"] == [{"text": PLAN, "provider": "local", "model": MODEL}]
    assert item["metadata"]["sampling"]["ended"] == "end"
    assert "empty" not in item["metadata"]


def test_a_thought_never_closed_is_reasoning_and_the_item_is_empty(monkeypatch):
    out = run_generate(monkeypatch, Tok(("<|channel>", "<channel|>")),
                       ["<|channel>thought\n1. Plan the story"], max_tokens=8)
    item = out["items"][0]
    assert item["text"] == ""
    assert item["reasoning"][0]["text"] == "1. Plan the story"
    assert item["metadata"]["sampling"]["ended"] == "empty"
    assert item["metadata"]["empty"]["cause"] == "reasoning"
    assert out["ended"]["empty"] == 1


def test_the_prefill_stays_at_the_front_of_the_prose(monkeypatch):
    out = run_generate(monkeypatch, Tok(("<|channel>", "<channel|>")),
                       ["<|channel>thought\nHm.<channel|>Once, a fox."],
                       continue_prefill=True)
    assert out["items"][0]["text"] == "Once, a fox."


@pytest.mark.parametrize("vocab", [("<|channel>", "<channel|>"), ()])
@pytest.mark.parametrize("reply", ["A story.", "  A story.\n", "", "a <b> c",
                                   "1. plan<channel|>A story."])
def test_a_sample_without_a_reasoning_section_is_unchanged(monkeypatch, vocab, reply):
    item = run_generate(monkeypatch, Tok(vocab), [reply])["items"][0]
    assert item["text"] == reply
    assert "reasoning" not in item
    assert set(item["metadata"]) == {"coords", "model", "sampling"}


def test_a_model_without_the_tokens_keeps_its_markup_as_written(monkeypatch):
    item = run_generate(monkeypatch, Tok(), [FLASH_48])["items"][0]
    assert item["text"] == FLASH_48 and "reasoning" not in item


@pytest.mark.skipif(os.environ.get("MECHBENCH_LIVE_BENCH") != "1",
                    reason="reads a stored result from the bench API")
def test_the_stored_flash_48_splits_at_its_channel():
    from mechbench_compute import bench

    items = bench.result("j_n1z5dyw5shg1qjve8v2r", "gen")["items"]
    stored = next(i for i in items if i["id"] == "flash-s48")
    reasoning, prose = split_reasoning(stored["text"], GEMMA4)
    assert len(reasoning) == 1
    assert reasoning[0].endswith("(This leads to the final output.)")
    assert prose.startswith("The silence was geological.")
    assert "<|channel>" not in prose and "<channel|>" not in prose
    others = [i for i in items if i["id"] != "flash-s48"]
    assert all(split_reasoning(i["text"], GEMMA4) == ([], i["text"]) for i in others), (
        json.dumps([i["id"] for i in others if "<|channel>" in i["text"]]))
