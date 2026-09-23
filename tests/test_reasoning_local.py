"""A local model's reasoning leaves its text, and a model that cannot
reason is left exactly as it was.

Of the families this package runs (Gemma 4, Gemma 3, Llama, Qwen 2.5),
only Gemma 4's tokenizer declares reasoning markup: `<|channel>` and
`<channel|>` as special tokens, written `<|channel>thought\\n…<channel|>`.
"""

from __future__ import annotations

import glob
import os

import pytest

from mechbench_compute import chat as chat_mod
from mechbench_compute import model_ref as mr
from mechbench_compute.chat.split_reasoning import DELIMITERS, find_delimiters, split_reasoning

GEMMA4 = DELIMITERS[0]
HUB = os.path.expanduser("~/.cache/huggingface/hub")


def pinned(name):
    found = sorted(glob.glob(f"{HUB}/models--mlx-community--{name}/snapshots/*/tokenizer.json"))
    if not found:
        pytest.skip(f"{name} is not in the local cache")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.dirname(found[0]))


@pytest.mark.parametrize("name,declares", [
    ("gemma-4-E4B-it-bf16", True),
    ("gemma-3-4b-it-bf16", False),
    ("Llama-3.2-3B-Instruct-bf16", False),
    ("Qwen2.5-3B-Instruct-bf16", False),
])
def test_only_gemma_4_declares_reasoning_markup(name, declares):
    found = find_delimiters(pinned(name))
    assert (found == GEMMA4) if declares else found is None


class TestSplit:
    def test_a_thought_leaves_the_text(self):
        assert split_reasoning("<|channel>thought\nAdd them.<channel|>Four.", GEMMA4) == (
            ["Add them."], "Four.")

    def test_an_empty_placeholder_thought_leaves_markup_and_no_entry(self):
        assert split_reasoning("<|channel>thought\n<channel|>Four.", GEMMA4) == ([], "Four.")

    def test_a_thought_cut_off_at_max_tokens_is_reasoning_not_text(self):
        assert split_reasoning("<|channel>thought\nFirst, the", GEMMA4) == (
            ["First, the"], "")

    @pytest.mark.parametrize("text", ["Four.", "  Four.\n", "", "a <b> c"])
    def test_text_without_markup_is_unchanged_byte_for_byte(self, text):
        assert split_reasoning(text, GEMMA4) == ([], text)
        assert split_reasoning(text, None) == ([], text)

    def test_a_model_without_the_tokens_is_never_split(self):
        text = "<|channel>thought\nnot markup here<channel|>ok"
        assert split_reasoning(text, None) == ([], text)


class FakeTok:
    def __init__(self, vocab=()):
        self.vocab = {t: i for i, t in enumerate(vocab, start=10)}
        self.unk_token_id = 3
        self.rendered: list[list[dict]] = []

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, **kw):
        self.rendered.append(turns)
        return repr(turns)


def run_local(monkeypatch, tok, reply):
    from mechbench_compute import distill, generate

    class FakeModel:
        tokenizer = tok

    monkeypatch.setattr(distill, "encode", lambda t, text: [1, 2, 3])
    monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
    monkeypatch.setattr(generate, "sample_completion_cached", lambda *a, **k: (reply, []))
    return chat_mod.run_local(FakeModel(), mr.parse("google/gemma-4-e4b-it"),
                              [{"id": "r0", "user": "What is 2+2?"}], {"n": 1, "seed": 3})


def test_gemma_4_reasoning_is_on_the_item_and_not_in_its_text(monkeypatch):
    tok = FakeTok(("<|channel>", "<channel|>"))
    item = run_local(monkeypatch, tok, "<|channel>thought\nAdd.<channel|>Four.")["items"][0]
    assert item["text"] == "Four."
    assert item["reasoning"] == [{"text": "Add.", "provider": "local",
                                  "model": "google/gemma-4-e4b-it"}]
    assert "empty" not in item["metadata"]


def test_a_local_reply_of_reasoning_alone_is_empty(monkeypatch):
    tok = FakeTok(("<|channel>", "<channel|>"))
    item = run_local(monkeypatch, tok, "<|channel>thought\nStill adding")["items"][0]
    assert item["text"] == "" and item["metadata"]["empty"]["cause"] == "reasoning"
    assert item["reasoning"][0]["text"] == "Still adding"


@pytest.mark.parametrize("reply", ["Four.", "<|channel>thought\nx<channel|>Four.", ""])
def test_a_model_that_cannot_reason_keeps_its_item_exactly(monkeypatch, reply):
    # Gemma 3, Llama and Qwen 2.5 declare no reasoning tokens.
    item = run_local(monkeypatch, FakeTok(), reply)["items"][0]
    assert item["text"] == reply
    assert "reasoning" not in item
    assert set(item["metadata"]) == {"coords", "model", "sampling"}


def test_a_local_turn_hands_its_reasoning_to_the_template_for_the_same_model():
    from mechbench_compute.chat.render_conversation import render_conversation
    from mechbench_compute.providers import messages as pm

    tok = FakeTok()
    turn = pm.Message(role="assistant", content=(
        pm.ReasoningPart(text="Add.", provider="local", model="gemma-4"),
        pm.TextPart("Four.")))
    ask = pm.Message(role="user", content=(pm.TextPart("And 3+3?"),))
    first = pm.Message(role="user", content=(pm.TextPart("2+2?"),))
    render_conversation(tok, pm.request({"model": "gemma-4", "messages": [first, turn, ask]}))
    render_conversation(tok, pm.request({"model": "llama-3", "messages": [first, turn, ask]}))
    same, other = tok.rendered
    assert same[1] == {"role": "assistant", "content": "Four.", "reasoning_content": "Add."}
    assert other[1] == {"role": "assistant", "content": "Four."}
