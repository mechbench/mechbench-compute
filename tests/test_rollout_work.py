"""What a decision read's rollout COSTS, asserted as work rather than
timed against a clock (task 000552).

A matrix of 312 reads once took 36 minutes where it had taken 4 (task
000506). That one was **not** a code regression at all: the runner's
launchd job carried `ProcessType: Background`, so the OS ran it on the
efficiency cores. No test of this shape would have caught it, and the
release-time stopwatch it left behind could not tell a slow path from a
slow machine.

What IS worth pinning here is the work the cached path promises, which a
stub counts in milliseconds: the prompt is encoded ONCE; every expansion
feeds only its own partial outcome (task 000227's whole point); each
node's children are picked without sorting the whole vocabulary. Those
are properties of the code, so they belong in the suite rather than on a
clock.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute.distill import expand_top_outcomes_cached

VOCAB = "abcdefgh\"" + "".join(chr(ord("i") + i) for i in range(7))
QUOTE = VOCAB.index('"')


class _Cache:
    """One layer's KV, shaped as the real one is: [batch, heads, seq, d],
    so `_copy_prefix_cache` finds arrays to re-materialize."""

    def __init__(self):
        self.keys = mx.zeros((1, 2, 4, 3))
        self.values = mx.zeros((1, 2, 4, 3))


class _Tok:
    def decode(self, ids):
        return "".join(VOCAB[int(i)] for i in ids)


class StubModel:
    """Every forward records what it was fed. The logits favour a short
    chain of tokens, then the terminator, so the expansion completes."""

    tokenizer = _Tok()

    def __init__(self):
        self.fed: list[list[int]] = []
        self.caches_made = 0

    def prompt_cache(self):
        self.caches_made += 1
        return [_Cache()]

    def lm(self, ids, cache=None):
        seq = [int(i) for i in np.array(ids)[0]]
        self.fed.append(seq)
        last = seq[-1]
        rows = []
        for _ in seq:
            logits = np.full(len(VOCAB), -12.0, dtype=np.float32)
            # Three plausible children, and the terminator once a path is
            # two tokens long, so every branch ends.
            logits[(last + 1) % 8] = 2.0
            logits[(last + 3) % 8] = 1.0
            logits[QUOTE] = 1.5
            rows.append(logits)
        return mx.array(np.stack(rows)[None, ...])


PROMPT = list(range(60))  # a prompt of the length a real battery renders
CFG = {"top_k": 5, "max_tokens": 4, "max_forwards": 48, "floor": 1e-3,
       "terminators": ['"']}


@pytest.fixture
def expanded():
    model = StubModel()
    out = expand_top_outcomes_cached(model, model.tokenizer, PROMPT, CFG)
    return model, out


def test_the_prompt_is_encoded_once_and_expansions_feed_only_their_own_tokens(expanded):
    model, out = expanded
    assert out["top_outcomes"], "the stub must complete some outcomes"
    assert model.fed[0] == PROMPT, "the first forward is the prefill"
    later = model.fed[1:]
    assert later, "the expansion must have run"
    # The regression: a forward that carries the prompt again.
    assert all(len(seq) <= CFG["max_tokens"] for seq in later), \
        f"an expansion fed {max(len(s) for s in later)} tokens; the prompt is {len(PROMPT)}"
    assert all(seq[0] not in PROMPT[10:] for seq in later)
    # Total work after the prefill is bounded by the outcome length, not
    # by the prompt — the promise of the cached path (000227), and the
    # one a future edit could quietly drop.
    assert sum(len(s) for s in later) <= CFG["max_forwards"] * CFG["max_tokens"]


def test_the_budget_of_forwards_is_honoured_and_reported(expanded):
    model, out = expanded
    assert out["forwards_used"] == len(model.fed) <= CFG["max_forwards"]


def test_children_are_picked_without_sorting_the_whole_vocabulary(monkeypatch):
    """A full argsort per node is ~20 ms on a real 262k vocabulary, and
    the path takes fifty children per node — so it partitions instead."""
    import mechbench_compute.distill as distill_mod

    def refuse(*a, **kw):
        raise AssertionError("the expansion sorted a whole vocabulary")

    monkeypatch.setattr(distill_mod.np, "argsort", refuse)
    model = StubModel()
    out = expand_top_outcomes_cached(model, model.tokenizer, PROMPT, CFG)
    assert out["top_outcomes"]
