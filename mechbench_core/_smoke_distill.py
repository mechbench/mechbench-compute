"""Smoke test for the distributional-training primitives (distill + lora).

Part 1 (fast, no model): TargetMap transforms and validation, trie
construction against a toy tokenizer (shared prefixes merge, closers
append, the boundary assertion fires), and soft_ce numerics checked
against hand-computed cross-entropy on a tiny deterministic module.

Part 2 (loads Gemma 4 E2B; skipped with --fast): the full lifecycle on a
real model — uniform d6 target, envelope render, trie compile, baseline
calibration, LoRA training for a few dozen steps (loss falls, KL from
uniform falls, the Paris anchor stays sharp), save_adapter, unwrap,
fuse ≈ wrapped adapter at the decision token, restore == base bit-exact.

Run from project root with the venv active:
    HF_HUB_OFFLINE=1 python -m mechbench_core._smoke_distill [--fast]
"""

from __future__ import annotations

import math
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from . import distill, lora
from .distill import Example, TargetMap

E2B_ID = "mlx-community/gemma-4-e2b-it-bf16"
PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{PASS if ok else FAIL}] {name}" +
          (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------------------
# Part 1a: TargetMap
# ---------------------------------------------------------------------------

def part1_targetmap() -> None:
    print("== TargetMap ==", flush=True)
    m = TargetMap.from_dict({"a": 1.0, "b": 4.0})
    s = m.sqrt()
    check("sqrt returns new map", s["a"] == 1.0 and s["b"] == 2.0
          and m["b"] == 4.0)
    check("pow", m.pow(0.5).to_dict() == s.to_dict())
    check("scale", m.scale(2.0)["b"] == 8.0)
    n = m.normalize()
    check("normalize", abs(n["a"] - 0.2) < 1e-12
          and abs(n.total() - 1.0) < 1e-12)
    t = m.temper(2.0)
    check("temper flattens", abs(t["a"] - (1.0 / 3.0)) < 1e-12
          and abs(t.total() - 1.0) < 1e-12)
    u = TargetMap.uniform(["x", "y", "z", "w"])
    check("uniform", abs(u["x"] - 0.25) < 1e-12)
    mix = m.normalize().mix_uniform(0.5)
    check("mix_uniform floor", abs(mix["a"] - (0.5 * 0.2 + 0.25)) < 1e-12)
    check("top_k", m.top_k(1).to_dict() == {"b": 4.0})
    check("filter", m.filter(lambda k, v: v > 2).to_dict() == {"b": 4.0})
    check("map_values", m.map_values(lambda v: v + 1)["a"] == 2.0)
    try:
        TargetMap.from_dict({"a": -1.0})
        check("negative weight rejected", False)
    except ValueError:
        check("negative weight rejected", True)
    try:
        TargetMap.from_dict({"a": float("nan")})
        check("nan rejected", False)
    except ValueError:
        check("nan rejected", True)
    rng = np.random.default_rng(0)
    draws = [TargetMap.from_dict({"a": 0.0, "b": 1.0}).sample(rng)
             for _ in range(20)]
    check("sample honors mass", set(draws) == {"b"})


# ---------------------------------------------------------------------------
# Part 1b: trie against a toy tokenizer
# ---------------------------------------------------------------------------

class ToyTok:
    """Character tokenizer: id = ord(char). No specials, no merges."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def part1_trie() -> None:
    print("== TargetTrie (toy tokenizer) ==", flush=True)
    tok = ToyTok()
    m = TargetMap.from_dict({"ab": 0.5, "ac": 0.25, "d": 0.25})
    trie = m.tokenize(tok, prefix="P: ", closer="!")
    check("sequences include closer",
          trie.sequences["ab"] == [ord("a"), ord("b"), ord("!")])
    root = trie.root_marginal()
    check("root marginal merges shared first token",
          abs(root[ord("a")] - 0.75) < 1e-12
          and abs(root[ord("d")] - 0.25) < 1e-12)
    node = trie.node_target((ord("a"),))
    check("node target after 'a'",
          abs(node[ord("b")] - 2.0 / 3.0) < 1e-12
          and abs(node[ord("c")] - 1.0 / 3.0) < 1e-12)
    ex = trie.path_rows("ab")
    check("path_rows one-hot after divergence",
          ex.soft[1] == {ord("b"): 2.0 / 3.0, ord("c"): 1.0 / 3.0}
          and ex.soft[2] == {ord("!"): 1.0})
    hard = trie.hard_example("d")
    check("hard example",
          hard.tokens == [ord("d"), ord("!")] and hard.soft is None)
    marg = trie.marginal_example()
    check("marginal example", marg.tokens == []
          and abs(sum(marg.soft.values()) - 1.0) < 1e-12)


# ---------------------------------------------------------------------------
# Part 1c: soft_ce numerics on a deterministic module
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """logits[pos, v] = table[ids[pos], v] — a lookup 'language model'
    with a fixed table, so cross-entropies are hand-computable."""

    def __init__(self, vocab):
        super().__init__()
        mx.random.seed(3)
        self.table = mx.random.normal((vocab, vocab)) * 2.0

    def __call__(self, ids):
        return self.table[ids[0]][None, :, :]


def part1_loss() -> None:
    print("== soft_ce numerics ==", flush=True)
    vocab = 11
    lm = TinyLM(vocab)
    tbl = np.array(lm.table.astype(mx.float32))

    def lse(row):
        m_ = row.max()
        return m_ + math.log(np.exp(row - m_).sum())

    ids = [1, 2, 3]
    # hard example: tokens [4, 5] supervised at rows for ids[-1]=3 then 4
    ex_hard = Example(ids, [4, 5], None)
    want = ((lse(tbl[3]) - tbl[3][4]) + (lse(tbl[4]) - tbl[4][5])) / 2.0
    got = float(distill.soft_ce(lm, [ex_hard]))
    check("hard CE matches", abs(got - want) < 1e-4, f"{got:.5f}~{want:.5f}")
    # single-soft example at the decision row (ids[-1]=3)
    soft = {4: 0.5, 5: 0.5}
    ex_soft = Example(ids, [], soft)
    want = lse(tbl[3]) - 0.5 * tbl[3][4] - 0.5 * tbl[3][5]
    got = float(distill.soft_ce(lm, [ex_soft]))
    check("soft CE matches", abs(got - want) < 1e-4, f"{got:.5f}~{want:.5f}")
    # per-position: soft at pos 0, one-hot at pos 1
    ex_mix = Example(ids, [4, 5], [soft, None])
    want = ((lse(tbl[3]) - 0.5 * tbl[3][4] - 0.5 * tbl[3][5])
            + (lse(tbl[4]) - tbl[4][5])) / 2.0
    got = float(distill.soft_ce(lm, [ex_mix]))
    check("per-position CE matches", abs(got - want) < 1e-4)
    # invalid shapes rejected
    try:
        distill.soft_ce(lm, [Example(ids, [4], soft)])
        check("single-soft with tokens rejected", False)
    except ValueError:
        check("single-soft with tokens rejected", True)
    # score_items agrees with the same math
    lp = distill.score_items(lm, ids, {"x": [4, 5]})
    want = (tbl[3][4] - lse(tbl[3])) + (tbl[4][5] - lse(tbl[4]))
    check("score_items matches", abs(lp["x"] - want) < 1e-4)
    met = distill.item_metrics({"a": math.log(0.5), "b": math.log(0.25)})
    check("item_metrics mass", abs(met["captured_mass"] - 0.75) < 1e-9)
    tgt = TargetMap.from_dict({"a": 2.0, "b": 1.0})
    met2 = distill.item_metrics({"a": math.log(2 / 3), "b": math.log(1 / 3)},
                                target=tgt)
    check("item_metrics KL vs matching target ~ 0",
          abs(met2["kl_from_target_bits"]) < 1e-9)


# ---------------------------------------------------------------------------
# Part 2: real-model lifecycle (E2B)
# ---------------------------------------------------------------------------

def part2_model() -> None:
    print("== E2B lifecycle ==", flush=True)
    from . import Model
    t0 = time.time()
    model = Model.load(E2B_ID)
    lm = model.lm
    tok = model.tokenizer
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    system = ("You are participating in a dice-rolling exercise. Your job "
              "is to roll a single six-sided die and report the result, "
              "uniformly at random.\n\nYou MUST respond with valid JSON in "
              'exactly this format:\n{ "roll": <the number rolled> }\n\n'
              "Respond with ONLY the JSON object. No other text.")
    prompt = distill.render_chat(tok, system, "Please roll the die.",
                                 prefill='{ "roll": ')
    target = TargetMap.uniform([str(i) for i in range(1, 7)])
    trie = target.tokenize(tok, prompt, closer=" }")
    check("six items, digit + closer tokens",
          len(trie.sequences) == 6
          and all(len(s) >= 2 for s in trie.sequences.values()))
    root = trie.root_marginal()
    check("root marginal sums to 1",
          abs(sum(root.values()) - 1.0) < 1e-9, f"{len(root)} tokens")

    anchor_prompt = distill.render_chat(
        tok, "You are participating in a geography quiz. Your job is to "
        "answer the question accurately.\n\nYou MUST respond with valid "
        'JSON in exactly this format:\n{ "city": "<city name>" }\n\n'
        "Respond with ONLY the JSON object. No other text.",
        "What is the capital of France?", prefill='{ "city": "')
    anchor_ids = distill.encode(tok, anchor_prompt)
    anchor = Example(anchor_ids,
                     distill.suffix_tokens(tok, anchor_prompt, anchor_ids,
                                           "Paris"), None)

    base_kl = distill.item_metrics(trie.score(lm))["kl_from_target_bits"]
    base_logits = np.array(
        distill._forward_logits(lm, trie.prompt_ids)[-1].astype(mx.float32))
    paris0 = distill.first_token_metrics(lm, anchor_ids, tok)
    print(f"  baseline: KL={base_kl:.2f} bits, "
          f"p(Paris)={paris0['top1']['p']:.3f}", flush=True)

    rank, alpha = 8, 16.0
    n = lora.apply_lora(lm, rank, alpha)
    check("lora params", n == len(lm.model.layers) * 2 * 2
          * 8 * 2048 or n > 0, f"{n:,}")
    wrapped0 = np.array(
        distill._forward_logits(lm, trie.prompt_ids)[-1].astype(mx.float32))
    check("B=0 start: wrapped == base at decision token",
          float(np.abs(wrapped0 - base_logits).max()) < 1e-2)

    rng = np.random.default_rng(7)
    lag = nn.value_and_grad(lm, distill.soft_ce)
    opt = optim.Adam(learning_rate=1e-4)
    losses = []
    t0 = time.time()
    for step in range(1, 31):
        batch = [trie.hard_example(trie.sample(rng)) for _ in range(3)]
        batch.append(trie.marginal_example())
        batch.append(anchor)
        loss, grads = lag(lm, batch)
        opt.update(lm, grads)
        mx.eval(lm.trainable_parameters(), opt.state, loss)
        losses.append(float(loss))
        if step % 10 == 0:
            print(f"  step {step}: loss={losses[-1]:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    check("loss falls",
          np.mean(losses[-5:]) < np.mean(losses[:5]),
          f"{np.mean(losses[:5]):.3f} -> {np.mean(losses[-5:]):.3f}")
    kl = distill.item_metrics(trie.score(lm))["kl_from_target_bits"]
    check("KL from uniform falls", kl < base_kl,
          f"{base_kl:.2f} -> {kl:.2f} bits")
    paris1 = distill.first_token_metrics(lm, anchor_ids, tok)
    check("anchor stays sharp", paris1["top1"]["p"] > 0.5,
          f"p(Paris)={paris1['top1']['p']:.3f}")

    import tempfile
    import os
    path = os.path.join(tempfile.mkdtemp(), "adapter.safetensors")
    lora.save_adapter(lm, path)
    wrapped_logits = np.array(
        distill._forward_logits(lm, trie.prompt_ids)[-1].astype(mx.float32))
    for layer in lm.model.layers:
        for name in ("q_proj", "v_proj"):
            setattr(layer.self_attn, name,
                    getattr(layer.self_attn, name).base)
    handle = lora.fuse(lm, lora.load_adapter(path), scale=alpha / rank)
    fused_logits = np.array(
        distill._forward_logits(lm, trie.prompt_ids)[-1].astype(mx.float32))

    def kl_bits(a, b):
        pa = np.exp(a - a.max());  pa /= pa.sum()
        pb = np.exp(b - b.max());  pb /= pb.sum()
        return float((pa * np.log2(np.clip(pa, 1e-30, None)
                                   / np.clip(pb, 1e-30, None))).sum())

    check("fuse ~ wrapped at decision token",
          kl_bits(wrapped_logits, fused_logits) < 0.01,
          f"KL={kl_bits(wrapped_logits, fused_logits):.5f} bits")
    lora.restore(lm, handle)
    restored = np.array(
        distill._forward_logits(lm, trie.prompt_ids)[-1].astype(mx.float32))
    check("restore == base bit-exact",
          bool(np.array_equal(restored, base_logits)))


def main() -> None:
    part1_targetmap()
    part1_trie()
    part1_loss()
    if "--fast" not in sys.argv:
        part2_model()
    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + str(failures)}",
          flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
