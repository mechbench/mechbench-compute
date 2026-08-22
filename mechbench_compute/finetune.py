"""Fine-tuning as an operation (epic 000259): the Regime D training
loop from experiment 002, generalized into core primitives the
Finetune block orchestrates.

Structure mirrors the proven recipe exactly:

- **Target items**: per training prompt, a TargetMap compiled into a
  TargetTrie against the rendered prompt; the trie's root marginal is
  the soft target (soft-target CE at the decision token), and its
  sequences supply face -> closer continuation rows (one-hot).
- **Anchor items**: one-hot known-answer rows (capability preservation
  pressure — the "capital of France" battery's training side).
- **The loop**: batched sampling from the three groups, soft_ce loss,
  Adam, per-step callback for progress/curves.

The loop mutates ``lm`` in place via apply_lora; callers own saving
the adapter (lora.save_adapter) and disposing of the mutated model
(the runner reloads a clean base afterward).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .distill import Example, TargetMap, encode, soft_ce, suffix_tokens


def target_map_from_spec(spec: Mapping[str, Any]) -> TargetMap:
    """Build a TargetMap from a data spec: {"uniform": [items...]} or
    {"weights": {item: weight, ...}} (normalized), or a fetched
    target_map object payload ({"kind": "target_map", "weights": ...})."""
    if "uniform" in spec:
        return TargetMap.uniform([str(x) for x in spec["uniform"]])
    weights = spec.get("weights")
    if weights:
        return TargetMap({str(k): float(v) for k, v in weights.items()}).normalize()
    raise ValueError(f"unrecognized target spec: {list(spec.keys())}")


def build_target_items(
    tokenizer, target: TargetMap, rendered_prompts: Sequence[str],
    closer: str = " }",
) -> tuple[list[Example], list[Example]]:
    """Compile the target against each rendered prompt: (marginal
    soft-target items, one-hot continuation items)."""
    marginals: list[Example] = []
    continuations: list[Example] = []
    for rendered in rendered_prompts:
        trie = target.tokenize(tokenizer, rendered, closer=closer)
        marginals.append(trie.marginal_example())
        for item in trie.items():
            seq = trie.sequences[item]
            if len(seq) >= 2:
                continuations.append(
                    Example(trie.prompt_ids + [seq[0]], [seq[1]]))
    return marginals, continuations


def build_anchor_items(
    tokenizer, anchors: Sequence[tuple[str, str]],
) -> list[Example]:
    """One-hot anchors from (rendered_prompt, answer_continuation)."""
    items = []
    for rendered, answer in anchors:
        ids = encode(tokenizer, rendered)
        first = suffix_tokens(tokenizer, rendered, ids, answer)[0]
        items.append(Example(ids, [first]))
    return items


def train_soft_ce(
    lm,
    groups: Mapping[str, list[Example]],
    batch_sizes: Mapping[str, int],
    *,
    steps: int,
    lr: float = 1e-4,
    seed: int = 7,
    on_step: Callable[[int, float], None] | None = None,
) -> float:
    """The Regime D loop: per step, sample `batch_sizes[g]` items from
    each non-empty group, take a soft-CE Adam step. Returns the final
    loss. ``on_step(step, loss)`` fires every step (progress hooks)."""
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    loss_and_grad = nn.value_and_grad(lm, soft_ce)
    opt = optim.Adam(learning_rate=lr)

    active = [(g, items, int(batch_sizes.get(g, 0)))
              for g, items in groups.items()
              if items and int(batch_sizes.get(g, 0)) > 0]
    if not active:
        raise ValueError("no non-empty training groups with batch size > 0")

    loss_val = 0.0
    for step in range(1, int(steps) + 1):
        batch: list[Example] = []
        for _, items, k in active:
            take = min(k, len(items))
            for i in rng.choice(len(items), take, replace=False):
                batch.append(items[int(i)])
        loss, grads = loss_and_grad(lm, batch)
        opt.update(lm, grads)
        mx.eval(lm.trainable_parameters(), opt.state, loss)
        loss_val = float(loss)
        if on_step:
            on_step(step, loss_val)
    return loss_val
