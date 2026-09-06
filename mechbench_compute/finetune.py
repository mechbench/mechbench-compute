"""Fine-tuning as an operation (epic 000259): the Regime D training
loop from experiment 002, generalized into core primitives the
Finetune block orchestrates.

Structure mirrors the proven recipes exactly:

- **Target items** (depth 1, the 002 die shape): per training prompt,
  a TargetMap compiled into a TargetTrie against the rendered prompt;
  the trie's root marginal is the soft target and its sequences
  supply continuation rows.
- **Sequence items** (depth N, the 016–018 deep-trie shape): per
  step, a freshly sampled depth-N sequence drawn per-slot from the
  target map(s), teacher-forced — plus (optionally) the first-slot
  marginal soft row. Which positions receive loss is configurable
  (``positions``): the position-0 cap from experiment 018's proposal
  is ``"skip_first"`` — the first token is conditioned on, never
  trained.
- **Anchor items**: one-hot known-answer rows (capability
  preservation pressure).
- **The loop**: batched sampling from fixed groups and per-step item
  FACTORIES (fresh sequences every step, as the proven trainers did),
  soft_ce loss, Adam, per-step callback.

Target specs are declarative — raw weights plus a TRANSFORM CHAIN
(sqrt/pow/temper/temper_to_entropy/mix_uniform/top_k), so a protocol
can carry one empirical-frequency object and express any point on the
target-shape dial as data.

The loop mutates ``lm`` in place via apply_lora; callers own saving
the adapter (lora.save_adapter) and disposing of the mutated model
(the runner reloads a clean base afterward).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .distill import Example, TargetMap, encode, soft_ce, suffix_tokens

# ---------------------------------------------------------------------------
# Target specs


def entropy_bits(target: TargetMap) -> float:
    """Shannon entropy (bits) of the normalized map."""
    weights = target.normalize().to_dict()
    return float(-sum(p * math.log2(p) for p in weights.values() if p > 0))


def temper_to_entropy(target: TargetMap, bits: float,
                      tolerance: float = 1e-4) -> TargetMap:
    """Solve the temperature T (bisection) so that ``target.temper(T)``
    has per-slot entropy ``bits``. Entropy is monotone increasing in T
    (T→0 sharpens toward the argmax, T→∞ flattens toward uniform), so
    the solve is exact up to ``tolerance``. ``bits`` must lie strictly
    between the map's min entropy (~0) and log2(support size)."""
    n = len(list(target.keys()))
    h_max = math.log2(n)
    if not 0.0 < bits < h_max:
        raise ValueError(
            f"temper_to_entropy: bits={bits} outside (0, log2({n})="
            f"{h_max:.3f})")
    lo, hi = 1e-4, 1e4
    for _ in range(200):
        mid = math.sqrt(lo * hi)  # geometric: T spans orders of magnitude
        h = entropy_bits(target.temper(mid))
        if abs(h - bits) <= tolerance:
            return target.temper(mid)
        if h < bits:
            lo = mid
        else:
            hi = mid
    return target.temper(math.sqrt(lo * hi))


_TRANSFORMS: dict[str, Callable[[TargetMap, Mapping[str, Any]], TargetMap]] = {
    "sqrt": lambda t, s: t.sqrt(),
    "pow": lambda t, s: t.pow(float(s["exponent"])),
    "temper": lambda t, s: t.temper(float(s["temperature"])),
    "temper_to_entropy": lambda t, s: temper_to_entropy(
        t, float(s["bits"]), float(s.get("tolerance", 1e-4))),
    "mix_uniform": lambda t, s: t.mix_uniform(float(s["epsilon"])),
    "top_k": lambda t, s: t.top_k(int(s["k"])),
    "normalize": lambda t, s: t.normalize(),
}


def target_map_from_spec(spec: Mapping[str, Any]) -> TargetMap:
    """Build a TargetMap from a data spec.

    Base (one of):
      {"uniform": [items...]}          equal weights over the items
      {"weights": {item: weight}}      arbitrary weights (e.g. raw
                                       corpus frequencies, or a fetched
                                       target_map object payload)

    Optional ``"transform"``: a LIST of steps applied in order, each
    {"op": name, ...args} with ops sqrt | pow(exponent) |
    temper(temperature) | temper_to_entropy(bits[, tolerance]) |
    mix_uniform(epsilon) | top_k(k) | normalize. The result is always
    normalized. This makes the target-shape dial declarative: carry
    ONE raw-frequency object and express identity / sqrt / uniform /
    tempered-inverse as transform chains.
    """
    if "uniform" in spec:
        target = TargetMap.uniform([str(x) for x in spec["uniform"]])
    else:
        weights = spec.get("weights")
        if not weights:
            raise ValueError(f"unrecognized target spec: {list(spec.keys())}")
        target = TargetMap({str(k): float(v) for k, v in weights.items()})
    for step in spec.get("transform") or []:
        op = step.get("op")
        if op not in _TRANSFORMS:
            raise ValueError(f"unknown target transform: {op!r} "
                             f"(known: {sorted(_TRANSFORMS)})")
        target = _TRANSFORMS[op](target, step)
    return target.normalize()


# ---------------------------------------------------------------------------
# Item builders (depth 1 — the 002 shape)


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


# ---------------------------------------------------------------------------
# Sequence items (depth N — the 016–018 deep-trie shape)


def position_runs(depth: int,
                  positions: str | Sequence[int]) -> list[tuple[int, int]]:
    """Resolve a positions spec into contiguous [start, end) trained
    runs over slots 0..depth-1.

    ``"all"`` trains every slot; ``"skip_first"`` (the 018 position-0
    cap) trains slots 1..depth-1; an explicit list of slot indices
    trains exactly those. Untrained slots are still CONDITIONED ON —
    each run becomes its own teacher-forced Example whose prompt
    includes every earlier slot's sampled token — so the sampled
    sequence stays coherent while loss lands only where asked."""
    if positions == "all":
        trained = set(range(depth))
    elif positions == "skip_first":
        trained = set(range(1, depth))
    else:
        trained = {int(i) for i in positions}
        bad = [i for i in trained if not 0 <= i < depth]
        if bad:
            raise ValueError(f"positions out of range for depth {depth}: {bad}")
    if not trained:
        raise ValueError("positions trains no slots")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(depth + 1):
        if i < depth and i in trained:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    return runs


def resolve_slot_targets(
    target_spec: Mapping[str, Any], depth: int,
) -> list[TargetMap]:
    """One TargetMap per slot. ``per_slot`` (a list of specs, length ==
    depth) gives each slot its own distribution; otherwise the
    top-level spec is shared by every slot (the iid recipe)."""
    per_slot = target_spec.get("per_slot")
    if per_slot is not None:
        if len(per_slot) != depth:
            raise ValueError(
                f"per_slot has {len(per_slot)} specs for depth {depth}")
        return [target_map_from_spec(s) for s in per_slot]
    return [target_map_from_spec(target_spec)] * depth


def naturalism_gate(
    tokenizer, targets: Sequence[TargetMap], rendered_prompts: Sequence[str],
    base_ids: Sequence[list[int]], *, join: str, closer: str,
    samples: int = 40, seed: int = 11,
) -> None:
    """The phase-0 gate from the deep-trie trainers, as a block
    invariant: every sampled sequence must tokenize to exactly one
    token per slot (plus the closer), so slot i ↔ token i and per-slot
    targets/caps mean what they say. Hard error on violation."""
    rng = np.random.default_rng(seed)
    depth = len(targets)
    for pi, (rendered, ids) in enumerate(zip(rendered_prompts, base_ids)):
        seen: dict[int, int] = {}
        for _ in range(samples):
            item = join.join(t.sample(rng) for t in targets)
            seq = suffix_tokens(tokenizer, rendered, list(ids), item + closer)
            d = len(seq) - (1 if closer else 0)
            seen[d] = seen.get(d, 0) + 1
        if list(seen) != [depth]:
            raise ValueError(
                f"NATURALISM VIOLATION prompt {pi}: sampled sequences "
                f"tokenized to depths {seen} (expected all {depth}) — "
                f"per-slot training is not meaningful for this "
                f"vocabulary/tokenizer pairing")


def build_sequence_factory(
    tokenizer, targets: Sequence[TargetMap], rendered_prompts: Sequence[str],
    *, join: str = "", closer: str = "",
    positions: str | Sequence[int] = "all",
) -> Callable[[np.random.Generator], list[Example]]:
    """A per-step item factory: each draw picks a prompt, samples one
    depth-N sequence per-slot from the targets, and returns the
    teacher-forced Example(s) for the trained position runs. Fresh
    sampling every step — the proven 016–018 recipe — rather than a
    fixed pool."""
    depth = len(targets)
    runs = position_runs(depth, positions)
    base_ids = [encode(tokenizer, r) for r in rendered_prompts]

    def factory(rng: np.random.Generator) -> list[Example]:
        pi = int(rng.integers(len(rendered_prompts)))
        item = join.join(t.sample(rng) for t in targets)
        seq = suffix_tokens(tokenizer, rendered_prompts[pi],
                            list(base_ids[pi]), item + closer)
        # Under the gate: seq[:depth] are the slot tokens, the tail is
        # the closer — trained with the final run so the model still
        # learns to stop.
        out: list[Example] = []
        for ri, (a, b) in enumerate(runs):
            end = len(seq) if ri == len(runs) - 1 else b
            out.append(Example(list(base_ids[pi]) + list(seq[:a]),
                               list(seq[a:end])))
        return out

    return factory


def build_marginal_items(
    tokenizer, target: TargetMap, rendered_prompts: Sequence[str],
) -> list[Example]:
    """First-slot marginal soft rows — the position-0 pressure term of
    the deep-trie recipe, split out so the cap can DROP it."""
    items: list[Example] = []
    for rendered in rendered_prompts:
        ids = encode(tokenizer, rendered)
        soft: dict[int, float] = {}
        for word, p in target.normalize().to_dict().items():
            tid = suffix_tokens(tokenizer, rendered, ids, word)[0]
            soft[tid] = soft.get(tid, 0.0) + p
        items.append(Example(ids, [], soft))
    return items


# ---------------------------------------------------------------------------
# The loop


def train_soft_ce(
    lm,
    groups: Mapping[str, list[Example]],
    batch_sizes: Mapping[str, int],
    *,
    steps: int,
    lr: float = 1e-4,
    seed: int = 7,
    factories: Mapping[str, Callable[[np.random.Generator],
                                     list[Example]]] | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> float:
    """The Regime D loop: per step, sample ``batch_sizes[g]`` items
    from each non-empty fixed group and DRAW ``batch_sizes[g]`` fresh
    item-lists from each factory group, take a soft-CE Adam step.
    Returns the final loss. ``on_step(step, loss)`` fires every step."""
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    loss_and_grad = nn.value_and_grad(lm, soft_ce)
    opt = optim.Adam(learning_rate=lr)

    active = [(g, items, int(batch_sizes.get(g, 0)))
              for g, items in groups.items()
              if items and int(batch_sizes.get(g, 0)) > 0]
    active_factories = [(g, f, int(batch_sizes.get(g, 0)))
                        for g, f in (factories or {}).items()
                        if int(batch_sizes.get(g, 0)) > 0]
    if not active and not active_factories:
        raise ValueError("no non-empty training groups with batch size > 0")

    loss_val = 0.0
    for step in range(1, int(steps) + 1):
        batch: list[Example] = []
        for _, items, k in active:
            take = min(k, len(items))
            for i in rng.choice(len(items), take, replace=False):
                batch.append(items[int(i)])
        for _, factory, k in active_factories:
            for _ in range(k):
                batch.extend(factory(rng))
        loss, grads = loss_and_grad(lm, batch)
        opt.update(lm, grads)
        mx.eval(lm.trainable_parameters(), opt.state, loss)
        loss_val = float(loss)
        if on_step:
            on_step(step, loss_val)
    return loss_val
