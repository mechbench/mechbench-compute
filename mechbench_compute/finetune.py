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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from .distill import Example, TargetMap, TargetTrie, encode, soft_ce, suffix_tokens

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
        # A fetched `target_map` object arrives as its payload, `{kind,
        # weights}` — the form the lexicon's own example writes
        # (`"weights": {"$fetch": …}`), which used to fail on `kind`.
        if isinstance(weights, Mapping) and isinstance(weights.get("weights"), Mapping):
            weights = weights["weights"]
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


def compile_tries(
    tokenizer, target: TargetMap, rendered_prompts: Sequence[str],
    closer: str = " }",
) -> list[TargetTrie]:
    """The target compiled against each rendered prompt, in order."""
    return [target.tokenize(tokenizer, rendered, closer=closer)
            for rendered in rendered_prompts]


def build_target_items(
    tokenizer, target: TargetMap, rendered_prompts: Sequence[str],
    closer: str = " }", *, tries: Sequence[TargetTrie] | None = None,
) -> tuple[list[Example], list[Example]]:
    """Compile the target against each rendered prompt: (marginal
    soft-target items, one-hot continuation items). Pass ``tries`` from
    ``compile_tries`` to reuse a compilation."""
    marginals: list[Example] = []
    continuations: list[Example] = []
    if tries is None:
        tries = compile_tries(tokenizer, target, rendered_prompts, closer)
    for trie in tries:
        marginals.append(trie.marginal_example())
        for item in trie.items():
            seq = trie.sequences[item]
            if len(seq) >= 2:
                continuations.append(
                    Example(trie.prompt_ids + [seq[0]], [seq[1]]))
    return marginals, continuations


def build_path_factory(
    tries: Sequence[TargetTrie],
) -> Callable[[np.random.Generator], list[Example]]:
    """Whole-trie items (task 000548, ``batch.path``): each draw picks a
    prompt, samples an outcome by its target mass, and trains a soft row
    at every token of its path — the closer included — each row the
    trie's next-token distribution at that node.

    The 002 items train the first token and one second token per item,
    drawn uniformly over items, which is exact only for outcomes of at
    most two tokens. Sampling paths by mass instead trains every node in
    proportion to the mass that reaches it: in expectation, the
    chain-rule decomposition of KL(target ‖ model) over complete
    outcomes, however long they are and however many share a prefix."""
    tries = list(tries)
    if not tries:
        raise ValueError("path items need at least one training prompt")

    def factory(rng: np.random.Generator) -> list[Example]:
        trie = tries[int(rng.integers(len(tries)))]
        return [trie.path_rows(trie.sample(rng))]

    return factory


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


def draw_slots(targets: Sequence[TargetMap], rng: np.random.Generator,
               replace: bool = True) -> list[str]:
    """One outcome per slot, slot by slot. With ``replace=False`` a slot
    draws only from the outcomes not yet drawn, their weights
    renormalized (weighted successive sampling), so no outcome repeats."""
    drawn: list[str] = []
    for i, target in enumerate(targets):
        if replace:
            drawn.append(target.sample(rng))
            continue
        keys = [k for k in target if k not in drawn]
        w = np.array([target[k] for k in keys], dtype=np.float64)
        if not keys or w.sum() <= 0:
            raise ValueError(
                f"replace: false — slot {i} has nothing left to draw once "
                f"{drawn} are taken")
        drawn.append(keys[rng.choice(len(keys), p=w / w.sum())])
    return drawn


def check_enough_to_draw(targets: Sequence[TargetMap]) -> None:
    """Without replacement, slot ``i`` must hold more than ``i`` outcomes,
    or a draw can run out before the sequence is full."""
    for i, target in enumerate(targets):
        if len(target) <= i:
            raise ValueError(
                f"replace: false needs more than {i} outcomes at slot {i}; "
                f"it has {len(target)}")


def naturalism_gate(
    tokenizer, targets: Sequence[TargetMap], rendered_prompts: Sequence[str],
    base_ids: Sequence[list[int]], *, join: str, closer: str,
    samples: int = 40, seed: int = 11, replace: bool = True,
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
            item = join.join(draw_slots(targets, rng, replace))
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
    positions: str | Sequence[int] = "all", replace: bool = True,
) -> Callable[[np.random.Generator], list[Example]]:
    """A per-step item factory: each draw picks a prompt, samples one
    depth-N sequence per-slot from the targets, and returns the
    teacher-forced Example(s) for the trained position runs. Fresh
    sampling every step — the proven 016–018 recipe — rather than a
    fixed pool. ``replace=False`` draws the slots without replacement."""
    depth = len(targets)
    runs = position_runs(depth, positions)
    base_ids = [encode(tokenizer, r) for r in rendered_prompts]

    def factory(rng: np.random.Generator) -> list[Example]:
        pi = int(rng.integers(len(rendered_prompts)))
        item = join.join(draw_slots(targets, rng, replace))
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
# Item slots (depth N of whole outcomes — task 000548)


class SlotTrie:
    """One slot's outcomes as token paths: each outcome's own tokens, then
    the slot's terminator (the join's tokens when another slot follows,
    the closer after the last). The terminator is what ends an outcome,
    so "Mystery" and "Mystery Thriller" part at the token after
    " Mystery": the join, or " Thr".

    ``node_target`` leaves out outcomes already drawn: their mass comes
    off every node on their path, and a child that only drawn outcomes
    passed through is gone. Counts decide that, not subtraction, so no
    ghost child survives on a float residue."""

    def __init__(self, weights: Mapping[str, float],
                 paths: Mapping[str, list[int]]):
        total = float(sum(weights.values()))
        self.weights = {k: float(v) / total for k, v in weights.items()}
        self.paths = {k: list(v) for k, v in paths.items()}
        self.keys = list(self.weights)
        self._w = np.array([self.weights[k] for k in self.keys],
                           dtype=np.float64)
        self._index = {k: i for i, k in enumerate(self.keys)}
        # prefix -> {next token: [mass, outcomes through it]}
        self._nodes: dict[tuple[int, ...], dict[int, list[float]]] = {}
        for item, seq in self.paths.items():
            for j, t in enumerate(seq):
                child = self._nodes.setdefault(tuple(seq[:j]), {}).setdefault(t, [0.0, 0])
                child[0] += self.weights[item]
                child[1] += 1

    def node_target(self, prefix: Sequence[int] = (),
                    excluded: Sequence[str] = ()) -> dict[int, float]:
        """The normalized next-token distribution after ``prefix``, with
        the ``excluded`` outcomes' mass removed."""
        prefix = tuple(prefix)
        node = {t: [m, n] for t, (m, n) in self._nodes[prefix].items()}
        for item in excluded:
            seq = self.paths.get(item)
            if seq is None or len(seq) <= len(prefix) or tuple(seq[:len(prefix)]) != prefix:
                continue
            child = node[seq[len(prefix)]]
            child[0] -= self.weights[item]
            child[1] -= 1
        kept = {t: max(m, 0.0) for t, (m, n) in node.items() if n > 0}
        z = sum(kept.values())
        if z <= 0:
            raise ValueError("every outcome through this node has been drawn")
        return {t: m / z for t, m in kept.items()}

    def sample(self, rng: np.random.Generator,
               excluded: Sequence[str] = ()) -> str:
        """An outcome by mass, from those not ``excluded``."""
        w = self._w
        gone = [self._index[k] for k in excluded if k in self._index]
        if gone:
            w = w.copy()
            w[gone] = 0.0
        return self.keys[rng.choice(len(self.keys), p=w / w.sum())]


def _common_prefix(seqs: Sequence[list[int]]) -> list[int]:
    first = seqs[0]
    n = len(first)
    for seq in seqs[1:]:
        n = min(n, len(seq))
        for j in range(n):
            if seq[j] != first[j]:
                n = j
                break
    return list(first[:n])


def compile_item_slots(
    tokenizer, targets: Sequence[TargetMap], rendered: str, *,
    join: str, closer: str, gate: bool = True,
) -> list[SlotTrie]:
    """One ``SlotTrie`` per slot for one rendered prompt.

    An outcome tokenizes differently first in the list (straight after
    the prefill) than after the join (`"Science"` against `" Science"`),
    so the first slot's paths are tokenized after the prompt and later
    slots' after an outcome and the join. The join's own tokens are what
    every later path shares at its start (`","` for `", "`, the space
    going with the next word); they become the terminator of the slot
    before.

    The gate (``gate=True``) checks every outcome where it can stand: first
    and followed by the join, last and followed by the closer, and (at
    depth 3 or more) in the middle. Its tokens there must be exactly its
    path's, or training would teach the model a tokenization it never
    produces. A violation is a hard error naming the outcomes."""
    depth = len(targets)
    if depth < 2:
        raise ValueError("item slots need depth 2 or more")
    if join == "":
        raise ValueError(
            "item slots need a join: without one, the end of an outcome "
            "and the start of the next are the same place")
    if closer == "":
        raise ValueError(
            "item slots need a closer: without one, an outcome that is the "
            "start of another (\"Mystery\", \"Mystery Thriller\") has "
            "nothing to end on in the last slot")
    ids = encode(tokenizer, rendered)
    first_items = list(targets[0].keys())
    later_items: list[str] = []
    for target in targets[1:]:
        later_items.extend(k for k in target if k not in later_items)
    first_body = {k: suffix_tokens(tokenizer, rendered, ids, k)
                  for k in first_items}
    # Later paths are tokenized after a first outcome and the join. The
    # outcome that stands there must keep its own tokens when the join
    # follows; one that merges with the join is passed over here, and the
    # gate below names it.
    joined: dict[str, list[int]] = {}
    for lead in first_items:
        ctx = rendered + lead
        ctx_ids = encode(tokenizer, ctx)
        try:
            joined = {k: suffix_tokens(tokenizer, ctx, ctx_ids, join + k)
                      for k in later_items}
            break
        except ValueError:
            continue
    else:
        raise ValueError(
            f"ITEM SLOT VIOLATION: every outcome changes its tokens when "
            f"the join {join!r} follows it")
    join_ids = _common_prefix(list(joined.values()))
    if not join_ids:
        raise ValueError(
            f"item slots: the join {join!r} has no tokens of its own — it "
            f"merges into the outcomes that follow it")
    later_body = {k: seq[len(join_ids):] for k, seq in joined.items()}
    empty = [k for k, seq in later_body.items() if not seq]
    if empty:
        raise ValueError(f"item slots: outcomes with no tokens after the join: {empty[:10]}")
    closer_ids = suffix_tokens(tokenizer, ctx, ctx_ids, closer) if closer else []

    if gate:
        tail = later_items[0]
        violations: list[str] = []

        def check(item: str, parts: list[str], expect: list[int]) -> None:
            text = join.join(parts) + closer
            try:
                got = suffix_tokens(tokenizer, rendered, ids, text)
            except ValueError:
                got = None
            if got != expect and item not in violations:
                violations.append(item)

        for k in first_items:
            check(k, [k, tail], first_body[k] + join_ids + later_body[tail] + closer_ids)
        for k in later_items:
            check(k, [lead, k], first_body[lead] + join_ids + later_body[k] + closer_ids)
            if depth >= 3:
                check(k, [lead, k, tail],
                      first_body[lead] + join_ids + later_body[k] + join_ids
                      + later_body[tail] + closer_ids)
        if violations:
            raise ValueError(
                f"ITEM SLOT VIOLATION: {len(violations)} outcome(s) tokenize "
                f"differently inside a {join!r}-joined list than on their own "
                f"path: {violations[:10]}")

    slots: list[SlotTrie] = []
    built: dict[tuple[int, bool, bool], SlotTrie] = {}
    for i, target in enumerate(targets):
        first, last = i == 0, i == depth - 1
        key = (id(target), first, last)
        if key not in built:
            terminator = closer_ids if last else join_ids
            body = first_body if first else later_body
            weights = target.normalize().to_dict()
            built[key] = SlotTrie(weights, {k: body[k] + terminator for k in weights})
        slots.append(built[key])
    return slots


def build_item_path_factory(
    tokenizer, targets: Sequence[TargetMap], rendered_prompts: Sequence[str],
    *, join: str, closer: str, replace: bool = True, gate: bool = True,
) -> Callable[[np.random.Generator], list[Example]]:
    """Item-slot paths (``unit: "item"``): each draw picks a prompt and
    draws an outcome for every slot — without replacement when
    ``replace=False`` — and returns one Example with a soft row at every
    token: at slot ``i``, that slot's next-token distribution over the
    outcomes still available. So the model is trained toward the
    no-duplicates conditional itself, not only shown samples of it."""
    if not replace:
        check_enough_to_draw(targets)
    compiled = [compile_item_slots(tokenizer, targets, r, join=join,
                                   closer=closer, gate=gate)
                for r in rendered_prompts]
    base_ids = [encode(tokenizer, r) for r in rendered_prompts]

    def factory(rng: np.random.Generator) -> list[Example]:
        pi = int(rng.integers(len(rendered_prompts)))
        tokens: list[int] = []
        soft: list[dict[int, float] | None] = []
        drawn: list[str] = []
        for slot in compiled[pi]:
            excluded = () if replace else drawn
            item = slot.sample(rng, excluded)
            path = slot.paths[item]
            for j in range(len(path)):
                soft.append(slot.node_target(path[:j], excluded))
                tokens.append(path[j])
            drawn.append(item)
        return [Example(list(base_ids[pi]), tokens, soft)]

    return factory


def batch_for(depth: int, unit: str,
              batch: Mapping[str, int] | None) -> dict[str, int]:
    """The per-step batch for a target's shape, with its default, refusing
    an item kind the shape does not build. A kind nobody builds would be
    silently skipped, and the protocol would train on less than it says."""
    if depth <= 1:
        default = {"target": 3, "anchor": 1, "continuation": 2}
        kinds = {"target", "continuation", "path", "anchor"}
        shape = "depth 1"
    elif unit == "item":
        default = {"path": 3, "anchor": 1}
        kinds = {"path", "anchor"}
        shape = "unit: item"
    else:
        default = {"sequence": 3, "target": 1, "anchor": 1}
        kinds = {"sequence", "target", "anchor"}
        shape = f"depth {depth}, unit: token"
    chosen = dict(batch) if batch else default
    unbuilt = sorted(k for k, n in chosen.items() if int(n) > 0 and k not in kinds)
    if unbuilt:
        raise ValueError(
            f"batch {unbuilt} is not an item kind of {shape}; it builds "
            f"{sorted(kinds)}")
    return chosen


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
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[dict], None] | None = None,
    resume_state: Mapping | None = None,
) -> float:
    """The Regime D loop: per step, sample ``batch_sizes[g]`` items
    from each non-empty fixed group and DRAW ``batch_sizes[g]`` fresh
    item-lists from each factory group, take a soft-CE Adam step.
    Returns the final loss. ``on_step(step, loss)`` fires every step.

    Resume (epic 000320): every ``checkpoint_every`` steps
    ``on_checkpoint(state)`` receives the full continuation state
    (weights, optimizer, step, sampling RNG); ``resume_state``
    restores one and continues from the step after it. The
    continuation is bit-identical to the uninterrupted loop: the
    generator object whose state is restored is the same one the
    factories draw from."""
    from mechbench_compute.resume import (
        capture_training_state,
        restore_training_state,
    )

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    loss_and_grad = nn.value_and_grad(lm, soft_ce)
    opt = optim.Adam(learning_rate=lr)
    start = 1
    if resume_state is not None:
        start = restore_training_state(lm, opt, rng, resume_state) + 1

    active = [(g, items, int(batch_sizes.get(g, 0)))
              for g, items in groups.items()
              if items and int(batch_sizes.get(g, 0)) > 0]
    active_factories = [(g, f, int(batch_sizes.get(g, 0)))
                        for g, f in (factories or {}).items()
                        if int(batch_sizes.get(g, 0)) > 0]
    if not active and not active_factories:
        raise ValueError("no non-empty training groups with batch size > 0")

    loss_val = 0.0
    for step in range(start, int(steps) + 1):
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
        if (checkpoint_every and on_checkpoint is not None
                and step % int(checkpoint_every) == 0 and step < int(steps)):
            on_checkpoint(capture_training_state(lm, opt, step, rng))
    return loss_val
