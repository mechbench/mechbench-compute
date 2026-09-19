"""Distributional-target training primitives (task 000114).

The core idea: instead of training on example *responses*, train directly
on a target *distribution* over responses. At a decision token, soft-target
cross-entropy against a distribution T has gradient P − T (predicted minus
target), so the model is pushed toward emitting the specified distribution —
uniform, shaped, or one-hot — rather than toward any single answer.

Two data structures carry the target:

- ``TargetMap`` — an immutable ``Map<String, Double>``: string items with
  non-negative weights. Load one from a dict or a flat JSON object, or
  build ``TargetMap.uniform(items)``. Whole-map transforms (``sqrt``,
  ``pow``, ``scale``, ``temper``, ``mix_uniform``, ``top_k``, ``filter``,
  ``map_values``) each return a new map.
- ``TargetTrie`` — the tokenized form, produced by
  ``TargetMap.tokenize(tokenizer, prefix, closer=...)``. Items become
  token paths in context (encoded against the rendered prompt so BPE
  boundaries are honest); shared token prefixes share trie nodes. The trie
  answers the questions training needs: the exact first-token marginal
  (``root_marginal``), per-node next-token distributions along an item's
  path (``path_rows``), and weighted sampling of items (``sample``).

Transforms live on the flat map; the trie is a compiled snapshot. For a
nonlinear transform like sqrt, transform-then-tokenize and
tokenize-then-transform differ — so there is exactly one blessed order:
transform first, then ``tokenize``.

Training composes three kinds of ``Example`` (see ``soft_ce``):

- sampled hard items — draw an item from the target and teacher-force its
  tokens plus the closer. In expectation over draws this equals the soft
  loss, and it exercises the full multi-token path.
- the exact root marginal — one soft row at the decision token.
- sharp anchors — one-hot examples of *confident* tasks mixed into every
  batch, so sharpness elsewhere is preserved rather than melted. Include
  closer tokens after each item (continuation anchors), or the flattened
  distribution leaks into free generation beyond the envelope.

Typical loop::

    from mechbench_compute import distill, lora

    target = distill.TargetMap.uniform([str(i) for i in range(1, 7)])
    prompt = distill.render_chat(tok, system, user, prefill='{ "roll": ')
    trie = target.tokenize(tok, prompt, closer=" }")

    n = lora.apply_lora(model.lm)
    loss_and_grad = nn.value_and_grad(model.lm, distill.soft_ce)
    for step in range(steps):
        batch = [trie.hard_example(trie.sample(rng)) for _ in range(3)]
        batch.append(trie.marginal_example())
        batch.append(anchor)                      # a hard Example
        loss, grads = loss_and_grad(model.lm, batch)
        opt.update(model.lm, grads)
        mx.eval(model.lm.trainable_parameters(), opt.state, loss)

Everything here operates on ``Model.lm`` (the text decoder, uniform across
families) via plain module calls — not the hook-aware forward, which is
for instrumentation. See README ("Two forward paths").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, NamedTuple

import mlx.core as mx
import numpy as np

__all__ = [
    "Example",
    "TargetMap",
    "TargetTrie",
    "encode",
    "first_token_metrics",
    "item_metrics",
    "render_chat",
    "score_items",
    "soft_ce",
    "suffix_tokens",
]


# ---------------------------------------------------------------------------
# TargetMap — the flat Map<String, Double>
# ---------------------------------------------------------------------------

class TargetMap:
    """An immutable weighted map over string items.

    Weights must be finite and non-negative. Transforms return new maps and
    do not renormalize unless the operation is distribution-semantic by
    nature (``temper``, ``mix_uniform``) — call ``normalize()`` explicitly
    when you want probabilities. ``tokenize`` and ``sample`` normalize
    internally, so an unnormalized map is fine to train from.
    """

    __slots__ = ("_w",)

    def __init__(self, weights: Mapping[str, float]):
        w = {}
        for k, v in weights.items():
            v = float(v)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(
                    f"TargetMap weight for {k!r} must be finite and >= 0, "
                    f"got {v}")
            w[str(k)] = v
        if not w:
            raise ValueError("TargetMap must contain at least one item")
        self._w = w

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_dict(cls, weights: Mapping[str, float]) -> "TargetMap":
        """Build from a ``{item: weight}`` mapping (weights need not sum
        to 1)."""
        return cls(weights)

    @classmethod
    def from_json(cls, path: str) -> "TargetMap":
        """Load a flat JSON object ``{"item": weight, ...}``."""
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a flat JSON object of item -> weight")
        return cls(data)

    @classmethod
    def uniform(cls, items: Iterable[str]) -> "TargetMap":
        """Equal weight on every item."""
        items = list(items)
        return cls({k: 1.0 / len(items) for k in items})

    # -- transforms (each returns a new TargetMap) -------------------------

    def map_values(self, fn: Callable[[float], float]) -> "TargetMap":
        """Apply ``fn`` to every weight; the general transform the named
        ones are shorthand for."""
        return TargetMap({k: fn(v) for k, v in self._w.items()})

    def sqrt(self) -> "TargetMap":
        """Square root of every weight (flattens a peaked map)."""
        return self.map_values(math.sqrt)

    def pow(self, exponent: float) -> "TargetMap":
        """Raise every weight to ``exponent``."""
        return self.map_values(lambda v: v ** exponent)

    def scale(self, factor: float) -> "TargetMap":
        """Multiply every weight by ``factor`` (must be >= 0)."""
        return self.map_values(lambda v: v * factor)

    def temper(self, temperature: float) -> "TargetMap":
        """Temperature-scale as a distribution: ``p ** (1/T)``, then
        normalize. T > 1 flattens toward uniform, T < 1 sharpens, T = 1 is
        identity."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        return self.pow(1.0 / temperature).normalize()

    def mix_uniform(self, epsilon: float) -> "TargetMap":
        """``(1 − ε)·p + ε·uniform`` over the current support, normalized.
        Guarantees every item a floor of mass ε/n."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        p = self.normalize()
        u = 1.0 / len(p._w)
        return TargetMap({k: (1 - epsilon) * v + epsilon * u
                          for k, v in p._w.items()})

    def top_k(self, k: int) -> "TargetMap":
        """Keep the ``k`` heaviest items (ties broken by key, for
        determinism)."""
        kept = sorted(self._w.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return TargetMap(dict(kept))

    def filter(self, predicate: Callable[[str, float], bool]) -> "TargetMap":
        """Keep items where ``predicate(item, weight)`` is true."""
        return TargetMap({k: v for k, v in self._w.items()
                          if predicate(k, v)})

    def normalize(self) -> "TargetMap":
        """Scale weights to sum to 1."""
        t = self.total()
        if t <= 0.0:
            raise ValueError("cannot normalize a TargetMap with zero total")
        return self.scale(1.0 / t)

    # -- views -------------------------------------------------------------

    def items(self):
        return self._w.items()

    def keys(self):
        return self._w.keys()

    def values(self):
        return self._w.values()

    def total(self) -> float:
        return sum(self._w.values())

    def to_dict(self) -> dict[str, float]:
        return dict(self._w)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._w, f, indent=1, sort_keys=True)

    def __getitem__(self, key: str) -> float:
        return self._w[key]

    def __contains__(self, key: str) -> bool:
        return key in self._w

    def __len__(self) -> int:
        return len(self._w)

    def __iter__(self) -> Iterator[str]:
        return iter(self._w)

    def __repr__(self) -> str:
        head = ", ".join(f"{k!r}: {v:.4g}"
                         for k, v in list(self._w.items())[:4])
        more = "" if len(self._w) <= 4 else f", … {len(self._w)} items"
        return f"TargetMap({{{head}}}{more})"

    def sample(self, rng: np.random.Generator) -> str:
        """Draw one item with probability proportional to its weight."""
        keys = list(self._w)
        w = np.array([self._w[k] for k in keys], dtype=np.float64)
        return keys[rng.choice(len(keys), p=w / w.sum())]

    # -- the bridge to token space ----------------------------------------

    def tokenize(self, tokenizer, prefix: str,
                 closer: str = "") -> "TargetTrie":
        """Compile this map into a ``TargetTrie`` against a tokenizer.

        ``prefix`` is the full rendered prompt (chat template + any
        prefill) the items will follow; each item is encoded *in context*
        (``prefix + item + closer``, then the prefix ids are sliced off),
        so BPE boundary effects are exactly those of real generation.
        ``closer`` (e.g. ``' }'`` to close a JSON envelope) is appended to
        every item's path as continuation-anchor tokens.
        """
        return TargetTrie(self, tokenizer, prefix, closer)


# ---------------------------------------------------------------------------
# TargetTrie — the tokenized form
# ---------------------------------------------------------------------------

class Example(NamedTuple):
    """One supervised training item for ``soft_ce``.

    - ``soft is None`` — teacher-forced hard example: every position in
      ``tokens`` gets one-hot cross-entropy.
    - ``soft`` is a ``{token_id: mass}`` dict — a single soft row at the
      decision position (immediately after the prompt); ``tokens`` must be
      empty.
    - ``soft`` is a list (same length as ``tokens``) — per-position
      targets: a dict for a soft row, ``None`` for one-hot on
      ``tokens[j]``.
    """

    prompt_ids: list[int]
    tokens: list[int]
    soft: dict[int, float] | list[dict[int, float] | None] | None = None


class TargetTrie:
    """A ``TargetMap`` compiled to token space. Built by
    ``TargetMap.tokenize``; not constructed directly in normal use."""

    def __init__(self, target: TargetMap, tokenizer, prefix: str,
                 closer: str = ""):
        norm = target.normalize()
        self.prefix = prefix
        self.closer = closer
        self.prompt_ids: list[int] = encode(tokenizer, prefix)
        self.weights: dict[str, float] = norm.to_dict()
        self.sequences: dict[str, list[int]] = {}
        for item in norm:
            seq = suffix_tokens(tokenizer, prefix, self.prompt_ids,
                                item + closer)
            if not seq:
                raise ValueError(
                    f"item {item!r} + closer tokenized to nothing")
            self.sequences[item] = seq
        # Trie: prefix-tuple -> {next_token_id: mass}
        self._nodes: dict[tuple[int, ...], dict[int, float]] = {}
        for item, seq in self.sequences.items():
            w = self.weights[item]
            for j, t in enumerate(seq):
                node = self._nodes.setdefault(tuple(seq[:j]), {})
                node[t] = node.get(t, 0.0) + w

    def items(self) -> list[str]:
        return list(self.weights)

    def node_target(self, prefix: tuple[int, ...] = ()) -> dict[int, float]:
        """Normalized next-token distribution at a trie node (keyed by the
        token path from the decision point). The root ``()`` is the
        distribution over first tokens."""
        node = self._nodes[tuple(prefix)]
        z = sum(node.values())
        return {t: w / z for t, w in node.items()}

    def root_marginal(self) -> dict[int, float]:
        """The exact target distribution over *first* tokens — each item's
        mass summed onto its first token id."""
        return self.node_target(())

    def sample(self, rng: np.random.Generator) -> str:
        """Draw an item with probability equal to its target mass."""
        keys = list(self.weights)
        w = np.array([self.weights[k] for k in keys], dtype=np.float64)
        return keys[rng.choice(len(keys), p=w / w.sum())]

    # -- Example builders --------------------------------------------------

    def hard_example(self, item: str) -> Example:
        """Teacher-forced one-hot example for ``item`` (tokens include the
        closer). Sampling items from the target and training on these is
        the soft loss in expectation."""
        return Example(self.prompt_ids, self.sequences[item], None)

    def marginal_example(self) -> Example:
        """One soft row at the decision token, targeting the exact
        first-token marginal."""
        return Example(self.prompt_ids, [], self.root_marginal())

    def path_rows(self, item: str) -> Example:
        """Per-node soft targets along ``item``'s full path: at each
        position, the trie's next-token distribution given the tokens so
        far (one-hot once the path is unshared, including the closer)."""
        seq = self.sequences[item]
        soft = [self.node_target(tuple(seq[:j])) for j in range(len(seq))]
        return Example(self.prompt_ids, seq, soft)

    def score(self, lm) -> dict[str, float]:
        """Teacher-forced ``log P(item + closer)`` for every item; feed to
        ``item_metrics`` for calibration diagnostics."""
        return score_items(lm, self.prompt_ids, self.sequences)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _forward_logits(lm, ids: list[int]) -> mx.array:
    out = lm(mx.array([ids]))
    return (out.logits if hasattr(out, "logits") else out)[0]


def soft_ce(lm, batch: list[Example]) -> mx.array:
    """Mean cross-entropy over a batch of ``Example``s, soft targets
    included; use with ``nn.value_and_grad(lm, soft_ce)``.

    At a soft row the gradient w.r.t. the logits is P − T. Rows are sliced
    out of the sequence *before* the fp32 cast — materializing the full
    [seq, vocab] logits in fp32 inside the gradient graph is what
    triggered Metal command-buffer watchdog kills on long prompts.
    """
    total = mx.zeros(())
    count = 0
    for ex in batch:
        ids, seq, soft = ex.prompt_ids, ex.tokens, ex.soft
        if isinstance(soft, dict):
            if seq:
                raise ValueError(
                    "single-soft Example must have empty tokens")
            row = _forward_logits(lm, ids)[len(ids) - 1].astype(mx.float32)
            tid = mx.array(list(soft.keys()))
            tw = mx.array(list(soft.values()))
            total = total + (mx.logsumexp(row)
                             - mx.sum(tw * mx.take(row, tid)))
            count += 1
            continue
        if not seq:
            raise ValueError("hard Example must have at least one token")
        if soft is not None and len(soft) != len(seq):
            raise ValueError("per-position soft list must match tokens")
        fed = (ids + seq[:-1]) if len(seq) > 1 else ids
        logits = _forward_logits(lm, fed)
        for j, t in enumerate(seq):
            row = logits[len(ids) - 1 + j].astype(mx.float32)
            tgt = soft[j] if soft is not None else None
            if tgt is None:
                total = total + (mx.logsumexp(row) - row[t])
            else:
                tid = mx.array(list(tgt.keys()))
                tw = mx.array(list(tgt.values()))
                total = total + (mx.logsumexp(row)
                                 - mx.sum(tw * mx.take(row, tid)))
            count += 1
    return total / count


# ---------------------------------------------------------------------------
# Prompt/envelope helpers
# ---------------------------------------------------------------------------

def render_chat(tokenizer, system: str, user: str, prefill: str = "",
                date_string: str | None = None) -> str:
    """Render a single-turn prompt (system and user merged into one user
    message, matching how the instruction-tuned chat templates here are
    exercised) plus an assistant prefill (e.g. ``'{ "roll": '``).

    Reproducibility hazard: some chat templates (e.g. Llama 3.2) inject
    the *live* date ("Today Date: ...") into every render, so identical
    code produces different prompts — and different measurements — on
    different days. Pass ``date_string`` (e.g. ``"05 Aug 2026"``) to pin
    it for such templates; templates without a date ignore it. Left
    unset, the template's own default (usually today) applies.
    """
    merged = (system + "\n\n" + user) if system else user
    kwargs = {} if date_string is None else {"date_string": date_string}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": merged}],
        tokenize=False, add_generation_prompt=True, **kwargs) + prefill


@dataclass(frozen=True)
class Rendered:
    """A record, rendered: the token ids the model sees, the text they
    came from, whether the chat template applied, and the prompt's
    length in tokens — every id, for a record without a trace, which is
    what `"generated"` positions count from."""

    ids: list[int]
    text: str
    chat: bool

    @property
    def prompt_len(self) -> int:
        return len(self.ids)

    @property
    def array(self):
        return mx.array([self.ids], dtype=mx.int32)

    def tokens(self, tokenizer) -> list[str]:
        return [tokenizer.decode([int(t)]) for t in self.ids]


def render(model, record: Mapping[str, Any], *, date_string: str | None = None) -> Rendered:
    """One way from a record to token ids.

    A condition — `user`, optional `system`, optional `prefill` — renders
    through the model's chat template as one user turn, the assistant's
    turn begun with the prefill; that is where a decision is read and
    what every model op sees. A record that says `template: false`, or
    that carries only `text` or `prompt`, is tokenized raw (its prefill,
    if any, appended); `template: "chat"` on such a record renders it as
    the user turn. The retired `template: "raw"` reads as `false`.
    """
    template = record.get("template")
    raw = template is False or template == "raw"
    text = record.get("user") or record.get("prompt") or record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"record {record.get('id')!r} has no prompt: expected `user` "
            "(a condition), `prompt`, or `text` (a document)")
    chat = (template is True or template == "chat"
            or ("user" in record and not raw))
    tok = model.tokenizer
    prefill = str(record.get("prefill") or "")
    if chat:
        kw = {} if date_string is None else {"date_string": date_string}
        rendered = render_chat(tok, str(record.get("system") or ""), text, prefill, **kw)
    else:
        rendered = text + prefill
    return Rendered(encode(tok, rendered), rendered, chat)


def encode(tokenizer, text: str) -> list[int]:
    """Tokenize without re-adding special tokens (tolerating tokenizers
    that lack the kwarg)."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def suffix_tokens(tokenizer, prefix: str, prefix_ids: list[int],
                  text: str) -> list[int]:
    """Token ids ``text`` contributes when it follows ``prefix`` —
    ``encode(prefix + text)`` minus the prefix ids, asserting the prefix
    tokenization is unchanged (BPE can otherwise shift the boundary)."""
    full = encode(tokenizer, prefix + text)
    if full[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"tokenizing {text!r} in context changed the prefix "
            f"tokenization; adjust the envelope so the boundary is stable")
    return full[len(prefix_ids):]


# ---------------------------------------------------------------------------
# Calibration scoring (the measurement half)
# ---------------------------------------------------------------------------

def score_items(lm, prompt_ids: list[int],
                sequences: Mapping[str, list[int]]) -> dict[str, float]:
    """Teacher-forced ``log P(sequence)`` for each item, in nats."""
    out = {}
    for item, seq in sequences.items():
        fed = (prompt_ids + seq[:-1]) if len(seq) > 1 else prompt_ids
        logits = _forward_logits(lm, fed)
        lp = 0.0
        for j, t in enumerate(seq):
            row = logits[len(prompt_ids) - 1 + j].astype(mx.float32)
            lp += float(row[t] - mx.logsumexp(row))
        out[item] = lp
    return out


def score_items_batched(lm, prompt_ids: list[int],
                        sequences: Mapping[str, list[int]],
                        chunk: int = 16) -> dict[str, float]:
    """Batched teacher-forced ``log P(sequence)`` — same math as
    ``score_items``, restructured for throughput on large item sets
    (e.g. 200-name calibration batteries).

    Items are grouped by tokenized length (one tensor shape per group —
    keeps the Metal buffer cache stable), stacked into forwards of up to
    ``chunk`` items, and each item's whole log-prob is reduced on-graph,
    so the host syncs once per chunk instead of once per token position.

    Numerical note: batched matmuls tile differently than single-item
    forwards, so results can differ from ``score_items`` at bf16 rounding
    level — measurable in flat-target KL diagnostics (see the
    living-experiments RERUN records). Opt in deliberately; don't swap it
    into an experiment mid-comparison.
    """
    L = len(prompt_ids)
    groups: dict[int, list[str]] = {}
    for item, seq in sequences.items():
        groups.setdefault(len(seq), []).append(item)
    out: dict[str, float] = {}
    for n, items in sorted(groups.items()):
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            fed = mx.array([prompt_ids + sequences[it][:-1] for it in part]
                           if n > 1 else [prompt_ids for _ in part])
            o = lm(fed)
            logits = o.logits if hasattr(o, "logits") else o
            rows = logits[:, L - 1: L - 1 + n, :].astype(mx.float32)
            tgt = mx.array([sequences[it] for it in part])
            lp = (mx.take_along_axis(rows, tgt[..., None], axis=-1)[..., 0]
                  - mx.logsumexp(rows, axis=-1))
            for it, v in zip(part, np.array(lp.sum(axis=1))):
                out[it] = float(v)
    return out


def score_items_fast(model, prompt_ids: list[int],
                     sequences: Mapping[str, list[int]],
                     chunk: int = 16) -> dict[str, float]:
    """Teacher-forced ``log P(sequence)`` via supervised-rows-only
    unembedding (task 000227): the trunk runs once per chunk, and the
    lm-head is applied only to the 1–5 supervised rows per item instead
    of every position. Same batching/bucketing/sync structure as
    ``score_items_batched``; takes a ``mechbench_compute.Model`` (not a bare
    module) because the trunk/head split is family-forked.

    Fidelity: the head sees a ``[B, n, D]`` rows-block instead of
    ``[B, S, D]``, so deep-tail logPs carry the usual bf16 tiling
    envelope vs the ``score_items`` oracle (characterization table in the
    README); mass-bearing results are unchanged. Switch consumers only
    per the re-run practice.
    """
    L = len(prompt_ids)
    groups: dict[int, list[str]] = {}
    for item, seq in sequences.items():
        groups.setdefault(len(seq), []).append(item)
    out: dict[str, float] = {}
    for n, items in sorted(groups.items()):
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            fed = mx.array([prompt_ids + sequences[it][:-1] for it in part]
                           if n > 1 else [prompt_ids for _ in part])
            h = model.trunk_hidden(fed)
            rows = model.head_logits(
                h[:, L - 1: L - 1 + n, :]).astype(mx.float32)
            tgt = mx.array([sequences[it] for it in part])
            lp = (mx.take_along_axis(rows, tgt[..., None], axis=-1)[..., 0]
                  - mx.logsumexp(rows, axis=-1))
            for it, v in zip(part, np.array(lp.sum(axis=1))):
                out[it] = float(v)
    return out


def _copy_prefix_cache(cache):
    """Independent per-item view of a filled prompt cache: shallow-copy
    each layer cache and re-materialize its batch-1 K/V arrays so the
    suffix pass can grow them without mutating the shared prefix state.

    Batch-1 only, deliberately: batched (B>1) *cached* decoding is broken
    upstream at mlx 0.31.2 (mlx-lm 0.31.3 / mlx-vlm 0.6.1) — with a
    natively built B=4 cache and four identical rows, the batched prompt
    pass is row-uniform but the cached suffix step corrupts every row
    after the first (top-1 becomes incoherent). Reproduced on both the
    mlx-lm and mlx-vlm stacks; revisit batching when upstream fixes land
    (it is the ~5–10× path for battery scoring)."""
    import copy as _copy
    out = []
    for c in cache:
        n = _copy.copy(c)
        for k, v in vars(c).items():
            if isinstance(v, mx.array) and v.ndim >= 3 and v.shape[0] == 1:
                setattr(n, k, mx.repeat(v, 1, axis=0))
        out.append(n)
    return out


def score_items_cached(model, prompt_ids: list[int],
                       sequences: Mapping[str, list[int]]) -> dict[str, float]:
    """Teacher-forced ``log P(sequence)`` with prefix reuse (task
    000227): the shared prompt is encoded **once** into a KV cache, and
    each item scores by feeding only its own 1–4 suffix tokens against a
    per-item copy of that cache — ~20× fewer trunk token-positions than
    the oracle on a 200-item battery, and the head runs only on suffix
    rows by construction.

    Positions and attention semantics are exact by design (the cache
    offset supplies real positions; a cached suffix attends the full
    prompt plus itself causally — precisely teacher forcing). Residual
    deltas vs the ``score_items`` oracle are the usual bf16 envelope from
    decomposed attention (measured mass-region ≤ ~0.21 nats on E2B);
    switch consumers only per the re-run practice.
    """
    cache = model.prompt_cache()
    lm = model.lm
    o = lm(mx.array([prompt_ids]), cache=cache)
    prompt_row = (o.logits if hasattr(o, "logits")
                  else o)[0, -1, :].astype(mx.float32)
    lse0 = mx.logsumexp(prompt_row)
    mx.eval(prompt_row)
    for c in cache:
        mx.eval([v for v in vars(c).values() if isinstance(v, mx.array)])
    out: dict[str, float] = {}
    for item, seq in sequences.items():
        lp = float(prompt_row[seq[0]] - lse0)
        if len(seq) > 1:
            cc = _copy_prefix_cache(cache)
            o = lm(mx.array([seq[:-1]]), cache=cc)
            rows = (o.logits if hasattr(o, "logits")
                    else o)[0].astype(mx.float32)
            tgt = mx.array(seq[1:])
            lp += float((mx.take_along_axis(rows, tgt[:, None],
                                            axis=-1)[:, 0]
                         - mx.logsumexp(rows, axis=-1)).sum())
        out[item] = lp
    return out


def complete_items(items: Any) -> list[str]:
    """The outcomes a complete read scores: a list of strings, or a target
    spec (`{"weights": …}` or `{"uniform": […]}`, with any `transform`),
    whose outcomes are its support after the transforms, heaviest first,
    so a `top_k` reads exactly the vocabulary a rung trained on."""
    if isinstance(items, (list, tuple)):
        out = [str(x) for x in items]
    elif isinstance(items, Mapping):
        from mechbench_compute.finetune import target_map_from_spec

        weights = target_map_from_spec(items).to_dict()
        out = sorted(weights, key=lambda k: (-weights[k], k))
    else:
        raise TypeError(
            "complete.items is a list of outcomes or a target spec "
            "({\"weights\": …} or {\"uniform\": […]})")
    if not out:
        raise ValueError("complete.items names no outcomes")
    return out


def score_complete(model, tokenizer, rendered: str, prompt_ids: list[int],
                   spec: Mapping[str, Any]) -> tuple[dict[str, dict], float, float]:
    """Exact probabilities of complete outcomes at a decision point (task
    000548): each outcome, between ``spec["opener"]`` and
    ``spec["closer"]``, tokenized as a continuation of the rendered prompt
    and scored by teacher forcing. The closer is what makes an outcome
    complete — "Mystery" is scored as `Mystery"`, so the mass of "Mystery
    Thriller" is not counted twice. The opener is text before each
    outcome that is not part of its name: after a list's `,` the next
    genre is scored as " Humor" and recorded as "Humor" (a prompt ending
    in the space would tokenize the space alone, as no generation does).

    Returns (entries by outcome, total mass, entropy in bits of the mass
    renormalized over the set). Each entry is ``{text, tokens, p, logp}``;
    ``p`` keeps eight decimals, since a wide vocabulary puts outcomes well
    below the five a token read keeps."""
    opener = str(spec.get("opener", ""))
    closer = str(spec.get("closer", '"'))
    names = complete_items(spec.get("items"))
    sequences = {name: suffix_tokens(tokenizer, rendered, prompt_ids, opener + name + closer)
                 for name in names}
    logps = score_items_fast(model, prompt_ids, sequences)
    entries: dict[str, dict] = {}
    ps = []
    for name in names:
        lp = float(logps[name])
        p = math.exp(lp)
        ps.append(p)
        entries[name] = {"text": name, "tokens": len(sequences[name]),
                         "p": round(p, 8), "logp": round(lp, 4)}
    mass = sum(ps)
    entropy = (-sum((p / mass) * math.log2(p / mass) for p in ps if p > 0)
               if mass > 0 else 0.0)
    return entries, mass, entropy


def item_metrics(logps: Mapping[str, float],
                 target: TargetMap | None = None) -> dict:
    """Distribution diagnostics for teacher-forced item log-probs.

    Renormalizes over the scored support, then reports captured mass,
    item-level entropy (bits), KL from ``target`` (bits; uniform over the
    support when ``target`` is None), and the top item.
    """
    keys = list(logps)
    lp = np.array([logps[k] for k in keys], dtype=np.float64)
    p = np.exp(lp)
    mass = float(p.sum())
    q = p / p.sum()
    if target is None:
        t = np.full(len(keys), 1.0 / len(keys))
    else:
        tn = target.normalize()
        t = np.array([tn[k] for k in keys], dtype=np.float64)
        t = t / t.sum()
    kl = float(np.sum(q * np.log2(np.clip(q / np.clip(t, 1e-300, None),
                                          1e-12, None))))
    h = float(-(q * np.log2(np.clip(q, 1e-12, None))).sum())
    top = int(np.argmax(q))
    return {"captured_mass": mass, "kl_from_target_bits": kl,
            "item_entropy_bits": h,
            "top1": {"item": keys[top], "p": float(q[top])}}


def first_token_metrics(lm, prompt_ids: list[int], tokenizer=None) -> dict:
    """Full-vocabulary entropy and top token at the decision position."""
    logits = np.array(_forward_logits(lm, prompt_ids)[-1]
                      .astype(mx.float32)).astype(np.float64)
    z = logits - logits.max()
    p = np.exp(z) / np.exp(z).sum()
    order = np.argsort(-p)
    nz = p[p > 0]
    out = {"entropy_bits": float(-(nz * np.log2(nz)).sum()),
           "top1": {"token_id": int(order[0]), "p": float(p[order[0]])}}
    if tokenizer is not None:
        out["top1"]["token"] = tokenizer.decode([int(order[0])])
    return out


def prefill_decision(model, prompt_ids: list[int], *, interventions=None):
    """Encode a prompt once into a KV cache and return
    ``(cache, last_row)`` where ``last_row`` is the float32 logits row
    at the decision position (task 000253, on the 000227 cache
    machinery). One model call serves both the decision-token
    distribution read and, via ``expand_top_outcomes_cached``, every
    subsequent expansion forward.

    With `interventions` the prompt runs through the hooked forward
    instead, the hooks live over every prompt position, and the cache
    holds what they left (000601). Without, the native forward: the
    bytes an un-intervened sample has always had.
    """
    cache = model.prompt_cache()
    if interventions:
        res = model.run(mx.array([prompt_ids]), interventions=list(interventions),
                        kv_cache=cache)
        row = res.logits[0, -1, :].astype(mx.float32)
    else:
        lm = model.lm
        o = lm(mx.array([prompt_ids]), cache=cache)
        row = (o.logits if hasattr(o, "logits") else o)[0, -1, :].astype(mx.float32)
    mx.eval(row)
    for c in cache:
        mx.eval([v for v in vars(c).values() if isinstance(v, mx.array)])
    return cache, row


def expand_top_outcomes_cached(model, tokenizer, prompt_ids: list[int],
                               cfg: Mapping, *, prefill=None) -> dict:
    """Best-first expansion of complete outcomes with prefix reuse: the
    prompt is encoded **once** (``prefill_decision``); every expansion
    node then feeds only its own partial-outcome tokens (a handful)
    against a per-node copy of the prompt cache, instead of re-encoding
    the ~hundreds-of-token prompt per forward.

    Semantics — branch floor, terminators, per-node top-50 children,
    optimality cut against the K-th completed outcome, and the mass
    accounting — are identical to the uncached expansion this replaces
    (mechbench-runner's original). ``forwards_used`` counts model calls
    including the prefill, so cached and uncached numbers stay
    comparable. Numerics carry the usual cached-suffix bf16 envelope
    (task 000227): switch consumers only per the re-run practice.

    ``prefill``: optional ``(cache, last_row)`` from a prior
    ``prefill_decision`` call, so the decision read and the expansion
    share one prompt encode; computed here when absent.
    """
    import heapq

    top_k = int(cfg.get("top_k", 10))
    max_tokens = int(cfg.get("max_tokens", 8))
    max_forwards = int(cfg.get("max_forwards", 128))
    branch_floor = float(cfg.get("floor", 1e-3))
    terminators = cfg.get("terminators", ['"'])

    cache, root_row = prefill if prefill is not None \
        else prefill_decision(model, prompt_ids)
    forwards = 1  # the prefill

    def _dist(row: mx.array) -> np.ndarray:
        lp = np.array(row - mx.logsumexp(row))
        return np.exp(lp.astype(np.float64))

    heap: list[tuple[float, list[int]]] = [(0.0, [])]
    completed: list[tuple[float, str]] = []
    pruned_mass = 0.0
    while heap and forwards < max_forwards:
        neg_lp, partial = heapq.heappop(heap)
        if (len(completed) >= top_k
                and -neg_lp <= completed[top_k - 1][0]):
            heapq.heappush(heap, (neg_lp, partial))
            break
        if partial:
            cc = _copy_prefix_cache(cache)
            o = model.lm(mx.array([partial]), cache=cc)
            row = (o.logits if hasattr(o, "logits")
                   else o)[0, -1, :].astype(mx.float32)
            forwards += 1
        else:
            row = root_row  # the prefill already produced this position
        probs = _dist(row)
        # The fifty most probable children, most probable first. A full
        # argsort over a 262k vocabulary costs ~20 ms per node and the
        # node needs fifty of them: partition first, sort the fifty
        # (ties by token id, so the order is the same on every run).
        # `kth` must be inside the array: a vocabulary narrower than the
        # branch width raised, where it should simply take all of it.
        top = np.argpartition(-probs, min(50, probs.size - 1))[:50]
        order = top[np.lexsort((top, -probs[top]))]
        for t in order:
            p_child = float(probs[t])
            total = float(np.exp(-neg_lp)) * p_child
            if total < branch_floor:
                pruned_mass += float(np.exp(-neg_lp)) * p_child
                continue
            piece = tokenizer.decode([int(t)])
            if any(term in piece for term in terminators):
                text = tokenizer.decode(partial).strip()
                if text:
                    completed.append((total, text))
                    completed.sort(key=lambda x: -x[0])
            elif len(partial) < max_tokens:
                heapq.heappush(
                    heap,
                    (neg_lp - float(np.log(max(p_child, 1e-300))),
                     partial + [int(t)]))
    frontier_mass = float(sum(np.exp(-h[0]) for h in heap))
    return {
        "top_outcomes": [
            {"text": text, "p": round(p, 5)}
            for p, text in completed[:top_k]
        ],
        "completed_mass": round(float(sum(p for p, _ in completed)), 4),
        "frontier_mass_bound": round(frontier_mass, 4),
        "forwards_used": forwards,
    }
