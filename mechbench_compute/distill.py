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


class TargetMap:
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

    @classmethod
    def from_dict(cls, weights: Mapping[str, float]) -> "TargetMap":
        return cls(weights)

    @classmethod
    def from_json(cls, path: str) -> "TargetMap":
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a flat JSON object of item -> weight")
        return cls(data)

    @classmethod
    def uniform(cls, items: Iterable[str]) -> "TargetMap":
        items = list(items)
        return cls({k: 1.0 / len(items) for k in items})

    def map_values(self, fn: Callable[[float], float]) -> "TargetMap":
        return TargetMap({k: fn(v) for k, v in self._w.items()})

    def sqrt(self) -> "TargetMap":
        return self.map_values(math.sqrt)

    def pow(self, exponent: float) -> "TargetMap":
        return self.map_values(lambda v: v ** exponent)

    def scale(self, factor: float) -> "TargetMap":
        return self.map_values(lambda v: v * factor)

    def temper(self, temperature: float) -> "TargetMap":
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        return self.pow(1.0 / temperature).normalize()

    def mix_uniform(self, epsilon: float) -> "TargetMap":
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        p = self.normalize()
        u = 1.0 / len(p._w)
        return TargetMap({k: (1 - epsilon) * v + epsilon * u
                          for k, v in p._w.items()})

    def top_k(self, k: int) -> "TargetMap":
        kept = sorted(self._w.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return TargetMap(dict(kept))

    def filter(self, predicate: Callable[[str, float], bool]) -> "TargetMap":
        return TargetMap({k: v for k, v in self._w.items()
                          if predicate(k, v)})

    def normalize(self) -> "TargetMap":
        t = self.total()
        if t <= 0.0:
            raise ValueError("cannot normalize a TargetMap with zero total")
        return self.scale(1.0 / t)

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
        keys = list(self._w)
        w = np.array([self._w[k] for k in keys], dtype=np.float64)
        return keys[rng.choice(len(keys), p=w / w.sum())]

    def tokenize(self, tokenizer, prefix: str,
                 closer: str = "") -> "TargetTrie":
        return TargetTrie(self, tokenizer, prefix, closer)


class Example(NamedTuple):
    prompt_ids: list[int]
    tokens: list[int]
    soft: dict[int, float] | list[dict[int, float] | None] | None = None


class TargetTrie:
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
        self._nodes: dict[tuple[int, ...], dict[int, float]] = {}
        for item, seq in self.sequences.items():
            w = self.weights[item]
            for j, t in enumerate(seq):
                node = self._nodes.setdefault(tuple(seq[:j]), {})
                node[t] = node.get(t, 0.0) + w

    def items(self) -> list[str]:
        return list(self.weights)

    def node_target(self, prefix: tuple[int, ...] = ()) -> dict[int, float]:
        node = self._nodes[tuple(prefix)]
        z = sum(node.values())
        return {t: w / z for t, w in node.items()}

    def root_marginal(self) -> dict[int, float]:
        return self.node_target(())

    def sample(self, rng: np.random.Generator) -> str:
        keys = list(self.weights)
        w = np.array([self.weights[k] for k in keys], dtype=np.float64)
        return keys[rng.choice(len(keys), p=w / w.sum())]

    def hard_example(self, item: str) -> Example:
        return Example(self.prompt_ids, self.sequences[item], None)

    def marginal_example(self) -> Example:
        return Example(self.prompt_ids, [], self.root_marginal())

    def path_rows(self, item: str) -> Example:
        seq = self.sequences[item]
        soft = [self.node_target(tuple(seq[:j])) for j in range(len(seq))]
        return Example(self.prompt_ids, seq, soft)

    def score(self, lm) -> dict[str, float]:
        return score_items(lm, self.prompt_ids, self.sequences)


def _forward_logits(lm, ids: list[int]) -> mx.array:
    out = lm(mx.array([ids]))
    return (out.logits if hasattr(out, "logits") else out)[0]


# external: MLX/Metal — a full [seq, vocab] float32 logits tensor in the gradient graph trips the command-buffer watchdog on long prompts; slice rows before the cast
def soft_ce(lm, batch: list[Example]) -> mx.array:
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


def render_chat(tokenizer, system: str, user: str, prefill: str = "",
                date_string: str | None = None) -> str:
    merged = (system + "\n\n" + user) if system else user
    kwargs = {} if date_string is None else {"date_string": date_string}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": merged}],
        tokenize=False, add_generation_prompt=True, **kwargs) + prefill


@dataclass(frozen=True)
class Rendered:
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
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def suffix_tokens(tokenizer, prefix: str, prefix_ids: list[int],
                  text: str) -> list[int]:
    full = encode(tokenizer, prefix + text)
    if full[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"tokenizing {text!r} in context changed the prefix "
            f"tokenization; adjust the envelope so the boundary is stable")
    return full[len(prefix_ids):]


def score_items(lm, prompt_ids: list[int],
                sequences: Mapping[str, list[int]]) -> dict[str, float]:
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


# external: mlx-lm, mlx-vlm — batched (B>1) cached decoding corrupts every row after the first on the suffix step; keep prefix-cache scoring batch-1
def _copy_prefix_cache(cache):
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
    import heapq

    top_k = int(cfg.get("top_k", 10))
    max_tokens = int(cfg.get("max_tokens", 8))
    max_forwards = int(cfg.get("max_forwards", 128))
    branch_floor = float(cfg.get("floor", 1e-3))
    terminators = cfg.get("terminators", ['"'])

    cache, root_row = prefill if prefill is not None \
        else prefill_decision(model, prompt_ids)
    forwards = 1

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
            row = root_row
        probs = _dist(row)
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
