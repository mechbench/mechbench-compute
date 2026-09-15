"""The value types and item constructors every op builds its output
from (docs/LEXICON.md §4–§5).

Six ops used to carry a float vector and each spelled "what it is of",
"which space" and "its norm" differently; five wrote "the top tokens"
five ways. This module is the one place those shapes are made:

  space(...)         `{model, layer, point, head?, d}` — where a vector
                     lives; two vectors are comparable only when their
                     spaces agree (`same_space`).
  token(...)         `{id, text}` — a token, once.
  vector(...)        an `activations/vector` item: `{id, coords, space,
                     vector, norm, token?, n_pooled?}`.
  distribution(...)  a `logits/distribution`: `{entropy_bits, top,
                     tracked}` from a log-probability vector and the
                     tokens the caller asked about.
  grid(...)          an `activations/grid` item: `{id, axes, measures,
                     tokens?}` — a scalar field over model axes.
  coordinate(...)    an `activations/coordinate` item: a vector's scalar
                     coordinate along a direction.

Readers of the shapes the bench stored before these existed (a `layer`
on the item and `point` on the header; `label` instead of a coordinate;
`top_tokens` with `p` only) go through `space_of`, `layer_of`,
`label_of` and `distribution_of`, which read either spelling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points as P

_ROUND_VECTOR = 5
_ROUND_P = 5
_ROUND_LOGP = 4

# --- value types ------------------------------------------------------------------------


def space(*, model: str | None, layer: int | None, point: str, d: int,
          head: int | None = None) -> dict[str, Any]:
    """Where a vector lives. `layer` is None for a whole-model point
    (the embedding, the final norm); `head` is None except for a per-head
    source. Every space has the same five fields, so spaces compare and
    sort against each other."""
    return {
        "model": model,
        "layer": None if layer is None else int(layer),
        "point": str(point),
        "head": None if head is None else int(head),
        "d": int(d),
    }


def same_space(a: Mapping[str, Any], b: Mapping[str, Any], *, what: str = "vectors") -> None:
    """Refuse two things that do not share a space: layer, point, head
    and width must agree; the model must agree when both are known."""
    sa, sb = space_of(a), space_of(b)
    for k in ("layer", "point", "head", "d"):
        if sa.get(k) != sb.get(k):
            raise ValueError(
                f"{what} live in different spaces: {k} {sa.get(k)!r} vs {sb.get(k)!r}")
    ma, mb = sa.get("model"), sb.get("model")
    if ma is not None and mb is not None and ma != mb:
        raise ValueError(f"{what} live in different spaces: model {ma!r} vs {mb!r}")


def token(tokenizer, tid: int) -> dict[str, Any]:
    """A token as `{id, text}`."""
    return {"id": int(tid), "text": tokenizer.decode([int(tid)])}


# --- items ------------------------------------------------------------------------------


def vector(vec: Any, sp: Mapping[str, Any], *, id: Any = None,
           coords: Mapping[str, Any] | None = None,
           token: Mapping[str, Any] | None = None,
           n_pooled: int | None = None, **extra: Any) -> dict[str, Any]:
    """An `activations/vector` item."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size != int(sp["d"]):
        raise ValueError(f"vector has {v.size} dims; its space says {sp['d']}")
    out: dict[str, Any] = {}
    if id is not None:
        out["id"] = id
    out["coords"] = dict(coords or {})
    out["space"] = dict(sp)
    out["vector"] = [round(float(x), _ROUND_VECTOR) for x in v]
    out["norm"] = round(float(np.linalg.norm(v)), 6)
    if token is not None:
        out["token"] = dict(token)
    if n_pooled is not None:
        out["n_pooled"] = int(n_pooled)
    out.update({k: v_ for k, v_ in extra.items() if v_ is not None})
    return out


def coordinate(coord: float, sp: Mapping[str, Any], direction: Mapping[str, Any], *,
               id: Any = None, coords: Mapping[str, Any] | None = None,
               **extra: Any) -> dict[str, Any]:
    """An `activations/coordinate` item: the scalar coordinate of a vector
    along a direction, with the space and the direction's identity."""
    out: dict[str, Any] = {}
    if id is not None:
        out["id"] = id
    out["coords"] = dict(coords or {})
    out["space"] = dict(sp)
    out["direction"] = direction_ref(direction)
    out["coord"] = round(float(coord), 6)
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def direction_ref(d: Mapping[str, Any]) -> dict[str, Any]:
    """A direction's identity without its vector: its space and how it
    was made."""
    prov = d.get("derivation") or {}
    out: dict[str, Any] = {"space": space_of(d), "method": prov.get("method")}
    for k in ("axis", "positive", "negative", "labels", "sources"):
        if prov.get(k) is not None:
            out[k] = prov[k]
    return out


def distribution(logp: np.ndarray, tokenizer, *, top_k: int,
                 tracked: Mapping[str, int] | None = None) -> dict[str, Any]:
    """A `logits/distribution` from a log-probability vector over the
    vocabulary: entropy in bits, the `top_k` most likely tokens ranked
    (each `{token, p, logp}`), and `tracked` — name → `{token, p, logp}`
    for the token ids the caller asked about, by the names it gave."""
    lp = np.asarray(logp, dtype=np.float64).reshape(-1)
    probs = np.exp(lp)
    nz = probs[probs > 0]
    # A stable sort: tokens with exactly equal log-probability (common at a
    # high-entropy layer read through the unembedding, where bf16 logits
    # tie by the hundred) rank by token id, so the same logits give the
    # same `top` every time and on every machine. An unstable sort made
    # thirty of a funnel's 140 top tokens differ between two identical
    # runs.
    out: dict[str, Any] = {
        "entropy_bits": round(float(-(nz * np.log2(nz)).sum()), 4),
        "top": [_entry(tokenizer, int(t), lp) for t in np.argsort(-lp, kind="stable")[: int(top_k)]],
    }
    if tracked:
        out["tracked"] = {str(name): _entry(tokenizer, int(tid), lp)
                          for name, tid in tracked.items()}
    return out


def _entry(tokenizer, tid: int, lp: np.ndarray) -> dict[str, Any]:
    return {"token": token(tokenizer, tid),
            "p": round(float(math.exp(lp[tid])), _ROUND_P),
            "logp": round(float(lp[tid]), _ROUND_LOGP)}


def grid(id: Any, axes: Sequence[str], measures: Mapping[str, Any], *,
         tokens: Sequence[str] | None = None,
         coords: Mapping[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """An `activations/grid` item: named measures, each a nested list
    indexed in `axes` order; `tokens` when an axis is `position`."""
    out: dict[str, Any] = {"id": id, "coords": dict(coords or {}),
                           "axes": list(axes), "measures": dict(measures)}
    if tokens is not None:
        out["tokens"] = list(tokens)
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def model_id_of(model: Any) -> str | None:
    """The identity a loaded model was asked for, for a space's `model`:
    the requested id when the model kept it, else the architecture's."""
    for attr in ("requested_ref", "requested", "model_id"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    arch = getattr(model, "arch", None)
    v = getattr(arch, "model_id", None)
    return v if isinstance(v, str) and v else None


# --- readers of either spelling ---------------------------------------------------------


def space_of(item: Mapping[str, Any], header: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The item's `space`, or one assembled from the older flattened
    spelling (`layer`/`head` on the item, `point`/`d_model`/`model` on
    the header, `d` on a direction)."""
    sp = item.get("space")
    if isinstance(sp, Mapping):
        return dict(sp)
    h = header or {}
    vec = item.get("vector")
    d = item.get("d") or h.get("d_model") or (len(vec) if isinstance(vec, list) else None)
    point = item.get("point") or h.get("point") or "resid_post"
    return space(model=item.get("model") or h.get("model")
                 or (item.get("derivation") or {}).get("model"),
                 layer=item.get("layer"), point=P.normalize(str(point)),
                 d=int(d) if d is not None else 0, head=item.get("head"))


def layer_of(item: Mapping[str, Any]) -> int | None:
    sp = item.get("space")
    if isinstance(sp, Mapping):
        return sp.get("layer")
    return item.get("layer")


def head_of(item: Mapping[str, Any]) -> int | None:
    sp = item.get("space")
    if isinstance(sp, Mapping):
        return sp.get("head")
    return item.get("head")


def coords_of(item: Mapping[str, Any]) -> dict[str, Any]:
    """The item's coordinates, with the retired `label` field folded in
    as the `label` coordinate."""
    coords = dict(item.get("coords") or {})
    if item.get("label") is not None and "label" not in coords:
        coords["label"] = item["label"]
    return coords


def label_of(item: Mapping[str, Any], axis: str | None) -> Any:
    """The item's value on `axis` — a coordinate; the retired `label`
    field is read as the `label` axis."""
    coords = item.get("coords") or {}
    if axis is None or axis == "label":
        v = coords.get("label")
        return v if v is not None else item.get("label")
    v = coords.get(axis)
    return v if v is not None else (item.get(axis) if axis in item else None)


def distribution_of(item: Mapping[str, Any]) -> dict[str, Any]:
    """A `logits/distribution` view of an item written before the shape
    existed: `top_tokens` (`{token, p}`), `outcome_mass` (outcome → p),
    `track_logp`, `tracks` (name → logp), or the `top` of a readout
    (`{token, logp}`). A current item is returned as is."""
    top = item.get("top")
    if isinstance(top, list) and top and isinstance(top[0].get("token"), Mapping):
        return dict(item)
    out = dict(item)
    ranked = []
    for t in (item.get("top") or item.get("top_tokens") or []):
        tok = t.get("token")
        text = tok["text"] if isinstance(tok, Mapping) else tok
        p, lp = t.get("p"), t.get("logp")
        if p is None and lp is not None:
            p = math.exp(float(lp))
        if lp is None and p is not None and p > 0:
            lp = math.log(float(p))
        ranked.append({"token": {"id": tok.get("id") if isinstance(tok, Mapping) else None,
                                 "text": text},
                       "p": p, "logp": lp})
    out["top"] = ranked
    out.pop("top_tokens", None)
    tracked: dict[str, Any] = dict(item.get("tracked") or {})
    for name, p in (item.get("outcome_mass") or {}).items():
        tracked.setdefault(str(name), {"token": {"id": None, "text": str(name)},
                                       "p": float(p),
                                       "logp": math.log(float(p)) if p and p > 0 else None})
    for name, lp in (item.get("tracks") or {}).items():
        tracked.setdefault(str(name), {"token": {"id": None, "text": str(name)},
                                       "p": math.exp(float(lp)), "logp": float(lp)})
    if item.get("track_logp") is not None:
        lp = float(item["track_logp"])
        tracked.setdefault("track", {"token": {"id": None, "text": None},
                                     "p": math.exp(lp), "logp": lp})
    if tracked:
        out["tracked"] = tracked
    for k in ("outcome_mass", "tracks", "track_logp"):
        out.pop(k, None)
    return out


def measures_of(item: Mapping[str, Any], legacy: Mapping[str, str]) -> dict[str, Any]:
    """A grid item's `measures`, or the retired per-op fields named by
    `legacy` (measure name → old field name)."""
    m = item.get("measures")
    if isinstance(m, Mapping):
        return dict(m)
    return {name: item[old] for name, old in legacy.items() if old in item}


__all__ = [
    "coordinate", "coords_of", "direction_ref", "distribution", "distribution_of", "grid",
    "head_of", "label_of", "layer_of", "measures_of", "same_space", "space",
    "space_of", "token", "vector",
]
