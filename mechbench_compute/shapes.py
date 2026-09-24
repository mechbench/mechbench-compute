from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points as P

_ROUND_VECTOR = 5
_ROUND_P = 5
_ROUND_LOGP = 4


def space(*, model: str | None, layer: int | None, point: str, d: int,
          head: int | None = None) -> dict[str, Any]:
    return {
        "model": model,
        "layer": None if layer is None else int(layer),
        "point": str(point),
        "head": None if head is None else int(head),
        "d": int(d),
    }


def same_space(a: Mapping[str, Any], b: Mapping[str, Any], *, what: str = "vectors") -> None:
    sa, sb = space_of(a), space_of(b)
    for k in ("layer", "point", "head", "d"):
        if sa.get(k) != sb.get(k):
            raise ValueError(
                f"{what} live in different spaces: {k} {sa.get(k)!r} vs {sb.get(k)!r}")
    ma, mb = sa.get("model"), sb.get("model")
    if ma is not None and mb is not None and ma != mb:
        raise ValueError(f"{what} live in different spaces: model {ma!r} vs {mb!r}")


def token(tokenizer, tid: int) -> dict[str, Any]:
    return {"id": int(tid), "text": tokenizer.decode([int(tid)])}


def vector(vec: Any, sp: Mapping[str, Any], *, id: Any = None,
           coords: Mapping[str, Any] | None = None,
           token: Mapping[str, Any] | None = None,
           n_pooled: int | None = None, **extra: Any) -> dict[str, Any]:
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
    prov = d.get("derivation") or {}
    out: dict[str, Any] = {"space": space_of(d), "method": prov.get("method")}
    for k in ("axis", "positive", "negative", "labels", "sources"):
        if prov.get(k) is not None:
            out[k] = prov[k]
    return out


def distribution(logp: np.ndarray, tokenizer, *, top_k: int,
                 tracked: Mapping[str, int] | None = None) -> dict[str, Any]:
    lp = np.asarray(logp, dtype=np.float64).reshape(-1)
    probs = np.exp(lp)
    nz = probs[probs > 0]
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
    out: dict[str, Any] = {"id": id, "coords": dict(coords or {}),
                           "axes": list(axes), "measures": dict(measures)}
    if tokens is not None:
        out["tokens"] = list(tokens)
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def model_id_of(model: Any) -> str | None:
    for attr in ("requested_ref", "requested", "model_id"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    arch = getattr(model, "arch", None)
    v = getattr(arch, "model_id", None)
    return v if isinstance(v, str) and v else None


def space_of(item: Mapping[str, Any], header: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    coords = dict(item.get("coords") or {})
    if item.get("label") is not None and "label" not in coords:
        coords["label"] = item["label"]
    return coords


def label_of(item: Mapping[str, Any], axis: str | None) -> Any:
    coords = item.get("coords") or {}
    if axis is None or axis == "label":
        v = coords.get("label")
        return v if v is not None else item.get("label")
    v = coords.get(axis)
    return v if v is not None else (item.get(axis) if axis in item else None)


def distribution_of(item: Mapping[str, Any]) -> dict[str, Any]:
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
    m = item.get("measures")
    if isinstance(m, Mapping):
        return dict(m)
    return {name: item[old] for name, old in legacy.items() if old in item}


__all__ = [
    "coordinate", "coords_of", "direction_ref", "distribution", "distribution_of", "grid",
    "head_of", "label_of", "layer_of", "measures_of", "same_space", "space",
    "space_of", "token", "vector",
]
