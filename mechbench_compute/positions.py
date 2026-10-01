from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SELECTOR_DOC = (
    '`"last"`, `"all"`, a list of indices (negative from the end), '
    '`{"tokens": [...]}`, `{"range": [a, b]}`, `{"after": n}`, '
    '`{"segment": "thinking"}` (a named span of the trace), '
    '`"subject"` (the last token of the record\'s `subject` string), '
    '`"generated"` (the positions of the generated tokens, steps 1 onward) or `{"step": k}` (the pass '
    'that writes the k-th generated token, counted from 0; also a list of '
    'steps, `"all"`, `"last"`, `{"range": [a, b]}`, `{"after": k}` or '
    '`{"tokens": [...]}` over steps).'
)

STEP_FORMS = ('an int, a list of ints, `"all"`, `"last"`, `{"range": [a, b]}`, '
              '`{"after": k}` or `{"tokens": [...]}`')

REDUCES = ("mean", "max")


def _slice(n: int, a: Any, b: Any) -> list[int]:
    return list(range(n))[None if a is None else int(a): None if b is None else int(b)]


def resolve(selector: Any, n: int, *, tokens: Sequence[str] | None = None,
            record: Mapping[str, Any] | None = None,
            prompt_len: int | None = None, gen_start: int | None = None,
            segmentations: Sequence[Mapping[str, Any]] | None = None,
            absent: str = "error") -> list[int]:
    if selector is None or selector == "last" or selector == "final":
        return [n - 1] if n else []
    if selector == "all":
        return list(range(n))
    if isinstance(selector, bool):
        raise ValueError(f"unknown positions {selector!r}")
    if isinstance(selector, int):
        return [(selector + n) % n] if n else []
    if isinstance(selector, str):
        if selector == "subject":
            return [_subject(tokens, record)]
        if selector == "generated":
            start = gen_start if gen_start is not None else prompt_len
            if start is None:
                raise ValueError(
                    "positions 'generated' needs a trace with generation_spans, "
                    "or a record whose prompt length is known")
            return list(range(max(0, min(int(start), n)), n))
        raise ValueError(f"unknown positions {selector!r}: {SELECTOR_DOC}")
    if isinstance(selector, Mapping):
        if "range" in selector:
            a, b = selector["range"]
            return _slice(n, a, b)
        if "after" in selector:
            return _slice(n, selector["after"], None)
        if "step" in selector:
            start = gen_start if gen_start is not None else prompt_len
            return _steps(selector["step"], n, start, tokens, absent)
        if "segment" in selector:
            try:
                return _segment(selector["segment"], n, segmentations)
            except ValueError:
                if absent == "none":
                    return []
                raise
        if "tokens" in selector:
            if tokens is None:
                raise ValueError("positions {\"tokens\": …} needs the decoded tokens")
            want = {str(t).strip().casefold() for t in selector["tokens"]}
            hit = [i for i, t in enumerate(tokens[:n]) if str(t).strip().casefold() in want]
            if not hit:
                if absent == "none":
                    return []
                raise ValueError(f"none of {sorted(want)} among the prompt's tokens")
            return hit
        raise ValueError(f"unknown positions {selector!r}: {SELECTOR_DOC}")
    if isinstance(selector, Sequence):
        return [(int(p) + n) % n for p in selector] if n else []
    raise ValueError(f"unknown positions {selector!r}: {SELECTOR_DOC}")


def _steps(sel: Any, n: int, start: int | None, tokens: Sequence[str] | None,
           absent: str) -> list[int]:
    if start is None:
        raise ValueError(
            "positions {\"step\": …} counts from where generation began; this sequence "
            "has no generation span and no known prompt length")
    start = int(start)
    if start < 1:
        raise ValueError("positions {\"step\": …} needs a prompt: step 0 is read at its last token")
    count = max(0, n - start + 1)
    ahead = absent == "none"
    if isinstance(sel, bool) or sel is None:
        raise ValueError(f"unknown step {sel!r}: {STEP_FORMS}")
    if isinstance(sel, int) or (isinstance(sel, Sequence) and not isinstance(sel, (str, Mapping))):
        picked: list[int] = []
        for k in ([sel] if isinstance(sel, int) else list(sel)):
            if isinstance(k, bool) or not isinstance(k, int):
                raise ValueError(f"unknown step {k!r}: a step is an int")
            at = k + count if k < 0 else k
            if 0 <= at < count:
                picked.append(at)
            elif not ahead:
                raise ValueError(
                    f"step {k} is past the end: this sequence holds {count} step"
                    f"{'' if count == 1 else 's'} (0 to {count - 1})")
        return [start - 1 + k for k in picked]
    if sel in ("all", "last") or (isinstance(sel, Mapping) and len(sel) == 1
                                  and next(iter(sel)) in ("range", "after", "tokens")):
        seen = list(tokens[start - 1:n]) if tokens is not None else None
        return [start - 1 + k for k in resolve(sel, count, tokens=seen, absent=absent)]
    raise ValueError(f"unknown step {sel!r}: {STEP_FORMS}")


def furthest_step(selector: Any) -> int | None:
    if not isinstance(selector, Mapping) or "step" not in selector:
        return None
    sel = selector["step"]
    if isinstance(sel, bool):
        return None
    if isinstance(sel, int):
        return sel if sel >= 0 else None
    if isinstance(sel, Mapping):
        if "after" in sel and isinstance(sel["after"], int) and sel["after"] >= 0:
            return int(sel["after"])
        if "range" in sel:
            a = (list(sel["range"]) + [None])[0]
            return int(a) if isinstance(a, int) and a >= 0 else None
        return None
    if isinstance(sel, Sequence) and not isinstance(sel, str):
        ks = [k for k in sel if isinstance(k, int) and not isinstance(k, bool) and k >= 0]
        return max(ks) if ks else None
    return None


def _segment(role: Any, n: int, segmentations: Sequence[Mapping[str, Any]] | None) -> list[int]:
    want = str(role)
    have: list[str] = []
    for seg_set in (segmentations or []):
        for seg in (seg_set.get("segments") or []):
            name = str(seg.get("role", ""))
            have.append(name)
            if name != want:
                continue
            a = max(0, min(int(seg.get("token_start", 0)), n))
            b = max(a, min(int(seg.get("token_end", n)), n))
            if a == b:
                raise ValueError(f"segment {want!r} is empty in this sequence")
            return list(range(a, b))
    raise ValueError(
        f"no {want!r} segment here"
        + (f"; this document has {sorted(set(have))}" if have
           else "; this document carries no named spans"))


def one(selector: Any, n: int, **kw: Any) -> int:
    idx = resolve(selector, n, **kw)
    if len(idx) != 1:
        raise ValueError(
            f"position {selector!r} names {len(idx)} positions; one is needed here")
    return idx[0]


def _subject(tokens: Sequence[str] | None, record: Mapping[str, Any] | None) -> int:
    subject = (record or {}).get("subject")
    if not isinstance(subject, str) or not subject:
        raise ValueError(
            f"record {(record or {}).get('id')!r}: position 'subject' needs a "
            "`subject` field naming a substring of the prompt")
    if tokens is None:
        raise ValueError("position 'subject' needs the decoded tokens")
    folded = subject.casefold()
    hits = [(len(t.strip()), i) for i, t in enumerate(tokens)
            if t.strip() and t.strip().casefold() in folded]
    if not hits:
        raise ValueError(
            f"record {record.get('id')!r}: subject {subject!r} not found among "
            "the prompt's tokens")
    return max(hits)[1]


def pool_spec(params: Mapping[str, Any]) -> dict[str, Any] | None:
    pool = params.get("pool")
    if pool is None:
        return None
    if not isinstance(pool, Mapping):
        raise ValueError(
            f"`pool` is an object {{\"reduce\": \"mean\" | \"max\", \"over\": "
            f"<positions>}}, not {pool!r} — `first_k` after `skip` is "
            f"{{\"range\": [skip, skip + k]}}, `last_k` is {{\"range\": [-k, null]}}")
    reduce = str(pool.get("reduce", "mean"))
    if reduce not in REDUCES:
        raise ValueError(f"unknown pool reduce {reduce!r}: one of {REDUCES}")
    return {"reduce": reduce, "over": pool.get("over", "all")}


def pooled(mat: Any, idx: Sequence[int], reduce: str) -> tuple[Any, int]:
    sub = mat[list(idx)] if idx else mat[-1:]
    v = sub.max(axis=0) if reduce == "max" else sub.mean(axis=0)
    return v, int(sub.shape[0])
