from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple


class Delimiters(NamedTuple):
    opening: str
    closing: str
    channel: str = ""


THINK_TAGS = Delimiters("<think>", "</think>")


def list_delimiters() -> tuple[Delimiters, ...]:
    from mechbench_compute.architectures import ARCHITECTURES

    found: list[Delimiters] = []
    for a in ARCHITECTURES:
        for d in a.reasoning:
            if d not in found:
                found.append(d)
    return tuple(found)

THINKING = "thinking"
ANSWER = "answer"
SCHEMA = "reasoning"


def delimiter_ids(tokenizer: Any) -> tuple[int, int] | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    for open_s, close_s, _ in list_delimiters():
        try:
            a, b = convert(open_s), convert(close_s)
        except Exception:  # noqa: BLE001
            continue
        if a is None or b is None or a == unk or b == unk or a == b:
            continue
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        return int(a), int(b)
    return None


def segments(ids: Sequence[int], *, start: int, pair: tuple[int, int] | None) -> list[dict[str, Any]]:
    if pair is None:
        return []
    open_id, close_id = pair
    tail = list(ids)[start:]
    try:
        opened = tail.index(open_id)
    except ValueError:
        return []
    try:
        closed = tail.index(close_id, opened + 1)
    except ValueError:
        return [{"role": THINKING, "token_start": start + opened + 1,
                 "token_end": len(ids), "terminated": False}]
    out = [{"role": THINKING, "token_start": start + opened + 1,
            "token_end": start + closed, "terminated": True}]
    if start + closed + 1 < len(ids):
        out.append({"role": ANSWER, "token_start": start + closed + 1,
                    "token_end": len(ids)})
    return out


def segmentation(ids: Sequence[int], *, start: int,
                 pair: tuple[int, int] | None) -> dict[str, Any] | None:
    segs = segments(ids, start=start, pair=pair)
    return {"schema_name": SCHEMA, "segments": segs} if segs else None


def answer_text(text: str, pair_text: tuple[str, ...] = THINK_TAGS) -> str:
    open_s, close_s = pair_text[:2]
    a = text.find(open_s)
    if a == -1:
        return text
    b = text.find(close_s, a + len(open_s))
    return text if b == -1 else (text[:a] + text[b + len(close_s):]).lstrip()


def split_thought(text: str, pair_text: tuple[str, ...] = THINK_TAGS) -> tuple[str | None, str]:
    open_s, close_s = pair_text[:2]
    a = text.find(open_s)
    if a == -1:
        return None, text
    b = text.find(close_s, a + len(open_s))
    if b == -1:
        return None, text
    inner = text[a + len(open_s):b].strip()
    rest = (text[:a] + text[b + len(close_s):]).lstrip()
    return (inner or None), rest
