"""Where a model's reasoning is, in the tokens it produced.

Some models reason out loud before answering, and mark it: Qwen3 and the
R1 distills emit `<think>` and `</think>` as **special tokens** — entries
in the tokenizer's own vocabulary, not prose the model happened to
write. That distinction is the whole basis of this module. A regex over
the text would mis-segment a model discussing the string "<think>", and
would invent a boundary on a model that has no such notion; asking the
tokenizer means the span exists only where the model's own vocabulary
says it does.

What comes out is a segmentation — the same shape `text/generate`
already writes for `prompt` / `body` — so every reader that understands
named spans over a trace understands this one, and a position selector
can name it.

A summary of hidden reasoning, as a remote provider returns, is NOT
this. It is text about tokens nobody can see, and it belongs in
metadata, never in a span over a trace that does not contain it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Delimiter pairs, by the name of the schema that uses them. A pair
#: counts only when BOTH tokens are in the tokenizer's vocabulary as
#: single tokens — a model that writes the characters without meaning
#: them has no such tokens.
DELIMITERS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),          # Qwen3, DeepSeek-R1 distills
    ("<reasoning>", "</reasoning>"),  # several finetunes
)

#: The role names a thinking segmentation uses.
THINKING = "thinking"
ANSWER = "answer"
SCHEMA = "reasoning"


def delimiter_ids(tokenizer: Any) -> tuple[int, int] | None:
    """The (open, close) token ids this tokenizer declares for reasoning,
    or None when it declares none.

    Only single-token delimiters count. `convert_tokens_to_ids` answers
    `unk` (or None) for a string the vocabulary lacks, and an id that
    round-trips to different text is not the delimiter — both are
    rejected, so a coincidence cannot become a span.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    for open_s, close_s in DELIMITERS:
        try:
            a, b = convert(open_s), convert(close_s)
        except Exception:  # noqa: BLE001 — a tokenizer that refuses is a tokenizer without them
            continue
        if a is None or b is None or a == unk or b == unk or a == b:
            continue
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        return int(a), int(b)
    return None


def segments(ids: Sequence[int], *, start: int, pair: tuple[int, int] | None) -> list[dict[str, Any]]:
    """The reasoning segments of `ids`, generation having begun at
    `start`. Empty when the model declares no delimiters or wrote none.

    The span covers what is INSIDE the delimiters: the tokens the model
    reasoned with, not the markers that announce them. What follows the
    close is the answer; a thought never closed has no answer segment and
    says `terminated: false`, because a model cut off mid-thought did not
    produce one.
    """
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
    """The reasoning segmentation to add beside the envelope, or None."""
    segs = segments(ids, start=start, pair=pair)
    return {"schema_name": SCHEMA, "segments": segs} if segs else None


def answer_text(text: str, pair_text: tuple[str, str] = DELIMITERS[0]) -> str:
    """`text` with a complete leading thought removed — what a reader who
    asked for the answer should see. Text without a complete pair is
    returned whole: a thought that never closed leaves no answer to cut
    to, and inventing one would hide that the model ran out of room.
    """
    open_s, close_s = pair_text
    a = text.find(open_s)
    if a == -1:
        return text
    b = text.find(close_s, a + len(open_s))
    return text if b == -1 else (text[:a] + text[b + len(close_s):]).lstrip()


def split_thought(text: str, pair_text: tuple[str, str] = DELIMITERS[0]) -> tuple[str | None, str]:
    """`(thinking, answer)` for a turn of generated text.

    A conversation's transcript keeps both, and what the next turn SEES
    is the answer: a participant's scratchpad is not the room's business,
    and replaying it would let a conversation feed on its own reasoning
    without anyone choosing that.

    A thought that never closed leaves the text whole and `None` for the
    thinking — there is no answer to separate, and pretending otherwise
    would hide that the turn ran out of room.
    """
    open_s, close_s = pair_text
    a = text.find(open_s)
    if a == -1:
        return None, text
    b = text.find(close_s, a + len(open_s))
    if b == -1:
        return None, text
    inner = text[a + len(open_s):b].strip()
    rest = (text[:a] + text[b + len(close_s):]).lstrip()
    return (inner or None), rest
