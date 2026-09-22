from __future__ import annotations

from typing import Any

#: The reasoning markup a local chat template can declare: the opening
#: and closing special tokens, and the channel name that follows the
#: opening. Gemma 4 writes `<|channel>thought\n…<channel|>`; `<think>`
#: is the Qwen3 and R1-distill convention, declared by no family this
#: package runs today and kept so one that does is split the same way.
DELIMITERS: tuple[tuple[str, str, str], ...] = (
    ("<|channel>", "<channel|>", "thought\n"),
    ("<think>", "</think>", ""),
)


def find_delimiters(tokenizer: Any) -> tuple[str, str, str] | None:
    """The reasoning markup this tokenizer declares, or None. A pair
    counts only when both are single tokens of its vocabulary: a model
    without them cannot mark reasoning, and its text is left alone."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    for opening, closing, channel in DELIMITERS:
        try:
            a, b = convert(opening), convert(closing)
        except Exception:  # noqa: BLE001 — a tokenizer that refuses has no such tokens
            continue
        if (isinstance(a, int) and isinstance(b, int) and a != b
                and a != unk and b != unk):
            return opening, closing, channel
    return None


def split_reasoning(text: str, delimiters: tuple[str, str, str] | None
                    ) -> tuple[list[str], str]:
    """`(reasoning, prose)` for a turn of decoded text. Every marked
    span leaves the prose; one never closed runs to the end of the turn
    and is reasoning, because a model cut off mid-thought wrote no reply
    after it. An empty span (a template's placeholder thought) leaves no
    entry. Text with no markup is returned unchanged, byte for byte."""
    if delimiters is None:
        return [], text
    opening, closing, channel = delimiters
    if opening not in text:
        return [], text
    reasoning: list[str] = []
    prose: list[str] = []
    rest = text
    while True:
        at = rest.find(opening)
        if at == -1:
            prose.append(rest)
            break
        prose.append(rest[:at])
        inner = rest[at + len(opening):]
        end = inner.find(closing)
        body, rest = (inner, "") if end == -1 else (inner[:end], inner[end + len(closing):])
        if channel and body.startswith(channel):
            body = body[len(channel):]
        if body.strip():
            reasoning.append(body.strip())
        if end == -1:
            break
    return reasoning, "".join(prose).strip()
