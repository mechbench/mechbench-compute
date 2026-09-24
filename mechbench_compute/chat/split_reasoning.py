from __future__ import annotations

from typing import Any

DELIMITERS: tuple[tuple[str, str, str], ...] = (
    ("<|channel>", "<channel|>", "thought\n"),
    ("<think>", "</think>", ""),
)


def find_delimiters(tokenizer: Any) -> tuple[str, str, str] | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    for opening, closing, channel in DELIMITERS:
        try:
            a, b = convert(opening), convert(closing)
        except Exception:  # noqa: BLE001
            continue
        if (isinstance(a, int) and isinstance(b, int) and a != b
                and a != unk and b != unk):
            return opening, closing, channel
    return None


def split_reasoning(text: str, delimiters: tuple[str, str, str] | None
                    ) -> tuple[list[str], str]:
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
