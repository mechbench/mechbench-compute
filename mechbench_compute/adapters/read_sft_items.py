from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from mechbench_compute.adapters.record_refused import RecordRefused
from mechbench_compute.adapters.tokenize_conversation import tokenize_conversation
from mechbench_compute.distill import encode


class SftItem(NamedTuple):
    ids: list[int]
    trained: list[bool]


def read_sft_items(tokenizer, records: Sequence[Mapping[str, Any]],
                   params: Mapping[str, Any]) -> tuple[list[SftItem], dict[str, Any]]:
    max_tokens = params.get("max_tokens", 512)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 2:
        raise ValueError(f"adapter/train: `max_tokens` is a whole number of tokens, at least 2, "
                         f"not {max_tokens!r}")
    truncation = params.get("truncation", "cut")
    if truncation not in ("cut", "fail"):
        raise ValueError(f"adapter/train: `truncation` is 'cut' or 'fail', not {truncation!r}")
    naturalism = bool(params.get("naturalism", True))
    if not records:
        raise ValueError("adapter/train: `objective: \"sft\"` trains on records, and none arrived")
    items: list[SftItem] = []
    counts = {"n_documents": 0, "n_conversations": 0, "n_tokens": 0}
    truncated = 0
    for record in records:
        rid = record.get("id")
        text = record.get("text")
        if record.get("messages") is not None:
            ids, trained = tokenize_conversation(tokenizer, record, naturalism=naturalism)
            counts["n_conversations"] += 1
        elif isinstance(text, str) and text:
            ids = encode(tokenizer, text)
            trained = [False] + [True] * (len(ids) - 1)
            counts["n_documents"] += 1
        else:
            raise RecordRefused(
                "NO_TEXT", rid,
                "has neither `text` (a document) nor `messages` (a conversation) to train on")
        if len(ids) > max_tokens:
            if truncation == "fail":
                raise RecordRefused(
                    "TOO_LONG", rid,
                    f"is {len(ids)} tokens, over `max_tokens` ({max_tokens}); raise it, or "
                    f"`truncation: \"cut\"` trains on the first {max_tokens}")
            ids, trained = ids[:max_tokens], trained[:max_tokens]
            truncated += 1
        if not any(trained):
            raise RecordRefused(
                "NOTHING_TO_TRAIN", rid,
                f"has no token to train within its first {max_tokens}: a document needs two "
                f"tokens, a conversation an assistant token before the cut")
        last = max(i for i, t in enumerate(trained) if t)
        items.append(SftItem(ids[:last + 1], trained[:last + 1]))
        counts["n_tokens"] += sum(trained)
    return items, {**counts, "max_tokens": max_tokens, "truncation": truncation,
                   "truncated": truncated, "naturalism": naturalism}
