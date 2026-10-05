from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.adapters.record_refused import RecordRefused
from mechbench_compute.distill import encode
from mechbench_compute.providers import messages as pm


def tokenize_conversation(tokenizer, record: Mapping[str, Any], *,
                          naturalism: bool = True,
                          date_string: str | None = None) -> tuple[list[int], list[bool]]:
    rid = record.get("id")
    turns = read_turns(record)
    pinned = {} if date_string is None else {"date_string": date_string}

    def render(upto: int, opening: bool) -> str:
        try:
            return tokenizer.apply_chat_template(turns[:upto], tokenize=False,
                                                 add_generation_prompt=opening, **pinned)
        except Exception as e:  # noqa: BLE001
            raise RecordRefused("TEMPLATE_REFUSED", rid,
                                f"is refused by the model's chat template: {e}") from None

    whole = render(len(turns), False)
    ids = encode(tokenizer, whole)
    trained = [False] * len(ids)
    for k, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        before, through = render(k, True), render(k + 1, False)
        if not (through.startswith(before) and whole.startswith(through)):
            raise RecordRefused(
                "TEMPLATE_UNSEGMENTED", rid,
                f"cannot be segmented at message {k}: the chat template's rendering through that "
                f"assistant turn does not continue its rendering up to the turn's generation "
                f"prompt, or the whole conversation does not continue it, so the template does "
                f"not say where the turn begins and ends")
        start, end = encode(tokenizer, before), encode(tokenizer, through)
        if naturalism and (ids[:len(start)] != start or ids[:len(end)] != end):
            raise RecordRefused(
                "NATURALISM_VIOLATION", rid,
                f"has an assistant turn at message {k} that does not begin and end on a token "
                f"boundary of the whole conversation, so its tokens in context are not its own; "
                f"`naturalism: false` takes each boundary at the token count of the text before it")
        for i in range(len(start), min(len(end), len(ids))):
            trained[i] = True
    return ids, trained


def read_turns(record: Mapping[str, Any]) -> list[dict[str, str]]:
    rid = record.get("id")
    messages = record["messages"]
    if not isinstance(messages, list):
        raise RecordRefused("UNREADABLE_MESSAGE", rid,
                            "has `messages` that is not a list of `{role, content}`")
    turns: list[dict[str, str]] = []
    for i, m in enumerate(messages):
        if isinstance(m, Mapping) and "role" not in m and "participant" in m:
            raise RecordRefused(
                "UNREADABLE_MESSAGE", rid,
                f"has a transcript's message {i}, which names a participant, not a role: a "
                f"transcript is trained from one participant's side, as `text/render` writes it")
        if isinstance(m, Mapping) and "content" not in m:
            raise RecordRefused("UNREADABLE_MESSAGE", rid, f"has message {i} with no `content`")
        try:
            message = pm.message(m)
        except (TypeError, ValueError) as e:
            raise RecordRefused("UNREADABLE_MESSAGE", rid, f"has message {i}: {e}") from None
        parts = sorted({p.to_wire()["type"] for p in message.content
                        if not isinstance(p, pm.TextPart)})
        if parts:
            raise RecordRefused(
                "UNREADABLE_MESSAGE", rid,
                f"has message {i} carrying {', '.join(parts)}: a conversation is trained on its "
                f"text, so keep only its text parts")
        turns.append({"role": message.role, "content": message.text()})
    if not any(t["role"] == "assistant" for t in turns):
        raise RecordRefused("NO_ASSISTANT_TURN", rid, "has no assistant message to train on")
    system = str(record.get("system") or "")
    if system:
        if turns[0]["role"] != "user":
            raise RecordRefused(
                "UNREADABLE_MESSAGE", rid,
                "has a `system` to join to its first message, and that message is not the user's")
        turns[0]["content"] = f"{system}\n\n{turns[0]['content']}"
    return turns
