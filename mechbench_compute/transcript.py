"""A transcript as a value (task 000617).

A conversation is a fold over turns: render the shared transcript for
the participant whose turn it is, ask that participant's model, append
what it said. `text/converse` did all three inside one loop; these are
the two pieces of that loop that are not "ask the model" — `render`
(transcript × participant → the messages a chat node sends) and
`extend` (transcript × reply → transcript) — so a conversation can be
composed from `text/chat` rather than duplicated beside it.

`sees` (task 000593) lives here too: what a participant re-reads of the
room's reasoning is a param of the render step, and so sweepable, not
a setting on the participant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import thinking as THINK

TRANSCRIPT_KIND = "text/transcript"
MAIN = "main"
#: The reserved participant name for scripted lines (the opening, a
#: human-authored injection). It is never attributed in a rendering:
#: "user: hello" would read as a participant called "user".
SCRIPT_SPEAKER = "user"
#: Kept for a transcript that does not say who its participants are.
#: Where it does, the list is the rule and this name is not special.
PERSPECTIVES = ("others_as_user_attributed", "others_as_user_merged")

#: What a participant re-reads of the room's reasoning, by default:
#: nothing. A scratchpad is written to be thrown away, and a
#: conversation that replays it by accident is feeding on its own
#: reasoning (task 000592). Anything else is a choice — a declared,
#: sweepable one.
SEES_DEFAULT: dict[str, Any] = {"own_thinking": "none", "others_thinking": "none"}
_SEES_KEYS = ("own_thinking", "others_thinking")


def parse_sees(value: Any) -> dict[str, Any]:
    """The `sees` clause, checked: each of `own_thinking` and
    `others_thinking` is `"none"`, `"full"`, `{"last_turns": n}` or
    `{"truncate_words": n}`. Absent keys take the default."""
    if value is None:
        return dict(SEES_DEFAULT)
    if isinstance(value, bool):
        # The 000592 boolean, read as the clause it meant.
        return {"own_thinking": "full" if value else "none", "others_thinking": "none"}
    if not isinstance(value, Mapping):
        raise ValueError("`sees` is an object: {own_thinking, others_thinking}")
    unknown = set(value) - set(_SEES_KEYS)
    if unknown:
        raise ValueError(f"`sees`: unknown key(s) {', '.join(sorted(unknown))}; "
                         f"the keys are {list(_SEES_KEYS)}")
    out = dict(SEES_DEFAULT)
    for key in _SEES_KEYS:
        if key not in value:
            continue
        policy = value[key]
        if policy in ("none", "full"):
            out[key] = policy
        elif isinstance(policy, Mapping) and len(policy) == 1 and (
                "last_turns" in policy or "truncate_words" in policy):
            (name, n), = policy.items()
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise ValueError(f"`sees.{key}.{name}` is a non-negative integer, not {n!r}")
            out[key] = {name: n}
        else:
            raise ValueError(
                f"`sees.{key}` is \"none\", \"full\", {{\"last_turns\": n}} or "
                f"{{\"truncate_words\": n}}, not {policy!r}")
    return out


#: How a transcript is cut down to fit. `truncate_oldest` keeps the
#: tail; `sliding` keeps the tail AND the opening turn, because the
#: opening usually carries the task.
WINDOW_POLICIES = ("none", "truncate_oldest", "sliding")


def _words(messages: Sequence[Mapping[str, Any]]) -> int:
    """A cheap length, in words: the window is a budget, not a
    tokenizer, and a participant's own model is what would count."""
    return max(1, sum(len(str(m.get("text", "")).split()) for m in messages))


def windowed(messages: Sequence[Mapping[str, Any]],
             window: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, dropped) under a window policy. How much of a transcript a
    participant sees is part of what it sees, which is why this belongs
    beside the rendering and not in a loop (000617)."""
    messages = [dict(m) for m in messages]
    if not window:
        return messages, []
    kind = str(window.get("policy", "none"))
    if kind not in WINDOW_POLICIES:
        raise ValueError(
            f"unknown window policy {kind!r} — one of {', '.join(WINDOW_POLICIES)}")
    budget = int(window.get("words", 0))
    if kind == "none" or budget <= 0 or not messages:
        return messages, []
    kept, dropped = list(messages), []
    while len(kept) > 1 and _words(kept) > budget:
        # `sliding` protects the opening turn as well as the tail.
        dropped.append(kept.pop(1 if kind == "sliding" and len(kept) > 2 else 0))
    return kept, dropped


def _shown(policy: Any, thinking: str | None, turns_back: int) -> str | None:
    """The thinking a policy lets through for a message `turns_back`
    turns from the end of that speaker's (or the others') turns —
    0 being the most recent."""
    if not thinking or policy == "none":
        return None
    if policy == "full":
        return thinking
    (name, n), = policy.items()
    if name == "last_turns":
        return thinking if turns_back < n else None
    words = thinking.split()
    return " ".join(words[:n]) if words[:n] else None


def render(messages: Sequence[Mapping[str, Any]], *, participant: str,
           channels: Sequence[str] = (MAIN,), perspective: str = "others_as_user_attributed",
           sees: Mapping[str, Any] | None = None,
           participants: Sequence[str] | None = None,
           window: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """The shared transcript as THIS participant sees it: `[{role,
    content}]`, the messages a chat node sends.

    Own messages become `assistant`; everyone else's become `user`,
    attributed by name under the attributed perspective. A line nobody
    in `participants` said is a SCRIPTED one — an opening, an injection
    — and is never attributed, because "user: hello" would read as a
    participant called user. (A transcript that lists no participants
    falls back to the reserved name itself.) Consecutive user-side
    messages merge into one,
    because providers require strict alternation and merging is the
    honest way to give it to them — the alternative is reordering
    someone's words. A message on a channel this participant is not on
    (a judge's verdict) is not part of the room.

    `sees` says what reasoning comes back: a participant's own
    scratchpad ahead of its own words, another's ahead of theirs,
    marked `(thinking)`.
    """
    if perspective not in PERSPECTIVES:
        raise ValueError(f"unknown perspective {perspective!r} — one of {PERSPECTIVES}")
    sees = parse_sees(sees)
    seen = set(channels)
    visible = [m for m in messages if str(m.get("channel", MAIN)) in seen]
    # What is beyond the window was not seen at all — dropped before the
    # perspective is applied, so the turns that remain still alternate.
    visible, _dropped = windowed(visible, window)
    # How many turns back each message is, among its own kind — the
    # speaker's own turns for `own_thinking`, everyone else's for
    # `others_thinking` — counted from the end.
    own_back: dict[int, int] = {}
    others_back: dict[int, int] = {}
    for i in range(len(visible) - 1, -1, -1):
        who = str(visible[i].get("participant", ""))
        table = own_back if who == participant else others_back
        table[i] = len(table)
    turns: list[tuple[str, str]] = []
    for i, m in enumerate(visible):
        who = str(m.get("participant", ""))
        text = str(m.get("text", ""))
        if who == participant:
            thought = _shown(sees["own_thinking"], m.get("thinking"), own_back[i])
            turns.append(("assistant", f"{thought}\n\n{text}" if thought else text))
            continue
        spoke = who in set(participants) if participants else who != SCRIPT_SPEAKER
        attributed = perspective == "others_as_user_attributed" and spoke
        thought = _shown(sees["others_thinking"], m.get("thinking"), others_back[i])
        line = f"{who}: {text}" if attributed else text
        if thought:
            line = (f"{who} (thinking): {thought}\n\n{line}" if attributed
                    else f"(thinking) {thought}\n\n{line}")
        turns.append(("user", line))
    merged: list[tuple[str, str]] = []
    for role, text in turns:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + "\n\n" + text)
        else:
            merged.append((role, text))
    return [{"role": r, "content": t} for r, t in merged]


# --- the ops ------------------------------------------------------------------------


def _transcripts(value: Any) -> list[dict[str, Any]]:
    from mechbench_compute.lexicon import kinds as K

    items = K.items_of(value or [])
    for t in items:
        if not isinstance(t.get("messages"), list):
            raise ValueError(
                f"transcript {t.get('id')!r} has no `messages`; a text/transcript "
                "carries its messages at the top level")
    return items


def render_records(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`text/render`: every transcript rendered for one participant, as
    the chat-shaped records `text/chat` takes — `messages`, `system`,
    the transcript's coords plus `participant` and `conversation`."""
    from mechbench_compute.lexicon import kinds as K

    participant = str(params["participant"])
    perspective = str(params.get("perspective") or "others_as_user_attributed")
    sees = parse_sees(params.get("sees"))
    channels = tuple(params.get("channels") or (MAIN,))
    window = params.get("window")
    system = str(params.get("system") or "")
    rows = []
    for t in _transcripts(inputs.get("transcripts")):
        cid = str(t.get("id"))
        names = [str(n) for n in (t.get("participants") or [])]
        messages = render(t["messages"], participant=participant, channels=channels,
                          perspective=perspective, sees=sees, participants=names,
                          window=window)
        coords = {**(t.get("coords") or {}), "conversation": cid, "participant": participant}
        row: dict[str, Any] = {"id": cid, "coords": coords, "messages": messages}
        if system:
            row["system"] = _fill(system, participant=participant, participants=names,
                                  turn=len(t["messages"]))
        rows.append(row)
    return K.collection("records/record", rows, participant=participant,
                        perspective=perspective, sees=sees,
                        **({"window": dict(window)} if window else {}),
                        description=f"Every transcript as {participant} sees it.")


def _fill(text: str, *, participant: str, participants: Sequence[str], turn: int) -> str:
    """A system prompt with the conversation's own variables filled in:
    who is here, who this is, which turn it is."""
    others = [p for p in participants if p != participant]
    table = {"name": participant, "participants": ", ".join(participants),
             "others": ", ".join(others), "turn": str(turn)}
    for _ in range(3):
        before = text
        for key, value in table.items():
            text = text.replace("{" + key + "}", value)
        if text == before:
            break
    return text


def extend(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`text/extend`: each transcript with one more turn — the reply on
    `replies` whose `coords.conversation` names it — spoken by
    `participant`. The reply's thinking (when its model marks
    reasoning) is kept on the message and out of its `text`, which is
    what the room hears; its provider call rides along as `call`.
    A conversation with two replies is two conversations, which is a
    map, not a turn: more than one reply per transcript is refused."""
    from mechbench_compute.lexicon import kinds as K

    participant = str(params["participant"])
    channel = str(params.get("channel") or MAIN)
    keep_fields = [str(f) for f in (params.get("keep_fields") or [])]
    stop_phrases = [str(x) for x in (params.get("stop_phrases") or []) if str(x)]
    max_messages = params.get("max_messages")
    replies: dict[str, list[Mapping[str, Any]]] = {}
    for d in K.items_of(inputs.get("replies") or []):
        cid = (d.get("coords") or {}).get("conversation")
        if cid is None:
            cid = (d.get("metadata") or {}).get("coords", {}).get("conversation")
        if cid is None:
            raise ValueError(
                f"reply {d.get('id')!r} names no conversation: a reply to a rendered "
                "transcript carries `coords.conversation`, which text/render sets")
        replies.setdefault(str(cid), []).append(d)
    items = []
    for t in _transcripts(inputs.get("transcripts")):
        cid = str(t.get("id"))
        got = replies.get(cid, [])
        if len(got) != 1:
            raise ValueError(
                f"transcript {cid!r} has {len(got)} replies; a turn is one reply "
                "(sample n=1, or map over samples to make n conversations)")
        d = got[0]
        thought, said = THINK.split_thought(str(d.get("text", "")))
        message: dict[str, Any] = {
            "index": len(t["messages"]), "participant": participant,
            "role_as_seen": "assistant", "text": said,
        }
        if channel != MAIN:
            # Recorded, but not part of the room: only a participant
            # whose `channels` include this one will be rendered it.
            message["channel"] = channel
        if thought:
            message["thinking"] = thought
        call = (d.get("metadata") or {}).get("call")
        if call is not None:
            message["call"] = dict(call)
        names = [str(n) for n in (t.get("participants") or [])]
        if participant not in names:
            names.append(participant)
        known = {"id", "kind", "coords", "participants", "messages", "stopped",
                 "text", "turns", "metadata"}
        # The transcript's own carried fields stand until a reply
        # replaces one.
        standing = {k: v for k, v in t.items() if k not in known}
        # Why the conversation is over, when this turn ended it: a stop
        # phrase in what was said, or the message cap reached. A fold's
        # `until: {"field": "stopped"}` reads it.
        stopped = str(t.get("stopped") or "")
        messages = [*t["messages"], message]
        if not stopped:
            hit = next((ph for ph in stop_phrases if ph.lower() in said.lower()), None)
            if hit is not None:
                stopped = f"stop_phrase:{hit}"
            elif max_messages is not None and len(messages) >= int(max_messages):
                stopped = "max_messages"
        # What the reply said about itself, carried onto the transcript:
        # a verdict, a rating, whose turn is next. The op does not know
        # what any of them mean — it carries what it was told to.
        carried = {**standing, **{f: d[f] for f in keep_fields if f in d}}
        items.append(transcript_item(cid, messages, participants=names,
                                     stopped=stopped, carried=carried,
                                     coords=dict(t.get("coords") or {}),
                                     metadata=dict(t.get("metadata") or {})))
    return K.collection(TRANSCRIPT_KIND, items, fidelity="segments",
                        description=f"Each transcript extended by one turn of {participant}.")


def transcript_item(cid: str, messages: Sequence[Mapping[str, Any]], *,
                    participants: Sequence[str], stopped: str = "",
                    coords: Mapping[str, Any] | None = None,
                    metadata: Mapping[str, Any] | None = None,
                    carried: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One conversation as its kind declares it: `messages`,
    `participants` and `stopped` at the top level, `turns` and `text`
    (the visible turns) for the browser, and coords on the item as
    every record's are."""
    visible = [m for m in messages if str(m.get("channel", MAIN)) == MAIN]
    return {
        **dict(carried or {}),
        "id": cid,
        "kind": TRANSCRIPT_KIND,
        "coords": dict(coords or {}),
        "participants": list(participants),
        "messages": [dict(m) for m in messages],
        "stopped": stopped,
        "text": "\n\n".join(f"{m.get('participant')}: {m.get('text', '')}" for m in visible),
        "turns": [{"role": m.get("participant"), "text": m.get("text", "")} for m in visible],
        "metadata": {**dict(metadata or {}), "coords": dict(coords or {})},
    }
