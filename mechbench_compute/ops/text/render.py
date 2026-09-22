from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.transcript.constants import MAIN
from mechbench_compute.transcript.read_transcripts import read_transcripts

#: One `sees` policy: a word, or an object with one of these fields.
_POLICY_FIELDS = (
    P("last_turns", "int", "Replay only the n most recent such turns.", None),
    P("truncate_words", "int", "Replay the first n words of each.", None),
)


#: The `sees` clause, wherever a transcript is rendered for a participant.
_SEES = P("sees", "object",
          "What the participant re-reads of the room's reasoning: its own "
          "scratchpad ahead of its own words, the others' ahead of theirs. "
          "Each of `own_thinking` and `others_thinking` is `\"none\"` (the "
          "default), `\"full\"`, `{\"last_turns\": n}` (only the n most recent "
          "such turns) or `{\"truncate_words\": n}` (the first n words of "
          "each). A variable, not a setting: `{\"$param\": \"sees\"}` and a "
          "sweep over policies is one protocol.",
          None, fields=(
              P("own_thinking", "string | object", "The policy for its own reasoning.", "none",
                choices=("none", "full"), fields=_POLICY_FIELDS),
              P("others_thinking", "string | object", "The policy for everyone else's.", "none",
                choices=("none", "full"), fields=_POLICY_FIELDS),
          ))


OP = Op(
    name="text/render",
    summary=(
        "A transcript as one participant sees it: its own turns as "
        "assistant, the others' as user, under a perspective and a `sees` "
        "clause — the chat-shaped records a chat node sends."
    ),
    description="""\
A conversation is a fold over turns — render the shared transcript for
the participant whose turn it is, ask its model, append what it said —
and this is the first step. A conversation is therefore composed from
`text/chat`, not a second kind of chat beside it.

Every transcript on `transcripts` becomes one record: `messages` (`[{role,
content}]`, what a chat node sends), `system` when given (with `{name}`,
`{participants}`, `{others}`, `{turn}` filled in), and the transcript's
coords plus `conversation` and `participant` — which is how `text/extend`
knows which reply belongs to which transcript.

**Perspective.** The participant's own messages become `assistant`;
everyone else's become `user`, attributed by name
(`others_as_user_attributed`) or not (`others_as_user_merged`); a scripted
line (the opening) is never attributed. Consecutive user-side messages
merge into one, because providers require strict alternation and merging
is the honest way to give it to them — the alternative is reordering
someone's words. A message on a channel the participant is not on (a
judge's verdict) is not part of the room.

**How much it sees.** `window` cuts the transcript down when it has
outgrown its budget — the tail alone, or the tail and the opening turn,
which usually carries the task. What falls outside was not seen at all:
it is dropped before the perspective is applied, so the turns that
remain still alternate.

**What it sees of the reasoning.** A model that marks its reasoning has
that reasoning kept on the transcript and out of the message's `text`.
`sees` says what comes back on a later turn: the participant's own
scratchpad ahead of its own words; another's, marked `(thinking)`, ahead
of theirs. The default replays nothing. Replaying is off-distribution —
reasoning models are generally trained with their thinking discarded
between turns — which is exactly why it is a variable here and not a
setting: the question is what changes.
""",
    inputs=(
        In("transcripts", "text/transcript",
           "The conversations to render, one record each.", many=True),
    ),
    output=Output('records/record', collection=True, doc='One record per transcript: `id` (the conversation\'s), `messages` (`[{role, content}]`), `system` when given, `coords` (the transcript\'s, plus `conversation` and `participant`). The header carries `participant`, `perspective` and `sees` as resolved.'),
    params=(
        P("participant", "string", "Whose view: the participant's name."),
        P("perspective", "string", "How the others' messages read.",
          "others_as_user_attributed", choices=("others_as_user_attributed", "others_as_user_merged")),
        _SEES,
        P("channels", "list[string]", "The channels the participant is on.", ["main"]),
        P("window", "object",
          "How much of the transcript this participant sees, when it has "
          "outgrown what it should be shown. How much is part of what — "
          "which is why it is decided here and not by whoever runs the "
          "turns.",
          None, fields=(
              P("policy", "string",
                "`\"truncate_oldest\"` keeps the tail; `\"sliding\"` keeps the "
                "tail and the opening turn, because the opening usually "
                "carries the task; `\"none\"` keeps everything.",
                "none", choices=("none", "truncate_oldest", "sliding")),
              P("words", "int",
                "The budget, in words. A window is a budget, not a "
                "tokenizer: the participant's own model is what would count "
                "tokens, and it is not here.", 0),
          )),
        P("system", "string",
          "A system prompt for the record, with `{name}`, `{participants}`, "
          "`{others}` and `{turn}` filled in.",
          None),
    ),
    example={"participant": "ana", "sees": {"own_thinking": {"last_turns": 1}}},
    example_inputs={"transcripts": {"$ref": {"bench": "you/lab/debates"}}},
)


def run(ctx, inputs, params):
    return render_records(inputs, params)


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


def _count_words(messages: Sequence[Mapping[str, Any]]) -> int:
    """A cheap length, in words: the window is a budget, not a
    tokenizer, and a participant's own model is what would count."""
    return max(1, sum(len(str(m.get("text", "")).split()) for m in messages))


def apply_window(messages: Sequence[Mapping[str, Any]],
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
    while len(kept) > 1 and _count_words(kept) > budget:
        # `sliding` protects the opening turn as well as the tail.
        dropped.append(kept.pop(1 if kind == "sliding" and len(kept) > 2 else 0))
    return kept, dropped


def _show_thinking(policy: Any, thinking: str | None, turns_back: int) -> str | None:
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
    visible, _dropped = apply_window(visible, window)
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
            thought = _show_thinking(sees["own_thinking"], m.get("thinking"), own_back[i])
            turns.append(("assistant", f"{thought}\n\n{text}" if thought else text))
            continue
        spoke = who in set(participants) if participants else who != SCRIPT_SPEAKER
        attributed = perspective == "others_as_user_attributed" and spoke
        thought = _show_thinking(sees["others_thinking"], m.get("thinking"), others_back[i])
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
    for t in read_transcripts(inputs.get("transcripts")):
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
