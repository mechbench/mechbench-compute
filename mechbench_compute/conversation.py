"""`~canonical/ops/conversation/1` — topologies as data (task 000339,
epic 000334).

Two models talking to each other is not a special mode of the chat
block; it is a graph of PARTICIPANTS plus three pieces of data:

**The perspective map.** Each participant sees the same shared
transcript through its own eyes: its own messages as `assistant`,
everyone else's as `user` — attributed by name, or merged anonymously.
That is the whole trick behind "Claude and GPT-5 each think they are
talking to a person": nobody is deceived, the roles are simply
rendered per viewer, and the transcript records both the participant
who produced a message AND the role each side saw it as.

**The turn policy.** Who speaks next, and when it stops: round robin,
the speaker naming their successor, a moderator agent choosing, a
judge agent deciding whether it is finished, or a stop phrase. Every
policy is bounded by `max_turns`, because a loop between two models
that never agree is the expensive kind of bug.

**The window policy.** Long conversations exceed context, and the fix
is a declared one — truncate the oldest, slide, or SUMMARIZE (itself a
model call, with its own provenance, cost and share of the cap).

Providers impose one thing on all of this: strict user/assistant
alternation. So consecutive messages from other participants MERGE
into a single user message before any adapter sees them, and the
rendering is checked rather than hoped for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mechbench_compute import chat as chat_mod
from mechbench_compute import tools as tool_mod
from mechbench_compute.providers import Budget, budget_from, make_transport
from mechbench_compute.providers import limiter as pl
from mechbench_compute.providers import messages as pm

TRANSCRIPT_KIND = "~canonical/kinds/transcript"

#: How a participant sees everyone else.
PERSPECTIVES = ("others_as_user_attributed", "others_as_user_merged")

#: Who speaks next.
TURN_POLICIES = ("round_robin", "speaker_names_next", "moderator",
                 "until_judge", "until_stop")

#: What to do when the conversation outgrows a context window.
WINDOW_POLICIES = ("none", "truncate_oldest", "sliding", "summarize")

#: Channels are how a conversation carries messages participants must
#: NOT see: a judge's verdicts, a moderator's routing. The perspective
#: map drops any channel a participant does not subscribe to.
MAIN = "main"

#: The reserved participant name for scripted lines (the opening, a
#: human-authored injection). It is never attributed in a rendering:
#: "user: hello" would read as a participant called "user".
SCRIPT_SPEAKER = "user"


@dataclass
class Agent:
    """A participant: which model, what it was told, what it may call.
    `budget_usd` is mandatory for a remote participant — the api
    refuses the node without one, and this is the second gate."""

    name: str
    model: Any
    system: str = ""
    tools: tuple[Any, ...] = ()
    provider_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 1024
    budget_usd: float | None = None
    channels: tuple[str, ...] = (MAIN,)
    perspective: str | None = None

    @staticmethod
    def parse(value: Any, *, index: int = 0) -> Agent:
        if isinstance(value, Agent):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "a participant is an agent object "
                '{"name", "model", "system", …}, not '
                f"{type(value).__name__}")
        raw = dict(value)
        if raw.get("kind") == "agent":
            raw.pop("kind")
        name = str(raw.get("name") or f"participant-{index + 1}")
        if "model" not in raw:
            raise ValueError(f"participant {name!r} declares no model")
        known = {f.name for f in Agent.__dataclass_fields__.values()}
        unknown = set(raw) - known - {"description", "provenance"}
        if unknown:
            raise ValueError(
                f"participant {name!r}: unknown field(s) "
                f"{', '.join(sorted(unknown))}")
        return Agent(
            name=name, model=raw["model"], system=str(raw.get("system", "")),
            tools=tuple(raw.get("tools") or ()),
            provider_options=dict(raw.get("provider_options") or {}),
            temperature=raw.get("temperature"), top_p=raw.get("top_p"),
            max_tokens=int(raw.get("max_tokens", 1024)),
            budget_usd=raw.get("budget_usd"),
            channels=tuple(raw.get("channels") or (MAIN,)),
            perspective=raw.get("perspective"),
        )

    def is_endpoint(self) -> bool:
        from mechbench_compute import model_ref as mr

        return mr.parse(self.model).is_endpoint


@dataclass
class Message:
    """One turn of the shared transcript, before any perspective is
    applied. `channel` is what makes a judge's verdict invisible to the
    participants it judges."""

    index: int
    participant: str
    text: str
    channel: str = MAIN
    tool_calls: tuple[Any, ...] = ()
    call: Mapping[str, Any] | None = None

    def to_wire(self, *, role_as_seen: str = "assistant") -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index, "participant": self.participant,
            "role_as_seen": role_as_seen, "text": self.text,
        }
        if self.tool_calls:
            out["tool_calls"] = [dict(t) for t in self.tool_calls]
        if self.call is not None:
            out["call"] = dict(self.call)
        if self.channel != MAIN:
            out["channel"] = self.channel
        return out


# --- the perspective map ---------------------------------------------------------


def render_for(agent: Agent, history: Sequence[Message], *,
               perspective: str) -> list[pm.Message]:
    """The shared transcript as THIS participant sees it.

    Own messages become `assistant`; everyone else's become `user`.
    Consecutive user-side messages merge into one, because providers
    require strict alternation and merging is the honest way to give it
    to them — the alternative is reordering someone's words.
    """
    if perspective not in PERSPECTIVES:
        raise ValueError(
            f"unknown perspective {perspective!r} — one of {PERSPECTIVES}")
    seen = set(agent.channels)
    turns: list[tuple[str, str]] = []
    for m in history:
        if m.channel not in seen:
            continue          # a judge's verdict is not part of the room
        if m.participant == agent.name:
            turns.append(("assistant", m.text))
            continue
        attributed = (perspective == "others_as_user_attributed"
                      and m.participant != SCRIPT_SPEAKER)
        text = f"{m.participant}: {m.text}" if attributed else m.text
        turns.append(("user", text))
    merged: list[tuple[str, str]] = []
    for role, text in turns:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + "\n\n" + text)
        else:
            merged.append((role, text))
    # A conversation that opens with this participant has nothing to
    # answer; providers want a user turn first, so the system prompt
    # carries the instruction and an empty opening is refused upstream.
    return [pm.Message(role=r, content=(pm.TextPart(t),)) for r, t in merged]


def system_for(agent: Agent, *, turn: int, participants: Sequence[str],
               values: Mapping[str, Any] | None = None) -> str:
    """A participant's system prompt with the conversation's own
    variables filled in: who is here, who this is, which turn it is."""
    others = [p for p in participants if p != agent.name]
    table = {
        "name": agent.name,
        "participants": ", ".join(participants),
        "others": ", ".join(others),
        "turn": str(turn),
        **{str(k): str(v) for k, v in (values or {}).items()},
    }
    text = agent.system
    for _ in range(3):
        before = text
        for key, value in table.items():
            text = text.replace("{" + key + "}", value)
        if text == before:
            break
    return text


# --- turn policies ----------------------------------------------------------------


def _named_participant(text: str, names: Sequence[str]) -> str | None:
    """The participant a speaker handed off to, if any. Matched on a
    word boundary so 'Claudia' does not answer for 'Claude'."""
    tail = text[-400:]
    hits = [(m.start(), n) for n in names
            for m in re.finditer(rf"\b{re.escape(n)}\b", tail)]
    return max(hits)[1] if hits else None


class TurnPolicy:
    """Who speaks next, and when to stop."""

    def __init__(self, params: Mapping[str, Any], names: Sequence[str]) -> None:
        self.kind = str(params.get("policy", "round_robin"))
        if self.kind not in TURN_POLICIES:
            raise ValueError(
                f"unknown turn policy {self.kind!r} — one of {TURN_POLICIES}")
        self.max_turns = int(params.get("max_turns", 6))
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.stop_phrases = tuple(params.get("stop_phrases") or ())
        self.names = list(names)
        self.moderator = params.get("moderator")
        self.judge = params.get("judge")

    def spoken(self, history: Sequence[Message]) -> list[Message]:
        """Turns the PARTICIPANTS took. A scripted opening and a judge's
        verdict are part of the transcript but not of the rotation, and
        not of `max_turns` — otherwise a two-line opening would silently
        buy two fewer model calls."""
        return [m for m in history
                if m.channel == MAIN and m.participant in self.names]

    def next_speaker(self, history: Sequence[Message]) -> str:
        spoken = self.spoken(history)
        if self.kind == "speaker_names_next" and history:
            named = _named_participant(history[-1].text, self.names)
            if named and (not spoken or named != spoken[-1].participant):
                return named
        return self.names[len(spoken) % len(self.names)]

    def stopped(self, history: Sequence[Message]) -> str:
        """Why the conversation is over, or "" to keep going."""
        if len(self.spoken(history)) >= self.max_turns:
            return "max_turns"
        if self.stop_phrases and history:
            text = history[-1].text.lower()
            for phrase in self.stop_phrases:
                if str(phrase).lower() in text:
                    return f"stop_phrase:{phrase}"
        return ""


# --- window policies ---------------------------------------------------------------


def apply_window(history: list[Message], *, policy: Mapping[str, Any],
                 count_tokens) -> tuple[list[Message], list[Message]]:
    """Fit the history into the window. Returns (kept, dropped).

    `truncate_oldest` and `sliding` differ in what they protect: the
    first keeps the tail, the second keeps the tail AND the opening
    turn, because the opening usually carries the task.
    """
    kind = str(policy.get("policy", "none"))
    if kind not in WINDOW_POLICIES:
        raise ValueError(
            f"unknown window policy {kind!r} — one of {WINDOW_POLICIES}")
    if kind == "none" or not history:
        return history, []
    budget = int(policy.get("tokens", 0))
    if budget <= 0:
        return history, []
    kept = list(history)
    dropped: list[Message] = []
    while len(kept) > 1 and count_tokens(kept) > budget:
        if kind == "sliding" and len(kept) > 2:
            dropped.append(kept.pop(1))     # keep the opening turn
        else:
            dropped.append(kept.pop(0))
    return kept, dropped


# --- the block ----------------------------------------------------------------------


def _transports(participants: Sequence[Agent], *, secrets, dry_run, params):
    """One transport per participant, built once. Local participants
    get None — the executor holds the model, and the caller samples
    through it."""
    out: dict[str, Any] = {}
    for agent in participants:
        from mechbench_compute import model_ref as mr

        ref = mr.parse(agent.model)
        if not ref.is_endpoint:
            out[agent.name] = None
            continue
        creds = dict((secrets or {}).get(ref.provider) or {})
        out[agent.name] = (
            make_transport(ref.provider, creds or None,
                           base_url=params.get("base_url"), dry_run=dry_run),
            ref, creds)
    return out


def run(params: Mapping[str, Any], *, inputs: Mapping[str, Any] | None = None,
        secrets: Mapping[str, Any] | None = None, limiter=None,
        job_budget: Budget | None = None, local_sampler=None,
        on_item=None, on_start=None, resume_items=None) -> dict[str, Any]:
    """Run one conversation per input record (or one, with none).

    `local_sampler(agent, system, messages, key=...) -> text` is
    injected by the executor for participants bound to local weights:
    this module knows conversation structure, not MLX.
    """
    inputs = inputs or {}
    raw_participants = (inputs.get("participants") or params.get("participants") or [])
    if isinstance(raw_participants, Mapping):
        raw_participants = (raw_participants.get("agents")
                            or raw_participants.get("records") or [])
    participants = [Agent.parse(p, index=i) for i, p in enumerate(raw_participants)]
    if len(participants) < 2:
        raise ValueError(
            "a conversation needs at least two participants — one model "
            "talking to itself is a chat node with n samples")
    names = [a.name for a in participants]
    if len(set(names)) != len(names):
        raise ValueError(f"participant names must be unique: {names}")

    policy = TurnPolicy(params.get("turns") or {}, names)
    perspective_map = params.get("perspective") or {}
    default_perspective = str(
        perspective_map.get("default", "others_as_user_attributed"))
    overrides = dict(perspective_map.get("overrides") or {})
    window = params.get("window") or {}
    max_tool_rounds = int(params.get("max_tool_rounds", 3))
    dry_run = bool(params.get("dry_run", False))
    budget = budget_from(params, required=any(a.is_endpoint() for a in participants))
    if job_budget is not None:
        budget = job_budget.child(budget.cap_usd)

    records = chat_mod._records(inputs.get("records") or params.get("records") or
                               [{"id": params.get("id", "conversation")}])
    transports = _transports(participants, secrets=secrets, dry_run=dry_run,
                             params=params)
    judge = Agent.parse(policy.judge, index=len(participants)) if policy.judge else None
    moderator = (Agent.parse(policy.moderator, index=len(participants))
                 if policy.moderator else None)
    if policy.kind == "moderator" and moderator is None:
        raise ValueError(
            "turn policy 'moderator' needs a moderator agent — who else "
            "would choose the speaker?")
    if moderator is not None and moderator.name not in transports:
        transports.update(_transports([moderator], secrets=secrets,
                                      dry_run=dry_run, params=params))
    if judge is not None and judge.name not in transports:
        transports.update(_transports([judge], secrets=secrets, dry_run=dry_run,
                                      params=params))
    summarizer_spec = window.get("summarizer")
    summarizer = Agent.parse(summarizer_spec) if summarizer_spec else None
    if summarizer is not None and summarizer.name not in transports:
        transports.update(_transports([summarizer], secrets=secrets,
                                      dry_run=dry_run, params=params))

    if on_start:
        on_start(len(records) * policy.max_turns)

    calls: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    def speak(agent: Agent, history: Sequence[Message], turn: int,
              values: Mapping[str, Any], key: str = "") -> tuple[
                  str, Mapping[str, Any] | None, tuple[Any, ...]]:
        """One participant's turn: render the room from where it sits,
        ask it, return what it said."""
        view = render_for(
            agent, history,
            perspective=str(overrides.get(agent.name)
                            or agent.perspective or default_perspective))
        system = system_for(agent, turn=turn, participants=names, values=values)
        entry = transports.get(agent.name)
        if entry is None:
            if local_sampler is None:
                raise ValueError(
                    f"participant {agent.name!r} runs local weights, but this "
                    "block was called without a local sampler — the executor "
                    "provides one")
            # The key seeds local sampling, so a resumed conversation
            # replays the same words at the same turn.
            return local_sampler(agent, system, view, key=key), None, ()
        transport, ref, creds = entry
        box = tool_mod.toolbox_from(agent.tools,
                                    block_runner=params.get("_block_runner"))
        req = pm.request({
            "model": ref.base,
            "system": system,
            "messages": [m.to_wire() for m in view],
            "max_tokens": int(agent.max_tokens),
            "tools": box.specs(),
            "temperature": agent.temperature,
            "top_p": agent.top_p,
            "provider_options": {**dict(ref.provider_options or {}),
                                 **dict(agent.provider_options or {})},
        })
        node_budget = (budget.child(agent.budget_usd) if agent.budget_usd
                       else budget)
        scope = pl.scope_for(ref.provider, creds)
        # A participant's tool loop is its own turn's business: the
        # room sees the answer, the transcript keeps what it called.
        for round_no in range(max_tool_rounds + 1):
            out = transport.chat(req, budget=node_budget, limiter=limiter,
                                 scope=scope)
            calls.append(out.call.to_wire())
            if not (out.tool_calls and box) or round_no == max_tool_rounds:
                break
            results = [box.call(c) for c in out.tool_calls]
            req = req.with_messages([
                *req.messages, out.as_message(),
                pm.Message(role="user", content=tuple(results)),
            ])
        record = out.call.to_wire()
        if box.runs:
            record = {**record, "tool_runs": [r.to_wire() for r in box.runs]}
        return out.text, record, out.tool_calls

    def count_tokens(history: Sequence[Message]) -> int:
        return max(1, sum(len(m.text.split()) for m in history))

    for rec in records:
        cid = str(rec.get("id", "conversation"))
        history: list[Message] = []
        # The opening is scripted: nobody was asked and nothing was
        # spent, which is why these messages carry no call record.
        for i, opening in enumerate(params.get("opening") or []):
            text = opening if isinstance(opening, str) else str(opening.get("text", ""))
            who = ("user" if isinstance(opening, str)
                   else str(opening.get("participant", "user")))
            # Record-scoped opening: the same conversation over a corpus
            # is one protocol, so `{field}` in an opening takes the
            # record's value the way a template node would.
            for field_name, value in rec.items():
                if isinstance(value, (str, int, float)):
                    text = text.replace("{" + str(field_name) + "}", str(value))
            history.append(Message(index=i, participant=who, text=text))
        stopped = ""
        spend_before = budget.spent_usd
        while not stopped:
            turn = len(history)
            key = f"{cid}:{turn}"
            if resume_items and key in resume_items:
                spooled = resume_items[key]
                history.append(Message(
                    index=turn, participant=str(spooled.get("participant", "")),
                    text=str(spooled.get("text", "")),
                    channel=str(spooled.get("channel", MAIN)),
                    call=spooled.get("call")))
                if on_item:
                    on_item(key, spooled, True)
                stopped = policy.stopped(history)
                continue
            chosen = policy.next_speaker(history)
            if policy.kind == "moderator" and moderator is not None:
                # The moderator is a participant in the room's business
                # but not in its conversation: it reads everything and
                # its routing rides a hidden channel.
                routing, mcall, _ = speak(moderator, history, turn, rec,
                                          f"{key}:moderator")
                history.append(Message(index=len(history),
                                       participant=moderator.name, text=routing,
                                       channel="moderator", call=mcall))
                named = _named_participant(routing, names)
                chosen = named or chosen
                turn = len(history)
                key = f"{cid}:{turn}"
            speaker = next(a for a in participants if a.name == chosen)
            kept, dropped = apply_window(history, policy=window,
                                         count_tokens=count_tokens)
            if dropped and str(window.get("policy")) == "summarize" and summarizer:
                text, call, _ = speak(summarizer, dropped, turn, rec,
                                      f"{key}:summary")
                kept = [Message(index=0, participant=summarizer.name,
                                text=f"[summary of {len(dropped)} earlier turns] {text}",
                                call=call), *kept]
            text, call, tool_calls = speak(speaker, kept, turn, rec, key)
            message = Message(index=turn, participant=speaker.name, text=text,
                              call=call, tool_calls=tuple(
                                  t.to_wire() for t in tool_calls))
            history.append(message)
            item = message.to_wire()
            item["conversation"] = cid
            if on_item:
                on_item(key, item)
            stopped = policy.stopped(history)
            if not stopped and judge is not None and policy.kind == "until_judge":
                verdict, jcall, _ = speak(judge, history, turn, rec,
                                          f"{key}:judge")
                history.append(Message(index=len(history), participant=judge.name,
                                       text=verdict, channel="judge", call=jcall))
                if _says_stop(verdict):
                    stopped = "judge"

        items.append(_transcript_item(cid, rec, participants, history, stopped,
                                      round(budget.spent_usd - spend_before, 8),
                                      default_perspective, overrides))

    return {
        "kind": "document_collection",
        "name": params.get("name", "conversations"),
        "description": params.get("description", ""),
        "fidelity": "segments",
        "item_kind": TRANSCRIPT_KIND,
        "items": items,
        "spend": chat_mod._summary(
            calls, budget,
            provider=",".join(sorted({c.get("provider", "") for c in calls})),
            dry_run=dry_run,
            replayed=sum(1 for c in calls if c.get("replayed"))),
    }


_STOP_RE = re.compile(r"\b(stop|done|finished|concluded|"
                      r"\"stop\"\s*:\s*true)\b", re.IGNORECASE)


def _says_stop(text: str) -> bool:
    return bool(_STOP_RE.search(text[-200:]))


def _transcript_item(cid: str, rec: Mapping[str, Any], participants,
                     history: Sequence[Message], stopped: str, spend: float,
                     default_perspective: str,
                     overrides: Mapping[str, Any]) -> dict[str, Any]:
    """One conversation as a document item: `turns` for the chat
    renderer, `text` so text/stats works unchanged, and the full
    transcript (with per-message provenance) in metadata."""
    visible = [m for m in history if m.channel == MAIN]
    return {
        "id": cid,
        "kind": TRANSCRIPT_KIND,
        "text": "\n\n".join(f"{m.participant}: {m.text}" for m in visible),
        "turns": [{"role": m.participant, "text": m.text} for m in visible],
        "metadata": {
            "coords": {**(rec.get("coords") or {})},
            "transcript": {
                "kind": "transcript",
                "id": cid,
                "participants": [a.name for a in participants],
                "messages": [
                    m.to_wire(role_as_seen="assistant") for m in history
                ],
                "stopped_because": stopped,
                "spend_usd": spend,
            },
            "perspective": {"default": default_perspective,
                            **({"overrides": dict(overrides)} if overrides else {})},
            "spend_usd": spend,
            "stopped_because": stopped,
        },
    }
