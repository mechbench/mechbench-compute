from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import thinking as THINK
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.transcript.constants import MAIN
from mechbench_compute.transcript.read_transcripts import read_transcripts

OP = Op(
    name="text/extend",
    summary=(
        "Each transcript with one more turn: the reply that names it, "
        "spoken by the participant — the last step of a conversation's "
        "fold."
    ),
    description="""\
The reply on `replies` whose `coords.conversation` names a transcript is
appended to it as `participant`'s turn. Reasoning the model marked is
kept on the message (`thinking`) and out of its `text`, which is what
the room hears; the reply's provider call rides along as `call`, with
`call.tool_runs` when the turn used the tools its chat node offered.
Every other field of the transcript is kept.

A transcript with two replies is two conversations, which is a map, not
a turn: more than one reply per transcript is refused, as is a reply
that names no conversation.

When the reply contains one of `stop_phrases`, or the transcript reaches
`max_messages`, its `stopped` says so — and a `records/fold` with `until:
{"field": "stopped"}` ends there.

A turn recorded on a `channel` other than `main` is on the transcript but
not in the room: only a participant whose `channels` name it will be
rendered it. That is all a platform should know about a turn that is not
for everyone — who reads it is the graph's business, not this op's.
""",
    inputs=(
        In("transcripts", "text/transcript", "The conversations so far.", many=True),
        In("replies", "records/record",
           "The replies, one per transcript: a chat node's documents over "
           "`text/render`'s records — or those documents after a "
           "`text/measure` has read something out of them, which is how a "
           "turn decides what happens next. Any record carrying `text` and "
           "naming its conversation is a turn.", many=True),
    ),
    output=Output('text/transcript', collection=True, doc='The same transcripts, each one message longer: `messages`, `participants` (the speaker added if new), `stopped`, `turns` and `text` for the browser.'),
    params=(
        P("participant", "string", "Who spoke: the participant's name."),
        P("channel", "string",
          "Which channel the turn is recorded on. A turn off `main` is kept "
          "on the transcript and shown only to a participant whose "
          "`channels` include it — which is how a turn can be part of the "
          "record without being part of the room.",
          "main"),
        P("keep_fields", "list[string]",
          "Fields to carry from the reply onto the transcript — a verdict, "
          "a rating, whose turn is next. What they mean is the graph's "
          "business; this op carries what it is told to. A field the "
          "transcript already has stands until a reply replaces it.",
          None),
        P("stop_phrases", "list[string]",
          "Phrases that end the conversation when the reply contains one "
          "(case-insensitive): the transcript's `stopped` becomes "
          "`stop_phrase:<phrase>`, which a fold's `until` reads.",
          None),
        P("max_messages", "int",
          "The message count at which the transcript's `stopped` becomes "
          "`max_messages`.",
          None),
    ),
    example={"participant": "ana", "stop_phrases": ["final answer"]},
    example_inputs={"transcripts": {"$ref": {"bench": "you/lab/debates"}},
                    "replies": {"$ref": {"bench": "you/lab/ana-replies"}}},
)


def run(ctx, inputs, params):
    return extend(inputs, params)


TRANSCRIPT_KIND = "text/transcript"


def extend(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`text/extend`: each transcript with one more turn — the reply on
    `replies` whose `coords.conversation` names it — spoken by
    `participant`. The reply's thinking (when its model marks
    reasoning) is kept on the message and out of its `text`, which is
    what the room hears; its provider call — with whatever tools the
    turn ran — rides along as `call`.
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
    for t in read_transcripts(inputs.get("transcripts")):
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
        meta = d.get("metadata") or {}
        call = meta.get("call")
        # What the turn actually did with the tools it was offered
        # rides along on the provider call, so the transcript records
        # the work and the room hears only the answer.
        runs = meta.get("tool_runs") or []
        if call is not None or runs:
            message["call"] = {
                **(dict(call) if call is not None else {}),
                **({"tool_runs": [dict(r) for r in runs]} if runs else {}),
            }
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
        items.append(build_transcript_item(cid, messages, participants=names,
                                           stopped=stopped, carried=carried,
                                           coords=dict(t.get("coords") or {}),
                                           metadata=dict(t.get("metadata") or {})))
    return K.collection(TRANSCRIPT_KIND, items, fidelity="segments",
                        description=f"Each transcript extended by one turn of {participant}.")


def build_transcript_item(cid: str, messages: Sequence[Mapping[str, Any]], *,
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
