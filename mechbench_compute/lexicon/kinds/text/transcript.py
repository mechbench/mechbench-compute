from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F, ID

KIND = Kind(
    "text/transcript",
    "One multi-party conversation: its messages in order, each with who said it and the role each side saw it as. A record, so every op that reads records reads transcripts.",
    fields={
        "id": ID,
        "messages": F("array", "`{index, participant, role_as_seen, text, call?, tool_calls?, channel?}`, in order.", items={"type": "object"}),
        "stopped": F("string", "Why the conversation ended: `max_turns`, or a stop phrase."),
        "participants": F("array", "The participant names.", items={"type": "string"}),
        "text": F("string", "The transcript rendered as text, for the browser.", **{"x-mechbench-text": True}),
    },
    required=("id", "messages"),
    extends="records/record",
    key=("id",),
    header={"name": "A label for the collection.", "description": "Free text beside the name.",
            "spend": "What the run bought from providers."},
    renderer={"primitive": "chat", "field_map": {"messages": "messages"}},
    doc="What a conversation writes — `text/render` → `text/chat` → `text/extend`, folded over turns: every "
        "message in order, each naming the participant who said it and "
        "the role each side saw it as, with any tool calls it made. `stopped` records why the conversation ended "
        "— the turn cap, or a stop phrase — so a transcript that ended early says so itself.",
)
