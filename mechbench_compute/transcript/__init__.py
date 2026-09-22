"""A transcript as a value.

A conversation is a fold over turns: render the shared transcript for
the participant whose turn it is, ask that participant's model, append
what it said. The two steps that are not "ask the model" live here —
`render` (transcript × participant → the messages a chat node sends)
and `extend` (transcript × reply → transcript) — so a conversation is
composed from `text/chat` rather than duplicated beside it.

`sees` lives here too: what a participant re-reads of the room's
reasoning is a param of the render step, and so sweepable, not a
setting on the participant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import thinking as THINK
from mechbench_compute.transcript.constants import MAIN  # noqa: F401
from mechbench_compute.transcript.read_transcripts import read_transcripts  # noqa: F401


# --- the ops ------------------------------------------------------------------------


