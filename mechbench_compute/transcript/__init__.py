"""A transcript as a value (task 000617).

A conversation is a fold over turns: render the shared transcript for
the participant whose turn it is, ask that participant's model, append
what it said. A retired op did all three inside one loop; these are
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
from mechbench_compute.transcript.constants import MAIN  # noqa: F401
from mechbench_compute.transcript.read_transcripts import read_transcripts  # noqa: F401


# --- the ops ------------------------------------------------------------------------


