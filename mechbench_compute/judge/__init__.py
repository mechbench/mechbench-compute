"""`eval/judge` — model-graded scoring.

A judge reads records (or transcripts) against a rubric and returns a
score, a label, or a preference — with its reasoning, its provenance,
and its cost.

Three commitments make a judge's numbers usable rather than merely
available:

**Votes, not a verdict.** `n_votes` repeats the call and aggregates —
mean for numeric scales, majority for categorical and pairwise — and
the object records the SPREAD. A judge that disagrees with itself is
telling you something about the rubric, and averaging that away would
throw out the finding.

**Position is randomized and recorded.** In pairwise mode the A/B order
flips per vote from a key-derived seed, and each vote records the order
it saw. Position bias is real and large; a judge that always picks A is
a result you can only see if you looked.

**Parsing is honest.** The judge is asked for JSON and read leniently
(a bare number, a quoted label), but a vote that could not be parsed is
recorded as unparsed rather than silently scored — an unreadable answer
is data about the rubric, not a zero.

Everything rides the chat block, so a judge inherits the budget cap,
the concurrency bound, item spooling and per-call provenance for free,
and a LOCAL judge (Gemma through the same path) is the cheap first
test rather than a special case.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import chat as chat_mod
from mechbench_compute.judge.constants import SCALES, FIRST_NUMBER  # noqa: F401
from mechbench_compute.judge.parse_json_object import parse_json_object  # noqa: F401
from mechbench_compute.judge.read_rationale import read_rationale  # noqa: F401


