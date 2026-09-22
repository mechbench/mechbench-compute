"""`text/chat` — one block for local weights and remote
endpoints (task 000337, epic 000334).

The same node, the same params, the same output kind, whichever side of
the network answers:

    model: "google/gemma-3-4b-it"                → MLX, on this machine
    model: {provider: "anthropic", model: ...}   → the Messages API

That is the point of the endpoint ModelRef. A protocol that compares a
local fine-tune against a frontier model is one graph with two chat
nodes, not two systems, and everything downstream — text/measure,
group-stats, eval — reads the same document collection either way.

What differs is what the run can promise. Local sampling is
`reproducible`: seeds make it a pure function of its item key. A remote
call is `exchangeable` at best — someone else's sampler, someone else's
weights, possibly a different dated model version tomorrow — so the
manifest records the version that ANSWERED, the cost, and the usage,
and resume tops the set up rather than claiming bit-identity.

Remote items are I/O bound, so they run on a bounded thread pool.
Output order stays canonical (records × sample index) regardless of
completion order; the SPOOL receives items as they land, which is what
lets an interrupted node keep what it paid for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from mechbench_compute.providers import Budget, budget_from, make_transport
from mechbench_compute.providers import limiter as pl
from mechbench_compute.providers import messages as pm
from mechbench_compute.tools import toolbox_from
from mechbench_compute.chat.build_request import build_request  # noqa: F401
from mechbench_compute.chat.by_cause import _by_cause  # noqa: F401
from mechbench_compute.chat.constants import ITEM_KIND  # noqa: F401
from mechbench_compute.chat.item import _item  # noqa: F401
from mechbench_compute.chat.new_box import _new_box  # noqa: F401
from mechbench_compute.chat.records import _records  # noqa: F401
from mechbench_compute.chat.refuse_remote_only import _REMOTE_ONLY, _refuse_remote_only  # noqa: F401
from mechbench_compute.chat.render_conversation import render_conversation  # noqa: F401
from mechbench_compute.chat.run_local import run_local  # noqa: F401
from mechbench_compute.chat.run_remote import run_remote  # noqa: F401
from mechbench_compute.chat.sandbox_and_tools import _sandbox_and_tools  # noqa: F401
from mechbench_compute.chat.summary import _summary  # noqa: F401


