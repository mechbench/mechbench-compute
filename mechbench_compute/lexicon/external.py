"""Ops that talk to something beyond the model in memory — a hosted
model endpoint, the Hugging Face hub, a benchmark harness — and the
ops that train and publish adapters.

Money. A node that calls a provider **must** declare `budget_usd`; the
platform refuses the protocol without it, and the executor refuses again
if one gets through. The cap is the most it may spend, and every call's
cost is recorded in the result so the bill is part of the measurement.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Op, P

_BUDGET = P("budget_usd", "float",
            "The most this node may spend on provider calls, in US dollars. "
            "Required when the model is a hosted endpoint; the node stops "
            "with what it has when the cap is reached. A job-level cap, if "
            "one is set, bounds it further.",
            None)


OPS: tuple[Op, ...] = (
    
    
)
