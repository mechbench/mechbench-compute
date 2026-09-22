"""Protocol execution: turn an experiment spec into results.

Lives in the compute layer, beside the primitives it drives, because it
is pure computation — it takes a spec, runs operations against a loaded
model, and returns typed payloads. It reaches no network and knows
nothing about jobs, queues or credentials; `mechbench-runner` owns all
of that and calls in here once it has claimed something to do.

The model is loaded once per process and reused. Load cost is
significant (minutes on first call, seconds on cached weights), so
callers should hold the executor for the process's lifetime rather
than instantiating per request.

`ProtocolExecutor` is composed from one mixin per topic, each a file of
its own, so that one question about it is answered by one file:

    pipeline.py       walking the graph: order, resume, emission
    dispatch.py       how a node reaches an operation's run()
    model.py          loading weights, fusing adapters
    remote.py         the nodes a provider answers, run in a wave
    tools.py          what a model may call mid-turn
    chat.py           the local half of text/chat
    memo.py           a node's memo of the remote calls it made
    legacy_kinds.py   the two spec kinds that are not graphs

A mixin is how a method keeps its `self`: every method reads the same
executor state whichever file it is written in.
"""

from __future__ import annotations

import json
from typing import Any

from mechbench_compute import Model
from mechbench_compute.protocol.chat import Chat
from mechbench_compute.protocol.dispatch import Dispatch
from mechbench_compute.protocol.is_remote import (  # noqa: F401
    REMOTE_BLOCKS,
    is_remote,
)
from mechbench_compute.protocol.legacy_kinds import LegacyKinds
from mechbench_compute.protocol.memo import Memo
from mechbench_compute.protocol.model import ModelLoading
from mechbench_compute.protocol.pipeline import Pipeline
from mechbench_compute.protocol.protocol_spec import ProtocolSpec
from mechbench_compute.protocol.remote import Remote
from mechbench_compute.protocol.serialize_params import serialize_params  # noqa: F401
from mechbench_compute.protocol.sort_edges import sort_edges  # noqa: F401
from mechbench_compute.protocol.summarize_node import summarize_node  # noqa: F401
from mechbench_compute.protocol.tools import Tools


class ProtocolExecutor(Chat, Dispatch, LegacyKinds, Memo, ModelLoading, Pipeline,
                       Remote, Tools):
    """The executor itself: what it is built with, and the one
    call that runs a spec. Everything it does is a topic beside
    this file — see the module docstring."""

    def __init__(self, on_download=None, on_download_bytes=None, *,
                 on_node_start=None, on_spool_item=None,
                 on_checkpoint=None, on_node_done=None, on_node_kept=None,
                 limiter=None, budget=None) -> None:
        self._model: Model | None = None
        self._model_id: str | None = None
        # Called just before weights are fetched, and only then: the runner
        # uses it to announce a wait that can run to gigabytes.
        self._on_download = on_download
        self._on_download_bytes = on_download_bytes
        # Resume plumbing. All optional; the runner wires them to its
        # spool. `on_node_start(nid, fp)` names the process identity a
        # node runs under; `on_spool_item(nid, key, item)` hands over
        # each completed item of an item-resumable block;
        # `on_checkpoint(nid, state)` a training checkpoint;
        # `on_node_done(nid, path, fp)` a finished node's emitted object;
        # `on_node_kept(nid, fp, result)` a finished node's result HELD
        # on the device instead of emitted (a run with `keep: "outputs"`),
        # so a resume on this device can pick it up without the bench
        # ever having seen it.
        self._on_node_start = on_node_start
        self._on_spool_item = on_spool_item
        self._on_checkpoint = on_checkpoint
        self._on_node_done = on_node_done
        self._on_node_kept = on_node_kept
        # Rate limits are the runner's business: it knows what else is
        # running against the same account. Absent one, remote calls are
        # unthrottled and only the provider's own 429s slow them down.
        self._limiter = limiter
        # A JOB-level budget, when the runner sets one: every remote
        # node's own cap is chained under it, so a graph whose node caps
        # sum past the job's cannot spend past the job's. The runner
        # reads `spent_usd` off it live to report spend.
        self._budget = budget

    def run(self, spec: ProtocolSpec, on_progress=None,
            secrets=None, resume=None, budget=None) -> Any:
        """Execute a job spec. `on_progress(done, total)` is invoked
        after each unit of work for kinds that have a natural unit
        (decision_distribution: one condition); it must be cheap and
        may be None.

        `resume`: `{node_id: {"fingerprint": str, and
        one of "done": <emitted object path> | "items": {key: item} |
        "checkpoint": <training state>}}`. Each entry is honoured
        only under an equal node fingerprint — otherwise that node
        restarts. The result is byte-identical to an uninterrupted
        run by construction; nothing about resumption is recorded in
        it."""
        if spec.kind == "layer_ablation":
            return self._run_layer_ablation(spec.prompt, spec.model_id)
        if spec.kind == "decision_distribution":
            return self._legacy_decision_distribution(spec, on_progress)
        if spec.kind == "pipeline":
            # A job-level budget arrives per RUN (the executor outlives
            # the job; the cap does not), and every remote node's cap is
            # chained under it.
            if budget is not None:
                self._budget = budget
            return self._run_pipeline(spec, on_progress, secrets=secrets,
                                      resume=resume)
        raise ValueError(f"unsupported protocolKind: {spec.kind!r}")


def canonical_json(payload: Any) -> str:
    """Python-side canonical JSON for the mechbench-api hash contract.

    Cross-language byte-identity is not available with JSON (Python
    emits `0.0`, JS emits `0`). The mechbench-api side hashes the bytes
    as received, so this form only has to be stable across Python
    invocations; pinning it across languages needs canonical CBOR.
    """
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
