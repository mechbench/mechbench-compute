from __future__ import annotations

import json
from typing import Any

from mechbench_compute import Model
from mechbench_compute.protocol.dispatch import Dispatch
from mechbench_compute.protocol.is_remote import is_remote  # noqa: F401
from mechbench_compute.protocol.legacy_kinds import LegacyKinds
from mechbench_compute.protocol.memo import Memo
from mechbench_compute.protocol.model import ModelLoading
from mechbench_compute.protocol.pipeline import Pipeline
from mechbench_compute.protocol.protocol_spec import ProtocolSpec
from mechbench_compute.protocol.remote import Remote
from mechbench_compute.protocol.sub import Sub
from mechbench_compute.protocol.serialize_params import serialize_params  # noqa: F401
from mechbench_compute.protocol.sort_edges import sort_edges  # noqa: F401
from mechbench_compute.protocol.summarize_node import summarize_node  # noqa: F401


class ProtocolExecutor(Dispatch, LegacyKinds, Memo, ModelLoading, Pipeline,
                       Remote, Sub):
    def __init__(self, on_download=None, on_download_bytes=None, *,
                 on_node_start=None, on_spool_item=None,
                 on_checkpoint=None, on_node_done=None, on_node_kept=None,
                 limiter=None, budget=None, on_token=None) -> None:
        self._model: Model | None = None
        self._model_id: str | None = None
        self._on_download = on_download
        self._on_download_bytes = on_download_bytes
        self._on_node_start = on_node_start
        self._on_spool_item = on_spool_item
        self._on_checkpoint = on_checkpoint
        self._on_node_done = on_node_done
        self._on_node_kept = on_node_kept
        self._limiter = limiter
        self._budget = budget
        self._on_token = on_token

    def run(self, spec: ProtocolSpec, on_progress=None,
            secrets=None, resume=None, budget=None) -> Any:
        if spec.kind == "layer_ablation":
            return self._run_layer_ablation(spec.prompt, spec.model_id)
        if spec.kind == "decision_distribution":
            return self._legacy_decision_distribution(spec, on_progress)
        if spec.kind == "pipeline":
            if budget is not None:
                self._budget = budget
            return self._run_pipeline(spec, on_progress, secrets=secrets,
                                      resume=resume)
        if spec.kind == "replay":
            if budget is not None:
                self._budget = budget
            return self._run_replay(spec, on_progress, secrets=secrets)
        raise ValueError(f"unsupported protocolKind: {spec.kind!r}")


    def _run_replay(self, spec: ProtocolSpec, on_progress=None, secrets=None) -> Any:
        from mechbench_compute.live.replay import replay

        extra = spec.extra
        events = list(extra.get("events") or [])

        def step_done(i, _step):
            if on_progress is not None:
                on_progress(i + 1, len(events))

        return replay(self, graph=extra["graph"], params=extra.get("params") or {},
                      outputs=extra["outputs"], events=events, state=extra["state"],
                      inputs=extra.get("inputs"),
                      secrets=secrets, on_step=step_done)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
