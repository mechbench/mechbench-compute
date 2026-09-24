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
    def __init__(self, on_download=None, on_download_bytes=None, *,
                 on_node_start=None, on_spool_item=None,
                 on_checkpoint=None, on_node_done=None, on_node_kept=None,
                 limiter=None, budget=None) -> None:
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
        raise ValueError(f"unsupported protocolKind: {spec.kind!r}")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
