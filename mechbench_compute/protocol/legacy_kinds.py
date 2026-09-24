from __future__ import annotations

from datetime import UTC
from typing import Any

import numpy as np
from mechbench_schema import (
    AblationPrompt,
    LayerAblationPayload,
    LayerAggregates,
)

from mechbench_compute import GLOBAL_LAYERS, N_LAYERS, Ablate, lexicon
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.protocol.protocol_spec import ProtocolSpec


class LegacyKinds:
    def _run_layer_ablation(
        self, prompt: str, model_id: str
    ) -> LayerAblationPayload:
        model = self._model_loaded(model_id)

        ids = model.tokenize(prompt)
        baseline = model.run(ids)
        baseline_lp = read_last_logp(baseline.logits)
        top1_id = int(np.argmax(baseline_lp))
        baseline_top1 = float(baseline_lp[top1_id])

        damage = np.zeros(N_LAYERS, dtype=np.float32)
        for layer in range(N_LAYERS):
            ids = model.tokenize(prompt)
            result = model.run(ids, interventions=[Ablate.layer(layer)])
            lp = read_last_logp(result.logits)
            damage[layer] = float(lp[top1_id]) - baseline_top1

        prompts = [
            AblationPrompt(
                text=prompt,
                target="",
                top1_id=top1_id,
                baseline_logprob=round(baseline_top1, 4),
                damage=[round(float(v), 4) for v in damage],
            )
        ]
        return LayerAblationPayload(
            protocol="mechbench-runner:layer_ablation",
            description=(
                "Single-prompt layer ablation: zero each decoder block's "
                "residual-stream update and measure Δ log p of the "
                "model's top-1 prediction."
            ),
            model=self.model_ref(model) or model_id,
            n_layers=N_LAYERS,
            global_layers=list(GLOBAL_LAYERS),
            prompts=prompts,
            aggregates=LayerAggregates(
                mean=[round(float(v), 4) for v in damage],
                median=[round(float(v), 4) for v in damage],
            ),
        )

    def _legacy_decision_distribution(self, spec: ProtocolSpec,
                                      on_progress=None) -> Any:
        from datetime import datetime

        import mechbench_schema as ms

        from mechbench_compute import __version__ as core_version

        extra = spec.extra or {}
        self._protocol_ref = (extra.get("protocolId"), extra.get("protocolVersion"))
        conditions = extra.get("conditions", [])
        result = self._run_op(
            "logits/read", {"conditions": conditions}, {"model": spec.model_id})
        prov = ms.Provenance(
            created_at=datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            produced_by=ms.ToolInfo(tool="mechbench-runner",
                                    version=core_version),
            inputs=[],
            params_fingerprint=ms.fingerprint_params(
                {k: v for k, v in extra.items() if k != "resultPath"}),
            schema_version=ms.__version__,
        )
        return ms.Emitted(
            payload=lexicon.canonical_collection(lexicon.collection(
                "logits/decision", lexicon.items_of(result),
                model=spec.model_id)),
            provenance=prov)
