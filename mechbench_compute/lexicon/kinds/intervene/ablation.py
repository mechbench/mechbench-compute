from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F, ID

KIND = Kind(
    "intervene/ablation",
    "The change in a target token's log-probability when one layer's component is removed, for one record and one layer.",
    fields={"id": ID, "layer": F("integer", "The layer ablated."),
            "delta_logp": F("number", "Ablated minus baseline log-probability.")},
    required=("id", "layer", "delta_logp"),
    key=("id", "layer"),
    header={"component": "What was removed at each layer.", "layers": "The layers swept.",
            "n_conditions": "How many records.",
            "n_off_top1": "How many conditions tracked a target the model would not itself have said (each such condition carries `own_top1`). Read this before any Δ.",
            "conditions": "Per record: `{id, target, variants, baseline_logp}` — the untouched read each delta is against, of the target's spellings with and without a leading space together.",
            "aggregates": "`{mean_delta, median_delta}` per layer across records."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="What `intervene/ablate-layers` produces: one item per (record, layer), the drop in the target's log-probability "
        "when that layer's sub-layer outputs are zeroed. The baseline each delta is measured against is on the "
        "header, per record, so a delta is never read without the number it is a difference from; a layer the "
        "answer runs through shows as a large negative delta.",
)
