from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import CELLS, F, TOKEN, VARIANTS

KIND = Kind(
    "intervene/trace",
    "A pair's causal trace: how much of the clean answer's probability comes back when the clean residual is patched into the corrupted run at each (layer, position).",
    extends="activations/grid",
    fields={"target": TOKEN, "variants": VARIANTS, "metric": F("string", "`logprob`, `prob` or `logit`."),
            "value_a": F("number", "The metric on prompt `a` (clean)."),
            "value_b": F("number", "The metric on prompt `b` (corrupted), the baseline."),
            "cells": CELLS({"recovery": "The change in the metric from the `b` baseline.",
                            "share": "The recovery as a fraction of the `a`−`b` gap; null when the pair has none."},
                           token=True)},
    required=("id", "axes", "measures"),
    key=("id",),
    header={"point": "The residual patched.", "metric": "`logprob` or `prob`.", "layers": "The layers, in row order."},
    doc="Axes `[layer, position]`; measures `recovery` — the change in the metric from the `b` baseline — and `share`, the same as a fraction of the `a`−`b` gap (0 is the corrupt run, 1 the clean one; absent when the pair has no gap); `tokens` are prompt `b`'s. `records/unnest field: cells` makes one record per (layer, position), `{address, point, layer, position, token, recovery, share}`, the point the header's and the position counted from the end, addressed as the circuit's components are.",
)
