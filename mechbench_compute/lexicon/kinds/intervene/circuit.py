from __future__ import annotations

from mechbench_compute.lexicon._base import Kind, Metric, P
from mechbench_compute.lexicon.values import COORDS, ID, F

COMPONENT = {
    "type": "object",
    "properties": {
        "address": {"type": "string",
                    "description": "`L{layer}.{point}[.H{head}]@{position}`, never null: the key `jaccard` and "
                                   "`records/diff` compare."},
        "point": {"type": "string",
                  "description": "The point it sits at: `attn.per_head_out`, `attn_out`, `mlp_out`, `resid_post`, …"},
        "layer": {"type": "integer", "description": "The layer."},
        "head": {"type": "integer", "description": "The head, at a point with a head axis."},
        "position": {"type": ["string", "integer"],
                     "description": "`all`, or a position counted from the end: -1 is the last token."},
        "token": {"type": "string", "description": "The token at that position, for a component from a trace."},
        "effect": {"type": "number", "description": "The source's measure at this component."},
    },
    "required": ["address", "point", "layer", "position", "effect"],
    "additionalProperties": False,
}

LINK = {
    "type": "object",
    "properties": {
        "from": {"type": "string", "description": "The sending component's address."},
        "to": {"type": "string", "description": "The receiving component's address."},
        "effect": {"type": "number", "description": "The effect along the link."},
    },
    "required": ["from", "to", "effect"],
    "additionalProperties": False,
}

KIND = Kind(
    "intervene/circuit",
    "A named set of components a behaviour runs through, with the metric, ablation and source it was found under.",
    doc="Written by `intervene/prune`, or by hand. A component is one spec item in the singular — a point, a "
        "layer, a head where the point has a head axis, a position — with an `address`, "
        "`L{layer}.{point}[.H{head}]@{position}`. Positions are counted from the end, so circuits found on "
        "prompts of different lengths line up. A residual location (`resid_*`) is where information is "
        "carried, not a component: `prune` keeps it as found, and `intervene/ablate-circuit` refuses it. "
        "The `universe` is what the circuit's complement is taken within, copied from the source's axes. "
        "`derivation.kept_share` is the summed effect of the kept components, in the sense of "
        "`derivation.sign`, over the same sum across the universe. A threshold that kept nothing is a circuit "
        "with no components. `links` is absent until a source carries them. Draw one through "
        "`records/unnest field: components` and `heat`.",
    extends="records/record",
    fields={"id": ID, "coords": COORDS,
            "components": F("array", "What the circuit is made of.", items=COMPONENT),
            "links": F("array", "`{from, to, effect}` between components, by address. Absent when the source "
                                "had none.", items=LINK),
            "metric": F("string", "The readout the effect was measured in: `logprob`, `prob`, `logit` or "
                                  "`entropy`."),
            "measure": F("string", "The source's measure that was thresholded: `mean_delta`, `share` or "
                                   "`recovery`."),
            "ablation": F("string", "How a component was removed when the effect was found: `zero` (a heads "
                                    "grid), `patch` (a trace) or `hand`."),
            "universe": F("object", "What the complement is taken within: `{points, layers, n_heads?, "
                                    "positions}`, positions `all` or a list counted from the end."),
            "source": F("object", "`{path, kind, method?, point?}`: the object the circuit was pruned from. "
                                  "`{kind: \"hand\"}` for a circuit written by hand."),
            "task": F("object", "What the circuit is for: `{ids, targets: [{id, target}], n_off_top1?}`, "
                                "copied from the source."),
            "derivation": F("object", "`{method, threshold?, top?, per_layer?, sign, kept, total, "
                                      "kept_share}`; `method` is `prune`, `hand` or `eap`, `total` the "
                                      "number of components in the universe.")},
    required=("id", "components", "metric", "measure", "ablation", "universe", "source", "task", "derivation"),
    key=("id",),
    header={"model": "The model, when the source carried it.",
            "thresholds": "The thresholds, when `prune` emitted one circuit per value."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    speak="{id}: {derivation.kept} of {derivation.total} components ({derivation.method}"
          "{'' if derivation.threshold is None else ' at ' + str(derivation.threshold)}"
          "{'' if derivation.top is None else ', top ' + str(derivation.top)}) carrying "
          "{round(derivation.kept_share, 2)} of the {metric} effect, found under {ablation} ablation"
          "{'' if source.path is None else '; from ' + source.path}.",
    metrics=(
        Metric("jaccard", "similarity", True,
               "The share of components two circuits have in common: |A∩B| / |A∪B| over their addresses, "
               "in [0, 1]. With `weighted`, Σmin(|effect|) / Σmax(|effect|) over the union, a component one "
               "circuit lacks counting as 0 there.",
               options=(P("weighted", "bool", "Weigh each component by the size of its effect.", False),)),
    ),
)
