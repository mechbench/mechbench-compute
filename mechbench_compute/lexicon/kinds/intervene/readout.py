from __future__ import annotations

from mechbench_compute.lexicon._base import Kind, Notable
from mechbench_compute.lexicon.values import COORDS, F, ID

KIND = Kind(
    "intervene/readout",
    "What one record's forward pass read out under one strength of an intervention: a next-token distribution. A capture readout is instead an `activations/vector` collection, `factor` on each vector.",
    extends="logits/distribution",
    fields={"id": ID, "coords": COORDS,
            "factor": F("number", "The sweep factor; 0 is the control. For a steer sweep, the alpha."),
            "cell": F("string", "Which cell of a multi-axis sweep this row is (`layer=3`, `factor=2/layer=3`, `control`) — absent when the sweep varies strength alone, whose cell is its `factor`."),
            "position": F("integer", "Before 0.110.0, for a capture: the position read."),
            "captures": F("object", "Before 0.110.0, for a capture: a collection of `activations/vector`, one per hook point. A capture readout is now itself that collection.")},
    required=("id", "factor"),
    key=("id", "factor", "cell"),
    header={"spec": "The intervention items as run, with objects replaced by their provenance; an operator's `f` in its canonical form, its `mask` by the directions' provenance.",
            "sweep": "The sweep as run: each axis (`strength`, and any of `layers`, `heads`, `positions`, `neurons`) and the values it took, `strength` including the 0 of an added control.",
            "readout": "`decision` or `capture`.",
            "layer": "For a steer sweep: the injection layer.",
            "direction": "For a steer sweep: the axis and values, norm and counts of the direction built.",
            "softcap": "For a decision readout on a model whose final logits pass through a softcap, `c·tanh(x/c)`, the cap `c`; each tracked answer then also carries `logit`, `precap_logit` and `saturated`."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="A decision readout carries `entropy_bits`, `top` and `tracked`; a capture readout carries `position` and `captures`, "
        "and a capture readout wires into another intervention's `source`.",
    notable=Notable("tracked.*.p", ("id", "factor", "cell"), "total-variation", 0.1,
                    "at factor {factor} on {id} '{name}' {change}", control={"factor": 0}),
)
