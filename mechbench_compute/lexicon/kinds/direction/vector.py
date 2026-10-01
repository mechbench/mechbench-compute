from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "direction/vector",
    "A unit vector in a model's activation space, with its derivation: how it was made, from what, on which model.",
    extends="activations/vector",
    fields={"unit": F("boolean", "Whether the vector is unit length."),
            "derivation": F("object", "`{method, sources, model, axis?, positive?, negative?, center?, exclude?, "
                                      "excluded_share?, …}` — how it was made. `center: true` when the vectors "
                                      "were centred before the fit; `exclude` lists dimensions left out (zero in "
                                      "`vector`), `excluded_share` the part of the squared difference they "
                                      "carried.")},
    required=("space", "vector", "derivation"),
    key=("id", "space"),
    header={"components": "How many principal components the collection holds, when `direction/decompose` made it "
                          "with `k`.",
            "explained": "The share of the variance the components explain together: the sum of each one's "
                         "`derivation.explained`.",
            "layer": "The layer the components were found at.",
            "point": "The point they were found at.",
            "model": "The model's wire form, or null when the vectors came from more than one.",
            "n_items": "How many vectors were decomposed."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="`norm` is the magnitude before normalisation, which some readings use.",
    speak="{derivation.method} direction at {space.point}"
          "{'' if space.layer is None else ' layer ' + str(space.layer)}"
          "{'' if derivation.positive is None else ', from ' + str(derivation.negative) + ' to '"
          " + str(derivation.positive)}"
          "{'' if derivation.center != True else ', on centred vectors'}"
          "{'' if derivation.exclude is None else ', dimensions ' + str(derivation.exclude) + ' left out'}"
          "{'' if derivation.excluded_share is None else ' (they carried '"
          " + str(round(derivation.excluded_share, 3)) + ' of the squared difference)'}"
          "; norm {round(norm, 3)}.",
)
