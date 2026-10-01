from __future__ import annotations

from mechbench_compute.lexicon._base import Draw, Kind
from mechbench_compute.lexicon.values import COORDS, ID, TOKEN, F

KIND = Kind(
    "activations/feature",
    "One dictionary feature's activation at one position of one record: the activations read in the dictionary's basis.",
    doc="Written by `dictionary/encode`, one item per (record, position, feature) where the feature fired, so a "
        "readout is sparse: a feature that did not fire at a position has no item there. `coords` are the "
        "record's, and `coords.model` says which reading the item belongs to — `base`, the node's model, or "
        "`adapted`, the same model with the adapter on the node's `adapted` port — so the two readings group "
        "and join like any two conditions. The header says how well the dictionary reconstructs the "
        "activations it was applied to (`fidelity`, and `adapted_fidelity` beside it when there was a second "
        "reading): a feature's change under an adapter means what it seems to only while the dictionary "
        "still reconstructs the adapted activations about as well as the base ones.",
    extends="records/record",
    fields={"id": ID, "coords": COORDS,
            "position": F("integer", "The position in the record's tokens, from 0."),
            "token": TOKEN,
            "feature": F("integer", "The feature's index in the dictionary."),
            "value": F("number", "The feature's activation there.")},
    required=("id", "position", "feature", "value"),
    key=("id", "coords", "position", "feature"),
    header={
        "dictionary": "The dictionary read through: its derivation, the spaces it reads, its width and source.",
        "model": "The node's model, in its wire form.",
        "point": "The point and layer read: `{point, layer}`.",
        "positions": "Which positions were read: `all` or `last`, and whether the beginning-of-sequence token "
                     "was left out.",
        "features": "When the readout kept only some features, their indices; the fidelity is over all of them.",
        "fidelity": "How well the dictionary reconstructs the base reading's activations: `variance_explained` "
                    "(1 − the squared error over the activations' total variance about their mean), `l0` (the "
                    "mean number of features firing per position), `mse` (the squared error per number), "
                    "`positions` (how many were read) and `inactive` (features that never fired on them), "
                    "with the dictionary's own published `l0` as `published_l0`.",
        "adapted_fidelity": "The same, on the adapted reading's activations, when there was one.",
        "fidelity_drop": "`fidelity.variance_explained` minus `adapted_fidelity.variance_explained`: how much "
                         "less of the adapted activations the dictionary explains.",
    },
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    speak="feature {feature} at {token.text} (position {position} of {id}): {round(value, 3)}. The dictionary "
          "explains {round(header.fidelity.variance_explained, 3)} of the variance here at L0 "
          "{round(header.fidelity.l0, 1)}{'' if header.adapted_fidelity is None else '; under the adapter, ' + "
          "str(round(header.adapted_fidelity.variance_explained, 3)) + ' at L0 ' + "
          "str(round(header.adapted_fidelity.l0, 1))}.",
    draw=Draw(mark="point", encoding={"x": "feature", "y": "value"}),
)
