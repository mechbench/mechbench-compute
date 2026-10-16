from __future__ import annotations

from mechbench_compute.lexicon._base import Draw, Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "direction/dictionary",
    "A sparse dictionary over a model's activations: one feature per item, each with its encoder and decoder weights, and the dictionary's derivation, source and shape on the header.",
    doc="Written by `dictionary/load` and `dictionary/train`. A dictionary is a basis, not a set: its encoder reads activations at the "
        "spaces in `reads` and gives each feature a value, and its decoder writes the sum of each feature's "
        "decoder row times that value, plus `b_dec`, back into the spaces in `writes`. How it was made is "
        "`derivation`, and it is the only difference between the methods: a sparse autoencoder (`sae`) reads "
        "and writes the same space; a transcoder reads one point and writes another (before the MLP and after "
        "it); a crosscoder reads and writes several layers, or one layer of several models, at once. Every "
        "item is one feature: `vector` is its decoder row, the direction it writes along, and `encoder` the "
        "column of the encoder that reads it. The weights live in the tensor store as shards beside the object, "
        "so a dictionary of any width is one object.",
    extends="records/record",
    fields={
        "index": F("integer", "The feature's index in the dictionary, from 0."),
        "vector": F("array", "The feature's decoder row: the direction it writes along, in the space it writes, "
                    "at the scale the dictionary trained it.", items={"type": "number"}),
        "encoder": F("array", "The feature's encoder column: the direction it reads along, in the space it reads.",
                     items={"type": "number"}),
        "norm": F("number", "The decoder row's length."),
        "b_enc": F("number", "The feature's encoder bias."),
        "threshold": F("number", "Under `jumprelu`: the pre-activation the feature must exceed to fire."),
        "against_vector": F("array", "In a crosscoder: the feature's decoder row in the second model, `against`; "
                            "`vector` is the row in `model`.", items={"type": "number"}),
        "against_encoder": F("array", "In a crosscoder: the encoder block that reads the second model.",
                             items={"type": "number"}),
        "against_norm": F("number", "In a crosscoder: the length of `against_vector`."),
        "difference": F("number", "In a crosscoder: the two decoder rows' lengths in the space the trainer "
                        "saw, `model`'s minus `against`'s over their sum: +1 only in `model`, −1 only in "
                        "`against`, 0 shared."),
        "frequency": F("number", "In a trained crosscoder: the share of the held-out tokens the feature fires on."),
        "mean_activation": F("number", "In a trained crosscoder: the feature's mean activation on the held-out "
                             "tokens it fires on."),
        "label": F("string", "What the feature responds to, in a few words, when something has described it."),
        "examples": F("array", "The records and positions it fires most on, when something has collected them.",
                      items={"type": "object"}),
    },
    required=("id", "index", "vector", "encoder", "norm", "b_enc"),
    key=("index",),
    header={
        "derivation": "How the dictionary was made: `sae` (reads and writes one space), `transcoder` (reads one "
                      "point, writes another) or `crosscoder` (reads and writes several layers or models).",
        "reads": "The spaces the encoder reads, each `{model, layer, point, head, d}`.",
        "writes": "The spaces the decoder writes, in the same form.",
        "model": "The checkpoint it was trained on: `{id, architecture}`.",
        "source": "Where it came from: `{hub: {repo, path, revision, commit, files}}`, each file with its sha256.",
        "width": "How many features.",
        "d_in": "The width of what the encoder reads.",
        "d_out": "The width of what the decoder writes.",
        "activation": "The encoder's nonlinearity: `{fn: jumprelu}` (each feature's own threshold), `{fn: relu}`, "
                      "or `{fn: topk, k}`; a trained dictionary adds the `training` that made it (`batchtopk` "
                      "with its `k`, or `jumprelu` with its `l0`).",
        "b_dec": "The decoder bias, `d_out` numbers.",
        "against_b_dec": "In a crosscoder: the decoder bias in the second model, `against`.",
        "published": "What the source says of itself: its own `l0`, measured on its training data.",
        "trained": "For a dictionary `dictionary/train` made, in place of `source` and `published`: the corpus "
                   "(each source's dataset, revision, commit and documents read), the hashes of the first "
                   "buffer's and the held-out windows' tokens, the tokens and steps trained, the seed, the "
                   "scale folded into the weights, the hyperparameters, the training curve and the compute "
                   "version.",
        "fidelity": "For a dictionary `dictionary/train` made: how well it reads the model on held-out tokens "
                    "never trained on: `variance_explained`, `l0`, `mse`, `dead` and `dense` (the shares of "
                    "features that never fire and that fire on more than `dense_rate` of the tokens), and "
                    "`loss`, the model's next-token loss `clean`, with the reconstruction `spliced` in at the "
                    "point and with the point `zero`, their differences and the share `recovered`.",
    },
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    speak="feature {index} of a {header.width}-wide {header.derivation} at {header.reads[0].point} layer "
          "{header.reads[0].layer}: decoder norm {round(norm, 3)}",
    draw=Draw(mark="point", encoding={"x": "index", "y": "norm"}),
)
