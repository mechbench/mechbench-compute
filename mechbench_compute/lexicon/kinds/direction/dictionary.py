from __future__ import annotations

from mechbench_compute.lexicon._base import Draw, Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "direction/dictionary",
    "A sparse dictionary over a model's activations: one feature per item, each with its encoder and decoder weights, and the dictionary's derivation, source and shape on the header.",
    doc="Written by `dictionary/load`. A dictionary is a basis, not a set: its encoder reads activations at the "
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
                      "or `{fn: topk, k}`.",
        "b_dec": "The decoder bias, `d_out` numbers.",
        "published": "What the source says of itself: its own `l0`, measured on its training data.",
    },
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    speak="feature {index} of a {header.width}-wide {header.derivation} at {header.reads[0].point} layer "
          "{header.reads[0].layer}: decoder norm {round(norm, 3)}",
    draw=Draw(mark="point", encoding={"x": "index", "y": "norm"}),
)
