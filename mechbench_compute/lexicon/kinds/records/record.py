from __future__ import annotations

from mechbench_compute.lexicon._base import Kind, Metric
from mechbench_compute.lexicon.values import COORDS, ID

KIND = Kind(
    "records/record",
    "The root record: an id, the coordinates it belongs to, and whatever fields the op that made it wrote.",
    fields={"id": ID, "coords": COORDS},
    required=("id",),
    key=("id",),
    header={"segments": "When the collection was made by `records/union`: the ports it came from and how many records each contributed."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="The root of the lattice: a condition, a pair, a document, a vector and a grid extend it, so a port "
        "typed `records/record` takes any of them. The ops that reshape and summarise records — `derive`, "
        "`filter`, `sort`, `join`, `group`, `union`, `unnest`, `tabulate`, `plot` — take any "
        "collection at all, whatever its item kind, since every item has an id and its fields. Fields beyond "
        "`id` and `coords` are whatever the producing op wrote; a consumer that needs one under another name "
        "gets it through `records/derive`, never through a parameter. When a model reads a record, its "
        "prompt is the first of `user`, `prompt` and `text` that it has. `user` is wrapped in the model's "
        "chat template, with the record's `system` as the system turn; `prompt` and `text` go in raw, as "
        "written. A record's `template` field overrides this: `\"chat\"` (or `true`) wraps any of the three, `\"raw\"` "
        "(or `false`) wraps none. A `prefill` is appended after the prompt either way.",
    metrics=(
        Metric("hamming", "distance", True,
               "How many coordinate axes two records differ on; an axis one of them lacks counts as a difference. The design's own factor structure, as a distance."),
    ),
)
