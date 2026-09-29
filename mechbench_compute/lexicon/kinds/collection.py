from __future__ import annotations

from mechbench_compute.lexicon._base import COLLECTION, Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    COLLECTION,
    "The one container: items of one kind, identified by the kind's key, sorted when stored by its `order_by` and then by that key, with the header fields the kind declares.",
    fields={"item_kind": F("string", "The kind of every item."),
            "key": F("array", "The item fields that identify an item.", items={"type": "string"}),
            "items": F("array", "The items.", items={"type": "object"}),
            "storage": F("string", "`\"tensor\"` when the items live in shards beside the object rather than in `items`."),
            "shards": F("array", "Under tensor storage: `{name, rows, size, sha256}` per shard, in order.", items={"type": "object"}),
            "n_items": F("integer", "Under tensor storage: how many rows the shards hold."),
            "d": F("integer", "Under tensor storage: the rows' width."),
            "order_by": F("array", "The item fields, in turn, that order the items before the key does: "
                          "`[\"rank\"]` after a sort.", items={"type": "string"})},
    required=("item_kind", "key", "items"),
    doc="Every plural result is this one shape. The item kind declares the `key` — the fields that identify an "
        "item — and its header, the collection-level facts that ride with the items (a model, a metric, a "
        "pass rate). Items are sorted by key before the object is hashed, so the same items in any order are "
        "the same object; order that matters is a property of the items, never of the container: of the key "
        "(`step`, `layer`), or of fields named by `order_by`, which orders the items before the key does — a "
        "sort writes each record's place into `rank` and declares `order_by: [\"rank\"]`. A port declared as `collection` takes any collection at all. A collection too large for "
        "one object — a per-token capture of a hundred thousand tokens — keeps its header here with "
        "`storage: \"tensor\"` and `items` empty, and its rows in safetensors shards stored beside it under "
        "`<label>/shards/`; a reader takes them one shard at a time.",
)
