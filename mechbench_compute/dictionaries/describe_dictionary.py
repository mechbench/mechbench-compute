from __future__ import annotations

from typing import Any

from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.resume import content_hash


def describe_dictionary(dictionary: Any) -> dict[str, Any]:
    header = read_header(dictionary)
    return {"kind": "direction/dictionary", "hash": content_hash(dictionary),
            **{k: header.get(k) for k in ("derivation", "reads", "width", "source")}}
