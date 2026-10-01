from __future__ import annotations

from typing import Any

import numpy as np

from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.dictionaries.check_sae import check_sae
from mechbench_compute.dictionaries.read_dictionary_weights import read_dictionary_weights
from mechbench_compute.lexicon.kinds import items_of


def read_feature(dictionary: Any, index: int) -> dict[str, Any]:
    header = read_header(dictionary)
    space, activation = check_sae(header)
    width = int(header["width"])
    if not 0 <= index < width:
        raise ValueError(f"feature {index} is not in the dictionary: its {width} features are 0 to {width - 1}")
    weights = read_dictionary_weights(dictionary) if activation["fn"] == "topk" else None
    items = items_of(dictionary)
    item = items[index] if index < len(items) and int(items[index]["index"]) == index else None
    if item is None:
        item = next((it for it in items if int(it["index"]) == index), None)
    if item is None:
        raise ValueError(f"the dictionary declares {width} features and does not carry feature {index}")
    writes = list(header.get("writes") or [space])
    return {"index": index, "reads": space, "writes": dict(writes[0]), "activation": activation,
            "encoder": np.asarray(item["encoder"], dtype=np.float32),
            "b_enc": np.float32(item["b_enc"]),
            "threshold": np.float32(item.get("threshold", 0.0)),
            "vector": np.asarray(item["vector"], dtype=np.float32),
            "weights": weights, "header": header}
