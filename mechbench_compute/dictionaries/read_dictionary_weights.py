from __future__ import annotations

from typing import Any

import numpy as np

from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.dictionaries.check_sae import check_sae
from mechbench_compute.lexicon.kinds import items_of


def read_dictionary_weights(dictionary: Any) -> dict[str, Any]:
    header = read_header(dictionary)
    space, activation = check_sae(header)
    width, d_in = int(header["width"]), int(header["d_in"])
    d_out = int(header.get("d_out") or d_in)
    w_enc = np.zeros((d_in, width), dtype=np.float32)
    w_dec = np.zeros((width, d_out), dtype=np.float32)
    b_enc = np.zeros(width, dtype=np.float32)
    threshold = np.zeros(width, dtype=np.float32)
    seen = np.zeros(width, dtype=bool)
    for item in items_of(dictionary):
        i = int(item["index"])
        w_enc[:, i] = np.asarray(item["encoder"], dtype=np.float32)
        w_dec[i] = np.asarray(item["vector"], dtype=np.float32)
        b_enc[i] = float(item["b_enc"])
        threshold[i] = float(item.get("threshold", 0.0))
        seen[i] = True
    if not seen.all():
        raise ValueError(f"the dictionary declares {width} features and carries {int(seen.sum())}")
    return {"space": space, "activation": activation, "w_enc": w_enc, "b_enc": b_enc,
            "threshold": threshold, "w_dec": w_dec,
            "b_dec": np.asarray(header.get("b_dec") or np.zeros(d_out), dtype=np.float32),
            "header": header}
