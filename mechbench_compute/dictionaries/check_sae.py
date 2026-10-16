from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.dictionaries.constants import ACTIVATIONS


def check_sae(header: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    derivation = header.get("derivation")
    if derivation == "crosscoder":
        raise ValueError("the dictionary is a 'crosscoder': its encoder reads two models at once, and this "
                         "reads one; read a crosscoder's features from its items (`difference`, "
                         "`frequency`, `vector`, `against_vector`)")
    if derivation != "sae":
        raise ValueError(f"the dictionary is a {derivation!r}; a feature is read and written through "
                         "sparse autoencoders (`sae`), which read and write one space, so far")
    reads = header.get("reads") or []
    if len(reads) != 1:
        raise ValueError(f"the dictionary reads {len(reads)} spaces; a sparse autoencoder reads one")
    activation = dict(header.get("activation") or {})
    if activation.get("fn") not in ACTIVATIONS:
        raise ValueError(f"the dictionary's nonlinearity {activation.get('fn')!r} is not one of "
                         f"{', '.join(ACTIVATIONS)}")
    return dict(reads[0]), activation
