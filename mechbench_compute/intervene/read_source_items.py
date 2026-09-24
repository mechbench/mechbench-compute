from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.intervene.spec_error import SpecError


def read_source_items(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    from mechbench_compute.lexicon import kinds as K

    if not isinstance(source, Mapping):
        raise SpecError("`source` must be a collection of activations/vector")
    ik = K.item_kind_of(source)
    if ik == "activations/vector":
        return list(K.items_of(source))
    if ik == "intervene/readout":
        out: list[Mapping[str, Any]] = []
        for item in K.items_of(source):
            caps = item.get("captures")
            if isinstance(caps, Mapping) and K.item_kind_of(caps) == "activations/vector":
                out.extend(K.items_of(caps))
        if out:
            return out
        raise SpecError("`source` is a readout collection with no captures")
    raise SpecError("`source` must be a collection of activations/vector "
                    "or a capture readout")
