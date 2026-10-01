from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon.kinds import item_kind_of


def resolve_feature(feature: Any, port: Any) -> tuple[Any, int]:
    if not isinstance(feature, Mapping) or not isinstance(feature.get("index"), int) \
            or isinstance(feature.get("index"), bool):
        raise ValueError('`feature` is `{"dictionary": {"$ref": …}, "index": 3}`: a dictionary, or one '
                         "on the `dictionary` port, and the feature's index in it")
    dictionary = feature.get("dictionary")
    if dictionary is None:
        dictionary = port
    if dictionary is None:
        raise ValueError("`feature` names no dictionary: give `feature.dictionary`, or bind a "
                         "`direction/dictionary` to the `dictionary` port")
    if not isinstance(dictionary, Mapping) or item_kind_of(dictionary) != "direction/dictionary":
        raise ValueError("`feature.dictionary` is a `direction/dictionary`, from `dictionary/load`")
    return dictionary, int(feature["index"])
