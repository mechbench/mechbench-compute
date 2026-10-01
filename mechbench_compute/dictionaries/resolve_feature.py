from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon.kinds import item_kind_of


def resolve_feature(feature: Any, port: Any) -> tuple[Any, int]:
    if not isinstance(feature, Mapping) or not isinstance(feature.get("index"), int) \
            or isinstance(feature.get("index"), bool):
        raise ValueError('`feature` is `{"index": 3}`: the feature\'s index in the dictionary on the '
                         "`dictionary` port")
    if "dictionary" in feature:
        raise ValueError("`feature.dictionary` is gone: bind the dictionary to the node's `dictionary` "
                         'port (`{"$ref": …}` there, or a `dictionary/load` node) and give `feature` '
                         "its `index` alone")
    return check_dictionary_port(port), int(feature["index"])


def check_dictionary_port(port: Any) -> Any:
    if port is None:
        raise ValueError("a feature needs its dictionary: bind a `direction/dictionary` to the node's "
                         "`dictionary` port")
    if not isinstance(port, Mapping) or item_kind_of(port) != "direction/dictionary":
        raise ValueError("the `dictionary` port takes a `direction/dictionary`, from `dictionary/load`")
    return port
