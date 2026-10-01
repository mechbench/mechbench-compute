from __future__ import annotations

from typing import Any

from mechbench_compute.dictionaries.resolve_feature import check_dictionary_port


def resolve_features(features: Any, port: Any) -> tuple[Any, list[int]]:
    if not isinstance(features, list) or not features \
            or any(not isinstance(i, int) or isinstance(i, bool) for i in features):
        raise ValueError("`features` is a non-empty list of feature indices, `[3, 71, 2048]`")
    if len(set(features)) != len(features):
        raise ValueError("`features` names a feature twice")
    return check_dictionary_port(port), [int(i) for i in features]
