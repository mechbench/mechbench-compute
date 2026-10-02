from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.dictionaries.read_feature import read_feature
from mechbench_compute.dictionaries.resolve_feature import resolve_feature
from mechbench_compute.intervene.operator_refused import OperatorRefused
from mechbench_compute.intervene.spec_error import SpecError

FEATURE_OPS = ("add", "project_out", "clamp", "rotate", "patch")


def fill_feature(item: Mapping[str, Any], port: Any) -> dict[str, Any]:
    item = dict(item)
    if item.get("feature") is None:
        return item
    if item.get("f") is not None:
        raise OperatorRefused("OPERATOR_FIELDS", "an operator reads its coordinates through `mask`; "
                              "give the feature's decoder row there as a direction, and leave `feature` "
                              "out", construct="feature")
    if item.get("neurons") is not None:
        raise SpecError("`feature` and `neurons` are one address in two bases, a dictionary's and the "
                        "model's own: name one of them on an item, not both")
    if item.get("parameter") is not None:
        raise SpecError("`feature` acts on an activation; an item that edits a `parameter` takes a "
                        "`direction`")
    if item.get("direction") is not None:
        raise SpecError("`feature` gives the item its direction, the feature's decoder row; leave "
                        "`direction` out")
    op = str(item.get("op", "zero"))
    if op not in FEATURE_OPS:
        raise SpecError(f"`feature` gives the item a direction, and op {op!r} takes none; one of "
                        f"{', '.join(FEATURE_OPS)}")
    try:
        dictionary, index = resolve_feature(item["feature"], port)
        feat = read_feature(dictionary, index)
    except ValueError as e:
        raise SpecError(str(e)) from None
    space = feat["writes"]
    point, layer = str(space["point"]), int(space["layer"])
    if item.get("point") not in (None, point):
        raise SpecError(f"feature {index} writes at {point}; the item names {item['point']!r}: leave "
                        "`point` out")
    if item.get("layers") not in (None, layer, [layer]):
        raise SpecError(f"feature {index} writes at layer {layer}; the item names {item['layers']!r}: "
                        "leave `layers` out")
    vector = feat["vector"]
    direction = {"kind": "direction/vector", "coords": {}, "space": dict(space),
                 "vector": [float(x) for x in vector],
                 "norm": round(float(np.linalg.norm(vector.astype(np.float64))), 6), "unit": False,
                 "derivation": {"method": "dictionary/feature", "index": index,
                                "model": space.get("model")}}
    return {**item, "point": point, "layers": [layer], "direction": direction,
            "feature": {"dictionary": dictionary, "index": index}}
