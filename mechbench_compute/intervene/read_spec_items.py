from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.intervene.spec_error import SpecError


def read_spec_items(inline: Any, inputs: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The spec items a node was given: its inline list when present,
    else the `items` of an `intervene/spec` object on its `intervention`
    port; [] when neither."""
    if inline:
        return [dict(it) for it in inline]
    port = (inputs or {}).get("intervention")
    if port is None:
        return []
    if not isinstance(port, Mapping) or not isinstance(port.get("items"), list):
        raise SpecError("the `intervention` port takes an intervene/spec: "
                        "an object with `items`, the spec items")
    return [dict(it) for it in port["items"]]
