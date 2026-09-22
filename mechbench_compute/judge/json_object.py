from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def _json_object(text: str) -> dict[str, Any]:
    m = _JSON_OBJECT.search(text)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}
