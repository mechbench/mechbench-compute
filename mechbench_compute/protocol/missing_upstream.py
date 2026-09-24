from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class MissingUpstream(RuntimeError):
    def __init__(self, nid: str, port: str, source: str, why: Mapping[str, Any]):
        self.nid, self.port, self.source = nid, port, source
        super().__init__(
            f"{nid} needs its {port!r} port, and {source} produced nothing: "
            f"{why.get('reason')}. That port's `on_missing` is 'fail' — the "
            f"default. A port declared `skip` passes the absence on; one "
            f"declared `placeholder` runs with an empty collection that says "
            f"it is one.")
