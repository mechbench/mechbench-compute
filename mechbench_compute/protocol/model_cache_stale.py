from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ModelCacheStale(ValueError):
    code = "MODEL_CACHE_STALE"

    def __init__(self, message: str, *, loaded: Mapping[str, Any], asked: Mapping[str, Any]) -> None:
        super().__init__(f"{self.code}: {message}")
        self.loaded = dict(loaded)
        self.asked = dict(asked)

    @property
    def issue(self) -> dict[str, Any]:
        return {"code": self.code, "loaded": self.loaded, "asked": self.asked,
                "message": str(self).removeprefix(f"{self.code}: ")}
