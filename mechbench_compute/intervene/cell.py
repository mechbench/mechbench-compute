from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class Cell:
    def __init__(self, factor: float, overrides: Mapping[str, Any],
                 coords: Mapping[str, Any], label: str | None) -> None:
        self.factor = float(factor)
        self.overrides = dict(overrides)
        self.coords = dict(coords)
        self.label = label

    @property
    def key(self) -> str:
        return self.label or f"{self.factor}"

    @property
    def slug(self) -> str:
        if not self.label:
            return f"f{self.factor:g}"
        return self.label.replace("=", "").replace("/", "-")

    @property
    def axes(self) -> dict[str, Any]:
        return {"factor": self.factor, **self.coords}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Cell(factor={self.factor}, overrides={self.overrides})"
