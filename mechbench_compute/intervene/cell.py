from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class Cell:
    """One cell of a sweep: the strength factor, the spec fields it
    overrides, and the coordinates it puts on every row it produces."""

    def __init__(self, factor: float, overrides: Mapping[str, Any],
                 coords: Mapping[str, Any], label: str | None) -> None:
        self.factor = float(factor)
        self.overrides = dict(overrides)
        #: The axis coordinates beside `factor` — `layer`, `head`, …
        self.coords = dict(coords)
        #: A one-word name for the cell, when the sweep runs axes beyond
        #: strength; None when it does not, so a strength-only sweep's
        #: rows are identified by `factor` exactly as they always were.
        self.label = label

    @property
    def key(self) -> str:
        return self.label or f"{self.factor}"

    @property
    def slug(self) -> str:
        """The cell in an id: `f1` for a strength-only sweep, as it has
        always been; `layer0`, `factor2-layer3` when other axes run."""
        if not self.label:
            return f"f{self.factor:g}"
        return self.label.replace("=", "").replace("/", "-")

    @property
    def axes(self) -> dict[str, Any]:
        """Every coordinate this cell puts on a row, `factor` included."""
        return {"factor": self.factor, **self.coords}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Cell(factor={self.factor}, overrides={self.overrides})"
