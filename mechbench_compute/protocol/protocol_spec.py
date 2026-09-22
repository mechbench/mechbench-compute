from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProtocolSpec:
    kind: str
    prompt: str
    model_id: str
    # Kind-specific spec payload (decision_distribution: conditions list).
    extra: dict[str, Any] | None = None
