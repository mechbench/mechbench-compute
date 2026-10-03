from __future__ import annotations

from typing import Any


class RecordRefused(ValueError):
    def __init__(self, code: str, record: Any, message: str) -> None:
        super().__init__(f"{code}: record {record!r} {message}")
        self.code = code
        self.record = record

    @property
    def issue(self) -> dict[str, Any]:
        return {"code": self.code, "record": self.record,
                "message": str(self).removeprefix(f"{self.code}: ")}
