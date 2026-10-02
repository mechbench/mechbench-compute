from __future__ import annotations

from mechbench_compute.intervene.spec_error import SpecError


class OperatorRefused(SpecError):
    def __init__(self, code: str, message: str, construct: str | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.construct = construct

    @property
    def issue(self) -> dict[str, str | None]:
        return {"code": self.code, "construct": self.construct,
                "message": str(self).removeprefix(f"{self.code}: ")}
