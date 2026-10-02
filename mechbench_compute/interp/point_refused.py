from __future__ import annotations


class PointRefused(ValueError):
    def __init__(self, code: str, message: str, point: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.point = point

    @property
    def issue(self) -> dict[str, str]:
        return {"code": self.code, "point": self.point,
                "message": str(self).removeprefix(f"{self.code}: ")}
