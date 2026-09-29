from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, NamedTuple

ERROR = "error"

WARNING = "warning"

SUBJECT = re.compile(r"[.:#]")


class Finding(NamedTuple):
    code: str
    at: str
    message: str
    severity: str = ERROR

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "at": self.at, "message": self.message, "severity": self.severity}


def read_subject(at: str) -> str:
    return SUBJECT.split(at, 1)[0]


def select_findings(findings: Iterable[Finding], codes: Iterable[str], *, subject: str | None = None,
                    at: str | None = None,
                    allowed: dict[str, frozenset[str]] | None = None) -> list[Finding]:
    wanted = frozenset(codes)
    allowed = allowed or {}
    return [f for f in findings
            if f.code in wanted
            and (subject is None or read_subject(f.at) == subject)
            and (at is None or f.at == at)
            and f.code not in allowed.get(read_subject(f.at), frozenset())]


def format_findings(findings: Iterable[Finding]) -> str:
    return "\n".join(f"{f.code} at {f.at}: {f.message}" for f in findings)


def has_errors(findings: Iterable[Finding]) -> bool:
    return any(f.severity == ERROR for f in findings)
