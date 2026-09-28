from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn


def raise_expr_error(op: str, error: Any, records: Sequence[Mapping[str, Any]] | None = None) -> NoReturn:
    index = getattr(error, "record", None)
    where = ""
    if index is not None and records is not None and 0 <= index < len(records):
        where = f" in record {records[index].get('id')!r}"
    raise ValueError(f"{op}: {error.detail}{where}: `{error.expr}`") from None
