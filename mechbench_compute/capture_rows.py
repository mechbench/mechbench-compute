from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

STORAGE_CHOICES = ("auto", "json", "tensor")

STORAGE_DOC = (
    "How a stored result holds its rows. `\"json\"`: in the object, under the "
    "op's ceiling, refused past it. `\"tensor\"`: as safetensors shards beside "
    "the object, with no ceiling. `\"auto\"`: json under the ceiling, tensor "
    "above. A result the run does not store (under `keep: \"outputs\"`, a node "
    "that is not an output) has no ceiling: it is held as rows in memory "
    "whatever its size, unless `\"tensor\"` is named.")

STORAGE_SECTION = """\
### Storage

A stored result is one object, and a stored object may not exceed 64 MiB,
so `storage: "json"` refuses a capture past the op's ceiling before the
model runs, naming `storage: "tensor"` as the lever. `"tensor"` writes the
rows as safetensors shards beside the object (`<label>/shards/…`), the
object itself the header with the shards named (`storage: "tensor"`,
`shards`, `n_items`, `d`, and `items` empty), with no ceiling. `"auto"`,
the default, is json under the ceiling and tensor above it, so a result
under it is the json it always was. The records operations read a tensor
collection as they read a json one, and a `records/derive` that drops
`vector` yields a json collection.

A result the run will not store has no ceiling at all: under
`keep: "outputs"` a node that is not an output is held in memory as rows
whatever its size, bounded only by the machine, and read by the nodes
downstream as any other result.
"""


def read_storage(params: Mapping[str, Any]) -> str:
    storage = str(params.get("storage", "auto"))
    if storage not in STORAGE_CHOICES:
        raise ValueError(f"storage is 'auto', 'json' or 'tensor', not {storage!r}")
    return storage


class CaptureRows:
    def __init__(self, storage: str, *, ceiling: int, stored: bool = True,
                 levers: str = "capture fewer layers or records") -> None:
        if storage not in STORAGE_CHOICES:
            raise ValueError(f"storage is 'auto', 'json' or 'tensor', not {storage!r}")
        self.storage = storage
        self.ceiling = int(ceiling)
        self.stored = bool(stored)
        self.levers = levers
        self.rows: list[dict[str, Any]] = []
        self.writer: Any = None
        if storage == "tensor":
            self.open_writer()

    def plan(self, total: int, what: str | None = None) -> None:
        if not self.stored or self.writer is not None or total <= self.ceiling:
            return
        if self.storage == "json":
            said = f"{what} = {total}" if what else str(total)
            raise ValueError(
                f"{said} floats exceeds the {self.ceiling} cap on a stored json "
                "result (a result object may not exceed 64 MiB) — set "
                "`storage: \"tensor\"` to write the rows as shards beside the "
                f"object, with no cap; or {self.levers}; or leave the result "
                "unstored (under `keep: \"outputs\"`, a node that is not an output)")
        self.open_writer()

    def open_writer(self) -> None:
        import tempfile

        from mechbench_compute import tensors

        self.writer = tensors.ShardWriter(tempfile.mkdtemp(prefix="mechbench-tensor-"))
        held, self.rows = self.rows, []
        self.extend(held)

    def add(self, row: dict[str, Any]) -> None:
        if self.writer is None:
            self.rows.append(row)
            return
        try:
            self.writer.add(row)
        except ValueError as e:
            raise ValueError(
                f"{e} — tensor storage holds rows of one width and one set of "
                "arrays; store this capture as json (`storage: \"json\"`), or "
                "split it so each node's rows share a width") from None

    def extend(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.add(row)

    def __len__(self) -> int:
        return self.writer.n_items if self.writer is not None else len(self.rows)

    def finish(self, item_kind: str, **header: Any) -> dict[str, Any]:
        from mechbench_compute import tensors
        from mechbench_compute.lexicon import kinds as K

        if self.writer is not None:
            return tensors.collection(item_kind, self.writer.close(), **header)
        return K.collection(item_kind, self.rows, **header)
