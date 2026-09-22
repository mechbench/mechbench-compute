"""The tensor store: a collection whose items live in shards, not in
its JSON.

Every object on the bench is JSON, and an `activations/vector`
collection caps at a few million floats — about 1,700 vectors of Gemma
E4B width. Serious work wants activations for 10⁵–10⁶ tokens at a
layer, to train a probe, a lens or a dictionary. That is a different
object: the collection's header stays a small JSON object with
`storage: "tensor"` and a `shards` list, and the rows live beside it
as safetensors files — content-addressed raw objects under
`<label>/shards/`, uploaded and downloaded by the same streaming paths
a checkpoint's files use.

A shard holds `vector` (`[rows, d]`), one tensor per numeric per-item
field (`surprisal`, `position`, …), and the non-numeric per-item fields
(`id`, `token`, `coords`, `space`) as a JSON table in the safetensors
header. Read back, a shard yields items in exactly the shape the JSON
collection's items have, one shard in memory at a time, so a reader
that iterates streams and a reader that builds a matrix chooses to.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

STORAGE = "tensor"
SHARDS_DIR = "shards"
#: A shard stays under the API's single-object ceiling with room for
#: its header: 48 MiB of rows.
MAX_SHARD_BYTES = 48 << 20
_COMPLETE_MARK = ".complete"

#: The per-item fields a vector item carries beside its vector, and
#: whether each is a number per item (a tensor column) or not (the
#: JSON table).
_VECTOR_KEY = "vector"


def is_tensor(collection: Any) -> bool:
    return isinstance(collection, Mapping) and collection.get("storage") == STORAGE


# --- writing ------------------------------------------------------------------------


def _split_item(item: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    """(vector, numeric fields, other fields) — numeric coords become
    columns too, under `coords.<name>`."""
    vec = np.asarray(item[_VECTOR_KEY], dtype=np.float32).reshape(-1)
    numeric: dict[str, float] = {}
    other: dict[str, Any] = {}
    for k, v in item.items():
        if k == _VECTOR_KEY:
            continue
        if k == "coords" and isinstance(v, Mapping):
            rest = {}
            for ck, cv in v.items():
                if isinstance(cv, (int, float)) and not isinstance(cv, bool):
                    numeric[f"coords.{ck}"] = float(cv)
                else:
                    rest[ck] = cv
            other["coords"] = rest
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric[k] = float(v)
        else:
            other[k] = v
    return vec, numeric, other


def _raw_numeric(item: Mapping[str, Any], key: str) -> Any:
    """The value as the item carried it, for `coords.<name>` or a top-level field."""
    if key.startswith("coords."):
        return (item.get("coords") or {}).get(key[len("coords."):])
    return item.get(key)


class ShardWriter:
    """Accumulates vector items and writes a shard whenever the rows
    would exceed `max_bytes`; `close()` writes the last one and returns
    the manifest entries."""

    def __init__(self, out_dir: str | Path, *, max_bytes: int = MAX_SHARD_BYTES,
                 max_rows: int | None = None) -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.max_rows = max_rows
        self._ints: dict[str, bool] = {}
        self._vectors: list[np.ndarray] = []
        self._numeric: dict[str, list[float]] = {}
        self._table: list[dict[str, Any]] = []
        self._bytes = 0
        self._d: int | None = None
        self.shards: list[dict[str, Any]] = []
        self.n_items = 0

    def add(self, item: Mapping[str, Any]) -> None:
        vec, numeric, other = _split_item(item)
        if self._d is None:
            self._d = int(vec.size)
        elif int(vec.size) != self._d:
            raise ValueError(
                f"a tensor collection holds one width: {self._d}, then {vec.size}")
        for k in set(self._numeric) | set(numeric):
            self._numeric.setdefault(k, [float("nan")] * len(self._table))
        for k in self._numeric:
            self._numeric[k].append(numeric.get(k, float("nan")))
        for k, v in numeric.items():
            # A column is integer until a value says otherwise.
            self._ints[k] = self._ints.get(k, True) and float(v).is_integer() and isinstance(
                _raw_numeric(item, k), int)
        self._vectors.append(vec)
        self._table.append(other)
        self._bytes += vec.nbytes + 8 * len(numeric)
        self.n_items += 1
        if self._bytes >= self.max_bytes or (self.max_rows and len(self._vectors) >= self.max_rows):
            self._flush()

    def _flush(self) -> None:
        if not self._vectors:
            return
        from safetensors.numpy import save_file

        k = len(self.shards)
        name = f"shard-{k:04d}.safetensors"
        tensors = {_VECTOR_KEY: np.stack(self._vectors).astype(np.float32)}
        for col, vals in self._numeric.items():
            # float64: a per-item number round-trips exactly, and it is
            # eight bytes beside a row of thousands.
            tensors[col] = np.asarray(vals, dtype=np.float64)
        path = self.out / name
        ints = sorted(k for k, is_int in self._ints.items() if is_int and k in tensors)
        save_file(tensors, str(path), metadata={
            "table": json.dumps(self._table, separators=(",", ":")),
            "ints": json.dumps(ints)})
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.shards.append({"name": name, "rows": len(self._vectors),
                            "size": path.stat().st_size, "sha256": h.hexdigest()})
        self._vectors, self._table, self._bytes = [], [], 0
        self._numeric = {k: [] for k in self._numeric}

    def close(self) -> dict[str, Any]:
        """What the collection's header records: the shards, the width,
        the row count, and where the files are until they are uploaded."""
        self._flush()
        return {"storage": STORAGE, "shards": list(self.shards), "n_items": self.n_items,
                "d": self._d, "_shard_dir": str(self.out)}


def collection(item_kind: str, writer_header: Mapping[str, Any], **header: Any) -> dict[str, Any]:
    """A tensor collection: the JSON header with no items, the shards
    named. `lexicon.kinds.collection`'s twin for rows that live apart."""
    from mechbench_compute.lexicon import kinds as K

    out = K.collection(item_kind, [], **header)
    out.update({k: v for k, v in writer_header.items()})
    return out


# --- reading ------------------------------------------------------------------------


class ShardedItems(Sequence[dict[str, Any]]):
    """The items of a tensor collection, read one shard at a time. A
    Sequence, so `len`, indexing and iteration all work; iteration is
    the streaming path (one shard in memory), indexing loads the shard
    the index falls in and keeps only that one."""

    def __init__(self, shard_paths: Sequence[Path], rows: Sequence[int]) -> None:
        self._paths = [Path(p) for p in shard_paths]
        self._rows = [int(r) for r in rows]
        self._starts = np.cumsum([0, *self._rows])
        self._loaded: tuple[int, list[dict[str, Any]]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as f:
            meta = f.metadata() or {}
            table = json.loads(meta.get("table", "[]"))
            ints = set(json.loads(meta.get("ints", "[]")))
            vectors = f.get_tensor(_VECTOR_KEY)
            numeric = {k: f.get_tensor(k) for k in f.keys() if k != _VECTOR_KEY}
        items = []
        for i, other in enumerate(table):
            item = dict(other)
            coords = dict(item.get("coords") or {})
            for k, col in numeric.items():
                v = float(col[i])
                if v != v:  # NaN: this item had no such field
                    continue
                if k in ints:
                    v = int(v)
                if k.startswith("coords."):
                    coords[k[len("coords."):]] = v
                else:
                    item[k] = v
            if coords or "coords" in item:
                item["coords"] = coords
            item[_VECTOR_KEY] = vectors[i]
            items.append(item)
        return items

    def shard(self, k: int) -> list[dict[str, Any]]:
        if self._loaded is None or self._loaded[0] != k:
            self._loaded = (k, self._read(self._paths[k]))
        return self._loaded[1]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for k in range(len(self._paths)):
            yield from self.shard(k)

    def __getitem__(self, i):  # type: ignore[override]
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        k = int(np.searchsorted(self._starts, i, side="right") - 1)
        return self.shard(k)[i - int(self._starts[k])]

    def shards(self) -> Iterator[list[dict[str, Any]]]:
        """Shard by shard, for a reader that accumulates."""
        for k in range(len(self._paths)):
            yield self.shard(k)


def items_of(collection: Mapping[str, Any]) -> ShardedItems:
    """The items of a materialized tensor collection; refuses one whose
    shards are not on disk yet, by name — the executor materializes a
    `$ref` before its consumer runs."""
    where = collection.get("_shard_dir")
    if not where:
        raise ValueError(
            "this tensor collection's shards are not on this machine: it must be "
            "materialized (the executor does this for a $ref) before its items are read")
    root = Path(where)
    shards = collection.get("shards") or []
    return ShardedItems([root / s["name"] for s in shards], [s["rows"] for s in shards])


def materialize(collection: Mapping[str, Any], label: str,
                fetch_file: Callable[[str], Any], cache_root: str | Path,
                on_bytes: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Fetch a tensor collection's shards into the cache, verified, and
    return the collection with `_shard_dir` set — the same discipline a
    checkpoint's files follow: keyed by the shards' hashes, verified
    file by file, marked complete only at the end."""
    shards = list(collection.get("shards") or [])
    key = hashlib.sha256(json.dumps([s.get("sha256") for s in shards]).encode()).hexdigest()[:24]
    target = Path(cache_root) / key
    mark = target / _COMPLETE_MARK
    if not mark.exists():
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        total = sum(int(s.get("size", 0)) for s in shards)
        done = 0
        for s in shards:
            name, want = str(s["name"]), str(s["sha256"])
            h = hashlib.sha256()
            with open(target / name, "wb") as f:
                data = fetch_file(f"{label}/{SHARDS_DIR}/{name}")
                for chunk in ([data] if isinstance(data, (bytes, bytearray)) else data):
                    h.update(chunk)
                    f.write(chunk)
                    done += len(chunk)
                    if on_bytes is not None:
                        on_bytes(done, total)
            if h.hexdigest() != want:
                shutil.rmtree(target)
                raise ValueError(
                    f"shard {name!r} of {label!r} arrived with the wrong hash — "
                    "refusing a corrupt materialization")
        mark.touch()
    else:
        mark.touch()
    return {**dict(collection), "_shard_dir": str(target)}


def upload(collection: Mapping[str, Any], label: str, put_file: Callable[[str, Path], Any],
           have: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Put a tensor collection's shards under `<label>/shards/`, skipping
    any already stored with the same hash, and return the collection as
    it is emitted: no local path, the shards as recorded."""
    where = collection.get("_shard_dir")
    if not where:
        raise ValueError("nothing to upload: the collection names no local shard dir")
    root = Path(where)
    have = have or {}
    for s in collection.get("shards") or []:
        name = str(s["name"])
        if have.get(f"{SHARDS_DIR}/{name}") == s.get("sha256"):
            continue
        put_file(f"{label}/{SHARDS_DIR}/{name}", root / name)
    return {k: v for k, v in collection.items() if k != "_shard_dir"}
