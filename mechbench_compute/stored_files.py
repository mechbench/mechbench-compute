from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from mechbench_compute.contained import check_file_name, check_target

STORAGE = "files"
FILES_DIR = "files"
FILE_KEY = "$file"
FILE_KIND = "result_file"
HEADROOM_BYTES = 1 << 20


def is_files(payload: Any) -> bool:
    return isinstance(payload, Mapping) and payload.get("storage") == STORAGE


def is_file_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {FILE_KEY}


def count_bytes(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(count_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_bytes(v) for v in value)
    return 0


def needs_files(payload: Any, limit: int | None = None) -> bool:
    if not isinstance(payload, Mapping) or "storage" in payload:
        return False
    if limit is None:
        from mechbench_compute.bench import MAX_OBJECT_BYTES as limit
    total = count_bytes(payload)
    if total == 0 or total <= limit // 2:
        return False
    from mechbench_schema import dump_canonical

    return len(dump_canonical(payload)) + HEADROOM_BYTES > limit


def offload(payload: Mapping[str, Any], target: str,
            put_file: Callable[[str, str], Any],
            have: Mapping[str, str] | None = None) -> dict[str, Any]:
    check_target(target)
    have = have or {}
    sent: dict[str, dict[str, Any]] = {}

    def send(data: bytes) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        name = f"{digest[:32]}.bin"
        if name in sent:
            return sent[name]
        path = f"{target}/{FILES_DIR}/{name}"
        check_target(path)
        if have.get(f"{FILES_DIR}/{name}") != digest:
            fd, tmp = tempfile.mkstemp(prefix="mechbench-file-")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                put_file(path, tmp)
            finally:
                os.unlink(tmp)
        sent[name] = {FILE_KEY: {"path": path, "bytes": len(data), "sha256": digest}}
        return sent[name]

    def walk(v: Any) -> Any:
        if isinstance(v, (bytes, bytearray)):
            return send(bytes(v))
        if isinstance(v, Mapping):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        return v

    return {**walk(payload), "storage": STORAGE}


def restore(payload: Mapping[str, Any],
            fetch_file: Callable[[str], bytes | Iterable[bytes]],
            on_bytes: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    refs: dict[str, Mapping[str, Any]] = {}

    def find(v: Any) -> None:
        if is_file_ref(v):
            ref = v[FILE_KEY]
            refs[str(ref.get("path"))] = ref
        elif isinstance(v, Mapping):
            for x in v.values():
                find(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                find(x)

    find(payload)
    total = sum(int(r.get("bytes", 0)) for r in refs.values())
    done = 0
    got: dict[str, bytes] = {}
    for path, ref in refs.items():
        check_target(path)
        check_file_name(path.rsplit("/", 1)[-1], what="file")
        data = fetch_file(path)
        chunks = [data] if isinstance(data, (bytes, bytearray)) else data
        h, parts = hashlib.sha256(), []
        for chunk in chunks:
            h.update(chunk)
            parts.append(bytes(chunk))
            done += len(chunk)
            if on_bytes is not None:
                on_bytes(done, total)
        if h.hexdigest() != ref.get("sha256"):
            raise ValueError(
                f"stored file {path!r} arrived with the wrong hash — refusing a "
                "corrupt read")
        got[path] = b"".join(parts)

    def walk(v: Any) -> Any:
        if is_file_ref(v):
            return got[str(v[FILE_KEY].get("path"))]
        if isinstance(v, Mapping):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    return {k: v for k, v in walk(payload).items() if k != "storage"}
