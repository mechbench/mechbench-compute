from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

FILE_NAME = re.compile(r"[A-Za-z0-9._-]+")


def check_target(target: str) -> None:
    import mechbench_schema as ms

    try:
        ms.parse_path(target)
    except ms.InvalidPathError as e:
        raise ms.InvalidPathError(f"cannot store at {target!r}: {e}") from e


def check_file_name(name: object, *, what: str) -> str:
    text = name if isinstance(name, str) else ""
    if not FILE_NAME.fullmatch(text) or text.startswith("."):
        raise ValueError(
            f"{what} {name!r} is not a plain file name: a stored name is one "
            f"path component of letters, digits, '.', '_' and '-', not starting "
            f"with '.' — refusing it before anything is written")
    return text


def check_relative_path(path: object, *, what: str) -> str:
    text = path if isinstance(path, str) else ""
    if (not text or "\0" in text or text.startswith("/")
            or any(p in ("", ".", "..") for p in text.split("/"))):
        raise ValueError(
            f"{what} {path!r} is not a relative path inside its root: no "
            f"absolute path, no empty, '.' or '..' component")
    return text


def resolve_inside(root: str | os.PathLike[str], relative: str, *, what: str) -> Path:
    base = Path(root).resolve()
    target = (base / relative).resolve()
    if target == base or not target.is_relative_to(base):
        raise ValueError(f"{what} {relative!r} resolves outside {base}")
    return target


def write_inside(root: str | os.PathLike[str], name: str, chunks: Iterable[bytes], *,
                 want: str, on_chunk: Callable[[int], None] | None = None) -> bool:
    dest = resolve_inside(root, check_file_name(name, what="file name"), what="file")
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".part-")
    h = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as f:
            for chunk in chunks:
                h.update(chunk)
                f.write(chunk)
                if on_chunk is not None:
                    on_chunk(len(chunk))
        if h.hexdigest() != want:
            return False
        os.replace(tmp, dest)
        tmp = ""
        return True
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
