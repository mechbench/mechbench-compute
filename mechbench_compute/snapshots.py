"""Filesystem snapshots as values (task 000358, epic 000334).

A sandbox's state is normally the least reproducible thing in a
system: a directory somebody mutated, whose history is gone. This
makes it a VALUE instead — a content-addressed tree — so that every
tool call is a function

    (snapshot, argv) -> (snapshot', stdout, stderr, exit)

and therefore an ordinary bench item with lineage. You can point at
the filesystem before and after any command, diff them, replay a run
byte-identically, and share a snapshot the way you share a corpus.

The whole scheme rests on one property: **identical content must
produce an identical hash**. So capture normalizes everything that is
not content — mtimes, ownership, the order the OS happened to return
entries in — and keeps only what a later run would need to reproduce
the tree. A snapshot that hashed differently because it was written on
a Tuesday would make the sandbox exactly as untrustworthy as the
directory it replaced.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

KIND = "fs_snapshot"

#: Blobs at or below this size ride inside the snapshot object; larger
#: ones are stored separately and referenced. 64 KiB keeps a snapshot
#: of scripts and small data self-contained — the common case — without
#: letting one large file turn the object into a download.
INLINE_MAX = 64 * 1024

#: Defaults, overridable per capture. A sandbox that can write an
#: unbounded tree is a sandbox that can fill the disk.
MAX_FILES = 10_000
MAX_BYTES = 256 * 1024 * 1024

#: The only mode distinction kept. Real permissions are host detail;
#: whether a thing is executable changes what a later run DOES.
MODE_FILE = 0o644
MODE_EXEC = 0o755


class SnapshotLimit(RuntimeError):
    """A tree exceeded a declared limit. Raised at capture, naming the
    limit and what was found — a sandbox that quietly truncated would
    produce a snapshot that is not the directory it claims to be."""


@dataclass(frozen=True)
class Entry:
    """One file. Directories are implied by paths and are not stored:
    an empty directory carries no content, and reproducing one is the
    materializer's job, not the snapshot's."""

    path: str
    size: int
    blob_hash: str
    executable: bool = False
    #: Present when the blob is small enough to travel with the tree.
    data: bytes | None = None

    def to_wire(self, *, inline: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path, "size": self.size,
            "blob_hash": self.blob_hash,
        }
        if self.executable:
            out["executable"] = True
        if inline and self.data is not None:
            out["data"] = self.data
        return out

    @staticmethod
    def from_wire(value: Mapping[str, Any]) -> Entry:
        data = value.get("data")
        return Entry(
            path=str(value["path"]), size=int(value["size"]),
            blob_hash=str(value["blob_hash"]),
            executable=bool(value.get("executable", False)),
            data=bytes(data) if data is not None else None,
        )


@dataclass(frozen=True)
class Mount:
    """A read-only tree grafted in at `at`, identified by the object it
    came from.

    **Never captured.** A mounted corpus cannot have changed — nothing
    can write to it — so re-hashing it after every tool call is pure
    waste, and on a 200 MB corpus it is the dominant cost of the whole
    sandbox. Its identity is the object it was mounted from.
    """

    at: str
    object: str
    digest: str = ""

    def to_wire(self) -> dict[str, Any]:
        out = {"at": self.at, "object": self.object}
        if self.digest:
            out["digest"] = self.digest
        return out

    @staticmethod
    def from_wire(v: Mapping[str, Any]) -> Mount:
        return Mount(at=str(v["at"]), object=str(v["object"]),
                     digest=str(v.get("digest", "")))


@dataclass(frozen=True)
class Snapshot:
    """A directory as a value.

    Entries are sorted by path **at construction**, not by each
    caller. `os.walk` is depth-first, so `capture` naturally produces
    `a.txt, z.txt, m/q.txt` while `from_wire` produced sorted order —
    and the digest walks `entries`, so the same tree hashed two
    different ways depending on how it was built. Normalizing here
    means there is one canonical order and no constructor can forget
    it.
    """

    entries: tuple[Entry, ...] = ()
    #: Read-only trees present in the sandbox but not part of its
    #: mutable state. They contribute to the digest by identity, never
    #: by content.
    mounts: tuple[Mount, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda e: e.path))
        if ordered != tuple(self.entries):
            object.__setattr__(self, "entries", ordered)
        mounts = tuple(sorted(self.mounts, key=lambda m: m.at))
        if mounts != tuple(self.mounts):
            object.__setattr__(self, "mounts", mounts)

    @property
    def n_files(self) -> int:
        return len(self.entries)

    @property
    def n_bytes(self) -> int:
        return sum(e.size for e in self.entries)

    def paths(self) -> tuple[str, ...]:
        return tuple(e.path for e in self.entries)

    def get(self, path: str) -> Entry | None:
        for e in self.entries:
            if e.path == path:
                return e
        return None

    def digest(self) -> str:
        """The tree's identity: content and paths, nothing else.

        Not a hash of `to_wire()` — that would fold in whether a blob
        happened to be inlined, which is a storage decision and not a
        fact about the tree.
        """
        h = hashlib.sha256()
        for e in self.entries:
            h.update(e.path.encode("utf-8"))
            h.update(b"\x00")
            h.update(e.blob_hash.encode("ascii"))
            h.update(b"\x01" if e.executable else b"\x00")
        for m in self.mounts:
            # By identity, not content: the point of a mount is that we
            # do not read it.
            h.update(b"\x02")
            h.update(m.at.encode("utf-8"))
            h.update(b"\x00")
            h.update((m.digest or m.object).encode("utf-8"))
        return "sha256:" + h.hexdigest()

    def to_wire(self, *, inline: bool = False) -> dict[str, Any]:
        """The stored form. Blobs are REFERENCES by default.

        Measured on a 2000-file tree: 8.43 MB with blobs inline against
        0.22 MB as hashes — 38x. A sandbox session emits one of these
        per tool call, so inlining would put the content of the whole
        tree on the wire for every `ls`. The blob store holds the
        bytes; this holds the shape.

        `inline=True` is for a self-contained fixture — a seeded tree
        small enough that carrying it beats needing a store beside it.
        """
        return {
            "kind": KIND,
            "version": 1,
            "digest": self.digest(),
            "n_files": self.n_files,
            "n_bytes": self.n_bytes,
            "entries": [(e.to_wire() if inline else e.to_wire(inline=False))
                        for e in self.entries],
            **({"mounts": [m.to_wire() for m in self.mounts]}
               if self.mounts else {}),
        }

    @staticmethod
    def from_wire(value: Mapping[str, Any]) -> Snapshot:
        if value.get("kind") != KIND:
            raise ValueError(
                f"not a filesystem snapshot: kind={value.get('kind')!r}")
        return Snapshot(
            tuple(Entry.from_wire(e) for e in value.get("entries", [])),
            tuple(Mount.from_wire(m) for m in value.get("mounts", [])))


#: The empty tree is a constant, and its digest is stable.
EMPTY = Snapshot()


def blob_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _walk(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Every regular file under `root`, in sorted order.

    Sorted because `os.walk` returns whatever the filesystem hands it,
    and a snapshot whose entry order depended on that would hash
    differently on two machines holding identical content.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            yield pathlib.Path(dirpath) / name


def capture(root: str | os.PathLike[str], *, inline_max: int = INLINE_MAX,
            max_files: int = MAX_FILES, max_bytes: int = MAX_BYTES,
            blobs: dict[str, bytes] | None = None,
            mounts: Sequence[Mount] = ()) -> Snapshot:
    """Read a directory into a snapshot.

    Symlinks are followed only within the tree and stored as the file
    they point at; a link escaping the root is refused rather than
    silently resolved, because a sandbox's snapshot must describe the
    sandbox and nothing outside it.

    `blobs`, when given, receives every blob keyed by hash — including
    the large ones left out of the entries. The caller stores them.
    """
    base = pathlib.Path(root).resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"not a directory: {base}")
    entries: list[Entry] = []
    total = 0
    skip = tuple(m.at.rstrip("/") + "/" for m in mounts)
    for path in _walk(base):
        rel = path.relative_to(base).as_posix()
        if any(rel == m.at.rstrip("/") or rel.startswith(p)
               for m, p in zip(mounts, skip, strict=True)):
            # A mount cannot have changed; reading it back is the
            # dominant cost on a large corpus and buys nothing.
            continue
        real = path.resolve()
        if not str(real).startswith(str(base) + os.sep) and real != base:
            raise SnapshotLimit(
                f"{path} leaves the snapshot root ({real}) — a snapshot "
                f"must describe the sandbox and nothing outside it")
        if not real.is_file():
            continue
        data = real.read_bytes()
        total += len(data)
        if len(entries) + 1 > max_files:
            raise SnapshotLimit(
                f"more than {max_files} files under {base}")
        if total > max_bytes:
            raise SnapshotLimit(
                f"more than {max_bytes} bytes under {base} "
                f"(reached {total} at {path.relative_to(base)})")
        digest = blob_hash(data)
        if blobs is not None:
            blobs[digest] = data
        entries.append(Entry(
            path=rel,
            size=len(data),
            blob_hash=digest,
            executable=bool(real.stat().st_mode & stat.S_IXUSR),
            data=data if len(data) <= inline_max else None,
        ))
    return Snapshot(tuple(entries), tuple(mounts))


def materialize(snapshot: Snapshot, root: str | os.PathLike[str], *,
                blobs: Mapping[str, bytes] | None = None) -> None:
    """Write a snapshot into a directory.

    Modes are normalized to 644/755 and mtimes are left to the OS: the
    snapshot deliberately does not carry them, so materializing is not
    a bit-for-bit restoration of a host directory. It is a restoration
    of the CONTENT, which is what a re-run needs.
    """
    base = pathlib.Path(root)
    base.mkdir(parents=True, exist_ok=True)
    for e in snapshot.entries:
        target = base / e.path
        if not str(target.resolve()).startswith(str(base.resolve())):
            raise SnapshotLimit(f"entry escapes the root: {e.path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = e.data
        if data is None:
            if blobs is None or e.blob_hash not in blobs:
                raise KeyError(
                    f"{e.path}: blob {e.blob_hash} is not inline and was "
                    f"not supplied — pass the blob store that captured it")
            data = blobs[e.blob_hash]
        if blob_hash(data) != e.blob_hash:
            raise ValueError(
                f"{e.path}: blob does not match its hash — the store is "
                f"corrupt or the wrong one")
        target.write_bytes(data)
        target.chmod(MODE_EXEC if e.executable else MODE_FILE)


@dataclass(frozen=True)
class Diff:
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def to_wire(self) -> dict[str, Any]:
        return {"added": list(self.added), "removed": list(self.removed),
                "changed": list(self.changed)}


def diff(before: Snapshot, after: Snapshot) -> Diff:
    """What a command did, as paths. Exact: a file whose content is
    unchanged never appears, even if it was rewritten."""
    a = {e.path: e for e in before.entries}
    b = {e.path: e for e in after.entries}
    changed = tuple(sorted(
        p for p in a.keys() & b.keys()
        if a[p].blob_hash != b[p].blob_hash
        or a[p].executable != b[p].executable))
    return Diff(added=tuple(sorted(b.keys() - a.keys())),
                removed=tuple(sorted(a.keys() - b.keys())),
                changed=changed)


def seeded(files: Mapping[str, bytes | str], *,
           executable: Sequence[str] = ()) -> Snapshot:
    """A snapshot built from literal content — the usual way a test or
    a protocol states its starting filesystem."""
    execs = set(executable)
    entries = []
    for path in sorted(files):
        raw = files[path]
        data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        entries.append(Entry(path=path, size=len(data),
                             blob_hash=blob_hash(data),
                             executable=path in execs,
                             data=data if len(data) <= INLINE_MAX else None))
    return Snapshot(tuple(entries))
