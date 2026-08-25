"""What weights this machine is holding, and what deleting them buys back.

Lives here, beside `hub.py`, because the runner and the web UI both want
the answer and neither should be reading the HuggingFace cache layout
itself. It is also one of the two modules that must load on a machine
with no compute backend at all — see the note in `__init__.py` — so it
imports nothing from MLX.

## Why not just sum the revisions

The cache shares blobs between revisions of the same repository: two
revisions of a 24 GB model that differ in one tensor occupy a little
over 24 GB, not 48. So a revision's *apparent* size is not what deleting
it returns, and summing revisions overstates a repository badly.

Both numbers are reported, because both get asked for:

* `size_bytes` — what this revision contains, shared blobs included.
* `reclaimable_bytes` — what actually comes back if it is deleted, which
  is only the blobs no other revision refers to.

The motivating case, measured 2026-08-22: 125 GB of cache with
`gemma-4-E4B-it-bf16` holding four revisions and `gemma-4-e2b-it-bf16`
three. Roughly 30 GB of that is revisions nobody chose to keep, and
nothing else reclaims it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RevisionInfo:
    """One cached commit of one repository."""

    commit: str
    #: Everything this revision contains, counting blobs it shares.
    size_bytes: int
    #: What deleting it would actually free — blobs nothing else refers to.
    reclaimable_bytes: int
    last_modified: datetime | None
    #: Ref names pointing here: `("main",)` for the current upstream tip
    #: as this machine last saw it. Empty means nothing points here,
    #: which is the definition of superseded.
    refs: tuple[str, ...]
    path: Path

    @property
    def short(self) -> str:
        return self.commit[:8]

    @property
    def superseded(self) -> bool:
        """Held only because it was downloaded once.

        Note that a *pinned* revision looks exactly like this: pins live
        in protocols, which this layer cannot see. Deleting is therefore
        always a decision for the caller, never something to infer.
        """
        return not self.refs


@dataclass(frozen=True)
class RepoInventory:
    repo_id: str
    #: True, deduplicated size of the repository on disk.
    disk_bytes: int
    revisions: tuple[RevisionInfo, ...]

    @property
    def main_commit(self) -> str | None:
        """What `main` resolves to locally — which is not necessarily what
        it resolves to upstream today."""
        for rev in self.revisions:
            if "main" in rev.refs:
                return rev.commit
        return None

    @property
    def superseded(self) -> tuple[RevisionInfo, ...]:
        return tuple(r for r in self.revisions if r.superseded)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(r.reclaimable_bytes for r in self.superseded)


def scan() -> list[RepoInventory]:
    """Every model repository in the local cache, largest first.

    A machine that has never downloaded a model has no cache DIRECTORY,
    and huggingface_hub raises CacheNotFound rather than reporting the
    empty truth. Every dev machine has the directory, so the release
    gate's fresh-venv smoke was the first thing to ever hit this
    (task 000300, first dry run)."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        info = scan_cache_dir()
    except CacheNotFound:
        return []
    out: list[RepoInventory] = []
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        revisions = [
            RevisionInfo(
                commit=rev.commit_hash,
                size_bytes=rev.size_on_disk,
                reclaimable_bytes=_freed_by(info, [rev.commit_hash]),
                last_modified=_as_utc(rev.last_modified),
                refs=tuple(sorted(rev.refs)),
                path=Path(rev.snapshot_path),
            )
            for rev in repo.revisions
        ]
        revisions.sort(key=lambda r: (-r.size_bytes, r.commit))
        out.append(
            RepoInventory(
                repo_id=repo.repo_id,
                disk_bytes=repo.size_on_disk,
                revisions=tuple(revisions),
            )
        )
    out.sort(key=lambda r: -r.disk_bytes)
    return out


def total_disk_bytes() -> int:
    from huggingface_hub import scan_cache_dir

    return int(scan_cache_dir().size_on_disk)


def find(repo_id: str) -> RepoInventory | None:
    for repo in scan():
        if repo.repo_id == repo_id:
            return repo
    return None


def delete_revisions(commits: list[str]) -> int:
    """Delete cached revisions by commit. Returns the bytes freed.

    Deliberately takes explicit commits rather than a policy: a revision
    that nothing points at may still be the one a protocol pins, and
    this layer cannot see protocols. Choosing is the caller's job.
    """
    if not commits:
        return 0
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        info = scan_cache_dir()
        known = {rev.commit_hash for repo in info.repos for rev in repo.revisions}
    except CacheNotFound:
        # A machine with no cache directory knows no commits, so every
        # commit asked for gets the same refusal below — the fresh-
        # machine truth scan() learned in 0.15.4, which this sibling
        # call missed. CI is a fresh machine on every run; that is how
        # this surfaced (every dev machine has the directory).
        info = None
        known = set()
    unknown = [c for c in commits if c not in known]
    if unknown:
        raise ValueError(
            f"not in the local cache: {', '.join(unknown)}. "
            f"Commits must be given in full, as `scan()` reports them."
        )
    strategy = info.delete_revisions(*commits)
    freed = int(strategy.expected_freed_size)
    strategy.execute()
    return freed


def _freed_by(info: object, commits: list[str]) -> int:
    """What the hub says deleting these would return, without deleting."""
    try:
        return int(info.delete_revisions(*commits).expected_freed_size)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — a size estimate is not worth an exception
        return 0


def _as_utc(value: float | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromtimestamp(value, tz=UTC)


def format_bytes(n: int) -> str:
    """`24.1 GB` — sized for a table, not for accounting."""
    step = 1000.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < step or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} TB"
