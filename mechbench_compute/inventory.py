from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RevisionInfo:
    commit: str
    size_bytes: int
    reclaimable_bytes: int
    last_modified: datetime | None
    refs: tuple[str, ...]
    path: Path

    @property
    def short(self) -> str:
        return self.commit[:8]

    @property
    def superseded(self) -> bool:
        return not self.refs


@dataclass(frozen=True)
class RepoInventory:
    repo_id: str
    disk_bytes: int
    revisions: tuple[RevisionInfo, ...]

    @property
    def main_commit(self) -> str | None:
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
    if not commits:
        return 0
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        info = scan_cache_dir()
        known = {rev.commit_hash for repo in info.repos for rev in repo.revisions}
    except CacheNotFound:
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
    try:
        return int(info.delete_revisions(*commits).expected_freed_size)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return 0


def _as_utc(value: float | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromtimestamp(value, tz=UTC)


def format_bytes(n: int) -> str:
    step = 1000.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < step or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} TB"
