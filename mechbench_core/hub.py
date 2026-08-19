"""Hub-ref plumbing (task 000260): revision pinning and offline
resolution against the HuggingFace cache layout.

Refs may pin a revision as ``repo/name@revision`` where revision is a
commit sha (or unambiguous prefix) or a ref name (branch/tag). Pinned
or not, callers should RECORD what actually resolved — reproducibility
by record first, strictness opt-in.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_model_ref(ref: str) -> tuple[str, str | None]:
    """Split ``repo@revision`` into (repo_id, revision|None)."""
    if "@" in ref:
        repo, _, rev = ref.partition("@")
        return repo, (rev or None)
    return ref, None


def hf_hub_cache() -> Path:
    env = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_dir(repo_id: str) -> Path:
    return hf_hub_cache() / ("models--" + repo_id.replace("/", "--"))


def resolve_cached_revision(repo_id: str,
                            revision: str | None = None) -> str | None:
    """Resolve a revision to a full commit sha from the local cache —
    offline-safe. None revision resolves the ``main`` ref. Returns None
    when the repo (or ref) isn't cached."""
    d = _repo_dir(repo_id)
    if not d.exists():
        return None
    snapshots = d / "snapshots"
    if revision:
        # Commit sha or prefix?
        if all(c in "0123456789abcdef" for c in revision.lower()) and len(revision) >= 7:
            matches = [s.name for s in snapshots.iterdir()
                       if s.name.startswith(revision.lower())] \
                if snapshots.exists() else []
            if len(matches) == 1:
                return matches[0]
        ref_file = d / "refs" / revision
        if ref_file.exists():
            return ref_file.read_text().strip()
        return None
    main = d / "refs" / "main"
    if main.exists():
        return main.read_text().strip()
    # Single-snapshot caches without refs: unambiguous.
    if snapshots.exists():
        snaps = [s.name for s in snapshots.iterdir() if s.is_dir()]
        if len(snaps) == 1:
            return snaps[0]
    return None


def snapshot_path(repo_id: str, commit_sha: str) -> Path | None:
    p = _repo_dir(repo_id) / "snapshots" / commit_sha
    return p if p.exists() else None
