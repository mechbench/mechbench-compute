from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def parse_model_ref(ref: str) -> tuple[str, str | None]:
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
    d = _repo_dir(repo_id)
    if not d.exists():
        return None
    snapshots = d / "snapshots"
    if revision:
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
    if snapshots.exists():
        snaps = [s.name for s in snapshots.iterdir() if s.is_dir()]
        if len(snaps) == 1:
            return snaps[0]
    return None


def snapshot_path(repo_id: str, commit_sha: str) -> Path | None:
    p = _repo_dir(repo_id) / "snapshots" / commit_sha
    return p if p.exists() else None


def is_offline() -> bool:
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        v = os.environ.get(var, "").strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
    return False


def ensure_model(
    ref: str,
    *,
    offline: bool | None = None,
    on_download: "Callable[[str, str | None], None] | None" = None,
    on_bytes: "Callable[[int, int], None] | None" = None,
) -> tuple[str, str, Path]:
    repo_id, revision = parse_model_ref(ref)

    sha = resolve_cached_revision(repo_id, revision)
    if sha is not None:
        cached = snapshot_path(repo_id, sha)
        if cached is not None:
            return repo_id, sha, cached

    if offline if offline is not None else is_offline():
        what = f"{repo_id}@{revision}" if revision else repo_id
        raise ValueError(
            f"{what} is not in the local HuggingFace cache and this machine is "
            f"configured for offline use (HF_HUB_OFFLINE). Fetch it on a "
            f"connected machine, or unset that variable."
        )

    if on_download is not None:
        on_download(repo_id, revision)

    from huggingface_hub import snapshot_download

    try:
        kwargs = {}
        if on_bytes is not None:
            kwargs["tqdm_class"] = _progress_tqdm(on_bytes)
        # external: huggingface_hub — snapshot_download lands in a directory named for the resolved commit
        path = Path(snapshot_download(repo_id, revision=revision, **kwargs))
    except Exception as exc:  # noqa: BLE001
        what = f"{repo_id}@{revision}" if revision else repo_id
        raise ValueError(f"could not fetch {what} from the HuggingFace hub: {exc}") from exc

    resolved = path.name if _looks_like_sha(path.name) else resolve_cached_revision(repo_id, revision)
    if resolved is None:
        raise ValueError(
            f"fetched {repo_id} but could not determine which commit it is; "
            f"the cache layout at {path} is not what was expected"
        )
    return repo_id, resolved, path


def _looks_like_sha(name: str) -> bool:
    return len(name) >= 7 and all(c in "0123456789abcdef" for c in name.lower())


def _progress_tqdm(on_bytes: "Callable[[int, int], None]"):
    from tqdm.auto import tqdm as _tqdm

    live: dict[int, tuple[int, int]] = {}

    class _Tqdm(_tqdm):  # type: ignore[misc]
        def update(self, n=1):  # noqa: ANN001, ANN201
            result = super().update(n)
            if self.unit in ("B", "iB") and self.total:
                live[id(self)] = (int(self.n), int(self.total))
                done = sum(d for d, _ in live.values())
                total = sum(t for _, t in live.values())
                if total:
                    on_bytes(done, total)
            return result

        def close(self):  # noqa: ANN201
            live.pop(id(self), None)
            return super().close()

    return _Tqdm
