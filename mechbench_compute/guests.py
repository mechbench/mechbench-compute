from __future__ import annotations

import gzip
import hashlib
import os
import pathlib
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuestMount:
    at: str
    host: str = ""
    url: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class Guest:
    name: str
    url: str
    sha256: str
    size: int
    source: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[GuestMount, ...] = ()

    @property
    def hosted(self) -> bool:
        return bool(self.url)


class GuestUnavailable(RuntimeError):
    pass


REGISTRY: dict[str, Guest] = {
    "mbshell": Guest(
        name="mbshell",
        url="https://github.com/mechbench/mechbench-compute/releases/download/mbshell-5090c2c1646c/mbshell-5090c2c1646c.wasm.gz",
        sha256="5090c2c1646c2b96f32baa7aac87535f8cf5e8895f01fd0cca65c3668a69ae62",
        size=15426977,
        source="guests/mbshell (go-busybox@13f3053 + go-busybox-wasi.patch + "
               "mvdan.cc/sh/v3@v3.12.0 + mvdan-sh-wasi.patch; go1.27.1 -trimpath -buildvcs=false)"),
    "cpython": Guest(
        name="cpython",
        url="https://github.com/mechbench/mechbench-compute/releases/download/cpython-0e9a1065ab0d/python.wasm.gz",
        sha256="0e9a1065ab0db40e5b42a49083f0b09c517f2cd4ff650914de8006ab9d079da3",
        size=29207565,
        source="guests/cpython (CPython v3.13.3 + wasi-sdk-34 + -ffile-prefix-map; host python3.13)",
        env={"PYTHONHOME": "/usr/local", "PYTHONDONTWRITEBYTECODE": "1"},
        mounts=(GuestMount(
            at="/usr/local/lib/python3.13",
            url="https://github.com/mechbench/mechbench-compute/releases/download/cpython-0e9a1065ab0d/stdlib.tar.gz",
            sha256="0e271acc56651af4a3031f307dcf6575a69c4a5c0a228c74e922a2a825ee6266"),)),
}


def register(guest: Guest) -> None:
    REGISTRY[guest.name] = guest


def is_registered(name: str) -> bool:
    return name in REGISTRY


def cache_dir() -> pathlib.Path:
    root = os.environ.get("MECHBENCH_GUEST_CACHE")
    base = pathlib.Path(root) if root else pathlib.Path.home() / ".mechbench" / "guests"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(name: str, *, fetch: bool = True) -> pathlib.Path:
    guest = REGISTRY.get(name)
    if guest is None:
        raise GuestUnavailable(
            f"no guest named {name!r} is registered — known: "
            f"{', '.join(sorted(REGISTRY)) or '(none)'}")
    target = cache_dir() / f"{guest.name}-{guest.sha256[:12]}.wasm"
    if target.exists():
        if _sha256_of(target) == guest.sha256:
            return target
        target.unlink()
    if not guest.hosted:
        raise GuestUnavailable(
            f"guest {name!r} is pinned (sha256 {guest.sha256[:12]}…) but not "
            f"hosted yet — build it from {guest.source or 'its source'} and "
            f"register the build with guests.install_local({name!r}, path)")
    if not fetch:
        raise GuestUnavailable(
            f"guest {name!r} is not cached and fetching is disabled")
    return _fetch(guest, target)


def _tls_context():
    import ssl

    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch(guest: Guest, target: pathlib.Path) -> pathlib.Path:
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{guest.name}-", suffix=".part",
                                        dir=str(target.parent))
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as out, urllib.request.urlopen(
                guest.url, timeout=300, context=_tls_context()) as resp:
            stream = gzip.GzipFile(fileobj=resp) if guest.url.endswith(".gz") else resp
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                out.write(chunk)
        got = _sha256_of(tmp)
        if got != guest.sha256:
            raise GuestUnavailable(
                f"guest {guest.name!r} from {guest.url} hashed to {got}, "
                f"expected {guest.sha256}. Not kept. Either the artifact "
                f"was replaced upstream or the registry is stale — both "
                f"need a human, not a retry.")
        size = tmp.stat().st_size
        if guest.size and size != guest.size:
            raise GuestUnavailable(
                f"guest {guest.name!r} is {size} bytes, registry says "
                f"{guest.size} — hash matched, so this is a registry error")
        os.replace(tmp, target)
        return target
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def install_local(name: str, path: str | os.PathLike[str], *,
                  source: str = "", replace: bool = False,
                  mounts: Sequence[tuple[str, str]] = (),
                  env: Mapping[str, str] | None = None) -> Guest:
    src = pathlib.Path(path)
    digest = _sha256_of(src)
    pinned = REGISTRY.get(name)
    if (pinned is not None and pinned.sha256 and pinned.sha256 != digest
            and not replace):
        raise GuestUnavailable(
            f"{src} hashes to {digest}, but {name!r} is pinned at "
            f"{pinned.sha256} (built from {pinned.source or '?'}). A build "
            f"that differs from the pin is worth understanding before it "
            f"runs under that name; pass replace=True to re-pin.")
    resolved = tuple(GuestMount(at=at, host=str(pathlib.Path(host).resolve()))
                     for host, at in mounts)
    merged_env = {**(dict(pinned.env) if pinned else {}), **(dict(env) if env else {})}
    guest = Guest(name=name, url=f"file://{src.resolve()}", sha256=digest,
                  size=src.stat().st_size,
                  source=source or (pinned.source if pinned else ""),
                  env=merged_env, mounts=resolved or (pinned.mounts if pinned else ()))
    target = cache_dir() / f"{name}-{digest[:12]}.wasm"
    if not target.exists():
        target.write_bytes(src.read_bytes())
    register(guest)
    return guest


def _ensure_mount(name: str, mount: GuestMount, *, fetch: bool) -> GuestMount:
    if mount.host and pathlib.Path(mount.host).is_dir():
        return mount
    if not mount.url:
        raise GuestUnavailable(
            f"guest {name!r} needs a runtime mount at {mount.at!r} but it is "
            f"neither built locally nor hosted — run the guest's build.sh and "
            f"install_local it with mounts=…")
    dest = cache_dir() / f"{name}-lib-{mount.sha256[:12]}"
    marker = dest / ".ok"
    if marker.is_file():
        return GuestMount(at=mount.at, host=str(dest), url=mount.url, sha256=mount.sha256)
    if not fetch:
        raise GuestUnavailable(
            f"guest {name!r} mount at {mount.at!r} is not unpacked and "
            f"fetching is disabled")
    import tarfile

    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{name}-lib-", suffix=".tgz",
                                        dir=str(cache_dir()))
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as out, urllib.request.urlopen(
                mount.url, timeout=300, context=_tls_context()) as resp:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
        got = _sha256_of(tmp)
        if got != mount.sha256:
            raise GuestUnavailable(
                f"guest {name!r} mount from {mount.url} hashed to {got}, "
                f"expected {mount.sha256}. Not kept.")
        import shutil

        staging = pathlib.Path(tempfile.mkdtemp(prefix=f"{name}-lib-",
                                                dir=str(cache_dir())))
        with tarfile.open(tmp, "r:gz") as tf:
            try:
                tf.extractall(staging, filter="data")
            except TypeError:
                tf.extractall(staging)
        (staging / ".ok").write_bytes(b"")
        try:
            os.replace(staging, dest)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
        return GuestMount(at=mount.at, host=str(dest), url=mount.url, sha256=mount.sha256)
    finally:
        tmp.unlink(missing_ok=True)


def resolve(guest: str | os.PathLike[str], *, fetch: bool = True
            ) -> tuple[pathlib.Path, tuple[GuestMount, ...], Mapping[str, str]]:
    if isinstance(guest, str) and is_registered(guest):
        g = REGISTRY[guest]
        wasm = ensure(guest, fetch=fetch)
        mounts = tuple(_ensure_mount(guest, m, fetch=fetch) for m in g.mounts)
        return wasm, mounts, g.env
    return pathlib.Path(guest), (), {}
