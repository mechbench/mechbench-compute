"""WASI guest binaries, fetched on first use and verified by hash
(task 000359, epic 000334).

The sandbox runs real programs — busybox, CPython — compiled to
WebAssembly. They are not vendored into the wheel: CPython-on-WASI is
20–30 MB and would triple what `pip install mechbench-compute` costs
for every user who never runs a sandbox. Instead each guest is a
**pinned artifact**: a URL, a sha256, and a size, built by us from a
pinned upstream commit and hosted as a GitHub release on this repo,
tagged by guest and hash. The runner fetches it the first time a
protocol asks, verifies the hash, and keeps it under
`~/.mechbench/guests/`.

The hash is the contract. A fetched file that does not match is
deleted, never used, and the failure names what was expected — the
same discipline as a hub revision (000260), because a sandbox running
a binary we did not pin is not a sandbox.
"""

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
    """A read-only directory preopened beside the working snapshot,
    carrying a guest's own runtime files. The CPython guest mounts its
    standard library here; the working tree stays at `/`.

    `at` is the guest path (fixed by the guest). `host` is the local
    directory that backs it — set by `install_local` for an unhosted
    build, or filled by unpacking `url` (a companion tarball, verified
    by `sha256`) once the guest is hosted. Never writable: a guest
    corrupting its own runtime would poison the shared cache."""

    at: str
    host: str = ""
    url: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class Guest:
    """One pinned artifact. `source` is where the bytes were BUILT
    from (a repo and commit), which is a different fact from `url`,
    where they are hosted; provenance wants both. An empty `url` means
    pinned but not yet hosted: the hash is settled, the bytes have to
    be built locally (`install_local`) until a release hosts them.

    `env` and `mounts` are the guest's RUNTIME needs — an interpreter
    that reads `PYTHONHOME` and a standard library it must find. They
    are not part of the wasm's hash; they are how it is run."""

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
    """The guest could not be produced in a state we are willing to
    run — unknown name, unreachable host, or a hash that did not match.
    Never a silent fallback to whatever bytes were there."""


#: name -> pinned artifact.
#:
#: mbshell is THE guest: go-busybox's applets behind an in-process
#: POSIX shell (mvdan/sh with a small WASI patch), compiled with
#: standard Go's wasip1 port — recipe and rationale in
#: `guests/mbshell/`. Plain busybox is not pinned because it has no
#: shell under wasm and never will (its ash is fork/exec).
#:
#: Hosted as a GitHub release on this repo, tagged by guest and hash,
#: with the third-party NOTICE beside it: go-busybox is MIT as declared
#: in its README (no LICENSE file in the tree; upstream issue #3),
#: mvdan/sh BSD-3-Clause, goawk MIT, golang.org/x BSD-3-Clause. The
#: hosted form is gzipped; the pin is the decompressed bytes.
REGISTRY: dict[str, Guest] = {
    "mbshell": Guest(
        name="mbshell",
        url="https://github.com/mechbench/mechbench-compute/releases/download/mbshell-5090c2c1646c/mbshell-5090c2c1646c.wasm.gz",
        sha256="5090c2c1646c2b96f32baa7aac87535f8cf5e8895f01fd0cca65c3668a69ae62",
        size=15426977,
        source="guests/mbshell (go-busybox@13f3053 + go-busybox-wasi.patch + "
               "mvdan.cc/sh/v3@v3.12.0 + mvdan-sh-wasi.patch; go1.27.1 -trimpath -buildvcs=false)"),
    # CPython 3.13.3 compiled to wasip1 with wasi-sdk — recipe in
    # `guests/cpython/`. Its standard library is a read-only MOUNT at
    # `/usr/local/lib/python3.13` (every module builds static, so the
    # library is pure .py), which the guest finds via PYTHONHOME. Not
    # hosted yet: the mount's `host` is filled by `install_local` from a
    # local build until a release hosts the wasm and a stdlib tarball.
    "cpython": Guest(
        name="cpython", url="",
        sha256="",  # filled when the build is pinned; see guests/cpython/build.sh
        size=0,
        source="guests/cpython (CPython v3.13.3 + wasi-sdk-34; host python3.13)",
        env={"PYTHONHOME": "/usr/local", "PYTHONDONTWRITEBYTECODE": "1"},
        mounts=(GuestMount(at="/usr/local/lib/python3.13"),)),
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
    """The guest's path on disk, fetching and verifying if needed.

    A cached file is re-hashed before it is trusted: a partial download
    from a previous crash, or a file edited by hand, must not run just
    because it has the right name.
    """
    guest = REGISTRY.get(name)
    if guest is None:
        raise GuestUnavailable(
            f"no guest named {name!r} is registered — known: "
            f"{', '.join(sorted(REGISTRY)) or '(none)'}")
    target = cache_dir() / f"{guest.name}-{guest.sha256[:12]}.wasm"
    if target.exists():
        if _sha256_of(target) == guest.sha256:
            return target
        # Wrong bytes under the right name: remove them so the next
        # attempt is a clean fetch rather than a repeat of this one.
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
    """A TLS context that trusts certifi's bundle. The python.org
    framework build on macOS ships with no system CA bundle wired in,
    so a plain urlopen fails with CERTIFICATE_VERIFY_FAILED on the
    first fetch — which is how this line got here."""
    import ssl

    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch(guest: Guest, target: pathlib.Path) -> pathlib.Path:
    """Download, decompress if the URL ends in .gz, verify, install.

    A standard-Go guest is 15 MB raw and 4 MB gzipped, and CloudFront
    will not compress objects over 10 MB, so the hosted form is the
    .gz. The pin is always the hash of the DECOMPRESSED bytes — what
    runs — so a re-compression upstream changes nothing.
    """
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
    """Register a guest from a file already on this machine — a fresh
    build, or a test fixture — computing its hash rather than trusting
    one. Copies it into the cache under its hash so later `ensure`
    calls find it without a network.

    If `name` is already pinned with a hash, the file must match it. A
    build that comes out different is a real event — a toolchain moved,
    or a source tree did — and running it under the pinned name would
    make the registry a lie. Pass `replace=True` to re-pin deliberately.
    A pin with an EMPTY hash is unpinned (a guest whose reproducible
    hash is not yet recorded) and accepts any build.

    `mounts` are `(host_dir, guest_path)` pairs — a guest's read-only
    runtime files, like CPython's standard library; `env` is the
    guest's runtime environment. When the name is pre-declared with
    mount TARGETS (the `at` paths) but no hosts, the hosts given here
    fill them; `env` merges over the declared env.
    """
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


def resolve(guest: str | os.PathLike[str], *, fetch: bool = True
            ) -> tuple[pathlib.Path, tuple[GuestMount, ...], Mapping[str, str]]:
    """`(wasm_path, mounts, env)` for a registered NAME (fetched and
    verified as needed) or a bare `.wasm` PATH (no mounts, no env). The
    sandbox calls this so a guest's runtime needs travel with it."""
    if isinstance(guest, str) and is_registered(guest):
        g = REGISTRY[guest]
        return ensure(guest, fetch=fetch), g.mounts, g.env
    return pathlib.Path(guest), (), {}
