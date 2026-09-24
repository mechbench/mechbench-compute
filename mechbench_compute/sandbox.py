from __future__ import annotations

import atexit
import os
import pathlib
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mechbench_compute import guests
from mechbench_compute import snapshots as fs

TRUNCATED = "\n[output truncated at {n} bytes]"

EXIT_DIR = "/.mechbench"


@dataclass(frozen=True)
class Limits:
    memory_mb: int = 256
    fuel: int = 100_000_000_000
    wall_seconds: float = 30.0
    output_bytes: int = 256 * 1024
    max_files: int = fs.MAX_FILES
    max_bytes: int = fs.MAX_BYTES

    def to_wire(self) -> dict[str, Any]:
        return {"memory_mb": self.memory_mb, "fuel": self.fuel,
                "wall_seconds": self.wall_seconds,
                "output_bytes": self.output_bytes,
                "max_files": self.max_files, "max_bytes": self.max_bytes}


@dataclass(frozen=True)
class Result:
    snapshot: fs.Snapshot
    stdout: str
    stderr: str
    exit_code: int
    changed: fs.Diff
    duration_ms: int
    fuel_used: int | None = None
    limit: str | None = None
    truncated: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.limit is None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "exit_code": self.exit_code,
            "stdout": self.stdout, "stderr": self.stderr,
            "snapshot_out": self.snapshot.digest(),
            "changed": self.changed.to_wire(),
            "duration_ms": self.duration_ms,
        }
        if self.fuel_used is not None:
            out["fuel_used"] = self.fuel_used
        if self.limit:
            out["limit"] = self.limit
        if self.truncated:
            out["truncated"] = list(self.truncated)
        return out


class SandboxError(RuntimeError):
    pass


STRICT_VIRTUAL = ("clock_res_get", "random_get")

ALWAYS_VIRTUAL = ("poll_oneoff", "clock_time_get")

STRICT_EPOCH_NS = 946_684_800 * 1_000_000_000
STRICT_TICK_NS = 1_000


EPOCH_TICK_S = 0.1

_ENGINE_LOCK = threading.Lock()
_ENGINE: Any = None
_MODULES: dict[tuple[str, int, int], Any] = {}


def _engine():
    global _ENGINE
    import wasmtime

    with _ENGINE_LOCK:
        if _ENGINE is None:
            config = wasmtime.Config()
            config.consume_fuel = True
            config.epoch_interruption = True
            _ENGINE = wasmtime.Engine(config)

            def tick(engine=_ENGINE) -> None:
                while True:
                    time.sleep(EPOCH_TICK_S)
                    engine.increment_epoch()

            threading.Thread(target=tick, daemon=True,
                             name="mechbench-sandbox-epoch").start()
        return _ENGINE


def _module(engine, path: pathlib.Path):
    import importlib.metadata

    import wasmtime

    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _ENGINE_LOCK:
        cached = _MODULES.get(key)
    if cached is not None:
        return cached
    version = importlib.metadata.version("wasmtime")
    cwasm = guests.cache_dir() / f"{path.stem}-{st.st_size}-wasmtime{version}.cwasm"
    module = None
    if cwasm.is_file():
        try:
            module = wasmtime.Module.deserialize(engine, cwasm.read_bytes())
        except Exception:  # noqa: BLE001
            module = None
    if module is None:
        try:
            module = wasmtime.Module.from_file(engine, str(path))
        except Exception as e:  # noqa: BLE001
            raise SandboxError(f"guest {path.name} will not load: {e}") from e
        try:
            tmp = cwasm.with_suffix(f".{os.getpid()}.part")
            tmp.write_bytes(module.serialize())
            os.replace(tmp, cwasm)
        except OSError:
            pass
    with _ENGINE_LOCK:
        _MODULES[key] = module
    return module


def _release_cached() -> None:
    global _ENGINE
    with _ENGINE_LOCK:
        _MODULES.clear()
        _ENGINE = None


atexit.register(_release_cached)


def _cap(text: bytes, limit: int, name: str,
         truncated: list[str]) -> str:
    if len(text) <= limit:
        return text.decode("utf-8", errors="replace")
    truncated.append(name)
    return (text[:limit].decode("utf-8", errors="replace")
            + TRUNCATED.format(n=limit))


def run(snapshot: fs.Snapshot, argv: Sequence[str], *,
        guest: str | os.PathLike[str], limits: Limits | None = None,
        strict: bool = False, env: Mapping[str, str] | None = None,
        stdin: bytes | str = b"", blobs: Mapping[str, bytes] | None = None,
        mounts: Sequence[tuple[str, fs.Snapshot]] = (),
        mount_blobs: Mapping[str, Mapping[str, bytes]] | None = None) -> Result:
    import wasmtime

    limits = limits or Limits()
    try:
        guest_path, guest_mounts, guest_env = guests.resolve(guest)
    except guests.GuestUnavailable as e:
        raise SandboxError(str(e)) from e
    if not guest_path.is_file():
        raise SandboxError(
            _resolve_hint(guest) or f"guest binary not found: {guest_path}")

    with tempfile.TemporaryDirectory(prefix="mechbench-sandbox-") as td:
        root = pathlib.Path(td) / "root"
        fs.materialize(snapshot, root, blobs=blobs)
        meta = pathlib.Path(td) / "meta"
        meta.mkdir()
        stdout_path = pathlib.Path(td) / "stdout"
        stderr_path = pathlib.Path(td) / "stderr"
        stdin_path = pathlib.Path(td) / "stdin"
        stdin_path.write_bytes(stdin.encode() if isinstance(stdin, str) else bytes(stdin))

        engine = _engine()
        module = _module(engine, guest_path)

        store = wasmtime.Store(engine)
        store.set_fuel(limits.fuel)
        store.set_limits(memory_size=limits.memory_mb * 1024 * 1024)

        wasi = wasmtime.WasiConfig()
        wasi.argv = list(argv)
        wasi.env = [(k, v) for k, v in {**dict(guest_env), **(env or {})}.items()]
        wasi.preopen_dir(str(root), "/")
        wasi.preopen_dir(str(meta), EXIT_DIR)
        for m in guest_mounts:
            if not m.host or not pathlib.Path(m.host).is_dir():
                raise SandboxError(
                    f"guest {guest_path.name} needs a mount at {m.at!r} but its "
                    f"host directory {m.host!r} is missing — install the guest "
                    f"with its runtime (guests.install_local(..., mounts=…))")
            wasi.preopen_dir(m.host, m.at, fs_mutable=False)
        for at, tree in mounts:
            if not at.startswith("/") or at.rstrip("/") in ("", EXIT_DIR):
                raise SandboxError(
                    f"mount path {at!r} must be absolute and outside the "
                    f"working tree and {EXIT_DIR}")
            mblobs = (mount_blobs or {}).get(at)
            mdir = _materialize_mount(
                tree, blobs=mblobs if mblobs is not None else tree.blobs)
            wasi.preopen_dir(str(mdir), at, fs_mutable=False)
        wasi.stdin_file = str(stdin_path)
        wasi.stdout_file = str(stdout_path)
        wasi.stderr_file = str(stderr_path)
        store.set_wasi(wasi)

        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        virtual = _Virtual(snapshot, argv, strict)
        virtual.install(linker, store, module)

        store.set_epoch_deadline(max(1, int(limits.wall_seconds / EPOCH_TICK_S)))

        started = time.monotonic()
        exit_code = 0
        limit: str | None = None
        instance_holder: list = []
        try:
            instance = linker.instantiate(store, module)
            instance_holder.append(instance)
            start = instance.exports(store).get("_start")
            if start is None:
                raise SandboxError(
                    f"guest {guest_path.name} exports no `_start` — not a "
                    f"WASI command")
            start(store)
        except wasmtime.ExitTrap as e:
            exit_code = _reported_exit(meta, int(e.code))
        except wasmtime.Trap as e:
            text = str(e)
            low = text.lower()
            exit_code = -1
            if "fuel" in low:
                limit = "fuel"
            elif "epoch" in low or "interrupt" in low:
                limit = "wall_seconds"
            elif "call stack exhausted" in low:
                limit = "stack"
            elif _out_of_memory(instance_holder, store, limits, text):
                limit = "memory_mb"
            else:
                stderr_path.write_bytes(stderr_path.read_bytes() + f"\n[trap] {text}".encode())
        except wasmtime.WasmtimeError as e:
            text = str(e)
            if "invalid exit status" in text:
                exit_code = _reported_exit(meta, 126)
            elif "memory minimum size" in text and "exceeds" in text:
                exit_code, limit = -1, "memory_mb"
                stderr_path.write_bytes(
                    stderr_path.read_bytes()
                    + f"\n[limit] memory_mb={limits.memory_mb} is below the guest's "
                      f"minimum: {text}".encode())
            else:
                raise SandboxError(
                    f"guest {guest_path.name} could not be run: {text}") from e
        duration_ms = int((time.monotonic() - started) * 1000)
        fuel_left = None
        try:
            fuel_left = store.get_fuel()
        except Exception:  # noqa: BLE001
            pass

        truncated: list[str] = []
        out = _cap(stdout_path.read_bytes(), limits.output_bytes, "stdout", truncated)
        err = _cap(stderr_path.read_bytes(), limits.output_bytes, "stderr", truncated)
        if limit is None and exit_code != 0 and _oom_exit(instance_holder, store, limits, err):
            limit = "memory_mb"
        try:
            after = fs.capture(root, max_files=limits.max_files,
                               max_bytes=limits.max_bytes,
                               mounts=snapshot.mounts)
        except fs.SnapshotLimit as e:
            return Result(snapshot=snapshot, stdout=out, stderr=err + f"\n[limit] {e}",
                          exit_code=-1, changed=fs.Diff(), duration_ms=duration_ms,
                          fuel_used=(limits.fuel - fuel_left) if fuel_left is not None else None,
                          limit="max_bytes" if "bytes" in str(e) else "max_files",
                          truncated=tuple(truncated))
        return Result(
            snapshot=after, stdout=out, stderr=err, exit_code=exit_code,
            changed=fs.diff(snapshot, after), duration_ms=duration_ms,
            fuel_used=(limits.fuel - fuel_left) if fuel_left is not None else None,
            limit=limit, truncated=tuple(truncated),
        )


def _reported_exit(meta: pathlib.Path, fallback: int) -> int:
    try:
        return int((meta / "exit").read_text().strip())
    except (OSError, ValueError):
        return fallback


_MOUNT_LOCK = threading.Lock()
_MOUNTS_READY: set[str] = set()


def _materialize_mount(tree: fs.Snapshot,
                       blobs: Mapping[str, bytes] | None = None) -> pathlib.Path:
    digest = tree.digest().split(":")[-1][:16]
    cache = guests.cache_dir()
    dest = cache / f"mount-{digest}"
    marker = cache / f".mount-{digest}.ok"
    with _MOUNT_LOCK:
        if digest in _MOUNTS_READY:
            return dest
    if marker.is_file() and dest.is_dir():
        with _MOUNT_LOCK:
            _MOUNTS_READY.add(digest)
        return dest
    staging = pathlib.Path(tempfile.mkdtemp(prefix="mount-", dir=str(cache)))
    import shutil
    try:
        fs.materialize(tree, staging, blobs=blobs if blobs is not None else tree.blobs)
        try:
            os.replace(staging, dest)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
        marker.write_bytes(b"")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    with _MOUNT_LOCK:
        _MOUNTS_READY.add(digest)
    return dest


def _resolve_hint(guest: str | os.PathLike[str]) -> str | None:
    if (isinstance(guest, str) and not guests.is_registered(guest)
            and os.sep not in guest and not guest.endswith(".wasm")):
        return (f"no guest named {guest!r} is registered — known: "
                f"{', '.join(sorted(guests.REGISTRY)) or '(none)'}")
    return None


def _pages(holder: list, store) -> int:
    if not holder:
        return 0
    try:
        mem = holder[0].exports(store).get("memory")
        return mem.size(store) if mem is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def _oom_exit(holder: list, store, limits: Limits, stderr: str) -> bool:
    if "out of memory" not in stderr.lower():
        return False
    return _pages(holder, store) * 65536 >= limits.memory_mb * 1024 * 1024 * 0.5


def _out_of_memory(holder: list, store, limits: Limits, trap_text: str) -> bool:
    low = trap_text.lower()
    if any(m in low for m in ("runtime.alloc", "out of memory", "malloc",
                              "memory allocation")):
        return True
    if "unreachable" not in low:
        return False
    return _pages(holder, store) * 65536 >= limits.memory_mb * 1024 * 1024 * 0.5


class _Virtual:
    # external: WASI preview1 — a poll_oneoff subscription is 48 bytes, an event 32
    SUB, EV = 48, 32

    def __init__(self, snapshot: fs.Snapshot, argv: Sequence[str], strict: bool):
        import hashlib
        self.strict = strict
        self.clock = STRICT_EPOCH_NS
        self.skipped = 0
        self.seed = hashlib.sha256(
            b"mechbench-sandbox-strict\0" + snapshot.digest().encode()
            + b"\0" + b"\0".join(a.encode() for a in argv)).digest()
        self.counter = 0

    def install(self, linker, store, module) -> None:
        import wasmtime

        names = set(ALWAYS_VIRTUAL) | (set(STRICT_VIRTUAL) if self.strict else set())
        linker.allow_shadowing = True
        for imp in module.imports:
            if imp.module != "wasi_snapshot_preview1" or imp.name not in names:
                continue
            if not isinstance(imp.type, wasmtime.FuncType):
                continue
            linker.define(store, "wasi_snapshot_preview1", imp.name,
                          wasmtime.Func(store, imp.type, getattr(self, imp.name),
                                        access_caller=True))

    def _now(self, clock_id: int) -> int:
        if self.strict:
            self.clock += STRICT_TICK_NS
            return self.clock
        base = time.time_ns() if clock_id == 0 else time.monotonic_ns()
        return base + self.skipped

    def _skip(self, ns: int) -> None:
        if self.strict:
            self.clock += ns
        else:
            self.skipped += ns

    def clock_time_get(self, caller, clock_id, _precision, out):
        import struct
        caller.get("memory").write(caller, struct.pack("<Q", self._now(clock_id)), out)
        return 0

    def clock_res_get(self, caller, _id, out):
        import struct
        caller.get("memory").write(caller, struct.pack("<Q", STRICT_TICK_NS), out)
        return 0

    def random_get(self, caller, buf, length):
        import hashlib
        chunks, need = [], length
        while need > 0:
            block = hashlib.sha256(self.seed + self.counter.to_bytes(8, "little")).digest()
            self.counter += 1
            chunks.append(block[:need])
            need -= len(block)
        caller.get("memory").write(caller, b"".join(chunks), buf)
        return 0

    def poll_oneoff(self, caller, subs, events, n, nevents_out):
        mem = caller.get("memory")
        for i in range(n):
            raw = bytes(mem.read(caller, subs + self.SUB * i, subs + self.SUB * (i + 1)))
            userdata, tag = raw[0:8], raw[8]
            if tag == 0:
                clock_id = int.from_bytes(raw[16:20], "little")
                timeout = int.from_bytes(raw[24:32], "little")
                absolute = int.from_bytes(raw[40:42], "little") & 1
                now = self._now(clock_id)
                deadline = timeout if absolute else now + timeout
                if deadline > now:
                    self._skip(deadline - now)
            event = userdata + b"\0\0" + bytes([tag]) + b"\0" * 5 + b"\0" * 16
            mem.write(caller, event, events + self.EV * i)
        mem.write(caller, int(n).to_bytes(4, "little"), nevents_out)
        return 0
