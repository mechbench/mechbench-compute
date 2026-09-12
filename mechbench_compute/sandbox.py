"""The WASI sandbox runtime (task 000359, epic 000334).

One function, and its signature is the design:

    run(snapshot, argv) -> (snapshot', stdout, stderr, exit)

The guest sees exactly one directory — the materialized snapshot —
and nothing else. No network (wasmtime's WASI has no network API to
switch off; absence is the default state). CPU is fuel-metered, wall
clock is epoch-interrupted, memory is capped, output is capped. Every
limit that trips is reported by name in the result rather than as a
truncated-looking success.

Architecture A, chosen on measurement (see 000359): the guest's file
I/O goes through wasmtime's WASI in Rust straight to the host, and
Python is never entered per syscall. The boundary — materialize
before, capture after — is what a call pays, and 000358 measured it.

The runtime takes a SNAPSHOT, not a path, on purpose. Whether it
materializes into a directory or, someday, serves the tree from the
content store is its own business, and nothing above it should know.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mechbench_compute import guests
from mechbench_compute import snapshots as fs

#: Marker appended when captured output exceeds its cap. Explicit, so a
#: model reading the result knows it saw a prefix and not the whole.
TRUNCATED = "\n[output truncated at {n} bytes]"


@dataclass(frozen=True)
class Limits:
    """Every ceiling a guest runs under. Defaults are meant for a tool
    call, not a build: generous for scripts, useless for mining."""

    memory_mb: int = 256
    #: wasmtime fuel, roughly one unit per wasm instruction. Calibrated
    #: on this machine: gzip over 8 MB burned 2.4e8 in 26 ms, so about
    #: 1e10 per second of tight compute. 1e11 is ~10 s — enough for a
    #: script, useless for mining.
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
    """What one command did. `snapshot` is the tree AFTER, `changed`
    is what it touched, and `limit` names the ceiling that ended it,
    if one did — `None` means the guest finished on its own."""

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
    """The runtime itself could not run the guest — a missing binary,
    a module that will not instantiate. Distinct from a guest that ran
    and failed, which is an ordinary Result with a non-zero exit."""


#: WASI preview1 imports that strict mode REPLACES with deterministic
#: ones. Each is a source of nondeterminism a reproducible re-run cannot
#: tolerate: a guest that reads the clock or the RNG can produce
#: different bytes from the same snapshot, and then the snapshot chain
#: is a record of nothing.
#:
#: Replaced, not denied. The first cut denied them, and every run died
#: at startup: the TinyGo runtime reads the clock before `main`, and
#: CPython seeds its hash from `random_get` before the first line of
#: the script. A language runtime needs these to exist. What it does
#: not need is for them to be real — so strict hands it a clock that
#: starts at a fixed instant and advances one microsecond per read, and
#: a byte stream seeded from the snapshot and argv. The run is then a
#: pure function of its inputs, which is the whole point.
STRICT_VIRTUAL = ("clock_time_get", "clock_res_get", "random_get")

#: Where the strict clock starts: 2000-01-01T00:00:00Z, in nanoseconds.
#: Any fixed instant would do; this one is recognizable in a log.
STRICT_EPOCH_NS = 946_684_800 * 1_000_000_000
STRICT_TICK_NS = 1_000

#: Denied ALWAYS. `poll_oneoff` is how a guest sleeps or waits, and a
#: guest blocked inside it is beyond the reach of epoch interruption —
#: the trap fires only when control returns to wasm, so `sleep 10`
#: under a 1 s wall cap ran for 10,002 ms (measured). Nothing a tool
#: call legitimately does needs to wait on the wall clock; stdin is a
#: file, so no read ever blocks. Denied, sleep fails in 0 ms.
ALWAYS_DENIED = ("poll_oneoff",)


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
        mount_blobs: Mapping[str, Mapping[str, bytes]] | None = None) -> Result:
    """Run `argv` inside `guest` over `snapshot`; return what happened.

    `guest` is a registered name (`"busybox"`), resolved through
    `guests.ensure` — fetched on first use and verified by hash — or a
    path to a `.wasm` file for a build under test. `argv[0]` is the
    applet name — busybox dispatches on it, so `["find", ".", …]` runs
    find. `strict=True` replaces the clock and the RNG with
    deterministic ones, so the same snapshot and argv give the same
    bytes back every time — see `STRICT_VIRTUAL`.
    """
    import wasmtime

    limits = limits or Limits()
    guest_path = _resolve_guest(guest)

    with tempfile.TemporaryDirectory(prefix="mechbench-sandbox-") as td:
        root = pathlib.Path(td) / "root"
        fs.materialize(snapshot, root, blobs=blobs)  # falls back to snapshot.blobs
        stdout_path = pathlib.Path(td) / "stdout"
        stderr_path = pathlib.Path(td) / "stderr"
        stdin_path = pathlib.Path(td) / "stdin"
        stdin_path.write_bytes(stdin.encode() if isinstance(stdin, str) else bytes(stdin))

        config = wasmtime.Config()
        config.consume_fuel = True
        config.epoch_interruption = True
        engine = wasmtime.Engine(config)
        try:
            module = wasmtime.Module.from_file(engine, str(guest_path))
        except Exception as e:  # noqa: BLE001 — a bad binary is a runtime fault
            raise SandboxError(f"guest {guest_path.name} will not load: {e}") from e

        store = wasmtime.Store(engine)
        store.set_fuel(limits.fuel)
        store.set_limits(memory_size=limits.memory_mb * 1024 * 1024)

        wasi = wasmtime.WasiConfig()
        wasi.argv = list(argv)
        wasi.env = [(k, v) for k, v in (env or {}).items()]
        wasi.preopen_dir(str(root), "/")
        wasi.stdin_file = str(stdin_path)
        wasi.stdout_file = str(stdout_path)
        wasi.stderr_file = str(stderr_path)
        store.set_wasi(wasi)

        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        _deny(linker, store, module, ALWAYS_DENIED, "blocked_call")
        if strict:
            _virtualize(linker, store, module, snapshot, argv)

        # Wall clock: the epoch ticks once per 100 ms from a thread, and
        # the store traps when its deadline passes. Coarse, and enough.
        deadline_ticks = max(1, int(limits.wall_seconds * 10))
        store.set_epoch_deadline(deadline_ticks)
        stop = threading.Event()

        def tick() -> None:
            while not stop.wait(0.1):
                engine.increment_epoch()

        ticker = threading.Thread(target=tick, daemon=True)
        ticker.start()

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
            exit_code = int(e.code)
        except wasmtime.Trap as e:
            text = str(e)
            low = text.lower()
            exit_code = -1
            if "fuel" in low:
                limit = "fuel"
            elif "epoch" in low or "interrupt" in low:
                limit = "wall_seconds"
            elif "[blocked_call]" in text:
                limit = "blocked_call"
            elif _out_of_memory(instance_holder, store, limits, text):
                limit = "memory_mb"
            else:
                stderr_path.write_bytes(stderr_path.read_bytes() + f"\n[trap] {text}".encode())
        finally:
            stop.set()
            ticker.join(timeout=1.0)
        duration_ms = int((time.monotonic() - started) * 1000)
        fuel_left = None
        try:
            fuel_left = store.get_fuel()
        except Exception:  # noqa: BLE001 — fuel accounting unavailable after a trap
            pass

        truncated: list[str] = []
        out = _cap(stdout_path.read_bytes(), limits.output_bytes, "stdout", truncated)
        err = _cap(stderr_path.read_bytes(), limits.output_bytes, "stderr", truncated)
        try:
            after = fs.capture(root, max_files=limits.max_files,
                               max_bytes=limits.max_bytes,
                               mounts=snapshot.mounts)
        except fs.SnapshotLimit as e:
            # The guest wrote more than the sandbox allows. Report it
            # as the limit it is; the tree is not captured, because a
            # truncated capture would be a snapshot of nothing real.
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


def _resolve_guest(guest: str | os.PathLike[str]) -> pathlib.Path:
    """A registered name goes through the loader; anything else is a
    path. A name that is neither registered nor a file is reported as
    the former — `run(guest="cpython")` before CPython is pinned should
    say "no guest named cpython", not "file not found: cpython"."""
    if isinstance(guest, str) and guests.is_registered(guest):
        try:
            return guests.ensure(guest)
        except guests.GuestUnavailable as e:
            raise SandboxError(str(e)) from e
    path = pathlib.Path(guest)
    if path.is_file():
        return path
    if isinstance(guest, str) and os.sep not in guest and not guest.endswith(".wasm"):
        raise SandboxError(
            f"no guest named {guest!r} is registered — known: "
            f"{', '.join(sorted(guests.REGISTRY)) or '(none)'}")
    raise SandboxError(f"guest binary not found: {path}")


def _out_of_memory(holder: list, store, limits: Limits, trap_text: str) -> bool:
    """Did the guest die because it hit the memory cap?

    An allocator that cannot grow does not trap with a memory error.
    TinyGo takes its fatal path, which is a plain `unreachable`, and a
    genuine guest panic (awk's `reflect` gap, say) looks the same in
    the trap text. Two pieces of evidence tell them apart:

    - the allocator is in the backtrace. Inside a capped linear memory
      an allocator fails for exactly one reason, so `runtime.alloc` (or
      `malloc`, for a C-built guest) above the trap IS the cap;
    - failing that, the memory was most of the way to the cap when the
      guest died. A doubling allocator trips at half; measured, TinyGo's
      `sort` died at 56 of 64 pages.
    """
    low = trap_text.lower()
    if any(m in low for m in ("runtime.alloc", "out of memory", "malloc",
                              "memory allocation")):
        return True
    if not holder or "unreachable" not in low:
        return False
    try:
        mem = holder[0].exports(store).get("memory")
        pages = mem.size(store) if mem is not None else 0
    except Exception:  # noqa: BLE001 — a store mid-trap may refuse
        return False
    return pages * 65536 >= limits.memory_mb * 1024 * 1024 * 0.5


def _deny(linker, store, module, names: Sequence[str], reason: str) -> None:
    """Shadow selected WASI imports with traps.

    Defined AFTER `define_wasi()` with shadowing allowed, so ours win.
    A guest calling one gets a trap that names the reason, not a zero
    it might mistake for a timestamp or a completed wait.
    """
    import wasmtime

    linker.allow_shadowing = True
    for imp in module.imports:
        if imp.module != "wasi_snapshot_preview1" or imp.name not in names:
            continue
        ftype = imp.type
        if not isinstance(ftype, wasmtime.FuncType):
            continue
        name = imp.name

        def trap(*_args, _name=name, _reason=reason):
            raise wasmtime.Trap(
                f"[{_reason}] {_name} is denied — a guest must not block "
                f"on the wall clock")

        linker.define(store, "wasi_snapshot_preview1", name,
                      wasmtime.Func(store, ftype, trap))


def _virtualize(linker, store, module, snapshot: fs.Snapshot,
                argv: Sequence[str]) -> None:
    """Replace the clock and the RNG with deterministic stand-ins.

    The clock starts at `STRICT_EPOCH_NS` and advances `STRICT_TICK_NS`
    per read — advancing, so a guest that spins until time passes
    still finishes, and by a fixed step, so it finishes the same way
    every time. The RNG is SHA-256 in counter mode over a seed drawn
    from the snapshot digest and argv: different inputs get different
    bytes, the same inputs get the same bytes, and neither is the
    host's entropy. WASI's errno for success is 0.
    """
    import hashlib
    import struct

    import wasmtime

    linker.allow_shadowing = True
    clock = [STRICT_EPOCH_NS]
    seed = hashlib.sha256(b"mechbench-sandbox-strict\0" + snapshot.digest().encode()
                          + b"\0" + b"\0".join(a.encode() for a in argv)).digest()
    counter = [0]

    def clock_time_get(caller, _id, _precision, out):
        clock[0] += STRICT_TICK_NS
        caller.get("memory").write(caller, struct.pack("<Q", clock[0]), out)
        return 0

    def clock_res_get(caller, _id, out):
        caller.get("memory").write(caller, struct.pack("<Q", STRICT_TICK_NS), out)
        return 0

    def random_get(caller, buf, length):
        chunks = []
        need = length
        while need > 0:
            block = hashlib.sha256(seed + counter[0].to_bytes(8, "little")).digest()
            counter[0] += 1
            chunks.append(block[:need])
            need -= len(block)
        caller.get("memory").write(caller, b"".join(chunks), buf)
        return 0

    stand_ins = {"clock_time_get": clock_time_get, "clock_res_get": clock_res_get,
                 "random_get": random_get}
    for imp in module.imports:
        if imp.module != "wasi_snapshot_preview1" or imp.name not in STRICT_VIRTUAL:
            continue
        if not isinstance(imp.type, wasmtime.FuncType):
            continue
        linker.define(store, "wasi_snapshot_preview1", imp.name,
                      wasmtime.Func(store, imp.type, stand_ins[imp.name],
                                    access_caller=True))
