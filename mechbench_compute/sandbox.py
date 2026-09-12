"""The WASI sandbox runtime (task 000359, epic 000334).

One function, and its signature is the design:

    run(snapshot, argv) -> (snapshot', stdout, stderr, exit)

The guest sees exactly one directory — the materialized snapshot —
and nothing else. No network (wasmtime's WASI has no network API to
switch off; absence is the default state). CPU is fuel-metered, wall
clock is epoch-interrupted, memory is capped, the wasm call stack is
capped, output is capped. Every limit that trips is reported by name
in the result rather than as a truncated-looking success.

The guest module is compiled once per process and kept — compiling
the 15 MB standard-Go guest is 1.1 s, deserializing its compiled form
is 10 ms, and a tool call is otherwise 5 ms. One engine, one epoch
ticker, a store per run.

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

#: A second, empty preopen the guest can write its TRUE exit status
#: into. WASI hosts reject `proc_exit` outside [0, 126) — wasmtime
#: reports "invalid exit status" with the number discarded — and 127
#: is what every shell says for "command not found". A guest that
#: wants to exit ≥ 126 writes the number to `<EXIT_DIR>/exit` and
#: exits 125; the host reads the file if it is there. Outside the
#: snapshot root, so never captured; invisible to `ls /`.
EXIT_DIR = "/.mechbench"


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
    #: One of `fuel`, `wall_seconds`, `memory_mb`, `stack`, `max_files`,
    #: `max_bytes` — or None.
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
#: at startup: a language runtime reads the clock before `main` and
#: seeds its hash from `random_get` before the first line. A runtime
#: needs these to exist. What it does not need is for them to be real —
#: so strict hands it a clock that starts at a fixed instant and
#: advances one microsecond per read, and a byte stream seeded from the
#: snapshot and argv. The run is then a pure function of its inputs,
#: which is the whole point.
STRICT_VIRTUAL = ("clock_res_get", "random_get")

#: Replaced ALWAYS, strict or not. `poll_oneoff` is how a guest sleeps
#: or waits, and a guest blocked inside it is beyond the reach of epoch
#: interruption — the trap fires only when control returns to wasm, so
#: `sleep 10` under a 1 s wall cap ran for 10,002 ms (measured). It
#: cannot be denied either: Go's runtime waits inside its own GC path,
#: so a denial killed every guest that grew its heap. Instead every
#: wait completes at once AND the clock jumps forward by the wait —
#: without the jump, Go's scheduler re-reads the clock, finds the
#: deadline unmet, and polls again until real time catches up. So the
#: clock is always ours too: host time plus every wait the guest has
#: skipped, or the strict counter. Nothing a sandboxed guest waits FOR
#: can happen — no network, no other process, stdin is a file — so a
#: completed wait is the truth, not a lie, and `sleep 5` followed by a
#: timestamp reads five seconds later either way.
ALWAYS_VIRTUAL = ("poll_oneoff", "clock_time_get")

#: Where the strict clock starts: 2000-01-01T00:00:00Z, in nanoseconds.
#: Any fixed instant would do; this one is recognizable in a log.
STRICT_EPOCH_NS = 946_684_800 * 1_000_000_000
STRICT_TICK_NS = 1_000


#: How often the shared engine's epoch advances. Wall-clock caps are
#: rounded up to this.
EPOCH_TICK_S = 0.1

_ENGINE_LOCK = threading.Lock()
_ENGINE: Any = None
_MODULES: dict[tuple[str, int, int], Any] = {}


def _engine():
    """The one engine, with its epoch ticker. Stores are per run and
    set their deadline relative to the epoch when they start, so a
    single ticker serves any number of concurrent runs."""
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
    """The compiled guest: from memory, else from the serialized copy
    beside the guest cache, else compiled and both are filled. The
    on-disk form is keyed by guest identity and wasmtime version;
    deserializing trusts the bytes, so only our own cache dir is read."""
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
        except Exception:  # noqa: BLE001 — a stale or foreign artifact; recompile
            module = None
    if module is None:
        try:
            module = wasmtime.Module.from_file(engine, str(path))
        except Exception as e:  # noqa: BLE001 — a bad binary is a runtime fault
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
        wasi.env = [(k, v) for k, v in (env or {}).items()]
        wasi.preopen_dir(str(root), "/")
        wasi.preopen_dir(str(meta), EXIT_DIR)
        wasi.stdin_file = str(stdin_path)
        wasi.stdout_file = str(stdout_path)
        wasi.stderr_file = str(stderr_path)
        store.set_wasi(wasi)

        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        virtual = _Virtual(snapshot, argv, strict)
        virtual.install(linker, store, module)

        # Wall clock: the shared ticker advances the epoch every
        # EPOCH_TICK_S, and the store traps when its deadline passes.
        # Coarse, and enough.
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
                # The wasm call stack, wasmtime's default 512 KiB —
                # unbounded recursion in the guest, not its heap.
                limit = "stack"
            elif _out_of_memory(instance_holder, store, limits, text):
                limit = "memory_mb"
            else:
                stderr_path.write_bytes(stderr_path.read_bytes() + f"\n[trap] {text}".encode())
        except wasmtime.WasmtimeError as e:
            text = str(e)
            if "invalid exit status" in text:
                # The guest exited ≥ 126 without using the side channel.
                # 126 is the floor of what we know.
                exit_code = _reported_exit(meta, 126)
            elif "memory minimum size" in text and "exceeds" in text:
                # The cap is below what the guest declares it needs to
                # start at all — a limit the protocol set, so a Result.
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
        except Exception:  # noqa: BLE001 — fuel accounting unavailable after a trap
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


def _reported_exit(meta: pathlib.Path, fallback: int) -> int:
    """The status the guest wrote to the side channel, else `fallback`."""
    try:
        return int((meta / "exit").read_text().strip())
    except (OSError, ValueError):
        return fallback


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


def _pages(holder: list, store) -> int:
    if not holder:
        return 0
    try:
        mem = holder[0].exports(store).get("memory")
        return mem.size(store) if mem is not None else 0
    except Exception:  # noqa: BLE001 — a store mid-trap may refuse
        return 0


def _oom_exit(holder: list, store, limits: Limits, stderr: str) -> bool:
    """Go's runtime does not trap on a refused grow: it prints
    `fatal error: out of memory` and exits 2. The phrase alone is not
    evidence — a guest can print anything — so it counts only with the
    memory most of the way to the cap when the guest died."""
    if "out of memory" not in stderr.lower():
        return False
    return _pages(holder, store) * 65536 >= limits.memory_mb * 1024 * 1024 * 0.5


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
    if "unreachable" not in low:
        return False
    return _pages(holder, store) * 65536 >= limits.memory_mb * 1024 * 1024 * 0.5


class _Virtual:
    """Stand-ins for the WASI calls through which a guest sees the
    outside world: the clock, the RNG, and waiting.

    Waiting is always virtual (`ALWAYS_VIRTUAL`). The clock and RNG
    are virtual only under strict: a clock from `STRICT_EPOCH_NS`
    advancing `STRICT_TICK_NS` per read — advancing, so a guest that
    spins until time passes still finishes, and by a fixed step, so it
    finishes the same way every time — and SHA-256 in counter mode over
    a seed drawn from the snapshot digest and argv. WASI's errno for
    success is 0.
    """

    #: WASI preview1 wire layouts: subscription is 48 bytes, event 32.
    SUB, EV = 48, 32

    def __init__(self, snapshot: fs.Snapshot, argv: Sequence[str], strict: bool):
        import hashlib
        self.strict = strict
        self.clock = STRICT_EPOCH_NS   # strict: the counter clock
        self.skipped = 0               # plain: nanoseconds of waits skipped
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
        # 0 realtime, 1 monotonic, 2/3 cpu time — all move with the
        # host, all carry the skipped waits.
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
        """Every subscription fires now. A clock subscription (tag 0)
        moves the strict clock to its deadline; fd subscriptions (1, 2)
        report ready — every fd a guest has is a file."""
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
