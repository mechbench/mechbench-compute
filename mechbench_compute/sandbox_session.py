from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from mechbench_compute import sandbox
from mechbench_compute import snapshots as fs

DEFAULT_GUEST = "mbshell"

PYTHON_GUEST = "cpython"

TOOL_NAMES = ("bash", "find", "grep", "python", "read_file", "write_file", "list")

DEFAULT_TOOLS = ("bash", "read_file", "write_file", "list")


class SandboxRefused(ValueError):
    pass


@dataclass(frozen=True)
class SandboxImage:
    base: str = DEFAULT_GUEST
    tools: tuple[str, ...] = DEFAULT_TOOLS
    limits: sandbox.Limits = field(default_factory=sandbox.Limits)
    strict: bool = False
    snapshot: fs.Snapshot = fs.EMPTY
    mounts: tuple[fs.Mount, ...] = ()
    object_mounts: tuple[tuple[str, fs.Snapshot], ...] = ()

    def __post_init__(self) -> None:
        unknown = [t for t in self.tools if t not in TOOL_NAMES]
        if unknown:
            raise SandboxRefused(
                f"image offers unknown tool(s) {unknown}; known: "
                f"{', '.join(TOOL_NAMES)}")

    @staticmethod
    def parse(value: Any) -> SandboxImage:
        if isinstance(value, SandboxImage):
            return value
        if not value:
            return SandboxImage()
        if not isinstance(value, Mapping):
            raise SandboxRefused(
                "a sandbox image is an object {base, tools, limits, strict, "
                f"snapshot, mounts}}, not {type(value).__name__}")
        limits = value.get("limits")
        lim = (limits if isinstance(limits, sandbox.Limits)
               else sandbox.Limits(**dict(limits)) if limits else sandbox.Limits())
        snap = value.get("snapshot")
        tree = fs.EMPTY if snap is None else _as_tree(snap)
        refs: list[fs.Mount] = []
        resolved: list[tuple[str, fs.Snapshot]] = []
        for m in (value.get("mounts") or ()):
            if isinstance(m, tuple) and len(m) == 2 and isinstance(m[1], fs.Snapshot):
                resolved.append((str(m[0]), m[1]))
                continue
            if isinstance(m, fs.Mount):
                refs.append(m)
                continue
            at = str(m["path"] if "path" in m else m["at"])
            src = m.get("snapshot")
            if src is not None:
                resolved.append((at, _as_tree(src)))
            elif m.get("object"):
                refs.append(fs.Mount(at=at, object=str(m["object"]),
                                     digest=str(m.get("digest", ""))))
            else:
                raise SandboxRefused(
                    f"mount at {at!r} needs a `snapshot` (a tree) or an "
                    f"`object` (a bench ref)")
        return SandboxImage(
            base=str(value.get("base") or DEFAULT_GUEST),
            tools=tuple(value.get("tools") or DEFAULT_TOOLS),
            limits=lim, strict=bool(value.get("strict", False)),
            snapshot=tree, mounts=tuple(refs), object_mounts=tuple(resolved))


@dataclass
class SandboxCall:
    tool: str
    argv: tuple[str, ...]
    exit_code: int
    snapshot_in: str
    snapshot_out: str
    duration_ms: int
    stdin: str = ""
    limit: str | None = None
    changed: Mapping[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tool": self.tool, "argv": list(self.argv),
            "exit_code": self.exit_code,
            "snapshot_in": self.snapshot_in, "snapshot_out": self.snapshot_out,
            "duration_ms": self.duration_ms,
        }
        if self.stdin:
            out["stdin"] = self.stdin
        if self.limit:
            out["limit"] = self.limit
        if self.changed:
            out["changed"] = dict(self.changed)
        if self.stdout:
            out["stdout"] = self.stdout
        if self.stderr:
            out["stderr"] = self.stderr
        return out


class SandboxSession:
    def __init__(self, image: SandboxImage | None = None, *,
                 guest: str | None = None) -> None:
        self.image = image or SandboxImage()
        self.snapshot = _with_mounts(self.image.snapshot, self.image.mounts)
        self.guest = guest or self.image.base
        self.calls: list[SandboxCall] = []

    def final_wire(self, inline_cap: int = 1 << 20) -> dict[str, Any] | None:
        working = replace(self.snapshot, mounts=())
        if not working.entries:
            return None
        inline = working.n_bytes <= inline_cap
        return working.to_wire(inline=inline)

    def bash(self, command: str, stdin: str = "") -> str:
        return self._run(("sh", "-c", str(command)), tool="bash", stdin=stdin)

    def find(self, path: str = ".", name: str | None = None,
             type: str | None = None) -> str:
        argv = ["find", str(path)]
        if name:
            argv += ["-name", str(name)]
        if type:
            argv += ["-type", str(type)]
        return self._run(tuple(argv), tool="find")

    def grep(self, pattern: str, path: str = ".", recursive: bool = True) -> str:
        argv = ["grep", "-n"]
        if recursive:
            argv.append("-r")
        argv += ["-e", str(pattern), str(path)]
        return self._run(tuple(argv), tool="grep")

    def python(self, code: str = "", script: str = "") -> str:
        if script:
            argv = ("python", str(script))
        elif code:
            argv = ("python", "-c", str(code))
        else:
            return "python: give either `code` (a snippet) or `script` (a path)"
        return self._run(argv, tool="python", guest=PYTHON_GUEST)

    def read_file(self, path: str) -> str:
        entry = self.snapshot.get(_norm(path))
        if entry is None:
            self._record("read_file", ("cat", str(path)), exit_code=1,
                         out="", err=f"read_file: {path}: no such file")
            return f"read_file: {path}: no such file"
        data = self._blob(entry)
        text = data.decode("utf-8", errors="replace")
        self._record("read_file", ("cat", str(path)), exit_code=0, out=text)
        return text

    def write_file(self, path: str, content: str) -> str:
        p = _norm(path)
        if not p or p.endswith("/"):
            msg = f"write_file: {path}: not a file path"
            self._record("write_file", ("write_file", str(path)), exit_code=1,
                         err=msg)
            return msg
        data = content.encode("utf-8")
        before = self.snapshot
        entry = fs.Entry(path=p, size=len(data), blob_hash=fs.blob_hash(data),
                         data=data if len(data) <= fs.INLINE_MAX else None)
        kept = tuple(e for e in before.entries if e.path != p)
        blobs = dict(before.blobs)
        if len(data) > fs.INLINE_MAX:
            blobs[entry.blob_hash] = data
        after = fs.Snapshot((*kept, entry), before.mounts, blobs)
        self._advance(before, after, "write_file", ("write_file", p), exit_code=0,
                      out=f"wrote {len(data)} bytes to {p}")
        return f"wrote {len(data)} bytes to {p}"

    def list(self, path: str = ".") -> str:
        prefix = "" if _norm(path) in ("", ".") else _norm(path).rstrip("/") + "/"
        names = sorted({e.path for e in self.snapshot.entries
                        if e.path.startswith(prefix)})
        listing = "\n".join(names)
        self._record("list", ("ls", str(path)), exit_code=0, out=listing)
        return listing

    def _run(self, argv: tuple[str, ...], *, tool: str, stdin: str = "",
             guest: str | None = None) -> str:
        before = self.snapshot
        try:
            result = sandbox.run(before, list(argv), guest=guest or self.guest,
                                  limits=self.image.limits, strict=self.image.strict,
                                  stdin=stdin, mounts=self.image.object_mounts)
        except sandbox.SandboxError as e:
            raise SandboxRefused(str(e)) from e
        self.snapshot = result.snapshot
        self.calls.append(SandboxCall(
            tool=tool, argv=argv, exit_code=result.exit_code,
            snapshot_in=before.digest(), snapshot_out=result.snapshot.digest(),
            duration_ms=result.duration_ms, stdin=stdin, limit=result.limit,
            changed=result.changed.to_wire() if not result.changed.empty else {},
            stdout=result.stdout, stderr=result.stderr))
        return _for_model(result.stdout, result.stderr, result.exit_code, result.limit)

    def _advance(self, before: fs.Snapshot, after: fs.Snapshot, tool: str,
                 argv: tuple[str, ...], *, exit_code: int, out: str = "",
                 err: str = "") -> None:
        self.snapshot = after
        changed = fs.diff(before, after)
        self.calls.append(SandboxCall(
            tool=tool, argv=argv, exit_code=exit_code,
            snapshot_in=before.digest(), snapshot_out=after.digest(),
            duration_ms=0, changed=changed.to_wire() if not changed.empty else {},
            stdout=out, stderr=err))

    def _record(self, tool: str, argv: tuple[str, ...], *, exit_code: int,
                out: str = "", err: str = "") -> None:
        d = self.snapshot.digest()
        self.calls.append(SandboxCall(
            tool=tool, argv=argv, exit_code=exit_code, snapshot_in=d,
            snapshot_out=d, duration_ms=0, stdout=out, stderr=err))

    def _blob(self, entry: fs.Entry) -> bytes:
        if entry.data is not None:
            return entry.data
        blob = self.snapshot.blobs.get(entry.blob_hash)
        if blob is None:
            raise SandboxRefused(
                f"file {entry.path!r} references blob {entry.blob_hash} which "
                "is not in the session — read_file needs the content inline or "
                "in the snapshot's blob sidecar")
        return blob

    def tool_defs(self, names: Sequence[str] | None = None) -> list[dict[str, Any]]:
        chosen = tuple(names) if names is not None else self.image.tools
        return [dict(TOOL_DEFS[n]) for n in chosen]


def _as_tree(value: Any) -> fs.Snapshot:
    if isinstance(value, fs.Snapshot):
        return value
    if isinstance(value, Mapping) and value.get("kind") in (fs.KIND, fs.LEGACY_KIND):
        return fs.Snapshot.from_wire(value)
    if isinstance(value, Mapping):
        return fs.seeded({str(k): v for k, v in value.items()})
    raise SandboxRefused("a tree must be a Snapshot or a {path: content} map")


def _norm(path: str) -> str:
    p = str(path).strip()
    for pre in ("./", "/"):
        while p.startswith(pre):
            p = p[len(pre):]
    return p


def _with_mounts(snapshot: fs.Snapshot, mounts: tuple[fs.Mount, ...]) -> fs.Snapshot:
    if not mounts:
        return snapshot
    return replace(snapshot, mounts=tuple(mounts))


def _for_model(stdout: str, stderr: str, exit_code: int, limit: str | None) -> str:
    parts = [stdout]
    if stderr:
        parts.append(stderr if stdout.endswith("\n") or not stdout
                     else "\n" + stderr)
    text = "".join(parts)
    if limit:
        text += f"\n[stopped: {limit}]"
    elif exit_code != 0:
        text += f"\n[exit {exit_code}]"
    return text if text else f"[exit {exit_code}]"


TOOL_DEFS: dict[str, dict[str, Any]] = {
    "bash": {
        "name": "bash",
        "description": "Run a shell command over the workspace and return "
                       "its output. Supports pipes, redirects, loops and "
                       "command substitution.",
        "schema": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The shell command."},
            "stdin": {"type": "string", "description": "Optional standard input."}},
            "required": ["command"]},
        "handler": {"sandbox": "bash"},
    },
    "find": {
        "name": "find",
        "description": "List files under a path, optionally filtered by name "
                       "pattern or type.",
        "schema": {"type": "object", "properties": {
            "path": {"type": "string"},
            "name": {"type": "string", "description": "Glob, e.g. *.txt"},
            "type": {"type": "string", "enum": ["f", "d"]}}},
        "handler": {"sandbox": "find"},
    },
    "grep": {
        "name": "grep",
        "description": "Search files for a pattern; returns matching lines "
                       "with their paths and line numbers.",
        "schema": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "recursive": {"type": "boolean"}},
            "required": ["pattern"]},
        "handler": {"sandbox": "grep"},
    },
    "python": {
        "name": "python",
        "description": "Run Python 3 over the workspace: a code snippet, "
                       "or a script file already in it. The interpreter "
                       "shares the files with the shell tools.",
        "schema": {"type": "object", "properties": {
            "code": {"type": "string", "description": "A snippet to run with -c."},
            "script": {"type": "string", "description": "A script path to run."}}},
        "handler": {"sandbox": "python"},
    },
    "read_file": {
        "name": "read_file",
        "description": "Return the contents of a file.",
        "schema": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]},
        "handler": {"sandbox": "read_file"},
    },
    "write_file": {
        "name": "write_file",
        "description": "Write text to a file, creating or replacing it.",
        "schema": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]},
        "handler": {"sandbox": "write_file"},
    },
    "list": {
        "name": "list",
        "description": "List the paths in the workspace under a prefix.",
        "schema": {"type": "object", "properties": {
            "path": {"type": "string"}}},
        "handler": {"sandbox": "list"},
    },
}
