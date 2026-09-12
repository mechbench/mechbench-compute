"""Guest conformance battery (task 000359): every applet against every
kind of input, asserting the robustness contract rather than outputs.

The contract: the guest never traps, never raises out of the runtime,
never panics (recovered or not), finishes inside a small budget, and
a read-only command leaves the tree alone. What a command PRINTS is
its business; that it comes back is ours. This is how coreutils got
hard — not by reasoning about each utility, but by feeding every one
of them everything. At a few milliseconds a run the whole matrix is
seconds, so it runs against the pinned hash on every push.

A new applet upstream fails `test_every_applet_is_covered` until it
is given a template here or an exclusion with a reason.
"""
from __future__ import annotations

import os
import pathlib
import random

import pytest

from mechbench_compute import guests, sandbox, snapshots as fs

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILT = pathlib.Path(os.environ.get(
    "MECHBENCH_MBSHELL_WASM", REPO / "guests" / "mbshell" / "build" / "mbshell.wasm"))
L = sandbox.Limits


@pytest.fixture(scope="module")
def guest(tmp_path_factory):
    if not BUILT.is_file():
        pytest.skip("mbshell.wasm not built on this machine — guests/mbshell/build.sh")
    os.environ["MECHBENCH_GUEST_CACHE"] = str(tmp_path_factory.mktemp("guests"))
    guests.install_local("mbshell", BUILT)   # refuses a build off the pin
    return "mbshell"


# ---------------------------------------------------------------- inputs

_rng = random.Random(20260912)

#: One tree with every kind of thing in it. `missing` is a path that
#: is not there.
BATTERY = fs.seeded({
    "text.txt": "one two three\nfour five\nsix\n",
    "empty.txt": "",
    "binary.bin": _rng.randbytes(64 * 1024),
    "large.txt": "".join(f"line {i:07d} {_rng.random():.12f}\n" for i in range(60_000)),
    "unicode.txt": ("héllo wörld\n日本語のテキスト\n👩‍👩‍👧‍👦 family\n"
                    "مرحبا بالعالم\né combining\n﻿BOM line\n"),
    "dir/inner.txt": "inside\n",
    "dir/sub/deeper.txt": "deeper\n",
    "deep/" + "/".join(f"d{i}" for i in range(40)) + "/leaf.txt": "leaf\n",
    **{f"many/f{i:03d}.txt": f"{i}\n" for i in range(200)},
})

KINDS = {
    "file": "text.txt", "empty": "empty.txt", "binary": "binary.bin",
    "large": "large.txt", "unicode": "unicode.txt", "directory": "dir",
    "deep": "deep", "many": "many", "missing": "missing",
}

# ------------------------------------------------------------- templates

#: (label, template, mutates). A list is run directly — busybox's own
#: dispatch — with X substituted; a string is `sh -c` with {x}.
X = "{x}"
TEMPLATES: list[tuple[str, list[str] | str, bool]] = [
    ("cat", ["cat", X], False),
    ("head", ["head", "-n", "2", X], False),
    ("head -c", ["head", "-c", "10", X], False),
    ("tail", ["tail", "-n", "2", X], False),
    ("wc", ["wc", X], False),
    ("wc -c", ["wc", "-c", X], False),
    ("sort", ["sort", X], False),
    ("sort -rn", ["sort", "-r", "-n", X], False),
    ("uniq", ["uniq", X], False),
    ("cut -c", ["cut", "-c1-3", X], False),
    ("cut -f", ["cut", "-d", " ", "-f1", X], False),
    ("grep -c", ["grep", "-c", "e", X], False),
    ("grep -rl", ["grep", "-r", "-l", "e", X], False),
    ("sed p", ["sed", "-n", "1p", X], False),
    ("sed s", ["sed", "s/e/E/g", X], False),
    ("awk NF", ["awk", "{print NF}", X], False),
    ("awk sum", ["awk", "{n+=length($0)} END {print n}", X], False),
    ("tr", "tr a-z A-Z < {x}", False),
    ("diff", ["diff", X, "text.txt"], False),
    ("diff -u", ["diff", "-u", "text.txt", X], False),
    ("gzip -c", ["gzip", "-c", X], False),
    ("gunzip -c", ["gunzip", "-c", X], False),
    ("tar c", ["tar", "cf", "-", X], False),
    ("tar t", ["tar", "tf", X], False),
    ("ls -la", ["ls", "-la", X], False),
    ("ls -R", ["ls", "-R", X], False),
    ("find", ["find", X], False),
    ("find -name", ["find", X, "-name", "*.txt"], False),
    ("find -type", ["find", X, "-type", "f"], False),
    ("echo", ["echo", X], False),
    ("printf", ["printf", "%s\\n", X], False),
    ("time", ["time", "cat", X], False),
    ("timeout", ["timeout", "1", "cat", X], False),
    ("xargs", "xargs echo < {x}", False),
    ("xargs -n1", "cat {x} | xargs -n1 echo", False),
    ("sh script", ["sh", X], False),
    ("sh pipeline", "cat {x} | wc -l", False),
    ("sh redirect", "grep -c e < {x} > /dev/null; echo $?", False),
    ("cp -r", ["cp", "-r", X, "out"], True),
    ("mv", ["mv", X, "out"], True),
    ("rm -r", ["rm", "-r", X], True),
    ("rmdir", ["rmdir", X], True),
    ("mkdir", ["mkdir", X], True),
    ("mkdir -p", ["mkdir", "-p", X + "/made"], True),
    ("gzip", ["gzip", X], True),
    ("gunzip", ["gunzip", X], True),
    ("sed -i", ["sed", "-i", "s/e/E/", X], True),
]

#: Applets that take no path. Run once each.
SOLO: list[list[str]] = [
    ["pwd"], ["nproc"], ["whoami"], ["logname"], ["users"], ["who"], ["w"],
    ["uptime"], ["free"], ["ps"], ["ss"],
    ["pidof", "sh"], ["pgrep", "sh"], ["pkill", "nosuch"], ["killall", "nosuch"],
    ["kill", "1"], ["kill", "-l"],
    ["nice", "-n", "5", "echo", "hi"], ["nohup", "echo", "hi"], ["setsid", "echo", "hi"],
    ["ionice", "-c", "2", "echo", "hi"], ["taskset", "1", "echo", "hi"],
    ["renice", "1", "1"], ["start-stop-daemon", "--help"],
    ["sleep", "0"], ["sleep", "5"],
    ["busybox"], ["busybox", "nosuch"],
    ["sh", "-c", "exit 127"], ["sh", "-c", "set -e; false"],
]

#: Network applets: nothing to connect to, ever. Must fail, fast.
NETWORK: list[list[str]] = [
    ["wget", "-O", "-", "http://127.0.0.1:9/"],
    ["nc", "-z", "127.0.0.1", "9"],
    ["dig", "localhost"],
]

#: Left out on purpose, with the reason.
EXCLUDED = {
    "top": "loops by design; with waits virtual it is a tight loop until the wall cap",
    "watch": "loops by design; same",
    "ash": "the same applet as sh",
}


def _argv(template: list[str] | str, x: str) -> list[str]:
    if isinstance(template, str):
        return ["sh", "-c", template.format(x=x)]
    return [a.replace(X, x) for a in template]


def _first_applet(template: list[str] | str) -> str:
    return template.split()[0] if isinstance(template, str) else template[0]


def _check(r: sandbox.Result, argv: list[str], kind: str, mutates: bool,
           limits_ok: tuple[str | None, ...] = (None,)) -> None:
    where = f"{argv} on {kind}"
    assert r.limit in limits_ok, f"{where}: hit {r.limit}: {r.stderr[-300:]}"
    assert 0 <= r.exit_code <= 255, f"{where}: exit {r.exit_code}: {r.stderr[-300:]}"
    text = r.stdout[-2000:] + r.stderr[-2000:]
    for bad in ("[trap]", "panic:", "fatal error:", "internal error", "goroutine "):
        assert bad not in text, f"{where}: {bad!r} in output: {text[-400:]}"
    assert r.duration_ms < 5000, f"{where}: {r.duration_ms} ms"
    if not mutates:
        assert r.changed.empty, f"{where}: a read-only command changed {r.changed.to_wire()}"


# ----------------------------------------------------------------- tests

@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("label,template,mutates", TEMPLATES, ids=[t[0] for t in TEMPLATES])
def test_applet_on_input(guest, label, template, mutates, kind):
    argv = _argv(template, KINDS[kind])
    r = sandbox.run(BATTERY, argv, guest=guest, limits=L(wall_seconds=10))
    _check(r, argv, kind, mutates)


@pytest.mark.parametrize("argv", SOLO, ids=[" ".join(a) for a in SOLO])
def test_applet_alone(guest, argv):
    r = sandbox.run(BATTERY, argv, guest=guest, limits=L(wall_seconds=10))
    _check(r, argv, "-", mutates=False)


@pytest.mark.parametrize("argv", NETWORK, ids=[a[0] for a in NETWORK])
def test_network_applet_fails_fast(guest, argv):
    r = sandbox.run(BATTERY, argv, guest=guest, limits=L(wall_seconds=3))
    _check(r, argv, "-", mutates=False)
    assert r.exit_code != 0 and r.duration_ms < 1000


def test_unbounded_recursion_is_a_named_limit(guest):
    # A shell function that calls itself forever. The interpreter
    # recurses on the wasm call stack, which wasmtime caps; the trap
    # is named, never generic.
    r = sandbox.run(BATTERY, ["sh", "-c", "f() { f; }; f"], guest=guest,
                    limits=L(memory_mb=64, wall_seconds=10))
    assert r.limit == "stack", (r.exit_code, r.stderr[-300:])
    assert "[trap]" not in r.stderr


def test_every_applet_is_covered(guest):
    r = sandbox.run(fs.EMPTY, ["busybox"], guest=guest)
    listed = set(r.stdout.split("\n")[1].split())
    covered = ({_first_applet(t) for _, t, _ in TEMPLATES}
               | {a[0] for a in SOLO} | {a[0] for a in NETWORK}
               | {"tr", "xargs", "sh"})
    assert listed - covered - set(EXCLUDED) == set(), "new applets without a template"
    assert set(EXCLUDED) <= listed, "an exclusion names an applet that is gone"
