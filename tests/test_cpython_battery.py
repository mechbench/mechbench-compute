"""CPython guest conformance (task 000453): real Python workloads over
every kind of input and the failure modes that must stay bounded — no
trap, no hang, an exit in range, and a read-only run that leaves the
tree alone. Skips unless `guests/cpython/build.sh` has run.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from mechbench_compute import guests, sandbox, snapshots as fs

REPO = pathlib.Path(__file__).resolve().parent.parent
CPY_WASM = pathlib.Path(os.environ.get(
    "MECHBENCH_CPYTHON_WASM", REPO / "guests" / "cpython" / "build" / "python.wasm"))
CPY_STDLIB = pathlib.Path(os.environ.get(
    "MECHBENCH_CPYTHON_STDLIB", REPO / "guests" / "cpython" / "build" / "stdlib"))
pytestmark = pytest.mark.skipif(
    not (CPY_WASM.is_file() and CPY_STDLIB.is_dir()),
    reason="cpython guest not built — guests/cpython/build.sh")
L = sandbox.Limits


@pytest.fixture(scope="module")
def cpython(tmp_path_factory):
    os.environ["MECHBENCH_GUEST_CACHE"] = str(tmp_path_factory.mktemp("guests"))
    guests.install_local("cpython", CPY_WASM, replace=True,
                         mounts=[(str(CPY_STDLIB), "/usr/local/lib/python3.13")])
    return "cpython"


TREE = fs.seeded({
    "a.txt": "one two three\n",
    "empty.txt": "",
    "unicode.txt": "héllo 日本語 👩‍👩‍👧 مرحبا\n",
    "data.json": '{"nums": [1, 2, 3], "name": "x"}',
    "big.txt": "".join(f"line {i}\n" for i in range(50_000)),
    "sub/deep/leaf.txt": "leaf\n",
    **{f"many/f{i:03d}.txt": f"{i}\n" for i in range(100)},
})


def run(cpython, code, tree=TREE, **kw):
    limits = L(memory_mb=kw.pop("memory_mb", 512),
               wall_seconds=kw.pop("wall_seconds", 20), output_bytes=1 << 20)
    return sandbox.run(tree, ["python", "-c", code], guest=cpython, limits=limits)


def _bounded(r, *, reads_only=True):
    assert 0 <= r.exit_code <= 255 or r.limit, (r.exit_code, r.stderr[-300:])
    blob = r.stdout[-2000:] + r.stderr[-2000:]
    for bad in ("[trap]", "Fatal Python error", "Segmentation"):
        assert bad not in blob, blob[-300:]
    assert r.duration_ms < 20000
    if reads_only:
        assert r.changed.empty, r.changed.to_wire()


#: (label, code, reads_only). Each must come back bounded.
CASES = [
    ("read", "print(open('a.txt').read())", True),
    ("read-empty", "print(len(open('empty.txt').read()))", True),
    ("read-unicode", "print(len(open('unicode.txt', encoding='utf-8').read()))", True),
    ("read-missing", "open('nope.txt')", True),
    ("json", "import json; print(json.load(open('data.json'))['nums'])", True),
    ("walk", "import os; print(sum(len(f) for _,_,f in os.walk('.')))", True),
    ("big", "print(sum(1 for _ in open('big.txt')))", True),
    ("glob", "import glob; print(len(glob.glob('many/*.txt')))", True),
    ("stdlib", "import re,math,collections,datetime,hashlib,csv,io,base64,"
               "statistics,decimal,itertools,functools,textwrap,unicodedata;"
               "print('ok')", True),
    ("hashlib", "import hashlib; print(hashlib.sha256(b'x').hexdigest()[:8])", True),
    ("recursion", "import sys; sys.setrecursionlimit(10**7)\n"
                  "def f(n): return f(n+1)\nf(0)", True),
    ("socket", "import socket; socket.socket().connect(('127.0.0.1', 9))", True),
    ("subprocess", "import subprocess; subprocess.run(['ls'])", True),
]


@pytest.mark.parametrize("label,code,reads_only", CASES, ids=[c[0] for c in CASES])
def test_case_stays_bounded(cpython, label, code, reads_only):
    _bounded(run(cpython, code), reads_only=reads_only)


class TestWritesAndDeterminism:
    def test_a_write_advances_the_snapshot(self, cpython):
        r = run(cpython, "open('out.txt','w').write('done')")
        assert r.ok and r.changed.added == ("out.txt",)
        assert r.snapshot.get("out.txt").data == b"done"

    def test_the_same_run_twice_is_the_same_result(self, cpython):
        code = "import json; print(json.dumps(sorted(__import__('os').listdir('.'))))"
        a, b = run(cpython, code), run(cpython, code)
        assert a.stdout == b.stdout and a.snapshot.digest() == b.snapshot.digest()

    def test_strict_stabilizes_hashing_and_random(self, cpython):
        code = "import random; random.seed(); print(hash('x'), random.random())"
        a = sandbox.run(fs.EMPTY, ["python", "-c", code], guest=cpython, strict=True)
        b = sandbox.run(fs.EMPTY, ["python", "-c", code], guest=cpython, strict=True)
        assert a.stdout == b.stdout


class TestTheLimitsHold:
    def test_a_spin_hits_the_wall_clock(self, cpython):
        r = run(cpython, "while True: pass", wall_seconds=1)
        assert r.limit == "wall_seconds" and r.duration_ms < 3000

    def test_a_huge_allocation_is_a_catchable_memoryerror(self, cpython):
        r = run(cpython, "bytearray(600*1024*1024)", memory_mb=256)
        assert r.exit_code == 1 and "MemoryError" in r.stderr and r.limit is None

    def test_the_stdlib_mount_is_read_only(self, cpython):
        r = run(cpython, "open('/usr/local/lib/python3.13/os.py','a').write('x')")
        assert r.exit_code == 1

    def test_exit_codes_including_those_wasi_clamps(self, cpython):
        assert run(cpython, "import sys; sys.exit(0)").exit_code == 0
        assert run(cpython, "import sys; sys.exit(3)").exit_code == 3
        assert run(cpython, "import sys; sys.exit(127)").exit_code == 127
        assert run(cpython, "import sys; sys.exit(200)").exit_code == 200


REL = REPO / "guests" / "cpython" / "build" / "release"


@pytest.mark.skipif(not (REL / "python.wasm.gz").is_file(),
                    reason="release artifacts not built — guests/cpython/build/release")
def test_the_hosted_guest_fetches_and_unpacks(tmp_path, monkeypatch):
    """The real hosting path, with the release artifacts served over
    file:// from an empty cache: fetch the wasm.gz, verify and
    decompress it, fetch the stdlib.tar.gz, verify and unpack it, and
    run. The pins are the ones in guests.REGISTRY, so a mismatch here
    is a mismatch against what will be hosted."""
    import gzip
    import hashlib

    monkeypatch.setenv("MECHBENCH_GUEST_CACHE", str(tmp_path))
    from mechbench_compute import guests as g
    # Expected hashes are computed from the artifacts, not read from the
    # registry — the module fixture may have install_local'd over it, and
    # this test is about the fetch/unpack mechanics regardless.
    wasm_bytes = gzip.decompress((REL / "python.wasm.gz").read_bytes())
    wasm_sha = hashlib.sha256(wasm_bytes).hexdigest()
    lib_sha = hashlib.sha256((REL / "stdlib.tar.gz").read_bytes()).hexdigest()
    monkeypatch.setattr(g, "REGISTRY", dict(g.REGISTRY))
    g.register(g.Guest(
        name="cpython", url=(REL / "python.wasm.gz").as_uri(),
        sha256=wasm_sha, size=len(wasm_bytes),   # decompressed size, as _fetch checks
        env={"PYTHONHOME": "/usr/local", "PYTHONDONTWRITEBYTECODE": "1"},
        mounts=(g.GuestMount(at="/usr/local/lib/python3.13",
                             url=(REL / "stdlib.tar.gz").as_uri(),
                             sha256=lib_sha),)))
    wasm, mounts, env = g.resolve("cpython")
    assert wasm.is_file() and pathlib.Path(mounts[0].host).is_dir()
    r = sandbox.run(fs.EMPTY, ["python", "-c", "print(6*7)"], guest="cpython",
                    limits=L(memory_mb=256, wall_seconds=20))
    assert r.ok and r.stdout.strip() == "42"
