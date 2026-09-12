"""The WASI sandbox runtime (task 000359) against the real guest.

Every test here runs go-busybox compiled to wasip1 — the actual
binary, not a stub — so what passes is what a protocol would get.
Skipped when the guest is not on this machine.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from mechbench_compute import guests, sandbox, snapshots as fs

BUILT = pathlib.Path(os.environ.get(
    "MECHBENCH_BUSYBOX_WASM",
    "/private/tmp/claude-501/-Users-benji-dev-redthreadlabs-evalcreativity/"
    "7062e481-d49e-40e7-b5a5-9752562db670/scratchpad/go-busybox/build/busybox.wasm"))
pytestmark = pytest.mark.skipif(not BUILT.is_file(),
                                reason="busybox.wasm not built on this machine")
L = sandbox.Limits


@pytest.fixture(scope="module")
def guest(tmp_path_factory):
    """The guest's registered NAME, so every run below resolves the
    way a protocol's would. install_local hashes the local build
    against the pin in `guests.REGISTRY`: a build that comes out
    different fails here, on purpose, rather than running under the
    pinned name."""
    os.environ["MECHBENCH_GUEST_CACHE"] = str(tmp_path_factory.mktemp("guests"))
    guests.install_local("busybox", BUILT)
    return "busybox"


SEED = fs.seeded({"a.txt": "one two three\n", "sub/b.txt": "four\nfive\n",
                  "c.md": "not a txt\n"})


class TestTheLoopWorks:
    def test_an_applet_reads_the_snapshot(self, guest):
        r = sandbox.run(SEED, ["find", ".", "-name", "*.txt"], guest=guest)
        assert r.ok and sorted(r.stdout.split()) == ["a.txt", "sub/b.txt"]

    def test_a_write_shows_in_the_capture(self, guest):
        r = sandbox.run(SEED, ["cp", "a.txt", "copied.txt"], guest=guest)
        assert r.ok
        assert r.changed.added == ("copied.txt",)
        assert r.snapshot.get("copied.txt").blob_hash == SEED.get("a.txt").blob_hash

    def test_the_input_snapshot_is_untouched(self, guest):
        before = SEED.digest()
        sandbox.run(SEED, ["rm", "a.txt"], guest=guest)
        assert SEED.digest() == before, "a run mutated its input value"

    def test_a_read_only_run_changes_nothing(self, guest):
        r = sandbox.run(SEED, ["wc", "-l", "a.txt"], guest=guest)
        assert r.ok and r.changed.empty and r.snapshot.digest() == SEED.digest()

    def test_the_same_run_twice_is_the_same_result(self, guest):
        a = sandbox.run(SEED, ["grep", "-r", "four", "."], guest=guest)
        b = sandbox.run(SEED, ["grep", "-r", "four", "."], guest=guest)
        assert (a.stdout, a.exit_code, a.snapshot.digest()) == (b.stdout, b.exit_code, b.snapshot.digest())

    def test_a_path_works_too(self, guest):
        r = sandbox.run(SEED, ["echo", "hi"], guest=BUILT)
        assert r.ok and r.stdout == "hi\n"

    def test_an_unregistered_name_says_so(self, guest):
        with pytest.raises(sandbox.SandboxError, match="no guest named 'cpython'"):
            sandbox.run(SEED, ["python3"], guest="cpython")

    def test_a_missing_applet_is_an_ordinary_failure(self, guest):
        r = sandbox.run(SEED, ["python3", "-c", "1"], guest=guest)
        assert r.exit_code != 0 and r.limit is None
        assert "applet not found" in r.stderr


class TestEveryLimitIsNamed:
    def test_fuel(self, guest):
        big = fs.seeded({"big.bin": os.urandom(4 << 20)})
        r = sandbox.run(big, ["gzip", "-c", "big.bin"], guest=guest,
                        limits=L(fuel=20_000_000, output_bytes=8 << 20))
        assert r.limit == "fuel" and r.fuel_used == 20_000_000

    def test_memory(self, guest):
        lines = fs.seeded({"l.txt": "".join(f"{os.urandom(8).hex()}\n" for _ in range(200_000))})
        r = sandbox.run(lines, ["sort", "l.txt"], guest=guest,
                        limits=L(memory_mb=4, output_bytes=1 << 20))
        assert r.limit == "memory_mb", r.stderr[-300:]

    def test_output(self, guest):
        r = sandbox.run(SEED, ["busybox"], guest=guest, limits=L(output_bytes=100))
        assert "stdout" in r.truncated and sandbox.TRUNCATED.format(n=100) in r.stdout

    def test_a_blocking_call_is_refused_at_once(self, guest):
        r = sandbox.run(SEED, ["sleep", "30"], guest=guest, limits=L(wall_seconds=1))
        assert r.limit == "blocked_call" and r.duration_ms < 1000

    def test_strict_makes_the_run_a_function_of_its_inputs(self, guest):
        # The "applet not found" listing prints a Go map, and Go seeds
        # map order from random_get — so plain runs shuffle it and a
        # strict run must not. A better witness than a clock applet:
        # it is the language runtime consuming entropy, not a program.
        plain = {sandbox.run(SEED, ["busybox", "nope"], guest=guest).stdout
                 for _ in range(4)}
        strict = {sandbox.run(SEED, ["busybox", "nope"], guest=guest, strict=True).stdout
                  for _ in range(4)}
        assert len(plain) > 1, "this guest no longer shuffles; find another witness"
        assert len(strict) == 1

    def test_strict_still_runs_ordinary_programs(self, guest):
        # The first cut DENIED the clock, and nothing ran: the runtime
        # reads it before main. Strict must be invisible to a program
        # that never asks the time.
        r = sandbox.run(SEED, ["find", ".", "-name", "*.txt"], guest=guest, strict=True)
        assert r.ok and sorted(r.stdout.split()) == ["a.txt", "sub/b.txt"]

    def test_the_tree_cap_refuses_rather_than_truncating(self, guest):
        r = sandbox.run(SEED, ["cp", "a.txt", "d.txt"], guest=guest,
                        limits=L(max_files=3))
        assert r.limit == "max_files" and r.snapshot.digest() == SEED.digest()


class TestWhatTheSandboxCannotDo:
    """Proven, not assumed."""

    def test_no_network(self, guest):
        r = sandbox.run(fs.EMPTY, ["wget", "-O", "x", "http://example.com/"],
                        guest=guest, limits=L(wall_seconds=5))
        assert r.exit_code != 0 and "x" not in r.snapshot.paths()

    def test_no_shell_in_this_guest(self, guest):
        # go-busybox stubs ash out under wasm: the native shell forks a
        # process per pipeline stage and WASI preview1 cannot spawn one.
        r = sandbox.run(SEED, ["sh", "-c", "echo hi"], guest=guest)
        assert r.exit_code != 0 and "not supported in wasm" in r.stderr

    def test_awk_panics_in_this_guest(self, guest):
        # TinyGo lacks reflect.Type.NumIn, which goawk needs. Recorded
        # so the next reader does not spend an hour rediscovering it.
        r = sandbox.run(SEED, ["awk", "{print NF}", "a.txt"], guest=guest)
        assert r.exit_code != 0 and r.limit is None
