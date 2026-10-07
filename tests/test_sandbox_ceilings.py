from __future__ import annotations

import math
import os
import pathlib

import pytest

from mechbench_compute import guests, sandbox
from mechbench_compute import snapshots as fs
from mechbench_compute.blocks.run_guest_json import run_guest_json
from mechbench_compute.sandbox_ceilings import Ceilings, read_ceilings, set_ceilings
from mechbench_compute.sandbox_io import CappedFile, DiskQuota
from mechbench_compute.sandbox_session import SandboxImage

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILT = pathlib.Path(os.environ.get(
    "MECHBENCH_MBSHELL_WASM", REPO / "guests" / "mbshell" / "build" / "mbshell.wasm"))
L = sandbox.Limits


@pytest.fixture(scope="module")
def guest(tmp_path_factory):
    if not BUILT.is_file():
        pytest.skip("mbshell.wasm not built on this machine — guests/mbshell/build.sh")
    os.environ["MECHBENCH_GUEST_CACHE"] = str(tmp_path_factory.mktemp("guests"))
    guests.install_local("mbshell", BUILT)
    return "mbshell"


@pytest.fixture
def ceilings():
    def lower(**given):
        set_ceilings(Ceilings(**given))
    yield lower
    set_ceilings(None)


class TestALimitIsAPositiveNumber:
    @pytest.mark.parametrize("given", [
        {"memory_mb": -1}, {"memory_mb": 0}, {"fuel": -5}, {"output_bytes": 0},
        {"max_files": True}, {"memory_mb": 1.5}, {"wall_seconds": math.nan},
        {"wall_seconds": math.inf}, {"wall_seconds": 0}, {"max_bytes": "9"},
    ])
    def test_anything_else_is_refused(self, given):
        with pytest.raises(ValueError, match=next(iter(given))):
            L(**given)

    def test_a_chat_image_refuses_a_negative_memory_limit(self):
        with pytest.raises(ValueError, match="memory_mb"):
            SandboxImage.parse({"limits": {"memory_mb": -1}})

    def test_the_defaults_are_within_the_default_ceilings(self):
        assert sandbox.clamp_limits(L(), Ceilings()) == (L(), {})


class TestTheRunnerSetsTheCeilings:
    def test_a_limit_above_a_ceiling_is_lowered_to_it(self):
        got, capped = sandbox.clamp_limits(L(memory_mb=6144, wall_seconds=5), Ceilings())
        assert got.memory_mb == 4096 and got.wall_seconds == 5
        assert capped == {"memory_mb": 4096}

    def test_a_limit_below_a_ceiling_is_kept(self):
        limits = L(memory_mb=32, fuel=10, wall_seconds=0.5, output_bytes=7)
        assert sandbox.clamp_limits(limits, Ceilings()) == (limits, {})

    def test_the_environment_sets_them(self, monkeypatch):
        monkeypatch.setenv("MECHBENCH_SANDBOX_CEILING_MEMORY_MB", "128")
        monkeypatch.setenv("MECHBENCH_SANDBOX_CEILING_WALL_SECONDS", "2.5")
        got = read_ceilings()
        assert (got.memory_mb, got.wall_seconds, got.fuel) == (128, 2.5, Ceilings().fuel)

    def test_a_bad_environment_value_is_named(self, monkeypatch):
        monkeypatch.setenv("MECHBENCH_SANDBOX_CEILING_FUEL", "lots")
        with pytest.raises(ValueError, match="MECHBENCH_SANDBOX_CEILING_FUEL"):
            read_ceilings()
        monkeypatch.setenv("MECHBENCH_SANDBOX_CEILING_FUEL", "-1")
        with pytest.raises(ValueError, match="fuel"):
            read_ceilings()

    def test_set_ceilings_wins_over_the_environment(self, monkeypatch, ceilings):
        monkeypatch.setenv("MECHBENCH_SANDBOX_CEILING_MEMORY_MB", "128")
        ceilings(memory_mb=64)
        assert read_ceilings().memory_mb == 64


class TestARunStaysUnderTheCeilings:
    def test_a_protocol_cannot_raise_the_wall_clock(self, guest, ceilings):
        ceilings(wall_seconds=1)
        r = sandbox.run(fs.EMPTY, ["sh", "-c", "while true; do :; done"], guest=guest,
                        limits=L(wall_seconds=600))
        assert r.limit == "wall_seconds" and r.duration_ms < 5000
        assert "this runner caps wall_seconds at 1" in r.stderr

    def test_a_records_op_names_the_ceiling(self, guest, ceilings):
        ceilings(wall_seconds=1)
        with pytest.raises(ValueError, match="this runner caps wall_seconds at 1"):
            run_guest_json("records/jq", "mbshell", ["sh", "-c", "while true; do :; done"], [],
                           memory_mb=64, seconds=600, output_mb=1)

    def test_output_past_the_ceiling_is_cut(self, guest, ceilings):
        ceilings(output_bytes=1000)
        r = sandbox.run(fs.EMPTY, ["sh", "-c", "i=0; while [ $i -lt 500 ]; do echo line$i; i=$((i+1)); done"], guest=guest,
                        limits=L(output_bytes=1 << 20))
        assert "stdout" in r.truncated and sandbox.TRUNCATED.format(n=1000) in r.stdout

    def test_a_guest_that_fills_the_disk_is_stopped(self, guest, ceilings):
        ceilings(disk_bytes=4 << 20)
        r = sandbox.run(fs.EMPTY, ["sh", "-c", "while true; do echo xxxxxxxxxxxxxxxx; done > big"],
                        guest=guest, limits=L(wall_seconds=20))
        assert r.limit == "disk_bytes" and r.exit_code == -1
        assert r.changed.empty and r.duration_ms < 15000


class TestTheHostKeepsItsOwnMemory:
    def test_output_is_read_only_to_the_cap(self, tmp_path):
        out = CappedFile(tmp_path / "stdout", 10)
        for _ in range(100):
            out(b"x" * 4096)
        assert out.read() == b"x" * 11

    def test_the_quota_counts_the_output_files(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        stdout = tmp_path / "stdout"
        stdout.write_bytes(os.urandom(64 * 1024))
        quota = DiskQuota(root, max_bytes=1024, max_entries=100, files=(str(stdout),))
        assert quota.check() and stdout.stat().st_size == 0

    def test_a_guest_that_floods_stdout_is_stopped_at_the_disk_ceiling(self, guest, ceilings):
        ceilings(disk_bytes=4 << 20)
        r = sandbox.run(fs.EMPTY, ["sh", "-c", "while true; do echo xxxxxxxxxxxxxxxx; done"],
                        guest=guest, limits=L(wall_seconds=20))
        assert r.limit == "disk_bytes" and r.duration_ms < 15000

    def test_the_quota_counts_and_truncates_only_inside_its_root(self, tmp_path):
        root, outside = tmp_path / "root", tmp_path / "outside"
        root.mkdir()
        outside.write_bytes(b"keep me" * 1000)
        (root / "link").symlink_to(outside)
        (root / "big").write_bytes(os.urandom(64 * 1024))
        quota = DiskQuota(root, max_bytes=1024, max_entries=100)
        assert quota.check() and quota.tripped
        assert (root / "big").stat().st_size == 0
        assert outside.read_bytes() == b"keep me" * 1000


class _Memory:
    def __init__(self, size):
        self.data = bytearray(size)

    def data_len(self, _caller):
        return len(self.data)

    def write(self, _caller, value, start):
        self.data[start:start + len(value)] = value

    def read(self, _caller, start, stop):
        return self.data[start:stop]


class _Caller:
    def __init__(self, size):
        self.memory = _Memory(size)

    def get(self, _name):
        return self.memory


class TestTheHostFunctionsCheckBoundsFirst:
    def test_random_get_past_the_memory_is_efault(self):
        v = sandbox._Virtual(fs.EMPTY, ["x"], strict=True)
        assert v.random_get(_Caller(65536), 65000, 4_000_000_000) == v.EFAULT
        assert v.counter == 0

    def test_a_signed_pointer_reads_as_unsigned(self):
        v = sandbox._Virtual(fs.EMPTY, ["x"], strict=True)
        assert v.random_get(_Caller(65536), -16, 8) == v.EFAULT

    def test_random_get_inside_the_memory_writes(self):
        v = sandbox._Virtual(fs.EMPTY, ["x"], strict=True)
        caller = _Caller(65536)
        assert v.random_get(caller, 100, 40) == 0
        assert any(caller.memory.data[100:140]) and not any(caller.memory.data[140:])

    def test_poll_oneoff_past_the_memory_is_efault(self):
        v = sandbox._Virtual(fs.EMPTY, ["x"], strict=True)
        assert v.poll_oneoff(_Caller(65536), 0, 0, 100_000_000, 0) == v.EFAULT
