"""Filesystem snapshots as values (task 000358).

The whole sandbox rests on one property: identical content must
produce an identical hash, on any machine, in any order. Most of what
follows tests that, because everything above it is meaningless if it
does not hold.
"""
from __future__ import annotations

import os
import pathlib
import time

import pytest

from mechbench_compute import snapshots as fs


def write(root: pathlib.Path, files: dict[str, str], mode: int | None = None):
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        if mode is not None:
            p.chmod(mode)
    return root


class TestTheHashIsContentAndNothingElse:
    def test_same_content_same_digest(self, tmp_path):
        a = write(tmp_path / "a", {"x.txt": "hello", "d/y.txt": "world"})
        b = write(tmp_path / "b", {"x.txt": "hello", "d/y.txt": "world"})
        assert fs.capture(a).digest() == fs.capture(b).digest()

    def test_mtime_does_not_change_the_digest(self, tmp_path):
        a = write(tmp_path / "a", {"x.txt": "hello"})
        before = fs.capture(a).digest()
        os.utime(a / "x.txt", (0, 0))
        assert fs.capture(a).digest() == before, "a snapshot carried a timestamp"

    def test_creation_order_does_not_change_the_digest(self, tmp_path):
        a = tmp_path / "a"; a.mkdir()
        for name in ("c.txt", "a.txt", "b.txt"):
            (a / name).write_text(name)
            time.sleep(0.001)
        b = tmp_path / "b"; b.mkdir()
        for name in ("a.txt", "b.txt", "c.txt"):
            (b / name).write_text(name)
        assert fs.capture(a).digest() == fs.capture(b).digest()

    def test_entries_come_back_sorted(self, tmp_path):
        a = write(tmp_path / "a", {"z.txt": "1", "a.txt": "2", "m/q.txt": "3"})
        assert list(fs.capture(a).paths()) == sorted(fs.capture(a).paths())

    def test_different_content_different_digest(self, tmp_path):
        a = write(tmp_path / "a", {"x.txt": "hello"})
        b = write(tmp_path / "b", {"x.txt": "hello!"})
        assert fs.capture(a).digest() != fs.capture(b).digest()

    def test_the_same_bytes_at_a_different_path_differ(self, tmp_path):
        a = write(tmp_path / "a", {"x.txt": "hello"})
        b = write(tmp_path / "b", {"y.txt": "hello"})
        assert fs.capture(a).digest() != fs.capture(b).digest()

    def test_the_executable_bit_is_content_enough_to_count(self, tmp_path):
        a = write(tmp_path / "a", {"run.sh": "echo hi"}, mode=0o644)
        plain = fs.capture(a).digest()
        (a / "run.sh").chmod(0o755)
        assert fs.capture(a).digest() != plain

    def test_inlining_is_a_storage_choice_not_an_identity_one(self, tmp_path):
        a = write(tmp_path / "a", {"x.txt": "hello"})
        big = fs.capture(a, inline_max=0)
        small = fs.capture(a, inline_max=1024)
        assert big.entries[0].data is None and small.entries[0].data is not None
        assert big.digest() == small.digest()


class TestRoundTrip:
    def test_capture_materialize_capture_is_stable(self, tmp_path):
        src = write(tmp_path / "src", {"a.txt": "one", "d/b.txt": "two"})
        snap = fs.capture(src)
        out = tmp_path / "out"
        fs.materialize(snap, out)
        assert fs.capture(out).digest() == snap.digest()
        assert (out / "d/b.txt").read_text() == "two"

    def test_the_executable_bit_survives(self, tmp_path):
        src = write(tmp_path / "src", {"run.sh": "echo hi"}, mode=0o755)
        out = tmp_path / "out"
        fs.materialize(fs.capture(src), out)
        assert os.access(out / "run.sh", os.X_OK)

    def test_a_large_blob_from_the_wire_needs_its_store(self, tmp_path):
        # In-process, a captured snapshot carries its large blobs. Once
        # it has crossed the wire the sidecar is gone — that is where a
        # store becomes mandatory, and where the error must be loud.
        src = write(tmp_path / "src", {"big.txt": "x" * 100})
        blobs: dict[str, bytes] = {}
        snap = fs.capture(src, inline_max=10, blobs=blobs)
        crossed = fs.Snapshot.from_wire(snap.to_wire())
        with pytest.raises(KeyError, match="not inline"):
            fs.materialize(crossed, tmp_path / "out")
        fs.materialize(crossed, tmp_path / "out2", blobs=blobs)
        assert (tmp_path / "out2" / "big.txt").read_text() == "x" * 100

    def test_a_corrupt_store_is_refused_not_written(self, tmp_path):
        src = write(tmp_path / "src", {"big.txt": "x" * 100})
        blobs: dict[str, bytes] = {}
        snap = fs.capture(src, inline_max=10, blobs=blobs)
        for k in blobs:
            blobs[k] = b"not what was captured"
        with pytest.raises(ValueError, match="does not match its hash"):
            fs.materialize(snap, tmp_path / "out", blobs=blobs)

    def test_the_wire_form_round_trips(self, tmp_path):
        src = write(tmp_path / "src", {"a.txt": "one", "d/b.txt": "two"})
        snap = fs.capture(src)
        back = fs.Snapshot.from_wire(snap.to_wire())
        assert back.digest() == snap.digest()
        assert back.paths() == snap.paths()

    def test_a_foreign_object_is_refused(self):
        with pytest.raises(ValueError, match="not a filesystem snapshot"):
            fs.Snapshot.from_wire({"kind": "records", "records": []})


class TestDiff:
    def test_it_names_exactly_what_changed(self, tmp_path):
        a = fs.seeded({"keep.txt": "same", "edit.txt": "before",
                       "gone.txt": "x"})
        b = fs.seeded({"keep.txt": "same", "edit.txt": "after",
                       "new.txt": "y"})
        d = fs.diff(a, b)
        assert d.added == ("new.txt",)
        assert d.removed == ("gone.txt",)
        assert d.changed == ("edit.txt",)

    def test_a_rewrite_with_the_same_bytes_is_not_a_change(self, tmp_path):
        a = fs.seeded({"x.txt": "same"})
        b = fs.seeded({"x.txt": "same"})
        assert fs.diff(a, b).empty

    def test_chmod_alone_is_a_change(self):
        a = fs.seeded({"run.sh": "echo"})
        b = fs.seeded({"run.sh": "echo"}, executable=["run.sh"])
        assert fs.diff(a, b).changed == ("run.sh",)


class TestLimits:
    def test_too_many_files_refuses(self, tmp_path):
        src = write(tmp_path / "src", {f"f{i}.txt": "x" for i in range(5)})
        with pytest.raises(fs.SnapshotLimit, match="more than 3 files"):
            fs.capture(src, max_files=3)

    def test_too_many_bytes_refuses_and_says_where(self, tmp_path):
        src = write(tmp_path / "src", {"a.txt": "x" * 50, "b.txt": "y" * 50})
        with pytest.raises(fs.SnapshotLimit, match="more than 60 bytes"):
            fs.capture(src, max_bytes=60)

    def test_a_symlink_out_of_the_tree_is_refused(self, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        src = tmp_path / "src"; src.mkdir()
        (src / "link.txt").symlink_to(outside)
        with pytest.raises(fs.SnapshotLimit, match="leaves the snapshot root"):
            fs.capture(src)


class TestTheEmptyTree:
    def test_it_is_a_constant_with_a_stable_digest(self, tmp_path):
        empty = tmp_path / "empty"; empty.mkdir()
        assert fs.capture(empty).digest() == fs.EMPTY.digest()
        assert fs.EMPTY.n_files == 0 and fs.EMPTY.n_bytes == 0

    def test_directories_alone_do_not_make_a_tree_nonempty(self, tmp_path):
        root = tmp_path / "d"
        (root / "a" / "b").mkdir(parents=True)
        assert fs.capture(root).digest() == fs.EMPTY.digest()


class TestOneCanonicalOrder:
    """`capture` walks depth-first and `from_wire` reads a list. Both
    must produce the same object, or the digest depends on how the
    snapshot was built rather than on what is in it."""

    def test_capture_and_from_wire_agree(self, tmp_path):
        src = write(tmp_path / "src",
                    {"z.txt": "1", "a.txt": "2", "m/q.txt": "3"})
        snap = fs.capture(src)
        assert snap.digest() == fs.Snapshot.from_wire(snap.to_wire()).digest()

    def test_construction_order_is_irrelevant(self):
        a = fs.seeded({"z.txt": "1", "a.txt": "2"})
        shuffled = fs.Snapshot(tuple(reversed(a.entries)))
        assert shuffled.digest() == a.digest()
        assert shuffled.paths() == a.paths()

    def test_a_nested_path_sorts_with_the_rest(self, tmp_path):
        src = write(tmp_path / "src",
                    {"z.txt": "1", "a.txt": "2", "m/q.txt": "3"})
        assert fs.capture(src).paths() == ("a.txt", "m/q.txt", "z.txt")


class TestMountsAreNotCaptured:
    """A read-only mount cannot have changed, so re-hashing it after
    every tool call is the dominant cost on a large corpus and buys
    nothing. Its identity is the object it came from."""

    def test_a_mounted_tree_is_skipped(self, tmp_path):
        root = write(tmp_path / "r", {"work.txt": "mine",
                                      "data/big.txt": "x" * 1000,
                                      "data/more.txt": "y" * 1000})
        m = fs.Mount(at="data", object="benji/lab/corpus", digest="sha256:abc")
        snap = fs.capture(root, mounts=[m])
        assert snap.paths() == ("work.txt",)
        assert snap.n_bytes == len("mine")
        assert snap.mounts == (m,)

    def test_the_mount_still_changes_the_digest(self, tmp_path):
        root = write(tmp_path / "r", {"work.txt": "mine"})
        a = fs.capture(root, mounts=[fs.Mount("data", "benji/lab/corpus-a")])
        b = fs.capture(root, mounts=[fs.Mount("data", "benji/lab/corpus-b")])
        assert a.digest() != b.digest(), "the mount is part of the state"

    def test_without_the_mount_the_files_come_back(self, tmp_path):
        root = write(tmp_path / "r", {"work.txt": "mine", "data/big.txt": "x"})
        assert len(fs.capture(root).entries) == 2

    def test_mounts_round_trip(self, tmp_path):
        root = write(tmp_path / "r", {"work.txt": "mine", "data/x.txt": "y"})
        snap = fs.capture(root, mounts=[fs.Mount("data", "benji/lab/c", "sha256:d")])
        back = fs.Snapshot.from_wire(snap.to_wire())
        assert back.mounts == snap.mounts and back.digest() == snap.digest()


class TestTheStoredFormIsReferences:
    """Measured: 2000 files at 4K is 8.43 MB with blobs inline and
    0.22 MB as hashes. A session emits one snapshot per tool call."""

    def test_blobs_are_not_inlined_by_default(self, tmp_path):
        root = write(tmp_path / "r", {"a.txt": "hello"})
        wire = fs.capture(root).to_wire()
        assert "data" not in wire["entries"][0]
        assert wire["entries"][0]["blob_hash"].startswith("sha256:")

    def test_inline_is_available_for_a_fixture(self, tmp_path):
        root = write(tmp_path / "r", {"a.txt": "hello"})
        wire = fs.capture(root).to_wire(inline=True)
        assert wire["entries"][0]["data"] == b"hello"

    def test_the_digest_is_the_same_either_way(self, tmp_path):
        root = write(tmp_path / "r", {"a.txt": "hello"})
        snap = fs.capture(root)
        assert (fs.Snapshot.from_wire(snap.to_wire()).digest()
                == fs.Snapshot.from_wire(snap.to_wire(inline=True)).digest())

    def test_the_reference_form_is_dramatically_smaller(self, tmp_path):
        root = write(tmp_path / "r", {f"f{i}.txt": "x" * 500 for i in range(40)})
        snap = fs.capture(root)
        from mechbench_schema import dump_canonical
        refs = len(dump_canonical(snap.to_wire()))
        inline = len(dump_canonical(snap.to_wire(inline=True)))
        assert inline > refs * 3, f"inline {inline} vs refs {refs}"


class TestBlobsRideWithTheValue:
    """A snapshot from `seeded()` or `capture()` carries its large
    blobs in-process, so it can be materialized without the caller
    threading a store through every call. Not identity, not wire."""

    def test_a_seeded_large_file_materializes_without_a_store(self, tmp_path):
        snap = fs.seeded({"big.bin": b"x" * (fs.INLINE_MAX + 1)})
        fs.materialize(snap, tmp_path / "out")
        assert (tmp_path / "out" / "big.bin").stat().st_size == fs.INLINE_MAX + 1

    def test_the_sidecar_is_not_identity(self, tmp_path):
        a = fs.seeded({"big.bin": b"x" * (fs.INLINE_MAX + 1)})
        b = fs.Snapshot(a.entries)  # same entries, no sidecar
        assert a.digest() == b.digest() and a == b

    def test_the_sidecar_is_not_on_the_wire(self):
        snap = fs.seeded({"big.bin": b"x" * (fs.INLINE_MAX + 1)})
        assert "blobs" not in snap.to_wire()
        assert "data" not in snap.to_wire()["entries"][0]

    def test_an_explicit_store_wins(self, tmp_path):
        snap = fs.seeded({"big.bin": b"x" * (fs.INLINE_MAX + 1)})
        wrong = {k: b"y" * len(v) for k, v in snap.blobs.items()}
        with pytest.raises(ValueError, match="does not match its hash"):
            fs.materialize(snap, tmp_path / "out", blobs=wrong)
