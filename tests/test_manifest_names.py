from __future__ import annotations

import hashlib

import pytest

from mechbench_compute import checkpoint, snapshots, tensors
from mechbench_compute.contained import check_file_name, write_inside
from mechbench_compute.protocol.resolver import Resolver

CRAFTED = ["../x", "/etc/passwd", "a/../../b", "", ".", "..", ".bashrc", "a/b",
           "..\\x", "x\0y", None, 7]


def refuse_fetch(name):
    raise AssertionError(f"fetched {name!r}")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("name", CRAFTED)
def test_a_crafted_name_is_not_a_file_name(name):
    with pytest.raises(ValueError, match="not a plain file name"):
        check_file_name(name, what="file")


@pytest.mark.parametrize("name", ["model-00001-of-00002.safetensors", "config.json",
                                  "tokenizer.model", "shard_0003.safetensors"])
def test_a_plain_name_is_accepted(name):
    assert check_file_name(name, what="file") == name


@pytest.mark.parametrize("name", CRAFTED)
def test_a_checkpoint_refuses_before_opening_anything(tmp_path, name):
    cache = tmp_path / "cache"
    manifest = {"files": [{"name": "ok.json", "size": 1, "sha256": sha(b"x")},
                          {"name": name, "size": 1, "sha256": sha(b"x")}]}
    with pytest.raises(ValueError, match="not a plain file name"):
        checkpoint.materialize(manifest, refuse_fetch, cache)
    assert not cache.exists()


@pytest.mark.parametrize("name", CRAFTED)
def test_a_tensor_collection_refuses_before_opening_anything(tmp_path, name):
    cache = tmp_path / "cache"
    collection = {"storage": tensors.STORAGE,
                  "shards": [{"name": name, "rows": 1, "size": 1, "sha256": sha(b"x")}]}
    with pytest.raises(ValueError, match="not a plain file name"):
        tensors.materialize(collection, "you/lab/t", refuse_fetch, cache)
    assert not cache.exists()


@pytest.mark.parametrize("name", ["../x", "/etc/passwd"])
def test_reading_or_uploading_shards_refuses_a_crafted_name(tmp_path, name):
    collection = {"storage": tensors.STORAGE, tensors.LOCAL_DIR: str(tmp_path),
                  "shards": [{"name": name, "rows": 1, "sha256": "x"}]}
    with pytest.raises(ValueError, match="not a plain file name"):
        tensors.items_of(collection)
    with pytest.raises(ValueError, match="not a plain file name"):
        tensors.upload(collection, "you/lab/t", refuse_fetch)


def test_a_checkpoint_materializes_through_a_rename(tmp_path):
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    manifest = {"files": [{"name": n, "size": len(b), "sha256": sha(b)}
                          for n, b in files.items()]}
    target = checkpoint.materialize(manifest, lambda n: files[n], tmp_path)
    assert {p.name for p in target.iterdir()} == {*files, ".complete"}
    assert (target / "model.safetensors").read_bytes() == b"weights"


def test_a_wrong_hash_leaves_no_file(tmp_path):
    assert not write_inside(tmp_path, "a.bin", [b"abc"], want=sha(b"other"))
    assert list(tmp_path.iterdir()) == []


def test_a_symlinked_name_cannot_leave_the_target(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    target = tmp_path / "target"
    target.mkdir()
    (target / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="resolves outside"):
        write_inside(target, "link", [b"evil"], want=sha(b"evil"))
    assert outside.read_bytes() == b"keep"


@pytest.mark.parametrize("path", ["../x", "/etc/passwd", "a/../../b", "", "a//b", "./a"])
def test_a_snapshot_entry_path_is_refused_before_anything_is_written(tmp_path, path):
    good = snapshots.seeded({"first.txt": "ok"}).entries[0]
    bad = snapshots.Entry(path=path, size=1, blob_hash=snapshots.blob_hash(b"x"),
                          executable=False, data=b"x")
    snap = snapshots.Snapshot((good, bad), (), {})
    root = tmp_path / "root"
    with pytest.raises(snapshots.SnapshotLimit, match="escapes the root"):
        snapshots.materialize(snap, root)
    assert list(root.iterdir()) == []


def test_a_nested_snapshot_entry_is_fine(tmp_path):
    snap = snapshots.seeded({"a/b/c.txt": "ok"})
    snapshots.materialize(snap, tmp_path)
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "ok"


def test_a_protocol_cannot_supply_a_local_shard_dir():
    r = Resolver(bound_params={"t": {"storage": "tensor", tensors.LOCAL_DIR: "/Users"}})
    inline = {"storage": "tensor", tensors.LOCAL_DIR: "/etc", "shards": []}
    assert tensors.LOCAL_DIR not in r.resolve_value(inline)
    assert tensors.LOCAL_DIR not in r.resolve_value({"$param": "t"})


@pytest.mark.parametrize("where", ["../secret", "/etc/passwd", "a/../../secret", "a//b"])
def test_a_conformance_input_stays_under_its_root(tmp_path, where):
    from mechbench_compute.conformance import read_inputs_from

    root = tmp_path / "inputs"
    root.mkdir()
    (tmp_path / "secret.json").write_text("{}")
    resolve = read_inputs_from(root)
    for key in ("file", "bench"):
        with pytest.raises(ValueError):
            resolve("x/op", {"records": {"$ref": {key: where}}})


def test_a_conformance_input_through_a_symlink_is_refused(tmp_path):
    from mechbench_compute.conformance import read_inputs_from

    root = tmp_path / "inputs"
    (root / "you").mkdir(parents=True)
    (tmp_path / "secret.json").write_text("{}")
    (root / "you" / "lab.json").symlink_to(tmp_path / "secret.json")
    with pytest.raises(ValueError, match="resolves outside"):
        read_inputs_from(root)("x/op", {"records": {"$ref": {"bench": "you/lab"}}})


def test_a_conformance_input_under_the_root_is_read(tmp_path):
    from mechbench_compute.conformance import read_inputs_from

    (tmp_path / "you" / "lab").mkdir(parents=True)
    (tmp_path / "you" / "lab" / "prompts.json").write_text('{"kind": "x"}')
    got = read_inputs_from(tmp_path)("x/op", {"records": {"$ref": {"bench": "you/lab/prompts"}}})
    assert got == {"records": {"kind": "x"}}
