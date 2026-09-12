"""The sandbox catalog (task 000361): the kind schemas are the contract
the composer (000341) and the UI trace/browser (000362) share, so they
are tested against the ACTUAL wire shapes compute emits — not against a
hand-written example that can drift from the code.
"""
from __future__ import annotations

import jsonschema
import pytest

from mechbench_compute import sandbox
from mechbench_compute import sandbox_kinds as sk
from mechbench_compute import snapshots as fs
from mechbench_compute.sandbox_session import SandboxImage, SandboxSession


def _valid(schema):
    # The schemas themselves must be well-formed JSON Schema.
    jsonschema.Draft202012Validator.check_schema(schema)


class TestTheSchemasAreValid:
    @pytest.mark.parametrize("schema", [
        sk.FS_SNAPSHOT_SCHEMA, sk.SANDBOX_IMAGE_SCHEMA, sk.SANDBOX_TOOL_CALL_SCHEMA])
    def test_each_is_well_formed(self, schema):
        _valid(schema)


class TestTheSchemasMatchTheCode:
    """A schema that does not describe what compute emits is worse than
    none — it lies to the composer. So validate real output."""

    def test_a_real_snapshot_validates(self):
        snap = fs.seeded({"a.txt": "hi\n", "sub/b.txt": "x" * (fs.INLINE_MAX + 5)})
        jsonschema.validate(snap.to_wire(), sk.FS_SNAPSHOT_SCHEMA)

    def test_the_empty_snapshot_validates(self):
        jsonschema.validate(fs.EMPTY.to_wire(), sk.FS_SNAPSHOT_SCHEMA)

    def test_a_snapshot_with_mounts_validates(self):
        snap = fs.Snapshot((), (fs.Mount(at="data", object="benji/corpus", digest="d"),))
        jsonschema.validate(snap.to_wire(), sk.FS_SNAPSHOT_SCHEMA)

    def test_real_sandbox_calls_validate(self):
        # Every provenance record a session produces must fit the
        # tool-call schema — the shape the 000362 trace reads.
        s = SandboxSession(SandboxImage.parse({"snapshot": {"a.txt": "one two\n"}}))
        s.write_file("r.txt", "x")
        s.read_file("a.txt")
        s.read_file("missing.txt")
        s.list()
        assert s.calls
        for c in s.calls:
            jsonschema.validate(c.to_wire(), sk.SANDBOX_TOOL_CALL_SCHEMA)

    def test_the_default_image_validates(self):
        jsonschema.validate(sk.default_image_wire(), sk.SANDBOX_IMAGE_SCHEMA)

    def test_a_declared_image_validates(self):
        wire = {"base": "mbshell", "tools": ["bash", "list"],
                "limits": {"memory_mb": 32, "wall_seconds": 5}, "strict": True,
                "mounts": [{"path": "data", "object": "benji/corpus"}]}
        jsonschema.validate(wire, sk.SANDBOX_IMAGE_SCHEMA)
        # …and it round-trips through the parser to the same tools.
        assert set(SandboxImage.parse(wire).tools) == {"bash", "list"}

    def test_the_limits_schema_matches_the_dataclass(self):
        jsonschema.validate(sandbox.Limits().to_wire(),
                            sk.SANDBOX_IMAGE_SCHEMA["properties"]["limits"])


class TestTheToolCatalog:
    def test_it_lists_every_tool_with_a_description(self):
        cat = sk.sandbox_tool_catalog()
        assert {t["name"] for t in cat} == set(sk.TOOL_NAMES)
        assert all(t["description"] and t["schema"] for t in cat)

    def test_it_omits_handlers(self):
        # The picker shows capabilities; the node wires handlers.
        assert all("handler" not in t for t in sk.sandbox_tool_catalog())


class TestRegistration:
    def test_the_fs_snapshot_manifest_is_registered(self):
        from mechbench_compute import platform_kinds
        paths = {m.path for m in platform_kinds.manifests()}
        assert sk.FS_SNAPSHOT_KIND in paths

    def test_the_manifest_renders_the_entries_as_a_table(self):
        from mechbench_compute import platform_kinds
        m = next(m for m in platform_kinds.manifests()
                 if m.path == sk.FS_SNAPSHOT_KIND)
        assert m.renderer.primitive == "table"
        assert m.renderer.field_map["rows"] == "entries"
