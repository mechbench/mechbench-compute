"""The sandbox's catalog: kind paths, JSON-Schema contracts, and the
tool catalog (task 000361, epic 000334).

Three shapes cross into the platform's catalog once a model can drive a
workspace:

- **fs-snapshot** — a content-addressed tree (000358). A browsable
  object: the UI's file browser (000362) renders its entries as a
  table.
- **sandbox-image** — the standard-library base a protocol declares
  and the composer (000341) edits: which tools, what limits, strict,
  mounts, starting tree.
- **sandbox-tool-call** — one entry in an item's `metadata.sandbox`
  (compute 0.54.0): what the model asked and what the filesystem did.
  Not a standalone document — a record shape the transcript trace
  (000362) reads — so it is a SCHEMA here, not a renderable kind.

The schemas are the contract these consumers share; the fs-snapshot
KindManifest (in `platform_kinds`) carries the renderer. The tool
catalog is the descriptions the composer's tool picker offers.

Registration follows the condition-set / lens-trajectory precedent:
renderer-bearing kinds live in compute's `platform_kinds`, by their
`~canonical/kinds/...` path. If the UI later needs these paths as
named constants across the repo boundary, they graduate to
mechbench-schema then; today one source is enough.
"""
from __future__ import annotations

from typing import Any

from mechbench_compute import sandbox
from mechbench_compute.sandbox_session import _TOOL_DEFS, TOOL_NAMES

#: Catalog paths. `fs_snapshot` (underscore) is the OBJECT wire tag in
#: `snapshots.py`; this dashed path is its CATALOG identity — the two
#: are deliberately distinct, one names bytes on the wire, the other a
#: registered kind.
FS_SNAPSHOT_KIND = "~canonical/kinds/fs-snapshot"
SANDBOX_IMAGE_KIND = "~canonical/kinds/sandbox-image"
SANDBOX_TOOL_CALL_KIND = "~canonical/kinds/sandbox-tool-call"

#: The tree object (`snapshots.Snapshot.to_wire`). Blobs are references
#: by default, so `data` is not part of the stored shape.
FS_SNAPSHOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["kind", "digest", "entries"],
    "properties": {
        "kind": {"const": "fs_snapshot"},
        "version": {"type": "integer"},
        "digest": {"type": "string", "description": "sha256:… over paths + "
                   "blob hashes + exec bits; the tree's identity."},
        "n_files": {"type": "integer"},
        "n_bytes": {"type": "integer"},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "size", "blob_hash"],
                "properties": {
                    "path": {"type": "string"},
                    "size": {"type": "integer"},
                    "blob_hash": {"type": "string"},
                    "executable": {"type": "boolean"},
                },
            },
        },
        "mounts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["at", "object"],
                "properties": {
                    "at": {"type": "string"},
                    "object": {"type": "string"},
                    "digest": {"type": "string"},
                },
            },
        },
    },
}

#: The limits sub-shape, shared by the image. Mirrors
#: `sandbox.Limits.to_wire`.
_LIMITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memory_mb": {"type": "integer", "minimum": 1},
        "fuel": {"type": "integer", "minimum": 1},
        "wall_seconds": {"type": "number", "exclusiveMinimum": 0},
        "output_bytes": {"type": "integer", "minimum": 1},
        "max_files": {"type": "integer", "minimum": 1},
        "max_bytes": {"type": "integer", "minimum": 1},
    },
}

#: The standard-library image (`SandboxImage`). `tools` is the allowlist
#: — the enforcement point, since one shell otherwise reaches every
#: applet.
SANDBOX_IMAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base": {"type": "string", "description": "Guest name, e.g. mbshell."},
        "tools": {"type": "array", "items": {"enum": list(TOOL_NAMES)},
                  "description": "The tools offered to the model."},
        "limits": _LIMITS_SCHEMA,
        "strict": {"type": "boolean", "description": "Virtualize the clock "
                   "and RNG so the run is a function of its inputs."},
        "mounts": {"type": "array", "items": {
            "type": "object", "required": ["object"],
            "properties": {"path": {"type": "string"},
                           "object": {"type": "string"}}}},
        "snapshot": {"description": "Starting tree: an fs-snapshot, or a "
                     "{path: content} map inline."},
    },
}

#: One tool call's provenance — an entry in an item's `metadata.sandbox`
#: (`SandboxCall.to_wire`). stdout/stderr ride along small; a node may
#: store large ones as objects and reference them (stdout_ref).
SANDBOX_TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["tool", "argv", "exit_code", "snapshot_in", "snapshot_out"],
    "properties": {
        "tool": {"enum": list(TOOL_NAMES)},
        "argv": {"type": "array", "items": {"type": "string"}},
        "stdin": {"type": "string"},
        "exit_code": {"type": "integer"},
        "limit": {"type": ["string", "null"], "description": "The ceiling "
                  "that stopped the call, if one did (fuel, wall_seconds, "
                  "memory_mb, stack, max_files, max_bytes)."},
        "snapshot_in": {"type": "string", "description": "Tree digest before."},
        "snapshot_out": {"type": "string", "description": "Tree digest after."},
        "changed": {"type": "object", "description": "added/removed/changed "
                    "paths, present only when the tree changed."},
        "duration_ms": {"type": "integer"},
        "stdout": {"type": "string"},
        "stderr": {"type": "string"},
        "stdout_ref": {"type": "string", "description": "Object ref when the "
                       "output was too large to inline."},
        "stderr_ref": {"type": "string"},
    },
}


def sandbox_tool_catalog() -> list[dict[str, Any]]:
    """The sandbox tool definitions, for the composer's tool picker
    (000341): name, description, schema — what a user sees when
    choosing capabilities to give a model. Handlers are omitted; the
    node wires those when it builds a session."""
    return [{"name": d["name"], "description": d["description"],
             "schema": dict(d["schema"])}
            for d in (_TOOL_DEFS[n] for n in TOOL_NAMES)]


def default_image_wire() -> dict[str, Any]:
    """A ready-to-edit default image for the composer: the workhorse
    plus the snapshot conveniences, default limits, non-strict."""
    from mechbench_compute.sandbox_session import DEFAULT_GUEST, DEFAULT_TOOLS

    return {"base": DEFAULT_GUEST, "tools": list(DEFAULT_TOOLS),
            "limits": sandbox.Limits().to_wire(), "strict": False}
