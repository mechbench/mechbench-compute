from __future__ import annotations

from typing import Any

from mechbench_compute import sandbox
from mechbench_compute.sandbox_session import TOOL_DEFS, TOOL_NAMES

FS_SNAPSHOT_KIND = "~canonical/kinds/sandbox/snapshot"
SANDBOX_IMAGE_KIND = "~canonical/kinds/sandbox/image"
SANDBOX_TOOL_CALL_KIND = "~canonical/kinds/sandbox/call"

FS_SNAPSHOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["kind", "digest", "entries"],
    "properties": {
        "kind": {"enum": ["sandbox/snapshot", "fs_snapshot"]},
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
    return [{"name": d["name"], "description": d["description"],
             "schema": dict(d["schema"])}
            for d in (TOOL_DEFS[n] for n in TOOL_NAMES)]


def default_image_wire() -> dict[str, Any]:
    from mechbench_compute.sandbox_session import DEFAULT_GUEST, DEFAULT_TOOLS

    return {"base": DEFAULT_GUEST, "tools": list(DEFAULT_TOOLS),
            "limits": sandbox.Limits().to_wire(), "strict": False}
