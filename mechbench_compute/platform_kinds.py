from __future__ import annotations

from typing import Any

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon._base import COLLECTION, KIND_ROOT, Kind


def item_schema(kind: Kind) -> dict[str, Any]:
    return {
        "type": "object",
        "required": list(kind.required),
        "properties": {name: dict(spec) for name, spec in K.all_fields(kind).items()},
    }


def superseded_paths(kind: Kind) -> list[str]:
    return sorted(old for old, (name, _plural) in K.KIND_ALIASES.items()
                  if old.startswith(KIND_ROOT) and name == kind.name)


def manifests():
    import mechbench_schema as ms

    from mechbench_compute import sandbox_kinds as sk

    out = []
    for kind in K.KINDS:
        if kind.renderer is None or kind.name == COLLECTION:
            continue
        schema = (sk.FS_SNAPSHOT_SCHEMA if kind.name == "sandbox/snapshot"
                  else item_schema(kind))
        out.append(ms.KindManifest(
            path=kind.path,
            title=kind.summary.rstrip(".") if len(kind.summary) <= 80 else kind.name,
            version="1",
            item_schema=schema,
            renderer=ms.RendererBinding(**kind.renderer),
            collection_renderer=(ms.RendererBinding(**kind.collection_renderer)
                                 if kind.collection_renderer else None),
            supersedes=superseded_paths(kind),
        ))
    return out


def _registered_version(e: Any) -> int | None:
    body = getattr(e, "body", None)
    if not isinstance(body, dict) or body.get("code") != "MANIFEST_PINNED":
        return None
    v = body.get("registeredVersion")
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def register_all() -> None:
    from mechbench_compute import bench

    for m in manifests():
        try:
            r = bench.register_kind(m)
        except bench.BenchError as e:
            held = _registered_version(e)
            if held is None:
                raise
            r = bench.register_kind(m.model_copy(update={"version": str(held + 1)}))
            was = (r.get("supersedes") or {}).get("version")
            print(f"superseded {r['path']} (version {was} -> {r.get('version')})")
            continue
        print(f"registered {r['path']}"
              + (f" v{r['version']}" if r.get("version") else "")
              + (" (idempotent)" if r.get("idempotent") else ""))


if __name__ == "__main__":
    register_all()
