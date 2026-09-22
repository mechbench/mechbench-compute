"""The platform's registered kinds, generated from the lexicon.

Every kind that declares a renderer (`mechbench_compute/lexicon/kinds.py`)
registers one `KindManifest` at its canonical path,
`~canonical/kinds/<family>/<kind>`: the item schema is the declaration's
fields (its ancestors' included), the renderer binding is the
declaration's, and `supersedes` names the other paths that resolve to
this kind. Every registered manifest stays registered and immutable, as
the registry promises; a superseded kind is marked on its successor,
never rewritten.

The filesystem snapshot is the one platform kind with a hand-written
item schema (`sandbox_kinds.FS_SNAPSHOT_SCHEMA`): its shape is the
snapshot codec's, not an op's.

Registration needs a platform-admin credential:

    MECHBENCH_API_URL=... MECHBENCH_API_KEY=... \\
        python -m mechbench_compute.platform_kinds
"""

from __future__ import annotations

from typing import Any

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon._base import COLLECTION, KIND_ROOT, Kind


def item_schema(kind: Kind) -> dict[str, Any]:
    """A JSON Schema for one item of `kind`: the declared fields with
    those inherited through `extends`, and the required set."""
    return {
        "type": "object",
        "required": list(kind.required),
        "properties": {name: dict(spec) for name, spec in K.all_fields(kind).items()},
    }


def superseded_paths(kind: Kind) -> list[str]:
    """The registered paths this kind replaces: every other spelling
    under the canonical root that the alias table resolves to it."""
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
    """The version the registry holds, when a refusal says a different
    manifest is registered at this path."""
    body = getattr(e, "body", None)
    if not isinstance(body, dict) or body.get("code") != "MANIFEST_PINNED":
        return None
    v = body.get("registeredVersion")
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def register_all() -> None:
    """Bring the registry up to the declarations.

    A kind whose declared fields have changed is registered as a NEW
    version at the same path: the registry refuses a changed manifest at
    a registered version and says which version it holds, and this
    re-registers at the next number. The manifest it replaces stays
    readable at `?version=N`, so the published contract keeps its
    history instead of freezing on the day a kind was first written.

    The number counts the times this kind has changed ON THIS REGISTRY,
    which is why the declaration does not carry it: two registries with
    different histories would disagree, and the one that matters is the
    one being written to.
    """
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
