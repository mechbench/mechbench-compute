"""Platform-registered document kinds shipped by mechbench-compute (task
000246): the open-source half of the Layer-2 extension seam. Each entry
is a KindManifest that core's release tooling registers at
~canonical/kinds/... via the bench client (platform-admin credential
required, per the 000166 promotion flow).

Run: MECHBENCH_API_URL=... MECHBENCH_API_KEY=... \
     python -m mechbench_compute.platform_kinds
"""

from __future__ import annotations


def manifests():
    import mechbench_schema as ms

    from mechbench_compute import sandbox_kinds as sk

    series_map = {"rows": "layers", "x": "layer", "y": "entropy_bits",
                  "label": "top1"}
    return [
        # Trajectory (task 000368): the residual stream read along one
        # axis — a position across layers, or a layer across positions.
        # Rows carry the vector; the table view shows step, layer,
        # position, token and norm, which is what a reader scans.
        ms.KindManifest(
            path="~canonical/kinds/trajectory",
            title="Residual trajectory",
            version="1",
            item_schema={
                "type": "object",
                "required": ["step", "layer", "position", "vector"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {},
                    "step": {"type": "integer"},
                    "layer": {"type": "integer"},
                    "position": {"type": "integer"},
                    "token": {"type": ["string", "null"]},
                    "norm": {"type": "number"},
                    "vector": {"type": "array", "items": {"type": "number"}},
                },
            },
            renderer=ms.RendererBinding(
                primitive="table", field_map={"rows": "rows"}),
        ),
        # Tokenizer diagnostics (task 000377): the depth inventory as
        # rows (depth, count, share) plus scalar summaries and the gate.
        ms.KindManifest(
            path="~canonical/kinds/tokenizer-stats",
            title="Tokenizer diagnostics",
            version="1",
            item_schema={
                "type": "object",
                "required": ["depth", "count", "share"],
                "properties": {
                    "depth": {"type": "integer"},
                    "count": {"type": "integer"},
                    "share": {"type": "number"},
                },
            },
            renderer=ms.RendererBinding(
                primitive="table", field_map={"rows": "rows"}),
        ),
        # ConditionSet (epic 000258): PromptFactory's output. Every
        # condition carries its axis COORDINATES as structured data —
        # downstream blocks (GroupBy, PairedDelta) operate on coords,
        # never by parsing the display id.
        ms.KindManifest(
            path="~canonical/kinds/condition-set",
            title="Condition set",
            version="1",
            item_schema={
                "type": "object",
                "required": ["id", "coords", "user"],
                "properties": {
                    "id": {"type": "string",
                           "description": "Display name; never parsed."},
                    "coords": {
                        "type": "object",
                        "description": "Axis coordinates (axis name -> "
                                       "value key), incl. batch for "
                                       "range-generated items.",
                        "additionalProperties": {"type": ["string", "integer"]},
                    },
                    "system": {"type": "string"},
                    "user": {"type": "string", "x-mechbench-text": True},
                    "prefill": {"type": "string"},
                },
            },
            renderer=ms.RendererBinding(
                primitive="table",
                field_map={"rows": "conditions"}),
        ),
        # v2: series-bound, with a collection-level compare view (task
        # 000250 — the funnel renders as overlaid curves). v1 remains
        # registered and immutable; collections opt into v2 by path.
        ms.KindManifest(
            path="~canonical/kinds/lens-trajectory/2",
            title="Logit-lens trajectory",
            version="2",
            item_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "x-mechbench-text": True},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "layers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "layer": {"type": "integer"},
                                        "top1": {"type": "string"},
                                        "p": {"type": "number"},
                                        "entropy_bits": {"type": "number"},
                                    },
                                },
                            }
                        },
                    },
                },
            },
            renderer=ms.RendererBinding(primitive="series",
                                        field_map=dict(series_map)),
            collection_renderer=ms.RendererBinding(
                primitive="series", field_map=dict(series_map)),
        ),
        ms.KindManifest(
            path="~canonical/kinds/lens-trajectory",
            title="Logit-lens trajectory",
            version="1",
            item_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "x-mechbench-text": True},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "layers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "layer": {"type": "integer"},
                                        "top1": {"type": "string"},
                                        "p": {"type": "number"},
                                        "entropy_bits": {"type": "number"},
                                    },
                                },
                            }
                        },
                    },
                },
            },
            renderer=ms.RendererBinding(
                primitive="table", field_map={"rows": "layers"}),
        ),
        # The sandbox's browsable object (task 000361): a
        # content-addressed tree, rendered as a file table. The UI's
        # snapshot browser (000362) reads this shape; a real tree
        # renderer is 000418's job (extend the renderer vocabulary),
        # and `table` is the honest default until then. The image and
        # the tool-call are contracts in `sandbox_kinds`, not renderable
        # kinds: an image is composer config, a tool-call is a record
        # inside a transcript.
        ms.KindManifest(
            path=sk.FS_SNAPSHOT_KIND,
            title="Filesystem snapshot",
            version="1",
            item_schema=sk.FS_SNAPSHOT_SCHEMA,
            renderer=ms.RendererBinding(
                primitive="table", field_map={"rows": "entries"}),
        ),
    ]


def register_all() -> None:
    from mechbench_compute import bench

    for m in manifests():
        try:
            r = bench.register_kind(m)
            print(f"registered {r['path']}"
                  + (" (idempotent)" if r.get("idempotent") else ""))
        except bench.BenchError as e:
            if "MANIFEST_PINNED" in str(e):
                # An earlier registration of this version exists with
                # different canonical bytes (typically schema evolution
                # adding optional fields). Pinned versions stay pinned;
                # changes go to a new version path.
                print(f"pinned    {m.path} (existing version retained)")
            else:
                raise


if __name__ == "__main__":
    register_all()
