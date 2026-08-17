"""Platform-registered document kinds shipped by mechbench-core (task
000246): the open-source half of the Layer-2 extension seam. Each entry
is a KindManifest that core's release tooling registers at
~canonical/kinds/... via the bench client (platform-admin credential
required, per the 000166 promotion flow).

Run: MECHBENCH_API_URL=... MECHBENCH_API_KEY=... \
     python -m mechbench_core.platform_kinds
"""

from __future__ import annotations


def manifests():
    import mechbench_schema as ms

    return [
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
    ]


def register_all() -> None:
    from mechbench_core import bench

    for m in manifests():
        r = bench.register_kind(m)
        print(f"registered {r['path']}"
              + (" (idempotent)" if r.get("idempotent") else ""))


if __name__ == "__main__":
    register_all()
