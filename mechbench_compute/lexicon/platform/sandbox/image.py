from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("sandbox/image", "A sandbox image: base, tools, limits, and the tree it starts from.", platform=True,
            doc="What a sandboxed conversation starts from. Two sessions on the same image start from the same tree, "
                "so their snapshots differ only by what the tools did.")
