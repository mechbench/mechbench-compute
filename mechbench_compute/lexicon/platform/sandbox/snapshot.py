from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("sandbox/snapshot", "A directory as a value: entries sorted by path, mounts by identity.", platform=True,
            renderer={"primitive": "table", "field_map": {"rows": "entries"}},
            doc="A filesystem tree as a stored object, so what a session's tools wrote is content-addressed like every "
                "other result: the same tree from two runs is the same object.")
