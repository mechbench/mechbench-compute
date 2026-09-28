from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("run/result", "A protocol run's result: every node's path, the manifest, the spend.", platform=True,
            doc="Written by the executor when a run completes: the stored object of every node, keyed by node id, "
                "with the manifest that fingerprints the run and what it spent.")
