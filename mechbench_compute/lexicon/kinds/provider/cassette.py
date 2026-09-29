from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("provider/cassette", "Recorded provider responses keyed by request hash, for replay.", platform=True,
            doc="The replies a run received from hosted models, keyed by the hash of the request that got them, so a "
                "re-run replays the same replies without calling the provider again.")
