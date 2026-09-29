from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("run/replay", "A live run replayed: each event's step and the state it ended in.", platform=True,
            doc="Written when a live run's recorded events are folded through its handler again, as a batch "
                "run: per event, its outputs (or, for a param change, the params from then on), and the final "
                "state and params.")
