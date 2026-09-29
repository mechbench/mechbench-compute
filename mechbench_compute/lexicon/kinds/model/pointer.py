from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("model/pointer", "Where a published model lives, so a later reference can load it.", platform=True,
            doc="What `adapter/merge` leaves behind: the location of a merged checkpoint, on the bench or on the hub, "
                "usable as the base of a later model reference.")
