from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("run/ladder", "A ladder of rungs, as the older experiments recorded one.", platform=True,
            doc="A sequence of training rungs recorded as one object, from before protocols were the unit of a "
                "run; kept so those results still read.")
