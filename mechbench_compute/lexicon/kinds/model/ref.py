from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("model/ref", "A model reference: a base and the adapters that are part of what it means.", platform=True,
            doc="What a protocol's `model` parameter names and what every space's `model` records. The adapters a "
                "reference carries are fused before anything else; an adapter arriving on a node's port is fused on "
                "top, for that node only. A reference's adapter is fused in every layer, or, written "
                "`{\"bench\": …, \"layers\": [3, 4, 5]}`, in those layers alone; a result's `fused` lists it with "
                "its `layers`. An `adapter/operator` among them is attached instead: the residual stream passes "
                "through it after the layers it was trained at, on every forward pass, and it takes no `layers`.")
