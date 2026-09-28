from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("model/ref", "A model reference: a base and the adapters that are part of what it means.", platform=True,
            doc="What a protocol's `model` parameter names and what every space's `model` records. The adapters a "
                "reference carries are fused before anything else; an adapter arriving on a node's port is fused on "
                "top, for that node only.")
