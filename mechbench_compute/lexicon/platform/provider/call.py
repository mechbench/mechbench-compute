from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("provider/call", "The provenance of one provider call: model version, usage, cost, latency.", platform=True,
            doc="What one request to a hosted model cost and what served it — the version the provider reported, the "
                "tokens in and out, the time taken.")
