from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("provider/completion", "One provider reply: text, parts, stop reason, usage, and its call.", platform=True,
            doc="One reply as the provider returned it, with the call that produced it, before it became a document "
                "or a transcript message.")
