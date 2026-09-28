from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind("sandbox/call", "One tool call inside a sandbox session, with what it read and wrote.", platform=True,
            doc="The record of one invocation: which tool, with what arguments, and the files it read and wrote.")
