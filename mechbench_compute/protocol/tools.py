"""What a model may call mid-turn.

The toolbox runs pure blocks itself. The handlers that need an executor —
a model block, a recording fetch off the bench — are this, which is what
makes `logits/read` available AS A TOOL: a model that can consult another
model, or the bench, in the middle of answering.
"""

from __future__ import annotations


class Tools:
    """Tools: see this module's docstring."""

    def _tool_block_runner(self, secrets=None):
        """Handlers the toolbox cannot run itself (task 000340): model
        blocks and, later, sub-protocols. This is what makes
        `decision-read` available AS A TOOL — a model that can consult
        another model, or the bench, mid-turn."""
        def run_block(ref, inputs, params):
            if ref in ("logits/read", "text/generate"):
                return self._run_op(ref, inputs, params, secrets=secrets)
            if ref == "tools/lookup":
                # The recording fetch, so a tool call's object shows up
                # in the run's resolved lineage like any other input.
                from mechbench_compute import bench

                return bench.fetch(str((inputs.get("arguments") or {}).get("path")))
            raise ValueError(
                f"{ref!r} is not available as a tool handler here — pure "
                "blocks, decision-read and generate are")

        return run_block
