from __future__ import annotations


class Tools:
    def _tool_block_runner(self, secrets=None):
        def run_block(ref, inputs, params):
            if ref in ("logits/read", "text/generate"):
                return self._run_op(ref, inputs, params, secrets=secrets)
            if ref == "tools/lookup":
                from mechbench_compute import bench

                return bench.fetch(str((inputs.get("arguments") or {}).get("path")))
            raise ValueError(
                f"{ref!r} is not available as a tool handler here — pure "
                "blocks, decision-read and generate are")

        return run_block
