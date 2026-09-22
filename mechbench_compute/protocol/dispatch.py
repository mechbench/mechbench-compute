"""How a node reaches an operation's `run()`.

One hop: the graph walk in protocol/pipeline.py has an operation's name,
its inputs and its params, and hands them here. `_run_op` finds the
operation's module (docs/OPS_LAYOUT.md: one operation, one file), lends
it a `Context` built from whatever the caller passed, and calls its
`run(ctx, inputs, params)`. An operation that declares weights to fuse an
adapter onto goes through the model-block wrapper on its way.
"""

from __future__ import annotations

from mechbench_compute import ops


class Dispatch:
    """Dispatch: see this module's docstring."""

    def _run_op(self, block, inputs, params, **lent):
        """Run an operation from its own file (docs/OPS_LAYOUT.md): lend
        it a context, and load the model and fuse an adapter around it
        when its declaration says there are weights to fuse onto."""
        mod = ops.find(block)
        ctx = ops.Context(executor=self, **lent)
        if ops.fuses_adapter(block):
            # The progress callbacks go through the wrapper as well as
            # the context: a test that stands in for the wrapper reports
            # progress from there, as the executor's methods once did.
            return self._run_model_block(
                lambda i, p, on_item=None, on_start=None: mod.run(ctx, i, p),
                inputs, params, on_item=lent.get("on_item"), on_start=lent.get("on_start"))
        return mod.run(ctx, inputs, params)
