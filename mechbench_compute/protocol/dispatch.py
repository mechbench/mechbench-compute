"""How a node reaches the work that answers for it.

`_run_node` makes the three-way choice: a result a remote wave already
produced while a sibling was running, a remote wave this node opens, or
an operation's own `run`. `_run_op` is that last hop — it finds the
operation's module (docs/OPS_LAYOUT.md: one operation, one file), lends
it a `Context` built from whatever the caller passed, and calls its
`run(ctx, inputs, params)`. An operation that declares weights to fuse an
adapter onto goes through the model-block wrapper on its way.

Both raise whatever the work raised: what a node's failure means is the
walk's decision, and its consumers' (protocol/pipeline.py).
"""

from __future__ import annotations

from mechbench_compute import ops
from mechbench_compute.protocol.is_remote import is_remote


def read_ahead(state, nid):
    """The result a wave already produced for this node."""
    done = state.ahead.pop(nid)
    if isinstance(done, BaseException):
        raise done
    return done


class Dispatch:
    """Dispatch: see this module's docstring."""

    def _run_node(self, state, nid, block, inputs, params, *, secrets,
                  resolver, progress, on_item, on_checkpoint, input_paths,
                  resume_kwargs):
        """One node's result, from whichever of the three answers for it."""
        if nid in state.ahead:
            # Already run, alongside its siblings. The bookkeeping stays
            # in the walk, in topological order, so the manifest, the
            # hashes and the emitted objects do not depend on which
            # branch finished first.
            return read_ahead(state, nid)
        if is_remote(block, params):
            # This node waits on somebody else's machine, so every other
            # remote node that is ready waits with it rather than after
            # it.
            state.ahead.update(self._run_remote_wave(
                nid, block, inputs, params, secrets,
                nodes=state.nodes, edges=state.edges, order=state.order,
                results=state.results, missing=state.missing,
                resolve_params=resolver.resolve_params,
                resolve_value=resolver.resolve_value,
                item_reporter=progress.open_items, expand=progress.expand,
                node_view=progress.node_view, report=progress.report,
                resume=state.resume, resume_kwargs=resume_kwargs))
            return read_ahead(state, nid)
        if ops.find(block) is not None:
            return self._run_op(
                block, inputs, params,
                on_item=on_item, on_start=progress.expand, secrets=secrets,
                on_checkpoint=on_checkpoint,
                input_paths=dict(input_paths),
                bindings=state.bound_params if state.declared else state.bindings,
                result_base=state.result_base,
                resume_items=resume_kwargs.get("resume_items"),
                resume_state=resume_kwargs.get("resume_state"))
        raise ValueError(f"unknown block: {block!r}")

    def _run_op(self, block, inputs, params, **lent):
        """Run an operation from its own file (docs/OPS_LAYOUT.md): lend
        it a context, and load the model and fuse an adapter around it
        when its declaration says there are weights to fuse onto."""
        mod = ops.find(block)
        ctx = ops.Context(executor=self, **lent)
        if ops.fuses_adapter(block):
            # The progress callbacks go through the wrapper as well as the
            # context, so a stand-in wrapper can report progress itself.
            return self._run_model_block(
                lambda i, p, on_item=None, on_start=None: mod.run(ctx, i, p),
                inputs, params, on_item=lent.get("on_item"), on_start=lent.get("on_start"))
        return mod.run(ctx, inputs, params)
