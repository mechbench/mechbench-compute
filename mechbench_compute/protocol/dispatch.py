from __future__ import annotations

from mechbench_compute import ops
from mechbench_compute.protocol.is_remote import is_remote
from mechbench_compute.registry import REGISTRY


def read_ahead(state, nid):
    done = state.ahead.pop(nid)
    if isinstance(done, BaseException):
        raise done
    return done


class Dispatch:
    def _run_node(self, state, nid, block, inputs, params, *, secrets,
                  resolver, progress, on_item, on_checkpoint, input_paths,
                  resume_kwargs):
        if nid in state.ahead:
            return read_ahead(state, nid)
        resolved = REGISTRY.find(block)
        if resolved is None:
            raise ValueError(f"unknown block: {block!r}")
        if is_remote(resolved, params):
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
        return self._run_op(
            resolved, inputs, params,
            on_item=on_item, on_start=progress.expand, secrets=secrets,
            on_checkpoint=on_checkpoint,
            input_paths=dict(input_paths),
            run_params=state.bound_params,
            result_base=state.result_base,
            resume_items=resume_kwargs.get("resume_items"),
            resume_state=resume_kwargs.get("resume_state"),
            on_token=self._node_on_token(nid))

    def _node_on_token(self, nid):
        sink = getattr(self, "_on_token", None)
        if sink is None:
            return None
        return lambda key, event: sink(nid, key, event)

    def _run_op(self, block, inputs, params, **lent):
        resolved = ops.resolve_op(block)
        run = resolved.module.run
        ctx = ops.Context.for_op(resolved.op, self, **lent)
        if ops.fuses_adapter(resolved) or (ops.fuses_adapter_locally(resolved)
                                           and not is_remote(resolved, params)):
            return self._run_model_block(
                lambda i, p, on_item=None, on_start=None: run(ctx, i, p),
                inputs, params, on_item=lent.get("on_item"), on_start=lent.get("on_start"))
        return run(ctx, inputs, params)
