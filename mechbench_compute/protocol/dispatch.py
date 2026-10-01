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
            on_token=self._node_on_token(nid), node=nid)

    def _node_on_token(self, nid):
        sink = getattr(self, "_on_token", None)
        if sink is None:
            return None
        return lambda key, event: sink(nid, key, event)

    def _run_op(self, block, inputs, params, **lent):
        resolved = ops.resolve_op(block)
        ctx = ops.Context.for_op(resolved.op, self, **lent)

        def run(i, p):
            with REGISTRY.within(ctx.scope):
                return resolved.module.run(ctx, i, p)

        try:
            if ops.fuses_adapter(resolved) or (ops.fuses_adapter_locally(resolved)
                                               and not is_remote(resolved, params)):
                result = self._run_model_block(
                    lambda i, p, on_item=None, on_start=None: run(i, p),
                    inputs, params, on_item=lent.get("on_item"), on_start=lent.get("on_start"))
            else:
                result = run(inputs, params)
        finally:
            fused = self._reference_restored(ctx.fused)
        if fused and isinstance(result, dict):
            result.setdefault("fused", fused)
        return place_fused(resolved.op, result)


def place_fused(op, result):
    from mechbench_compute.lexicon._base import DEFAULT_OUTPUT

    out = result.get(DEFAULT_OUTPUT) if isinstance(result, dict) else None
    if op.outputs and isinstance(out, dict) and "fused" in result:
        out.setdefault("fused", result.pop("fused"))
    return result
