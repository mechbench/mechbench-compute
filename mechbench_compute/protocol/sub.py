from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.protocol.protocol_spec import ProtocolSpec


class Sub:
    def run_sub(self, target, inputs, params, *, secrets=None, budget=None,
                on_item=None, on_start=None, on_token=None, outputs=None) -> Any:
        from mechbench_compute import dataflow as dataflow_mod

        child = type(self)(
            on_download=self._on_download,
            on_download_bytes=self._on_download_bytes,
            limiter=self._limiter, budget=self.open_child_budget(budget),
            on_token=on_token)
        child._model, child._model_id = self._model, self._model_id
        child._checkpoint_dirs = self.__dict__.setdefault("_checkpoint_dirs", {})
        try:
            if isinstance(target, str):
                return child._run_op(target, dict(inputs), dict(params), secrets=secrets,
                                     on_item=on_item, on_start=on_start)
            if not isinstance(target, Mapping):
                raise TypeError(
                    f"a sub-run's target is an operation's name or a graph, "
                    f"not {type(target).__name__}")
            if on_item is not None or on_start is not None:
                raise ValueError(
                    "a graph reports its progress through its own nodes: "
                    "on_item and on_start follow one operation")
            out = child.run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None,
                extra={"graph": {**target, "dataflow": dataflow_mod.DATAFLOW},
                       "params": dict(params),
                       "inputs": dict(inputs),
                       **({"outputs": [dict(o) for o in outputs]} if outputs else {})}),
                secrets=secrets)
            return dict(out.payload.get("outputs") or {})
        finally:
            self._model, self._model_id = child._model, child._model_id

    def open_child_budget(self, cap_usd):
        if cap_usd is None:
            return self._budget
        from mechbench_compute.providers.budget import Budget

        cap = float(cap_usd)
        if cap <= 0:
            raise ValueError(f"a sub-run's budget is a positive cap in USD, not {cap_usd!r}")
        return Budget(cap_usd=cap) if self._budget is None else self._budget.child(cap)
