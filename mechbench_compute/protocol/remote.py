"""Nodes whose work happens on somebody else's machine.

Two remote nodes that do not depend on each other have no reason to wait
for each other: the time is latency. `_run_remote_wave` runs the node the
graph walk asked for and every remote node already ready beside it, and
hands every result back for the caller to account for in topological
order.
"""

from __future__ import annotations

from typing import Any

from mechbench_compute import lexicon
from mechbench_compute.protocol.is_remote import is_remote
from mechbench_compute.protocol.sort_edges import sort_edges

#: How many remote nodes may be in flight at once. The provider's own
#: rate limiter bounds the requests WITHIN a node; this bounds the
#: nodes, so a twenty-branch fan-out does not open twenty connections'
#: worth of concurrency on top of it.
MAX_PARALLEL_NODES = 8


class Remote:
    """Remote: see this module's docstring."""

    def _run_remote_wave(self, nid, block, inputs, params, secrets, *,
                         nodes, edges, order, results, missing,
                         resolve_params, resolve_value, item_reporter, expand,
                         node_view, report, resume, resume_kwargs):
        """Run this remote node and every remote node ready beside it.

        "Ready beside it" is the whole of the scheduling: a node later in
        the topological order whose inputs are ALL computed already does
        not depend on this one, so waiting for this one buys nothing but
        latency. Two prompts to two providers, then a judge, is the shape
        this exists for: the wave costs the longer call, not the sum.

        What stays on the calling thread, deliberately: every result is
        returned and the caller does the hashing, the emitting and the
        progress accounting in topological order, so the manifest and
        the stored objects are identical to a serial run. A node with
        resume state is left out of the wave entirely — partial work is
        the one thing not worth racing.
        """
        from concurrent.futures import ThreadPoolExecutor

        ready: list[tuple[str, str, dict, dict]] = [(nid, block, inputs, params)]
        if not resume_kwargs:
            for other in order:
                if (other == nid or other in results or other in missing
                        or len(ready) >= MAX_PARALLEL_NODES):
                    continue
                node = nodes[other]
                try:
                    peer_block = lexicon.resolve(str(node.get("block")))
                except KeyError:
                    continue
                peer_params = resolve_params(node.get("params"))
                if not is_remote(peer_block, peer_params):
                    continue
                if resume.get(other) if isinstance(resume, dict) else None:
                    continue
                sources = {e["from"]["node"] for e in sort_edges(edges, other)}
                if not sources <= set(results):
                    continue        # it is waiting for something, not for us
                peer_inputs = {
                    e["to"]["port"]: results[e["from"]["node"]]
                    for e in sort_edges(edges, other)}
                for port, raw in (node.get("inputs") or {}).items():
                    if raw is not None and port not in peer_inputs:
                        peer_inputs[port] = resolve_value(raw)
                ready.append((other, peer_block, peer_inputs, peer_params))

        if len(ready) == 1:
            return {nid: self._dispatch_remote(
                block, inputs, params, secrets,
                on_item=item_reporter(nid), on_start=expand, **resume_kwargs)}

        node_view["parallel"] = [n for n, _b, _i, _p in ready]
        report()
        print(f"[graph] {len(ready)} remote nodes in flight: "
              f"{', '.join(n for n, _b, _i, _p in ready)}")
        out: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=len(ready)) as pool:
            futures = {
                pool.submit(self._dispatch_remote, b, i, p, secrets,
                            on_item=item_reporter(n),
                            on_start=(expand if n == nid else None)): n
                for n, b, i, p in ready}
            for fut, name in futures.items():
                try:
                    out[name] = fut.result()
                except Exception as exc:  # noqa: BLE001 — the caller decides
                    out[name] = exc
        node_view.pop("parallel", None)
        return out

    def _dispatch_remote(self, block, inputs, params, secrets, *,
                         on_item=None, on_start=None, **resume_kwargs):
        """The blocks whose work is a provider's. Separate from the
        executor's big dispatch so a thread runs exactly this and nothing
        that touches the loop's bookkeeping."""
        return self._run_op(block, inputs, params, secrets=secrets,
                            on_item=on_item, on_start=on_start,
                            **resume_kwargs)
