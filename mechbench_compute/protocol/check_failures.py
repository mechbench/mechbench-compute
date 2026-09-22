"""Whether a run that finished its walk actually succeeded."""

from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute.protocol.serialize_params import serialize_params


def check_failures(state, resolver) -> None:
    """Raise the first failure nobody answered for — which is every
    failure in a graph that declares no `on_missing` policy. It is
    raised here, once the sibling branches have finished, rather than
    where it happened.
    """
    from mechbench_compute import bench, dataflow

    orphaned = [nid for nid in state.order if nid in state.failures
                and nid not in state.tolerated]
    if not orphaned:
        return
    first = orphaned[0]
    if len(orphaned) > 1:
        print(f"[graph] {len(orphaned)} nodes failed: "
              f"{', '.join(orphaned)}")
    # A failed run keeps its intermediates even in discard mode: they
    # are the evidence of what went wrong. Stored where a kept run would
    # have stored them; a failure to store one is said, and does not
    # hide the failure being reported.
    for nid, (held_block, held_params) in state.held.items():
        try:
            out = bench.emit(
                f"{state.result_base}/{dataflow.INTERMEDIATES}/{nid}",
                state.results[nid],
                inputs=list(resolver.read_stored_inputs(state.nodes[nid])),
                operation=lexicon.canonical_path(held_block),
                params=serialize_params(held_params))
            state.node_paths[nid] = out["path"]
        except Exception as exc:  # noqa: BLE001 — evidence, not the verdict
            print(f"[graph] could not store the held result of {nid}: {exc}")
    raise state.failures[first]
