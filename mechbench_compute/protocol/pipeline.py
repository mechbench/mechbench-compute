"""Walking the graph: the one method that runs a protocol.

`_run_pipeline` reads the spec into a `RunState`, takes that state's
nodes in topological order, and for each one resolves, dispatches and
records: the resolver turns its references into values, `gather_inputs`
fills its ports, the `try` below sends it to an operation or a remote
wave, and `store_result` hashes and emits what came back. The manifest
`build_manifest` returns at the end is the record of all of that.

Each of those is a file of its own beside this one, so what is left
here is the walk itself.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from mechbench_compute import lexicon, ops
from mechbench_compute.protocol.copy_arch import copy_arch
from mechbench_compute.protocol.gather_inputs import gather_inputs
from mechbench_compute.protocol.is_remote import is_remote
from mechbench_compute.protocol.missing_upstream import MissingUpstream
from mechbench_compute.protocol.progress import Progress
from mechbench_compute.protocol.protocol_spec import ProtocolSpec
from mechbench_compute.protocol.read_resume_entry import read_resume_entry
from mechbench_compute.protocol.resolver import Resolver
from mechbench_compute.protocol.restore_node import restore_node
from mechbench_compute.protocol.run_state import RunState
from mechbench_compute.protocol.serialize_params import serialize_params
from mechbench_compute.protocol.sort_edges import sort_edges
from mechbench_compute.protocol.summarize_node import summarize_node
from mechbench_compute.protocol.total_spend import total_spend


class Pipeline:
    """Pipeline: see this module's docstring."""

    def _run_pipeline(self, spec: ProtocolSpec, on_progress=None,
                      secrets=None, resume=None) -> Any:
        """Execute a protocol graph: topological order over the nodes,
        every node an operation run in-process with the prefix cache. A
        node produces one value (edges' port names select inputs, not
        outputs), and the whole graph runs in this one job — the
        executor does no multi-job planning.

        Params may reference bindings: any string param "$name"
        resolves to spec bindings[name]."""
        from datetime import datetime

        import mechbench_schema as ms

        from mechbench_compute import __version__ as core_version
        from mechbench_compute import bench, dataflow
        from mechbench_compute import resume as resume_mod
        from mechbench_compute import tensors as tensors_mod
        from mechbench_compute.seeds import hardware_class

        state = RunState(spec, resume)
        resolver = Resolver(
            declared=state.declared, bindings=state.bindings,
            bound_params=state.bound_params, secrets=secrets,
            on_download=self._on_download,
            on_download_bytes=self._on_download_bytes)
        progress = Progress(on_progress, len(state.order),
                            on_spool_item=self._on_spool_item)
        current = {"nid": ""}

        for pos, nid in enumerate(state.order):
            node = state.nodes[nid]
            # Any spelling a protocol may carry — bare or stored —
            # becomes the one bare name here, once, before anything
            # hashes it. An unknown spelling refuses by name.
            try:
                block = lexicon.resolve(node["block"])
            except KeyError:
                raise ValueError(f"unknown block: {node['block']!r}") from None
            progress.start_node(pos, nid)
            params = resolver.resolve_node_params(node, block)
            # Edges in a canonical order: by port, then by the edge's own
            # `index` when it has one, then by source node id. Two
            # consequences. A VARIADIC port receives them as an ordered
            # list, so a node over several branches knows which branch is
            # which. And a node's fingerprint does not depend on the
            # order its author wrote the edges in, so moving a line in
            # the JSON restarts nothing downstream.
            in_edges = sort_edges(state.edges, nid)
            gathered = gather_inputs(state, nid, node,
                                     lexicon.BY_NAME.get(block), in_edges,
                                     resolver)
            if gathered is None:
                progress.bump()
                continue
            inputs, input_paths, inline_hashes = gathered
            # An input written under `params` is an unknown param, and
            # `check_params` below refuses it by name.
            #
            # Process identity for this node: what it computes is fixed
            # by the block, its wire params, its inputs' content, and the
            # compute version. A partial from a previous attempt is
            # reused only under an equal fingerprint.
            current["nid"] = nid
            on_item = progress.open_items(nid)
            self._current = current
            # Before anything runs: does this block actually read what
            # the protocol asked for, and take what was wired to it? A
            # silently ignored param is a wrong answer with no error;
            # an unwired required port is the same, earlier.
            from mechbench_compute.block_params import check_inputs, check_params
            check_params(block, serialize_params(params))
            inputs = check_inputs(block, inputs)
            # The stored identity, not the spelling: one fingerprint per
            # op however the protocol wrote it (docs/LEXICON.md §1).
            fingerprint = resume_mod.node_fingerprint(
                block=lexicon.canonical_path(block), params=serialize_params(params),
                input_hashes=[state.node_hashes.get(e["from"]["node"], "")
                              for e in in_edges] + inline_hashes,
                core_version=core_version,
                model=str(serialize_params(params).get("model", "")),
            )
            if self._on_node_start is not None:
                self._on_node_start(nid, fingerprint)
            entry = read_resume_entry(state, nid, fingerprint)
            if restore_node(state, nid, entry, block, params,
                            fingerprint, self._on_node_done):
                progress.bump()
                continue
            resume_kwargs: dict[str, Any] = {}
            if entry and entry.get("items") and resume_mod.item_resumable(block):
                resume_kwargs["resume_items"] = dict(entry["items"])
            if entry and entry.get("checkpoint") is not None:
                resume_kwargs["resume_state"] = entry["checkpoint"]
            on_checkpoint = (
                (lambda st, _n=nid: self._on_checkpoint(_n, st))
                if self._on_checkpoint is not None else None
            )
            # A node's failure is caught rather than thrown: each
            # consumer's port decides what an absent input means, and
            # sibling branches finish either way. A failure nothing
            # tolerates is raised at the end of the run.
            try:
                if nid in state.ahead:
                    # Already run, alongside its siblings. The
                    # bookkeeping below is this node's own and stays here,
                    # in topological order, so the manifest, the hashes
                    # and the emitted objects do not depend on which
                    # branch finished first.
                    done_ahead = state.ahead.pop(nid)
                    if isinstance(done_ahead, BaseException):
                        raise done_ahead
                    state.results[nid] = done_ahead
                elif is_remote(block, params):
                    # This node waits on somebody else's machine, so
                    # every other remote node that is ready waits with
                    # it rather than after it.
                    state.ahead.update(self._run_remote_wave(
                        nid, block, inputs, params, secrets,
                        nodes=state.nodes, edges=state.edges,
                        order=state.order, results=state.results,
                        missing=state.missing,
                        resolve_params=resolver.resolve_params,
                        resolve_value=resolver.resolve_value,
                        item_reporter=progress.open_items,
                        expand=progress.expand,
                        node_view=progress.node_view,
                        report=progress.report,
                        resume=state.resume, resume_kwargs=resume_kwargs))
                    done_ahead = state.ahead.pop(nid)
                    if isinstance(done_ahead, BaseException):
                        raise done_ahead
                    state.results[nid] = done_ahead
                elif ops.find(block) is not None:
                    state.results[nid] = self._run_op(
                        block, inputs, params,
                        on_item=on_item, on_start=progress.expand, secrets=secrets,
                        on_checkpoint=on_checkpoint,
                        input_paths=dict(input_paths),
                        bindings=state.bound_params if state.declared
                        else state.bindings,
                        result_base=state.result_base,
                        resume_items=resume_kwargs.get("resume_items"),
                        resume_state=resume_kwargs.get("resume_state"))
                else:
                    raise ValueError(f"unknown block: {block!r}")
            except MissingUpstream:
                raise
            except Exception as exc:  # noqa: BLE001 — recorded, then decided on
                state.failures[nid] = exc
                state.missing[nid] = {"reason": f"{type(exc).__name__}: {exc}",
                                      "source": [nid]}
                print(f"[graph] {nid} failed: {exc}")
                if self._on_node_done is not None:
                    self._on_node_done(nid, None, fingerprint)
                progress.bump()
                continue
            # Hash BEFORE emitting. The hash canonical-encodes the
            # result, so a result carrying a live object fails here,
            # locally and by name, rather than reaching the emit path and
            # failing as a transport fault. A collection is stored with
            # its items in key order: the same items in any order are the
            # same bytes.
            state.results[nid] = lexicon.canonical_collection(state.results[nid])
            # A result about a model's layers stays about them through
            # every records op: the landmarks on any input's header ride
            # onto an output that has none.
            state.results[nid] = copy_arch(inputs, state.results[nid])
            state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
            if state.result_base and state.discard and not state.outputs_of.get(nid):
                # Held, not emitted: the consumers read it from
                # memory, a resume from the spool, and the API never sees
                # the bytes. Its identity is its content hash, which the
                # manifest records and its consumers' lineage cites.
                state.held[nid] = (block, params)
                if self._on_node_kept is not None:
                    self._on_node_kept(nid, fingerprint, state.results[nid])
            elif state.result_base:
                names = state.outputs_of.get(nid, [])
                if state.declared_outputs is None:
                    target = f"{state.result_base}/{nid}"
                elif names:
                    target = f"{state.result_base}/{names[0]}"
                else:
                    target = f"{state.result_base}/{dataflow.INTERMEDIATES}/{nid}"
                to_emit = state.results[nid]
                if tensors_mod.is_tensor(to_emit):
                    # The rows go up first as raw shards under the
                    # result's label (retry-as-resume: a shard already
                    # there with the same hash is not sent again), then
                    # the header is emitted as the object itself. The
                    # in-memory result keeps its local shard dir: a
                    # consumer in this job reads the rows from there.
                    to_emit = tensors_mod.upload(
                        to_emit, target, lambda lab, path: bench.put_file(lab, path, kind="tensor_shard"),
                        have=bench.list_prefix_hashes(target))
                out = bench.emit(
                    target,
                    to_emit,
                    # Lineage names the inputs that EXIST. A node run
                    # under `on_missing` has an upstream that produced
                    # nothing and so stored nothing: there is no path to
                    # cite, and the absence is recorded under
                    # `nodes_missing` on the manifest instead. An upstream
                    # HELD rather than stored (discard mode) is cited by
                    # its content hash, which is a path form of its own.
                    inputs=list(dict.fromkeys([
                        *(cited for cited in (
                            state.node_paths.get(e["from"]["node"])
                            or (f"~hash/sha256:{state.node_hashes[e['from']['node']]}"
                                if e["from"]["node"] in state.held else None)
                            for e in in_edges) if cited is not None),
                        # …and the stored objects it read by reference.
                        *resolver.read_stored_inputs(node)])),
                    # Provenance records the stored identity.
                    operation=lexicon.canonical_path(block),
                    params=serialize_params(params),
                )
                state.node_paths[nid] = out["path"]
                # A node declared as several outputs is stored under each
                # name; the first is the one its consumers' lineage cites.
                for also in names[1:]:
                    bench.emit(f"{state.result_base}/{also}", to_emit,
                               inputs=[out["path"]],
                               operation=lexicon.canonical_path(block),
                               params=serialize_params(params))
            if (isinstance(state.results[nid], dict)
                    and state.results[nid].get("spend")):
                state.spend_by_node[nid] = state.results[nid]["spend"]
            # A held node was handed over by `on_node_kept`: it has no
            # emitted object for `on_node_done` to record.
            if self._on_node_done is not None and nid not in state.held:
                self._on_node_done(nid, state.node_paths.get(nid), fingerprint)
            # An expanded node's items already covered its worth; bumping
            # again would push `done` past `total` by one per such node.
            if not progress.expanded:
                progress.bump()

        # A failure nobody answered for fails the run — which is every
        # failure in a graph that declares no `on_missing` policy. It is
        # raised here, once the sibling branches have finished.
        orphaned = [nid for nid in state.order if nid in state.failures
                    and nid not in state.tolerated]
        if orphaned:
            first = orphaned[0]
            if len(orphaned) > 1:
                print(f"[graph] {len(orphaned)} nodes failed: "
                      f"{', '.join(orphaned)}")
            # A failed run keeps its intermediates even in discard mode:
            # they are the evidence of what went wrong. Stored where a
            # kept run would have stored them; a failure to store one is
            # said, and does not hide the failure being reported.
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

        terminals = [nid for nid in state.nodes
                     if not any(e["from"]["node"] == nid for e in state.edges)
                     and nid not in state.missing]

        def sanitize(v, at):
            """Manifests reference binary, never embed it: bytes are
            replaced by a stub pointing at the node object that holds
            the real payload."""
            if isinstance(v, bytes):
                return {"$binary": {"bytes": len(v), "stored_at": at}}
            if isinstance(v, dict):
                return {k: sanitize(x, at) for k, x in v.items()}
            if isinstance(v, list):
                return [sanitize(x, at) for x in v]
            return v

        if state.declared_outputs is None:
            kept = {nid: nid for nid in terminals}
        else:
            # By declared name. An output whose node produced nothing is
            # absent here and named under `nodes_missing`.
            kept = {o["name"]: o["from"]["node"] for o in state.declared_outputs
                    if o["from"]["node"] in state.results
                    and o["from"]["node"] not in state.missing}
        payload = {
            "kind": "run/result",
            "outputs": {name: sanitize(state.results[nid],
                                       state.node_paths.get(nid, ""))
                         for name, nid in kept.items()},
            **({"output_nodes": dict(kept)}
               if state.declared_outputs is not None else {}),
            "nodes_executed": [nid for nid in state.order
                               if nid not in state.missing],
            "node_paths": state.node_paths,
            # Every node's content hash and its upstream nodes: what a
            # result's lineage verifies against when an intermediate was
            # held rather than stored — and a record worth having either
            # way.
            "node_hashes": {nid: state.node_hashes[nid] for nid in state.order
                            if nid in state.node_hashes},
            "node_inputs": {nid: [e["from"]["node"]
                                  for e in sort_edges(state.edges, nid)]
                            for nid in state.order
                            if nid in state.results and nid not in state.missing},
            **({"keep": state.keep, "nodes_held": sorted(state.held)}
               if state.discard else {}),
            # What each node produced, small enough to read beside the
            # node in the composer without fetching its object: the kind,
            # and how many items or rows.
            "node_summaries": {nid: summarize_node(state.results[nid],
                                                   state.spend_by_node.get(nid))
                               for nid in state.order
                               if nid in state.results and nid not in state.missing},
            # What did not run, and why. A reader of this result must
            # never have to infer an absence from a shorter list.
            **({"nodes_missing": {nid: state.missing[nid] for nid in state.order
                                  if nid in state.missing}} if state.missing else {}),
            "resolved": resolver.resolved,
            # Where this ran. Recorded, never fingerprinted: bit-identity
            # is promised within a hardware class.
            "resources": {"hardware": hardware_class(),
                          **({"spend": total_spend(state.spend_by_node)}
                             if state.spend_by_node else {})},
        }
        prov = ms.Provenance(
            created_at=datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            produced_by=ms.ToolInfo(tool="mechbench-runner",
                                    version=core_version),
            inputs=[],
            params_fingerprint=ms.fingerprint_params(
                {"graph": state.graph, "bindings": state.bindings}),
            schema_version=ms.__version__,
        )
        return ms.Emitted(payload=payload, provenance=prov)
