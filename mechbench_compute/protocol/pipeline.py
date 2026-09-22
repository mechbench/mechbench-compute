"""Walking the graph: the one method that runs a protocol.

`_run_pipeline` takes a spec's graph, puts its nodes in topological
order, and for each one resolves its params, gathers its inputs off its
in-edges, decides whether a previous attempt's work can be reused, sends
it to an operation, hashes what came back, emits it, and accounts for
what it cost. The manifest at the end is the record of all of that.

The dispatch itself — which of an operation, a remote wave or a pure
block a node goes to — is the `try` near the end of the node loop, and
what it calls is `_run_op` in protocol/dispatch.py.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from mechbench_compute import lexicon, ops
from mechbench_compute.protocol.check_graph import check_graph
from mechbench_compute.protocol.copy_arch import copy_arch
from mechbench_compute.protocol.is_remote import is_remote
from mechbench_compute.protocol.missing_upstream import MissingUpstream
from mechbench_compute.protocol.progress import Progress
from mechbench_compute.protocol.protocol_spec import ProtocolSpec
from mechbench_compute.protocol.read_missing_policy import read_missing_policy
from mechbench_compute.protocol.resolver import Resolver
from mechbench_compute.protocol.serialize_params import serialize_params
from mechbench_compute.protocol.sort_edges import sort_edges
from mechbench_compute.protocol.sort_nodes import sort_nodes
from mechbench_compute.protocol.summarize_node import summarize_node
from mechbench_compute.protocol.total_spend import total_spend


class Pipeline:
    """Pipeline: see this module's docstring."""

    def _run_pipeline(self, spec: ProtocolSpec, on_progress=None,
                      secrets=None, resume=None) -> Any:
        """Execute a protocol graph: topological order over the nodes,
        pure blocks resolved from the core registry, model blocks
        executed in-process with the prefix cache. A node produces one
        value (edges' port names select inputs, not outputs), and the
        whole graph runs in this one job — the executor does no
        multi-job planning.

        Params may reference bindings: any string param "$name"
        resolves to spec bindings[name]."""
        from datetime import datetime

        import mechbench_schema as ms

        from mechbench_compute import __version__ as core_version
        from mechbench_compute.seeds import hardware_class

        from mechbench_compute import dataflow
        from mechbench_compute import tensors as tensors_mod

        extra = spec.extra or {}
        graph = extra.get("graph") or {}
        bindings = extra.get("bindings") or {}
        # The declared form: the run binds `params` and `inputs` by name,
        # and the graph refers to them with values no literal can be. It
        # is lowered here, so everything below — ordering, resume, missing
        # nodes, fingerprints — sees one graph shape.
        declared = dataflow.is_declared(graph)
        bound_params = extra.get("params") or {}
        if declared:
            graph = dataflow.lower(graph, extra.get("inputs") or {})
        nodes = {n["id"]: n for n in graph.get("nodes", [])}
        edges = graph.get("edges", [])
        # What the protocol declares it keeps: `[{name, from: {node}}]`,
        # from its signature. With these, a result is stored under its
        # declared NAME and every other node's value apart, as an
        # intermediate — so a node can be renamed without moving a result
        # anyone depends on. Without them, terminals are the outputs,
        # under their ids.
        declared_outputs = extra.get("outputs") if declared else None
        outputs_of: dict[str, list[str]] = {}
        for o in declared_outputs or []:
            source = o["from"]["node"]
            if source not in nodes:
                raise ValueError(
                    f"output {o['name']!r} comes from {source!r}, which is not a node")
            if o["name"] == dataflow.INTERMEDIATES:
                raise ValueError(
                    f"an output cannot be named {dataflow.INTERMEDIATES!r}: that is "
                    f"where intermediates are stored")
            outputs_of.setdefault(source, []).append(o["name"])
        # Eager discard: with `keep: "outputs"` a node that is not a
        # declared output is never emitted. Its result stays here for its
        # consumers, goes to the runner's spool for a resume, and is
        # cited downstream by content hash. The default keeps everything.
        keep = str(extra.get("keep") or "all")
        if keep not in ("all", "outputs"):
            raise ValueError(f"keep must be 'all' or 'outputs', not {keep!r}")
        discard = declared and keep == "outputs"
        #: Held results: node id -> (block, resolved params), what an
        #: emit of the evidence needs should the run fail.
        held: dict[str, tuple[str, Any]] = {}

        resolver = Resolver(
            declared=declared, bindings=bindings, bound_params=bound_params,
            secrets=secrets, on_download=self._on_download,
            on_download_bytes=self._on_download_bytes)

        order = sort_nodes(nodes, edges)
        results: dict[str, Any] = {}
        progress = Progress(on_progress, len(order),
                            on_spool_item=self._on_spool_item)
        current = {"nid": ""}

        from mechbench_compute import resume as resume_mod

        # Before anything runs: is this graph runnable at all? Blocks,
        # params and ports are decidable at load, and a graph that cannot
        # run says so in the first second rather than after the nodes
        # upstream of the mistake have been computed.
        check_graph(nodes, edges, order)
        if declared:
            dataflow.check_refs(nodes, bound_params)

        resume = resume or {}
        # A consumer may require a minimum resume level of an upstream
        # node (`require_resume: {port: level}`). A requirement the
        # upstream block cannot meet forces that node to restart
        # rather than reuse a partial; unknown level names refuse.
        forced_restart: set[str] = set()
        for nid_c, node_c in nodes.items():
            req = (node_c.get("params") or {}).get("require_resume") or {}
            for port, level in req.items():
                src = next((e["from"]["node"] for e in edges
                            if e["to"]["node"] == nid_c
                            and e["to"]["port"] == port), None)
                if src is None:
                    continue
                offered = resume_mod.resume_level(nodes[src]["block"],
                                                  nodes[src].get("params"),
                                                  nodes[src].get("inputs"))
                if not resume_mod.satisfies(offered, str(level)):
                    forced_restart.add(src)
        node_hashes: dict[str, str] = {}
        # Remote nodes run ahead of their turn, alongside a sibling;
        # their results wait here for the loop to reach them and do the
        # bookkeeping in topological order.
        ahead: dict[str, Any] = {}
        # Nodes that produced nothing, and why. A node lands here by
        # failing, or by being skipped because something
        # upstream of it did. `tolerated` records the ones some consumer
        # answered for; a failure nothing answered for is raised when
        # the run is otherwise over, so sibling branches still finish.
        missing: dict[str, dict[str, Any]] = {}
        failures: dict[str, BaseException] = {}
        tolerated: set[str] = set()

        # Per-node emission: every node's output becomes a bench object
        # under the job's result namespace, with
        # lineage inputs = its upstream nodes' paths and operation =
        # the block ref. The protocol graph and the lineage graph are
        # then the same graph, by construction.
        from mechbench_compute import bench

        result_base = extra.get("resultPath")
        node_paths: dict[str, str] = {}
        # What this run bought from other people: per node, and summed in
        # the manifest, so the bill is a property of the run rather than
        # something a reader reconstructs from items.
        spend_by_node: dict[str, dict[str, Any]] = {}

        for pos, nid in enumerate(order):
            node = nodes[nid]
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
            in_edges = sort_edges(edges, nid)
            by_port: dict[str, list[Any]] = {}
            for e in in_edges:
                by_port.setdefault(e["to"]["port"], []).append(e)
            op_here = lexicon.BY_NAME.get(block)
            # An upstream that produced nothing — it failed, or was
            # itself skipped — is answered by the port it was wired to.
            # `fail` is the default: the run stops here, with the
            # original error. `skip` passes the absence on.
            # `placeholder` hands the block an empty collection that
            # says it is one.
            absent = {
                port: [e for e in es if e["from"]["node"] in missing]
                for port, es in by_port.items()
            }
            absent = {p: es for p, es in absent.items() if es}
            if absent:
                policy_skip = False
                for port, es in sorted(absent.items()):
                    decl = op_here.port(port) if op_here else None
                    policy = read_missing_policy(decl, es)
                    source = es[0]["from"]["node"]
                    why = missing[source]
                    if policy == "fail":
                        raise MissingUpstream(nid, port, source, why) from None
                    if policy == "skip":
                        policy_skip = True
                    tolerated.update(e["from"]["node"] for e in es)
                if policy_skip:
                    src = sorted({e["from"]["node"]
                                  for es in absent.values() for e in es})
                    missing[nid] = {"reason": "an upstream is missing",
                                    "source": src}
                    print(f"[graph] {nid}: skipped ({', '.join(src)} missing)")
                    progress.bump()
                    continue
                # Only placeholders left: drop those edges and fill below.
                by_port = {
                    p: [e for e in es if e["from"]["node"] not in missing]
                    for p, es in by_port.items()
                }
                placeholders = {p: es for p, es in absent.items()}
            else:
                placeholders = {}
            inputs, input_paths = {}, {}
            for port, es in {p: es for p, es in by_port.items() if es}.items():
                decl = op_here.port(port) if op_here else None
                if decl is not None and decl.variadic:
                    inputs[port] = [
                        {"node": e["from"]["node"],
                         "value": results[e["from"]["node"]]} for e in es]
                    input_paths[port] = [
                        node_paths.get(e["from"]["node"], "") for e in es]
                else:
                    # One edge, as every port but a variadic one takes;
                    # more than one is refused at load by `_preflight`.
                    inputs[port] = results[es[-1]["from"]["node"]]
                    input_paths[port] = node_paths.get(es[-1]["from"]["node"], "")
            for port, es in sorted(placeholders.items()):
                decl = op_here.port(port) if op_here else None
                kind = (decl.kinds[0] if decl is not None else "records/record")
                if kind == lexicon.COLLECTION:
                    kind = "records/record"
                stand_in = lexicon.collection(kind, [], missing={
                    "reason": "the upstream produced nothing",
                    "source": sorted({e["from"]["node"] for e in es})})
                if port in inputs and isinstance(inputs[port], list):
                    inputs[port] = [*inputs[port],
                                    *({"node": e["from"]["node"],
                                       "value": stand_in} for e in es)]
                else:
                    inputs[port] = stand_in
            # An input given inline under the node's `inputs` — a
            # literal, or `{"$fetch": …}` of a stored object — fills a
            # port the way an edge does, and its content hash joins the
            # fingerprint the way an upstream node's would.
            inline_hashes: list[str] = []
            for port, raw in sorted((node.get("inputs") or {}).items()):
                if raw is None:
                    continue
                if port in inputs:
                    raise ValueError(
                        f"{nid}: port {port!r} is wired by an edge and also "
                        f"given under `inputs` — one or the other")
                inputs[port] = resolver.resolve_value(raw)
                if isinstance(raw, dict) and "$fetch" in raw:
                    input_paths[port] = str(resolver.resolve_value(raw["$fetch"]))
                inline_hashes.append(
                    f"{port}:{resume_mod.content_hash(inputs[port])}")
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
                input_hashes=[node_hashes.get(e["from"]["node"], "")
                              for e in in_edges] + inline_hashes,
                core_version=core_version,
                model=str(serialize_params(params).get("model", "")),
            )
            if self._on_node_start is not None:
                self._on_node_start(nid, fingerprint)
            entry = resume.get(nid) if isinstance(resume, dict) else None
            resume_kwargs: dict[str, Any] = {}
            if entry:
                if entry.get("fingerprint") != fingerprint:
                    print(f"[resume] {nid}: fingerprint changed; restarting")
                    entry = None
                elif nid in forced_restart:
                    print(f"[resume] {nid}: a consumer requires more than "
                          f"this block offers; restarting")
                    entry = None
            if entry and entry.get("done"):
                # Node skip: its object already exists on the bench
                # under this exact fingerprint. Fetch it as the result;
                # it was emitted by the earlier attempt.
                fetched = bench.fetch(entry["done"])
                results[nid] = (fetched.get("payload", fetched)
                                if isinstance(fetched, dict) else fetched)
                node_paths[nid] = entry["done"]
                node_hashes[nid] = resume_mod.content_hash(results[nid])
                if self._on_node_done is not None:
                    self._on_node_done(nid, node_paths[nid], fingerprint)
                progress.bump()
                continue
            if entry and "held" in entry and discard and not outputs_of.get(nid):
                # Node skip, discard mode: the earlier attempt held the
                # result on this device under this exact fingerprint. The
                # spool still has it; nothing to write again.
                results[nid] = entry["held"]
                node_hashes[nid] = resume_mod.content_hash(results[nid])
                held[nid] = (block, params)
                progress.bump()
                continue
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
                if nid in ahead:
                    # Already run, alongside its siblings. The
                    # bookkeeping below is this node's own and stays here,
                    # in topological order, so the manifest, the hashes
                    # and the emitted objects do not depend on which
                    # branch finished first.
                    done_ahead = ahead.pop(nid)
                    if isinstance(done_ahead, BaseException):
                        raise done_ahead
                    results[nid] = done_ahead
                elif is_remote(block, params):
                    # This node waits on somebody else's machine, so
                    # every other remote node that is ready waits with
                    # it rather than after it.
                    ahead.update(self._run_remote_wave(
                        nid, block, inputs, params, secrets,
                        nodes=nodes, edges=edges, order=order,
                        results=results, missing=missing,
                        resolve_params=resolver.resolve_params,
                        resolve_value=resolver.resolve_value,
                        item_reporter=progress.open_items,
                        expand=progress.expand,
                        node_view=progress.node_view,
                        report=progress.report,
                        resume=resume, resume_kwargs=resume_kwargs))
                    done_ahead = ahead.pop(nid)
                    if isinstance(done_ahead, BaseException):
                        raise done_ahead
                    results[nid] = done_ahead
                elif ops.find(block) is not None:
                    results[nid] = self._run_op(
                        block, inputs, params,
                        on_item=on_item, on_start=progress.expand, secrets=secrets,
                        on_checkpoint=on_checkpoint,
                        input_paths=dict(input_paths),
                        bindings=bound_params if declared else bindings,
                        result_base=extra.get("resultPath"),
                        resume_items=resume_kwargs.get("resume_items"),
                        resume_state=resume_kwargs.get("resume_state"))
                else:
                    raise ValueError(f"unknown block: {block!r}")
            except MissingUpstream:
                raise
            except Exception as exc:  # noqa: BLE001 — recorded, then decided on
                failures[nid] = exc
                missing[nid] = {"reason": f"{type(exc).__name__}: {exc}",
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
            results[nid] = lexicon.canonical_collection(results[nid])
            # A result about a model's layers stays about them through
            # every records op: the landmarks on any input's header ride
            # onto an output that has none.
            results[nid] = copy_arch(inputs, results[nid])
            node_hashes[nid] = resume_mod.content_hash(results[nid])
            if result_base and discard and not outputs_of.get(nid):
                # Held, not emitted: the consumers read it from
                # memory, a resume from the spool, and the API never sees
                # the bytes. Its identity is its content hash, which the
                # manifest records and its consumers' lineage cites.
                held[nid] = (block, params)
                if self._on_node_kept is not None:
                    self._on_node_kept(nid, fingerprint, results[nid])
            elif result_base:
                names = outputs_of.get(nid, [])
                if declared_outputs is None:
                    target = f"{result_base}/{nid}"
                elif names:
                    target = f"{result_base}/{names[0]}"
                else:
                    target = f"{result_base}/{dataflow.INTERMEDIATES}/{nid}"
                to_emit = results[nid]
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
                            node_paths.get(e["from"]["node"])
                            or (f"~hash/sha256:{node_hashes[e['from']['node']]}"
                                if e["from"]["node"] in held else None)
                            for e in in_edges) if cited is not None),
                        # …and the stored objects it read by reference.
                        *resolver.read_stored_inputs(node)])),
                    # Provenance records the stored identity.
                    operation=lexicon.canonical_path(block),
                    params=serialize_params(params),
                )
                node_paths[nid] = out["path"]
                # A node declared as several outputs is stored under each
                # name; the first is the one its consumers' lineage cites.
                for also in names[1:]:
                    bench.emit(f"{result_base}/{also}", to_emit,
                               inputs=[out["path"]],
                               operation=lexicon.canonical_path(block),
                               params=serialize_params(params))
            if isinstance(results[nid], dict) and results[nid].get("spend"):
                spend_by_node[nid] = results[nid]["spend"]
            # A held node was handed over by `on_node_kept`: it has no
            # emitted object for `on_node_done` to record.
            if self._on_node_done is not None and nid not in held:
                self._on_node_done(nid, node_paths.get(nid), fingerprint)
            # An expanded node's items already covered its worth; bumping
            # again would push `done` past `total` by one per such node.
            if not progress.expanded:
                progress.bump()

        # A failure nobody answered for fails the run — which is every
        # failure in a graph that declares no `on_missing` policy. It is
        # raised here, once the sibling branches have finished.
        orphaned = [nid for nid in order if nid in failures
                    and nid not in tolerated]
        if orphaned:
            first = orphaned[0]
            if len(orphaned) > 1:
                print(f"[graph] {len(orphaned)} nodes failed: "
                      f"{', '.join(orphaned)}")
            # A failed run keeps its intermediates even in discard mode:
            # they are the evidence of what went wrong. Stored where a
            # kept run would have stored them; a failure to store one is
            # said, and does not hide the failure being reported.
            for nid, (held_block, held_params) in held.items():
                try:
                    out = bench.emit(
                        f"{result_base}/{dataflow.INTERMEDIATES}/{nid}",
                        results[nid],
                        inputs=list(resolver.read_stored_inputs(nodes[nid])),
                        operation=lexicon.canonical_path(held_block),
                        params=serialize_params(held_params))
                    node_paths[nid] = out["path"]
                except Exception as exc:  # noqa: BLE001 — evidence, not the verdict
                    print(f"[graph] could not store the held result of {nid}: {exc}")
            raise failures[first]

        terminals = [nid for nid in nodes
                     if not any(e["from"]["node"] == nid for e in edges)
                     and nid not in missing]

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

        if declared_outputs is None:
            kept = {nid: nid for nid in terminals}
        else:
            # By declared name. An output whose node produced nothing is
            # absent here and named under `nodes_missing`.
            kept = {o["name"]: o["from"]["node"] for o in declared_outputs
                    if o["from"]["node"] in results
                    and o["from"]["node"] not in missing}
        payload = {
            "kind": "run/result",
            "outputs": {name: sanitize(results[nid], node_paths.get(nid, ""))
                         for name, nid in kept.items()},
            **({"output_nodes": dict(kept)} if declared_outputs is not None else {}),
            "nodes_executed": [nid for nid in order if nid not in missing],
            "node_paths": node_paths,
            # Every node's content hash and its upstream nodes: what a
            # result's lineage verifies against when an intermediate was
            # held rather than stored — and a record worth having either
            # way.
            "node_hashes": {nid: node_hashes[nid] for nid in order if nid in node_hashes},
            "node_inputs": {nid: [e["from"]["node"] for e in sort_edges(edges, nid)]
                            for nid in order if nid in results and nid not in missing},
            **({"keep": keep, "nodes_held": sorted(held)} if discard else {}),
            # What each node produced, small enough to read beside the
            # node in the composer without fetching its object: the kind,
            # and how many items or rows.
            "node_summaries": {nid: summarize_node(results[nid], spend_by_node.get(nid))
                               for nid in order if nid in results and nid not in missing},
            # What did not run, and why. A reader of this result must
            # never have to infer an absence from a shorter list.
            **({"nodes_missing": {nid: missing[nid] for nid in order
                                  if nid in missing}} if missing else {}),
            "resolved": resolver.resolved,
            # Where this ran. Recorded, never fingerprinted: bit-identity
            # is promised within a hardware class.
            "resources": {"hardware": hardware_class(),
                          **({"spend": total_spend(spend_by_node)}
                             if spend_by_node else {})},
        }
        prov = ms.Provenance(
            created_at=datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            produced_by=ms.ToolInfo(tool="mechbench-runner",
                                    version=core_version),
            inputs=[],
            params_fingerprint=ms.fingerprint_params(
                {"graph": graph, "bindings": bindings}),
            schema_version=ms.__version__,
        )
        return ms.Emitted(payload=payload, provenance=prov)
