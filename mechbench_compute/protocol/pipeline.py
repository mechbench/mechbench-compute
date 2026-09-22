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

from collections.abc import Mapping
from datetime import UTC
from typing import Any

from mechbench_compute import lexicon, ops
from mechbench_compute.protocol.check_graph import check_graph
from mechbench_compute.protocol.copy_arch import copy_arch
from mechbench_compute.protocol.is_remote import is_remote
from mechbench_compute.protocol.missing_upstream import MissingUpstream
from mechbench_compute.protocol.protocol_spec import ProtocolSpec
from mechbench_compute.protocol.read_missing_policy import read_missing_policy
from mechbench_compute.protocol.serialize_params import serialize_params
from mechbench_compute.protocol.sort_edges import sort_edges
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
        from mechbench_compute.blocks import PURE_BLOCKS
        from mechbench_compute.seeds import hardware_class

        from pathlib import Path

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

        # What actually resolved: every $fetch's content hash and every
        # model ref's snapshot commit, recorded into the result manifest
        # — reproducibility by record; pins opt into strictness.
        resolved: dict[str, dict] = {"objects": {}, "models": {}}

        def fetch_object(ref, want=None):
            from mechbench_compute import bench
            fetched, meta = bench.fetch(ref, with_meta=True)
            got = (meta or {}).get("content_hash") or ""
            resolved["objects"][str(ref)] = got
            if want and not got.endswith(str(want)):
                raise ValueError(
                    f"pinned object {ref!r} resolved to {got!r}, "
                    f"expected sha256 {want!r}")
            payload = fetched.get("payload", fetched) if isinstance(fetched, dict) else fetched
            if tensors_mod.is_tensor(payload):
                # A tensor collection's rows are shards beside it: fetched
                # into the cache, verified, and read one shard at a time.
                # The same progress callbacks a checkpoint's fetch uses.
                if self._on_download is not None:
                    self._on_download(str(ref), None)
                payload = tensors_mod.materialize(
                    payload, str(ref), bench.get_file_chunks,
                    Path.home() / ".mechbench" / "tensors",
                    on_bytes=self._on_download_bytes)
            return payload

        def stored_inputs_of(node):
            """The bench objects a node reads by reference, in the order it
            names them: its lineage inputs beside its upstream nodes. A
            frequency table fetched into a param is an input of the node
            that trained on it. Read off the node itself rather than
            collected as fetches happen, because remote nodes resolve
            alongside their siblings and a fetch's timing says nothing
            about whose it was."""
            found: list[str] = []

            def walk(v):
                if declared:
                    if dataflow.is_param_ref(v):
                        v = bound_params.get(v["$param"])
                    if dataflow.is_object_ref(v):
                        path = v["$ref"].get("bench")
                        if isinstance(path, str) and path not in found:
                            found.append(path)
                        return
                elif isinstance(v, dict) and "$fetch" in v:
                    path = v["$fetch"]
                    if isinstance(path, str) and path.startswith("$"):
                        path = bindings.get(path[1:])
                    if isinstance(path, str) and path not in found:
                        found.append(path)
                    return
                if isinstance(v, dict):
                    for x in v.values():
                        walk(x)
                elif isinstance(v, list):
                    for x in v:
                        walk(x)

            walk(node.get("inputs") or {})
            walk(node.get("params") or {})
            return found

        def resolve_declared(v, keep_reference=False):
            """The declared form's two references, and nothing else: a
            string that begins with `$` is a string here."""
            if dataflow.is_param_ref(v):
                name = v["$param"]
                if name not in bound_params:
                    raise ValueError(f"unbound param: {name!r}")
                return resolve_declared(bound_params[name], keep_reference)
            if dataflow.is_object_ref(v):
                which, source = dataflow.source_of(v)
                if keep_reference:
                    # The op asked for the address, not what is at it.
                    return dict(v["$ref"])
                if which == "bench":
                    return fetch_object(source, v["$ref"].get("sha256"))
                if which == "hf_dataset":
                    return resolve_hf_dataset(source)
                return resolve_hf_adapter(source)
            if isinstance(v, dict):
                return {k: resolve_declared(x) for k, x in v.items()}
            if isinstance(v, list):
                return [resolve_declared(x) for x in v]
            return v

        def resolve_value(v):
            """Recursive param resolution. Forms beyond literals:
            "$name"                    -> the run binding (a string).
            {"$fetch": ref}            -> the bench object's payload.
            {"$fetch": ref, "sha256":} -> same, verified against the
                                          pinned content hash.
            In a declared graph: {"$param": name} and {"$ref": source}."""
            if declared:
                return resolve_declared(v)
            if isinstance(v, str) and v.startswith("$"):
                name = v[1:]
                if name not in bindings:
                    raise ValueError(f"unbound hole: {v}")
                return bindings[name]
            if isinstance(v, dict) and set(v.keys()) == {"$hf_adapter"}:
                return resolve_hf_adapter(resolve_value(v["$hf_adapter"]))
            if isinstance(v, dict) and set(v.keys()) == {"$hf_dataset"}:
                return resolve_hf_dataset(resolve_value(v["$hf_dataset"]))
            if isinstance(v, dict) and "$fetch" in v                     and set(v.keys()) <= {"$fetch", "sha256"}:
                return fetch_object(resolve_value(v["$fetch"]), v.get("sha256"))
            if isinstance(v, dict):
                return {k: resolve_value(x) for k, x in v.items()}
            if isinstance(v, list):
                return [resolve_value(x) for x in v]
            return v

        def resolve_hf_dataset(spec):
            """{"$hf_dataset": {repo, split, config?, revision?, limit?,
            columns?: {id?, coords?: [...]}}} -> a record stream shaped
            like our own: {id, coords, values} per row, columns as
            values (Template substitutes them), declared coords
            columns lifted into coords. Resolution recorded (repo,
            requested revision, arrow fingerprint, rows)."""
            from datasets import load_dataset

            hf_token = (secrets or {}).get("hf", {}).get("token")
            repo = spec["repo"]
            split = spec.get("split", "train")
            config = spec.get("config")
            revision = spec.get("revision")
            limit = spec.get("limit")
            colmap = spec.get("columns") or {}
            kwargs = {"split": split}
            if revision:
                kwargs["revision"] = revision
            if hf_token:
                kwargs["token"] = hf_token
            ds = (load_dataset(repo, config, **kwargs) if config
                  else load_dataset(repo, **kwargs))
            n = min(int(limit), len(ds)) if limit else len(ds)
            id_col = colmap.get("id")
            coord_cols = list(colmap.get("coords") or [])
            records = []
            for i in range(n):
                row = ds[i]
                rid = str(row[id_col]) if id_col else f"{split}-{i}"
                coords = {c: str(row[c]) for c in coord_cols}
                values = {k: str(v) for k, v in row.items()
                          if k not in coord_cols}
                records.append({"id": rid,
                                 "coords": {"split": split, **coords},
                                 "values": values})
            key = f"{repo}@{revision}" if revision else repo
            resolved.setdefault("datasets", {})[key] = {
                "repo": repo, "config": config, "split": split,
                "revision": revision,
                "fingerprint": getattr(ds, "_fingerprint", None),
                "rows_total": len(ds), "rows_used": n}
            return lexicon.collection("records/record", records)

        def resolve_hf_adapter(spec):
            """{"$hf_adapter": {repo, revision?}} -> an adapter object
            payload imported from a hub PEFT LoRA repo. The resolved
            snapshot commit is recorded."""
            from huggingface_hub import snapshot_download

            from mechbench_compute.peft import peft_import

            repo = spec["repo"]
            revision = spec.get("revision")
            kwargs = {"allow_patterns": ["adapter_*", "*.json"]}
            if revision:
                kwargs["revision"] = revision
            hf_token = (secrets or {}).get("hf", {}).get("token")
            if hf_token:
                kwargs["token"] = hf_token
            local = snapshot_download(repo, **kwargs)
            commit = local.rstrip("/").rsplit("/", 1)[-1]
            payload = peft_import(local)
            resolved.setdefault("adapters", {})[
                f"{repo}@{revision}" if revision else repo] = {
                "repo": repo, "revision": revision, "commit": commit,
                "target_modules": payload["lora"]["target_modules"]}
            return payload

        def record_model(ref):
            if not isinstance(ref, str) or ref in resolved["models"]:
                return
            from mechbench_compute.hub import (
                parse_model_ref,
                resolve_cached_revision,
            )
            try:
                repo, rev = parse_model_ref(ref)
                resolved["models"][ref] = {
                    "repo": repo, "pinned": rev,
                    "commit": resolve_cached_revision(repo, rev)}
            except Exception:  # noqa: BLE001 — recording is best-effort
                resolved["models"][ref] = {"repo": ref, "pinned": None,
                                            "commit": None}

        def resolve_params(params, block=None):
            if declared and block is not None:
                # A param whose op asked for the reference itself gets the
                # address; every other is resolved to what is there.
                return {k: resolve_declared(v, dataflow.wants_reference(block, k))
                        for k, v in (params or {}).items()}
            return {k: resolve_value(v) for k, v in (params or {}).items()}

        # Topological order (Kahn). The API validated acyclicity, but a
        # runner never trusts its inputs to be well-formed.
        indeg = {nid: 0 for nid in nodes}
        for e in edges:
            indeg[e["to"]["node"]] += 1
        order = [nid for nid, d in indeg.items() if d == 0]
        i = 0
        while i < len(order):
            for e in edges:
                if e["from"]["node"] == order[i]:
                    t = e["to"]["node"]
                    indeg[t] -= 1
                    if indeg[t] == 0:
                        order.append(t)
            i += 1
        if len(order) != len(nodes):
            raise ValueError("pipeline graph has a cycle")

        # Progress: one unit per node, but model blocks expand the
        # denominator to their item count on entry and tick per item —
        # the board's bar moves per condition/story, not per node.
        #
        # Alongside the flat scalar, STRUCTURE: which node the run is in
        # and how far through it. A denominator that grows mid-run reads
        # as a bug to anyone watching; "node 3/5, 12/40" only ever counts
        # up. Passed as a third argument only when the callback accepts
        # one, so a two-argument callback keeps working.
        import inspect

        results: dict[str, Any] = {}
        total_units = len(order)
        done_units = 0
        node_view = {"index": 0, "count": len(order), "id": "",
                     "done": 0, "total": 0}
        expanded = False
        try:
            wants_node = (on_progress is not None and
                          len(inspect.signature(on_progress).parameters) >= 3)
        except (TypeError, ValueError):  # builtins, odd callables
            wants_node = False

        def report():
            if on_progress is None:
                return
            if wants_node:
                on_progress(done_units, total_units, dict(node_view))
            else:
                on_progress(done_units, total_units)

        def bump(n=1):
            nonlocal done_units
            done_units += n
            report()

        def expand(n_items):
            nonlocal total_units, expanded
            expanded = True
            if n_items > 1:
                total_units += n_items - 1
            node_view["total"] = max(int(n_items), 0)
            report()

        current = {"nid": ""}

        def item_reporter(nid: str):
            """One node's item callback. Bound to the node rather than
            reading a shared `current`, because two nodes can be in
            flight at once and an item spooled under the wrong node's id
            is a resumed job reusing another node's work."""

            def on_item(key=None, item=None, reused=False):
                # Blocks that know nothing of resume call this bare; an
                # item-resumable block names the item so the runner can
                # spool it. A reused item counts as progress and is not
                # spooled again.
                node_view["done"] += 1
                bump(1)
                if (self._on_spool_item is not None and key is not None
                        and not reused):
                    self._on_spool_item(nid, key, item)

            return on_item

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
            node_view.update(index=pos + 1, id=nid, done=0, total=0)
            expanded = False
            report()
            raw_params = node.get("params") or {}
            if block in ("records/map", "records/fold") and isinstance(raw_params.get("body"), Mapping):
                # A map's or a fold's body is the CHILD run's graph,
                # holes and all: `$topic` is bound per record by `bind`,
                # `$participant` per step by `over`, not by this run.
                # Resolving it here would refuse a hole that is not this
                # protocol's to fill.
                params = resolve_params(
                    {k: v for k, v in raw_params.items() if k != "body"}, block)
                params["body"] = raw_params["body"]
            else:
                params = resolve_params(raw_params, block)
            if "model" in params:
                # A binding may be a structured ModelRef. Normalize it
                # HERE — adapters are fetched through the same recording
                # path as $fetch, so a run's manifest names everything it
                # actually loaded.
                mval = params.get("model")
                if isinstance(mval, dict) or hasattr(mval, "adapter_labels"):
                    from mechbench_compute import model_ref as model_ref_mod

                    def _fetch_recording(label):
                        from mechbench_compute import bench

                        fetched, meta = bench.fetch(label, with_meta=True)
                        resolved["objects"][str(label)] = (
                            (meta or {}).get("content_hash") or ""
                        )
                        return (
                            fetched.get("payload", fetched)
                            if isinstance(fetched, dict)
                            else fetched
                        )

                    ref = model_ref_mod.resolve(mval, fetch=_fetch_recording)
                    params = {**params, "model": ref}
                    # An endpoint has no repo to pin; what a run records
                    # about it is the version that ANSWERED, per call.
                    if not ref.is_endpoint:
                        record_model(ref.base)
                else:
                    record_model(mval)
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
                    bump()
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
                inputs[port] = resolve_value(raw)
                if isinstance(raw, dict) and "$fetch" in raw:
                    input_paths[port] = str(resolve_value(raw["$fetch"]))
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
            on_item = item_reporter(nid)
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
                bump()
                continue
            if entry and "held" in entry and discard and not outputs_of.get(nid):
                # Node skip, discard mode: the earlier attempt held the
                # result on this device under this exact fingerprint. The
                # spool still has it; nothing to write again.
                results[nid] = entry["held"]
                node_hashes[nid] = resume_mod.content_hash(results[nid])
                held[nid] = (block, params)
                bump()
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
                        resolve_params=resolve_params,
                        resolve_value=resolve_value,
                        item_reporter=item_reporter, expand=expand,
                        node_view=node_view, report=report,
                        resume=resume, resume_kwargs=resume_kwargs))
                    done_ahead = ahead.pop(nid)
                    if isinstance(done_ahead, BaseException):
                        raise done_ahead
                    results[nid] = done_ahead
                elif ops.find(block) is not None:
                    results[nid] = self._run_op(
                        block, inputs, params,
                        on_item=on_item, on_start=expand, secrets=secrets,
                        on_checkpoint=on_checkpoint,
                        input_paths=dict(input_paths),
                        bindings=bound_params if declared else bindings,
                        result_base=extra.get("resultPath"),
                        resume_items=resume_kwargs.get("resume_items"),
                        resume_state=resume_kwargs.get("resume_state"))
                elif block in PURE_BLOCKS:
                    results[nid] = PURE_BLOCKS[block](inputs, params)
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
                bump()
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
                        *stored_inputs_of(node)])),
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
            if not expanded:
                bump()

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
                        inputs=list(stored_inputs_of(nodes[nid])),
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
            "resolved": resolved,
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
