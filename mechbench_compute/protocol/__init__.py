"""Protocol execution: turn an experiment spec into results.

Lives in the compute layer, beside the primitives it drives, because it
is pure computation — it takes a spec, runs blocks against a loaded
model, and returns typed payloads. It reaches no network and knows
nothing about jobs, queues or credentials; `mechbench-runner` owns all
of that and calls in here once it has claimed something to do.


Supports only `layer_ablation` in v0 — the same kind the job-runner
shim handled. The MCP `run_experiment` tool calls this directly;
the job-runner dispatches to it after claiming a queued job.

The model is loaded once per process and reused. Load cost is
significant (minutes on first call, seconds on cached weights), so
callers should hold the runner for the process's lifetime rather
than instantiating per request.
"""

from __future__ import annotations

import json
import re
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import mlx.core as mx
import numpy as np
from mechbench_schema import (
    AblationPrompt,
    LayerAblationPayload,
    LayerAggregates,
)

from mechbench_compute import GLOBAL_LAYERS, N_LAYERS, Ablate, Model, lexicon, ops
from mechbench_compute import thinking as THINK
from mechbench_compute.protocol.tokenizer_id import _tokenizer_id  # noqa: F401
from mechbench_compute.protocol.wire_model import _wire_model  # noqa: F401
from mechbench_compute.protocol.protocol_spec import ProtocolSpec  # noqa: F401
from mechbench_compute.protocol.wire_params import _wire_params  # noqa: F401


class ProtocolExecutor:
    def __init__(self, on_download=None, on_download_bytes=None, *,
                 on_node_start=None, on_spool_item=None,
                 on_checkpoint=None, on_node_done=None, on_node_kept=None,
                 limiter=None, budget=None) -> None:
        self._model: Model | None = None
        self._model_id: str | None = None
        # Called just before weights are fetched, and only then: the runner
        # uses it to announce a wait that can run to gigabytes.
        self._on_download = on_download
        self._on_download_bytes = on_download_bytes
        # Resume plumbing (epic 000320 / 000322). All optional; the
        # runner wires them to its spool. `on_node_start(nid, fp)`
        # names the process identity a node runs under;
        # `on_spool_item(nid, key, item)` hands over each completed
        # item of an item-resumable block; `on_checkpoint(nid, state)`
        # a training checkpoint; `on_node_done(nid, path, fp)` a
        # finished node's emitted object; `on_node_kept(nid, fp, result)`
        # a finished node's result HELD on the device instead of emitted
        # (a run with `keep: "outputs"`, 000561), so a resume on this
        # device can pick it up without the bench ever having seen it.
        self._on_node_start = on_node_start
        self._on_spool_item = on_spool_item
        self._on_checkpoint = on_checkpoint
        self._on_node_done = on_node_done
        self._on_node_kept = on_node_kept
        # Rate limits are the runner's business (000338): it knows what
        # else is running against the same account. Absent one, remote
        # calls are unthrottled and only the provider's own 429s slow
        # them down.
        self._limiter = limiter
        # A JOB-level budget (000338), when the runner sets one: every
        # remote node's own cap is chained under it, so a graph whose
        # node caps sum past the job's cannot spend past the job's. The
        # runner reads `spent_usd` off it live to report spend.
        self._budget = budget

    def _model_loaded(self, model_id) -> Model:
        """The model an operation declared, loading it if it is not resident.

        Accepts a bare "repo[@rev]" string or a ModelRef (000312 Arc A),
        whose BASE is what loads here — adapters are the fuse layer's
        business, not the loader's.

        `model_id` is required, and there is no fallback to whatever happens
        to be loaded. An operation that ran a model it did not name produces
        a result whose recorded model is a claim rather than a fact — which is
        precisely the bug this signature exists to prevent (the flat
        layer_ablation path did exactly that: it recorded the requested model
        and executed the warmed one).
        """
        if getattr(model_id, "is_endpoint", False):
            # An endpoint has no weights to load, and handing a provider
            # model id to the hub would produce a download error about a
            # repo that was never meant to exist.
            raise ValueError(
                f"{model_id.describe()} is a remote endpoint: this operation "
                "runs local weights and cannot use one. Use "
                "text/chat, which serves both.")
        if hasattr(model_id, "base"):
            # The ref's base names WHERE the weights come from, and a
            # bench base must be translated to its materialized local
            # directory HERE — in the one place every call site funnels
            # through. The wrapper used to translate and the block's own
            # no-op re-call did not, so the label reached the hub as if
            # it were a repo id ("Repo id must be in the form
            # 'namespace/repo_name'") — after a 54-minute download had
            # already succeeded (2026-08-25, the first read ever run on
            # a bench-based checkpoint).
            if getattr(model_id, "base_kind", None) == "bench":
                model_id = str(self._materialize_checkpoint(model_id.base))
            else:
                model_id = model_id.base
        if not model_id:
            raise ValueError(
                "this operation did not declare a model. Every operation that "
                "runs one names it in its own params, so the result can say "
                "which weights produced it."
            )
        # One model in memory at a time; swapping ids reloads.
        if self._model is None or (model_id and model_id != self._model_id):
            self._model = Model.load(model_id, on_download=self._on_download,
                                     on_download_bytes=self._on_download_bytes)
            self._model_id = model_id
        return self._model

    def run(self, spec: ProtocolSpec, on_progress=None,
            secrets=None, resume=None, budget=None) -> Any:
        """Execute a job spec. `on_progress(done, total)` is invoked
        after each unit of work for kinds that have a natural unit
        (decision_distribution: one condition); it must be cheap and
        may be None.

        `resume` (epic 000320): `{node_id: {"fingerprint": str, and
        one of "done": <emitted object path> | "items": {key: item} |
        "checkpoint": <training state>}}`. Each entry is honoured
        only under an equal node fingerprint — otherwise that node
        restarts. The result is byte-identical to an uninterrupted
        run by construction; nothing about resumption is recorded in
        it."""
        if spec.kind == "layer_ablation":
            return self._run_layer_ablation(spec.prompt, spec.model_id)
        if spec.kind == "decision_distribution":
            return self._legacy_decision_distribution(spec, on_progress)
        if spec.kind == "pipeline":
            # A job-level budget arrives per RUN (the executor outlives
            # the job; the cap does not), and every remote node's cap is
            # chained under it.
            if budget is not None:
                self._budget = budget
            return self._run_pipeline(spec, on_progress, secrets=secrets,
                                      resume=resume)
        raise ValueError(f"unsupported protocolKind: {spec.kind!r}")


    @staticmethod
    def model_ref(model: Model) -> str | None:
        """`repo@commit` for a loaded model — what a result should record."""
        if getattr(model, "repo_id", None) and getattr(model, "revision", None):
            return f"{model.repo_id}@{model.revision}"
        return None

    def _run_layer_ablation(
        self, prompt: str, model_id: str
    ) -> LayerAblationPayload:
        model = self._model_loaded(model_id)

        ids = model.tokenize(prompt)
        baseline = model.run(ids)
        baseline_lp = _last_logp(baseline.logits)
        top1_id = int(np.argmax(baseline_lp))
        baseline_top1 = float(baseline_lp[top1_id])

        damage = np.zeros(N_LAYERS, dtype=np.float32)
        for layer in range(N_LAYERS):
            ids = model.tokenize(prompt)
            result = model.run(ids, interventions=[Ablate.layer(layer)])
            lp = _last_logp(result.logits)
            damage[layer] = float(lp[top1_id]) - baseline_top1

        prompts = [
            AblationPrompt(
                text=prompt,
                target="",
                top1_id=top1_id,
                baseline_logprob=round(baseline_top1, 4),
                damage=[round(float(v), 4) for v in damage],
            )
        ]
        return LayerAblationPayload(
            protocol="mechbench-runner:layer_ablation",
            description=(
                "Single-prompt layer ablation: zero each decoder block's "
                "residual-stream update and measure Δ log p of the "
                "model's top-1 prediction."
            ),
            # The resolved commit, not the reference: a payload has to say
            # which weights produced it, and a moving ref does not.
            model=self.model_ref(model) or model_id,
            n_layers=N_LAYERS,
            global_layers=list(GLOBAL_LAYERS),
            prompts=prompts,
            aggregates=LayerAggregates(
                mean=[round(float(v), 4) for v in damage],
                median=[round(float(v), 4) for v in damage],
            ),
        )


    def _legacy_decision_distribution(self, spec: ProtocolSpec,
                                      on_progress=None) -> Any:
        """The pre-protocol decision_distribution kind, kept as a thin
        shim over the decision-read block (superseded-code cleanup,
        2026-08-19): same spec in, same payload shape out, one
        implementation."""
        from datetime import datetime

        import mechbench_schema as ms

        from mechbench_compute import __version__ as core_version

        extra = spec.extra or {}
        # The job spec names the protocol (id + version) that produced
        # this run. Kept for anything that wants a STABLE identity for a
        # node across compute releases — a memo label, for one.
        self._protocol_ref = (extra.get("protocolId"), extra.get("protocolVersion"))
        conditions = extra.get("conditions", [])
        result = self._run_op(
            "logits/read", {"conditions": conditions}, {"model": spec.model_id})
        prov = ms.Provenance(
            created_at=datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            produced_by=ms.ToolInfo(tool="mechbench-runner",
                                    version=core_version),
            inputs=[],
            params_fingerprint=ms.fingerprint_params(
                {k: v for k, v in extra.items() if k != "resultPath"}),
            schema_version=ms.__version__,
        )
        return ms.Emitted(
            payload=lexicon.canonical_collection(lexicon.collection(
                "logits/decision", lexicon.items_of(result),
                model=spec.model_id)),
            provenance=prov)

    def _run_pipeline(self, spec: ProtocolSpec, on_progress=None,
                      secrets=None, resume=None) -> Any:
        """Execute a protocol graph (epic 000258, arc B): topological
        order over the nodes, pure blocks resolved from the core
        registry, model blocks executed in-process with the prefix
        cache. v1 restrictions: single output per node (edges' port
        names select inputs but every node produces one value) and the
        whole graph runs in this one job — multi-job planning is the
        planner's future concern, not the executor's.

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
        # The declared form (epic 000553): the run binds `params` and
        # `inputs` by name, and the graph refers to them with values no
        # literal can be. It is lowered here into the shapes this executor
        # has always run, so everything below — ordering, resume, missing
        # nodes, fingerprints — sees what it saw before.
        declared = dataflow.is_declared(graph)
        bound_params = extra.get("params") or {}
        if declared:
            graph = dataflow.lower(graph, extra.get("inputs") or {})
        nodes = {n["id"]: n for n in graph.get("nodes", [])}
        edges = graph.get("edges", [])
        # What the protocol declares it keeps (000558): `[{name, from:
        # {node}}]`, from its signature. With these, a result is stored
        # under its declared NAME and every other node's value apart, as
        # an intermediate — so a node can be renamed without moving a
        # result anyone depends on. Without them (every legacy protocol)
        # nothing changes: terminals are the outputs, under their ids.
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
        # Eager discard (000561): with `keep: "outputs"` a node that is not
        # a declared output is never emitted. Its result stays here for
        # its consumers, goes to the runner's spool for a resume, and is
        # cited downstream by content hash. The default keeps everything,
        # as it always has.
        keep = str(extra.get("keep") or "all")
        if keep not in ("all", "outputs"):
            raise ValueError(f"keep must be 'all' or 'outputs', not {keep!r}")
        discard = declared and keep == "outputs"
        #: Held results: node id -> (block, resolved params), what an
        #: emit of the evidence needs should the run fail.
        held: dict[str, tuple[str, Any]] = {}

        # What actually resolved (task 000260): every $fetch's content
        # hash and every model ref's snapshot commit, recorded into the
        # result manifest — reproducibility by record; pins opt into
        # strictness.
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
                # A tensor collection's rows are shards beside it
                # (000613): fetched into the cache, verified, and the
                # consumer reads them one shard at a time. The same
                # progress callbacks a checkpoint's fetch uses.
                if self._on_download is not None:
                    self._on_download(str(ref), None)
                payload = tensors_mod.materialize(
                    payload, str(ref), bench.get_file_chunks,
                    Path.home() / ".mechbench" / "tensors",
                    on_bytes=self._on_download_bytes)
            return payload

        def stored_inputs_of(node):
            """The bench objects a node reads by reference, in the order it
            names them: its lineage inputs beside its upstream nodes
            (000557). A frequency table fetched into a param is an input of
            the node that trained on it, and until now lineage did not say
            so. Read off the node itself rather than collected as fetches
            happen, because remote nodes resolve alongside their siblings
            and a fetch's timing says nothing about whose it was."""
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
            payload imported from a hub PEFT LoRA repo. Token plumbing
            arrives with 000264 (huggingface_hub token= kwarg); the
            resolved snapshot commit is recorded."""
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
        # Alongside the flat scalar, STRUCTURE (000316): which node the
        # run is in and how far through it. A denominator that grows
        # mid-run reads as a bug to anyone watching; "node 3/5, 12/40"
        # only ever counts up. Passed as a third argument when the
        # callback accepts one, so an older runner keeps working.
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
            flight at once (task 000396) and an item spooled under the
            wrong node's id is a resumed job reusing another node's
            work."""

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
        # params and ports are decidable at load, and a graph that
        # cannot run should say so in the first second rather than after
        # the nodes upstream of the mistake have been computed (000512,
        # 000513).
        _preflight(nodes, edges, order)
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
        # Remote nodes run ahead of their turn, alongside a sibling
        # (task 000396); their results wait here for the loop to
        # reach them and do the bookkeeping in topological order.
        ahead: dict[str, Any] = {}
        # Nodes that produced nothing, and why (task 000399). A node
        # lands here by failing, or by being skipped because something
        # upstream of it did. `tolerated` records the ones some consumer
        # answered for; a failure nothing answered for is raised when
        # the run is otherwise over, so sibling branches still finish.
        missing: dict[str, dict[str, Any]] = {}
        failures: dict[str, BaseException] = {}
        tolerated: set[str] = set()

        # Per-node emission (arc B second half): every node's output
        # becomes a bench object under the job's result namespace, with
        # lineage inputs = its upstream nodes' paths and operation =
        # the block ref. The protocol graph and the lineage graph are
        # then the same graph, by construction.
        from mechbench_compute import bench

        result_base = extra.get("resultPath")
        node_paths: dict[str, str] = {}
        # What this run bought from other people (000337): per node, and
        # summed in the manifest, so the bill is a property of the run
        # rather than something a reader reconstructs from items.
        spend_by_node: dict[str, dict[str, Any]] = {}

        for pos, nid in enumerate(order):
            node = nodes[nid]
            # Any spelling a protocol may carry — bare, stored, with the
            # retired `/1`, or a pre-rename name — becomes the one bare
            # name here, once, before anything hashes it. A retired
            # spelling warns; an unknown one refuses by name.
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
                # `$participant` per step by `over`, not by this run
                # (tasks 000400, 000617). Resolving it here would refuse
                # a hole that is not this protocol's to fill.
                params = resolve_params(
                    {k: v for k, v in raw_params.items() if k != "body"}, block)
                params["body"] = raw_params["body"]
            else:
                params = resolve_params(raw_params, block)
            if "model" in params:
                # The model algebra (000312 Arc A): a binding may be a
                # structured ModelRef. Normalize it HERE — adapters are
                # fetched through the same recording path as $fetch, so
                # a run's manifest names everything it actually loaded.
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
            # Edges in a canonical order (task 000397): by port, then by
            # the edge's own `index` when it has one, then by source node
            # id. Two consequences. A VARIADIC port receives them as an
            # ordered list, so a node over several branches knows which
            # branch is which. And a node's fingerprint no longer depends
            # on the order its author happened to write the edges in —
            # before this, moving an edge in the JSON restarted every
            # cached and resumed thing downstream of it.
            in_edges = _ordered_edges(edges, nid)
            by_port: dict[str, list[Any]] = {}
            for e in in_edges:
                by_port.setdefault(e["to"]["port"], []).append(e)
            op_here = lexicon.BY_NAME.get(block)
            # An upstream that produced nothing — it failed, or was
            # itself skipped — is answered by the port it was wired to
            # (task 000399). `fail` is the default and is what every
            # graph did before: the run stops here, with the original
            # error. `skip` passes the absence on. `placeholder` hands
            # the block an empty collection that says it is one.
            absent = {
                port: [e for e in es if e["from"]["node"] in missing]
                for port, es in by_port.items()
            }
            absent = {p: es for p, es in absent.items() if es}
            if absent:
                policy_skip = False
                for port, es in sorted(absent.items()):
                    decl = op_here.port(port) if op_here else None
                    policy = _missing_policy(decl, es)
                    source = es[0]["from"]["node"]
                    why = missing[source]
                    if policy == "fail":
                        raise _MissingUpstream(nid, port, source, why) from None
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
            # An input under `params` — where ports lived before 0.78 —
            # was lifted onto its port with a warning until 0.82.0. It
            # is now an unknown param, and `check_params` below refuses
            # it by name, as the 0.78.0 notes promised.
            #
            # Process identity for this node (epic 000320): what it
            # computes is fixed by the block, its wire params, its
            # inputs' content, and the compute version. A partial from a
            # previous attempt is reused only under an equal fingerprint.
            current["nid"] = nid
            on_item = item_reporter(nid)
            self._current = current
            # Before anything runs: does this block actually read what
            # the protocol asked for, and take what was wired to it?
            # (000438 — a silently ignored param is a wrong answer with
            # no error; an unwired required port is the same, earlier.)
            from mechbench_compute.block_params import check_inputs, check_params
            check_params(block, _wire_params(params))
            inputs = check_inputs(block, inputs)
            # The stored identity, not the spelling: one fingerprint per
            # op however the protocol wrote it (docs/LEXICON.md §1).
            fingerprint = resume_mod.node_fingerprint(
                block=lexicon.canonical_path(block), params=_wire_params(params),
                input_hashes=[node_hashes.get(e["from"]["node"], "")
                              for e in in_edges] + inline_hashes,
                core_version=core_version,
                model=str(_wire_params(params).get("model", "")),
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
            # A node's failure is caught rather than thrown (task
            # 000399): each consumer's port decides what an absent
            # input means, and sibling branches finish either way. A
            # failure nothing tolerates is raised at the end of the
            # run, which is what every graph did before.
            try:
                if nid in ahead:
                    # Already run, alongside its siblings (000396). The
                    # bookkeeping below is this node's own and stays here,
                    # in topological order, so the manifest, the hashes
                    # and the emitted objects do not depend on which
                    # branch finished first.
                    done_ahead = ahead.pop(nid)
                    if isinstance(done_ahead, BaseException):
                        raise done_ahead
                    results[nid] = done_ahead
                elif _is_remote(block, params):
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
            except _MissingUpstream:
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
            # Hash BEFORE emitting (000488). The hash canonical-encodes
            # the result, so a result carrying a live object fails here,
            # locally and by name — instead of being serialized by the
            # emit path, rejected or dropped by the network, and read as
            # a transport fault. That ordering hid a 6 GB result for a
            # night; this one makes the same mistake a failing test.
            # A collection is stored with its items in key order: the same
            # items in any order are the same bytes.
            results[nid] = lexicon.canonical_collection(results[nid])
            # A result about a model's layers stays about them through
            # every records op (000624): the landmarks on any input's
            # header ride onto an output that has none.
            results[nid] = _carry_arch(inputs, results[nid])
            node_hashes[nid] = resume_mod.content_hash(results[nid])
            if result_base and discard and not outputs_of.get(nid):
                # Held, not emitted (000561): the consumers read it from
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
                    # under `on_missing` (000399) has an upstream that
                    # produced nothing and so stored nothing — there is
                    # no path to cite, and citing the absence as a path
                    # was a KeyError that failed the very run the policy
                    # was keeping alive. `nodes_missing` on the manifest
                    # is where the absence is recorded. An upstream HELD
                    # rather than stored (discard mode) is cited by its
                    # content hash, which is a path form of its own.
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
                    params=_wire_params(params),
                )
                node_paths[nid] = out["path"]
                # A node declared as several outputs is stored under each
                # name; the first is the one its consumers' lineage cites.
                for also in names[1:]:
                    bench.emit(f"{result_base}/{also}", to_emit,
                               inputs=[out["path"]],
                               operation=lexicon.canonical_path(block),
                               params=_wire_params(params))
            if isinstance(results[nid], dict) and results[nid].get("spend"):
                spend_by_node[nid] = results[nid]["spend"]
            # A held node was handed over by `on_node_kept`: it has no
            # emitted object for `on_node_done` to record.
            if self._on_node_done is not None and nid not in held:
                self._on_node_done(nid, node_paths.get(nid), fingerprint)
            # An expanded node's items already covered its worth — the
            # old unconditional bump made done overrun total by one per
            # expanded node ("59/57 steps").
            if not expanded:
                bump()

        # A failure nobody answered for fails the run — which is every
        # failure in a graph that declares no `on_missing` policy, so a
        # protocol written before this behaves exactly as it did. What
        # changed is WHEN: the sibling branches have finished by now.
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
                        params=_wire_params(held_params))
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
            # Every node's content hash and its upstream nodes (000561):
            # what a result's lineage verifies against when an
            # intermediate was held rather than stored — and a record
            # worth having either way.
            "node_hashes": {nid: node_hashes[nid] for nid in order if nid in node_hashes},
            "node_inputs": {nid: [e["from"]["node"] for e in _ordered_edges(edges, nid)]
                            for nid in order if nid in results and nid not in missing},
            **({"keep": keep, "nodes_held": sorted(held)} if discard else {}),
            # What each node produced, small enough to read beside the
            # node in the composer without fetching its object (task
            # 000525): the kind, and how many items or rows.
            "node_summaries": {nid: node_summary(results[nid], spend_by_node.get(nid))
                               for nid in order if nid in results and nid not in missing},
            # What did not run, and why (000399). A reader of this result
            # must never have to infer an absence from a shorter list.
            **({"nodes_missing": {nid: missing[nid] for nid in order
                                  if nid in missing}} if missing else {}),
            "resolved": resolved,
            # Where this ran (000402). Recorded, never fingerprinted:
            # bit-identity is promised within a hardware class.
            "resources": {"hardware": hardware_class(),
                          **({"spend": _spend_total(spend_by_node)}
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


    def _tool_block_runner(self, secrets=None):
        """Handlers the toolbox cannot run itself (task 000340): model
        blocks and, later, sub-protocols. This is what makes
        `decision-read` available AS A TOOL — a model that can consult
        another model, or the bench, mid-turn."""
        def run_block(ref, inputs, params):
            if ref in ("logits/read", "text/generate"):
                return self._run_op(ref, inputs, params, secrets=secrets)
            if ref == "tools/lookup":
                # The recording fetch, so a tool call's object shows up
                # in the run's resolved lineage like any other input.
                from mechbench_compute import bench

                return bench.fetch(str((inputs.get("arguments") or {}).get("path")))
            raise ValueError(
                f"{ref!r} is not available as a tool handler here — pure "
                "blocks, decision-read and generate are")

        return run_block


    #: An opened memo: the label it came from, and the cassette that
    #: will be written back to it.
    _Memo = namedtuple("_Memo", "label tape existing")

    def _open_memo(self, params):
        """Load this node's memo of remote calls, if it asked for one.

        `cache: "<bench label>"` — an explicit label, not a derivation
        from the node's identity. A memo keyed on node identity would
        be thrown away by every compute release, which is exactly
        backwards: the compute version is not part of what a provider
        was asked, and the REQUEST hash inside the memo is what
        decides a hit. A named memo survives a version bump, which is
        the point of having one.
        """
        label = params.get("cache")
        if not label:
            return None
        if label is True:
            # Derived from the PROTOCOL's identity and the node's id —
            # both stable across compute releases, which a node
            # fingerprint is not. The request hash inside the memo is
            # what decides a hit; this only decides where the memo
            # lives. Reconsidered 2026-09-11: refusing `cache: true` and
            # demanding a name was friction for no gain, since the job
            # spec already carries the identity needed.
            pid, _ = getattr(self, "_protocol_ref", (None, None))
            nid = (getattr(self, "_current", None) or {}).get("nid") \
                if hasattr(self, "_current") else None
            nid = nid or params.get("_nid")
            if not pid or not nid:
                raise ValueError(
                    "`cache: true` needs the run's protocol id and the node "
                    "id to derive a label, and this execution has neither "
                    "(a bare ProtocolSpec with no protocolId). Name it: "
                    '`cache: "<owner>/<project>/memos/<name>"`.')
            owner = params.get("_owner") or "memos"
            label = f"{owner}/memos/{pid}/{nid}"
        from mechbench_compute import bench
        from mechbench_compute.providers.cassette import Cassette

        label = str(label)
        try:
            obj = bench.fetch(label)
            payload = obj.get("payload", obj)
            tape = Cassette.from_wire(payload)
            existing = tape.n_responses
        except Exception:  # noqa: BLE001 — no memo yet is the ordinary case
            tape, existing = Cassette(provider="", label=label), 0
        return self._Memo(label, tape, existing)

    def _close_memo(self, memo, out):
        """Write the memo back, and say what it saved.

        Emitted even when nothing new was recorded: a run that was a
        complete hit is exactly the run worth being able to point at.
        """
        from mechbench_compute import bench

        added = memo.tape.n_responses - memo.existing
        summary = out.get("summary")
        if isinstance(summary, dict):
            calls = summary.get("calls") or 0
            summary["cache"] = {
                "label": memo.label,
                "hits": max(0, calls - added),
                "recorded": added,
                "entries": memo.tape.n_responses,
            }
        try:
            bench.emit(memo.label, memo.tape.to_wire(),
                       operation="chat/memo")
        except Exception as e:  # noqa: BLE001 — a run must not fail on its memo
            if isinstance(summary, dict):
                summary.setdefault("cache", {})["store_error"] = str(e)[:200]
        return out

    def _block_chat_local(self, inputs, params, on_item=None, on_start=None,
                          resume_items=None):
        from mechbench_compute import chat as chat_mod

        model = self._model_loaded(params.get("model"))
        records = inputs.get("records") or []
        return chat_mod.run_local(model, params.get("model"), records, params,
                                  inputs=inputs,
                                  on_item=on_item, on_start=on_start,
                                  resume_items=resume_items)


    def _run_remote_wave(self, nid, block, inputs, params, secrets, *,
                         nodes, edges, order, results, missing,
                         resolve_params, resolve_value, item_reporter, expand,
                         node_view, report, resume, resume_kwargs):
        """Run this remote node and every remote node ready beside it.

        "Ready beside it" is the whole of the scheduling: a node later in
        the topological order whose inputs are ALL computed already does
        not depend on this one, so waiting for this one buys nothing but
        latency. Two prompts to two providers, then a judge, is the shape
        this exists for — it used to take the sum of the two calls.

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
                if not _is_remote(peer_block, peer_params):
                    continue
                if resume.get(other) if isinstance(resume, dict) else None:
                    continue
                sources = {e["from"]["node"] for e in _ordered_edges(edges, other)}
                if not sources <= set(results):
                    continue        # it is waiting for something, not for us
                peer_inputs = {
                    e["to"]["port"]: results[e["from"]["node"]]
                    for e in _ordered_edges(edges, other)}
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

    def _run_model_block(self, fn, inputs, params, *args, **kwargs):
        """Model-block wrapper: load the bound model, fuse an adapter
        when one arrives (input port `adapter` or params.adapter), run
        the block, restore. Blocks themselves stay adapter-unaware —
        their own _model_loaded call returns the same fused instance."""
        mval = params.get("model")
        ref = mval if hasattr(mval, "adapter_payloads") else None
        # _model_loaded owns the ref-to-weights translation (bench bases
        # materialize there), so this call and the block's own no-op
        # re-call resolve identically.
        model = self._model_loaded(mval)
        skipped: list[str] = []
        with self._adapter_fused(model, inputs, params, ref=ref, skipped=skipped):
            result = fn(inputs, params, *args, **kwargs)
        if skipped and isinstance(result, dict):
            # Deltas the architecture could not take (lora.fuse): the
            # result says so, next to its numbers, never only in a log.
            result["adapter_skipped_modules"] = skipped
        # The model's depth landmarks ride on every result a model block
        # writes (000624), so a figure downstream can draw them without
        # being told: which layers attend globally, and where fresh keys
        # and values stop. One place, for every block present and future.
        arch = getattr(model, "arch", None)
        if isinstance(result, dict) and "arch" not in result and arch is not None:
            from mechbench_compute.blocks import arch_header

            result["arch"] = arch_header(arch)
        return result

    def _adapter_fused(self, model, inputs, params, ref=None, skipped=None):
        """Context manager: fuse the model's adapter STACK (000312 Arc
        B), run the block, restore in reverse.

        Layering, not precedence: the ModelRef's adapters are part of
        what "the model" MEANS and fuse first, in order; an adapter
        arriving as data flow (input port `adapter`, or params.adapter)
        is the node's own operand and fuses last, on top. That is the
        successive-rounds composition: read on {B, [a1]} plus an edge
        from this graph's train node reads through a1 then the new
        round. params.adapter_scale keeps its old meaning by applying
        to that last, node-level adapter only."""
        import contextlib

        from mechbench_compute.lora import (
            fuse_adapter_stack,
            restore_adapter_stack,
        )

        @contextlib.contextmanager
        def _cm():
            payloads = list(ref.adapter_payloads) if ref is not None else []
            node_level = inputs.get("adapter")
            if node_level:
                payloads.append(node_level)
            if not payloads:
                yield None
                return
            override = (
                float(params["adapter_scale"])
                if node_level and "adapter_scale" in params
                else None
            )
            handles = fuse_adapter_stack(
                model.lm, payloads, override,
                skip_missing=bool(params.get("adapter_skip_missing", False)),
                skipped=skipped)
            try:
                yield handles
            finally:
                restore_adapter_stack(model.lm, handles)
        return _cm()


    def _materialize_checkpoint(self, label: str):
        """A bench checkpoint label -> a local directory, cached by the
        manifest's identity and verified file-by-file (checkpoint.py).
        This is what makes {"base": {"bench": ...}} loadable. Memoized
        per executor: the loader resolves the same label at least twice
        per node (wrapper + the block's no-op re-call), and neither
        should re-fetch the manifest."""
        from pathlib import Path

        from mechbench_compute import bench, checkpoint

        memo = getattr(self, "_checkpoint_dirs", None)
        if memo is None:
            memo = self._checkpoint_dirs = {}
        if label in memo:
            return memo[label]

        manifest, _meta = bench.fetch(
            f"{label}/{checkpoint.MANIFEST_NAME}", with_meta=True)
        payload = (
            manifest.get("payload", manifest)
            if isinstance(manifest, dict)
            else manifest
        )
        if (not isinstance(payload, dict)
                or payload.get("kind") not in ("adapter/checkpoint", "checkpoint_manifest")):
            raise ValueError(
                f"{label!r} has no checkpoint manifest — is it a checkpoint "
                "prefix published by merge?")
        cache_root = Path.home() / ".mechbench" / "checkpoints"
        # The download callbacks are the same ones hub fetches use: the
        # runner turns them into watchdog stamps and board progress. A
        # silent 10 GB fetch reads as a wedge and gets killed as one.
        if self._on_download is not None:
            self._on_download(label, None)
        target = checkpoint.materialize(
            payload,
            lambda name: bench.get_file_chunks(f"{label}/{name}"),
            cache_root,
            on_bytes=self._on_download_bytes,
        )
        # A human-readable note of WHICH label this hash-keyed directory
        # holds, for `mechbench models` and the eviction report (000297).
        # The dot prefix keeps it out of every safetensors glob.
        (target / ".label").write_text(label)
        memo[label] = target
        return target


def node_summary(value: Any, spend: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What a node produced, in the terms a reader asks first: which kind,
    and how many. `{kind, collection, items}` for a collection (however it
    is spelled — a retired plural object, a bare list); `{kind,
    collection: false}` for one object, with `rows` when it is a table of
    them; `{}` for a value that carries no kind. `spend_usd` when the node
    called a provider."""
    out: dict[str, Any] = {}
    if isinstance(value, list):
        out = {"kind": lexicon.COLLECTION, "collection": True, "items": len(value)}
    elif isinstance(value, Mapping):
        item_kind = lexicon.item_kind_of(value)
        if item_kind is not None:
            out = {"kind": item_kind, "collection": True, "items": len(lexicon.items_of(value))}
        elif isinstance(value.get("kind"), str):
            try:
                name, _plural = lexicon.resolve_kind(value["kind"], warn=False)
            except KeyError:
                name = value["kind"]
            out = {"kind": name, "collection": False}
            if isinstance(value.get("rows"), list):
                out["rows"] = len(value["rows"])
    if spend and spend.get("cost_usd") is not None:
        out["spend_usd"] = round(float(spend["cost_usd"]), 8)
    return out


def _spend_total(by_node: dict[str, Any]) -> dict[str, Any]:
    """The run's bill: total, per provider, per node (task 000337)."""
    by_provider: dict[str, dict[str, Any]] = {}
    total = 0.0
    calls = 0
    for s in by_node.values():
        p = str(s.get("provider", ""))
        slot = by_provider.setdefault(p, {"cost_usd": 0.0, "calls": 0})
        slot["cost_usd"] = round(slot["cost_usd"] + float(s.get("cost_usd", 0.0)), 8)
        slot["calls"] += int(s.get("calls", 0))
        total += float(s.get("cost_usd", 0.0))
        calls += int(s.get("calls", 0))
    return {"cost_usd": round(total, 8), "calls": calls,
            "by_provider": by_provider,
            "by_node": {k: {"cost_usd": round(float(v.get("cost_usd", 0.0)), 8),
                            "calls": int(v.get("calls", 0)),
                            "provider": v.get("provider", "")}
                        for k, v in by_node.items()},
            "dry_run": all(bool(v.get("dry_run")) for v in by_node.values())}


#: What a port may do when its upstream produced nothing (task 000399).
MISSING_POLICIES = ("fail", "skip", "placeholder")

#: The blocks whose work is somebody else's network, not this machine's
#: (task 000396). Two of these that do not depend on each other have no
#: reason to wait for each other: the time is latency, and the provider
#: is answering other people's requests anyway. Everything else stays
#: serial — a local model node MUST (one model in memory, one fused
#: adapter at a time), and a pure block takes microseconds, where a
#: thread would be pure risk for no gain.
REMOTE_BLOCKS = ("text/chat", "eval/judge")


def _carry_arch(inputs: Mapping[str, Any], result: Any) -> Any:
    """`result` with the model's depth landmarks on it, when the result
    is an object without them. A bare list has no header to carry them
    on and is returned as is.

    They come from an input: a records op downstream of a model op
    passes them along. An op whose model work happens INSIDE it has no
    such input, and carries them itself — see `_block_map`."""
    if not isinstance(result, dict) or "arch" in result:
        return result
    for value in (inputs or {}).values():
        if isinstance(value, Mapping) and isinstance(value.get("arch"), Mapping):
            return {**result, "arch": dict(value["arch"])}
    return result

#: How many remote nodes may be in flight at once. The provider's own
#: rate limiter (000344) bounds the requests WITHIN a node; this bounds
#: the nodes, so a twenty-branch fan-out does not open twenty
#: connections' worth of concurrency on top of it.
MAX_PARALLEL_NODES = 8


def _is_remote(block: str, params: Mapping[str, Any]) -> bool:
    """Whether this node's work happens on somebody else's machine: a
    chat-shaped block whose model reference names a provider."""
    if block not in REMOTE_BLOCKS:
        return False
    model = params.get("model") or params.get("judge") or {}
    if isinstance(model, Mapping):
        return bool(model.get("provider"))
    return bool(getattr(model, "is_endpoint", False))


def _missing_policy(decl, edges) -> str:
    """What to do about an absent input on this port.

    The OP declares what its port can meaningfully do without the input;
    an EDGE may override, because whether a partial result is worth
    having is a question about the experiment, not about the operation.
    An edge that says nothing inherits the port's declaration, and a
    port that says nothing fails — silence never buys tolerance.
    """
    chosen = [e.get("on_missing") for e in edges if e.get("on_missing")]
    if chosen:
        return str(chosen[0])
    return decl.on_missing if decl is not None else "fail"


class _MissingUpstream(RuntimeError):
    """A node needed an input its upstream never produced, and the port
    it was wired to says that is fatal (task 000399).

    Carries the chain, because the useful question is never "what
    raised" but "what was this waiting for": the node, its port, the
    upstream that produced nothing, and why THAT happened.
    """

    def __init__(self, nid: str, port: str, source: str, why: Mapping[str, Any]):
        self.nid, self.port, self.source = nid, port, source
        super().__init__(
            f"{nid} needs its {port!r} port, and {source} produced nothing: "
            f"{why.get('reason')}. That port's `on_missing` is 'fail' — the "
            f"default. A port declared `skip` passes the absence on; one "
            f"declared `placeholder` runs with an empty collection that says "
            f"it is one.")


def _ordered_edges(edges, nid: str) -> list[Any]:
    """The edges into a node, in the one order the platform reads them:
    port, then the edge's declared `index`, then the source node's id
    (task 000397). Every use of a node's in-edges — its inputs, its
    lineage, its fingerprint — reads this order, so none of them depends
    on where in the graph's edge list an author put a line."""
    into = [e for e in edges if (e.get("to") or {}).get("node") == nid]
    return sorted(into, key=lambda e: (str(e["to"].get("port", "")),
                                       int(e.get("index", 0)),
                                       str((e.get("from") or {}).get("node", ""))))


def _preflight(nodes, edges, order) -> None:
    """Everything about a graph that is decidable before it runs, decided
    before it runs (tasks 000512, 000513).

    The executor used to check a node's block, params and ports when
    execution reached it, so a graph that could not run spent everything
    upstream of the first bad node proving so — one 014 trace took five
    resumes and most of a day to arrive at a param refusal that was
    decidable at load. Three things are known here, for every node:

    - its **operation** exists (a retired spelling names its
      replacement);
    - every **param** is one the op reads — names only, so nothing is
      fetched and no binding has to be resolved;
    - every **port** an edge or an `inputs` entry names exists, and
      every required port is filled by one of them.

    What a port is FILLED WITH is not knowable here — it is the upstream
    node's output — so the kind check stays where it is, in the loop.

    Every problem is reported, not the first: a protocol being carried
    forward usually has several, and fix-one-run-again over a long
    protocol is the expensive version of this bug.
    """
    from mechbench_compute.block_params import check_params

    problems: list[str] = []
    into: dict[str, set[str]] = {}
    edge_counts: dict[str, dict[str, int]] = {}
    for e in edges:
        to = e.get("to") or {}
        if isinstance(to.get("node"), str) and isinstance(to.get("port"), str):
            into.setdefault(to["node"], set()).add(to["port"])
            counts = edge_counts.setdefault(to["node"], {})
            counts[to["port"]] = counts.get(to["port"], 0) + 1

    for nid in order:
        node = nodes[nid]
        block = node.get("block")
        if not isinstance(block, str):
            problems.append(f"  {nid}: no block")
            continue
        try:
            name = lexicon.resolve(block)
        except KeyError:
            problems.append(f"  {nid}: {lexicon.explain_unknown(block)}")
            continue
        try:
            check_params(name, node.get("params") or {})
        except ValueError as e:
            problems.append(f"  {nid} ({name}): {e}")
        op = lexicon.BY_NAME[name]
        filled = into.get(nid, set()) | {
            k for k, v in (node.get("inputs") or {}).items() if v is not None}
        for port_name in sorted(filled):
            if op.port(port_name) is None:
                known = ", ".join(sorted(p.name for p in op.inputs)) or "none"
                problems.append(
                    f"  {nid} ({name}): no input port {port_name!r}. "
                    f"Its ports: {known}.")
        # How MANY edges reach each port (task 000397). A port that takes
        # one and is wired twice used to keep whichever edge came last in
        # the list — silently, and the winner depended on the order the
        # author wrote them in.
        for port_name, n in sorted(edge_counts.get(nid, {}).items()):
            decl = op.port(port_name)
            if decl is None:
                continue          # already reported above
            bad = decl.arity_error(n)
            if bad:
                problems.append(f"  {nid} ({name}): port {port_name!r} {bad}.")
        # What each port does when its upstream produces nothing (000399):
        # the policy must be one of the three, and `placeholder` needs a
        # kind with an empty value — an empty collection is a real value,
        # an empty `direction/vector` is not.
        for e in edges:
            if (e.get("to") or {}).get("node") != nid or not e.get("on_missing"):
                continue
            policy, port_name = str(e["on_missing"]), e["to"].get("port")
            decl = op.port(str(port_name))
            if policy not in MISSING_POLICIES:
                problems.append(
                    f"  {nid} ({name}): on_missing is "
                    f"{', '.join(MISSING_POLICIES)}, not {policy!r}.")
            elif policy == "placeholder" and decl is not None and not decl.many:
                problems.append(
                    f"  {nid} ({name}): port {port_name!r} takes one "
                    f"`{decl.kind}`, which has no empty value to stand in "
                    f"for a missing upstream. Use `skip`, or `fail`.")
        for port in op.inputs:
            if not port.required:
                continue
            if port.wildcard:
                if not filled:
                    problems.append(
                        f"  {nid} ({name}): needs at least one input edge "
                        f"(`{port.kind}` on a port of your naming).")
            elif port.name not in filled:
                problems.append(
                    f"  {nid} ({name}): needs an input on its {port.name!r} "
                    f"port (`{port.kind}`): wire an edge onto it, or give it "
                    f"under the node's `inputs`.")

    if problems:
        raise ValueError(
            f"this protocol cannot run — {len(problems)} problem"
            f"{'s' if len(problems) > 1 else ''} found before anything ran:\n"
            + "\n".join(problems))


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def canonical_json(payload: Any) -> str:
    """Python-side canonical JSON for the mechbench-api hash contract.

    Caveat: cross-language byte-identity is *aspirational* with JSON
    (Python emits `0.0`, JS emits `0`). The mechbench-api side hashes
    the bytes as received, so this canonical form only needs to be
    stable across Python invocations — task 000186 moves the
    contract to canonical CBOR, which pins the form across languages.
    """
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
