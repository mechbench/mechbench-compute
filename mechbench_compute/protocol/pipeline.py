from __future__ import annotations

from datetime import UTC
from typing import Any

from mechbench_compute import lexicon
from mechbench_compute.protocol.build_manifest import build_manifest
from mechbench_compute.protocol.check_failures import check_failures
from mechbench_compute.protocol.gather_inputs import gather_inputs
from mechbench_compute.protocol.missing_upstream import MissingUpstream
from mechbench_compute.protocol.progress import Progress
from mechbench_compute.protocol.protocol_spec import ProtocolSpec
from mechbench_compute.protocol.read_resume_entry import read_resume_entry
from mechbench_compute.protocol.resolver import Resolver
from mechbench_compute.protocol.restore_node import restore_node
from mechbench_compute.protocol.run_state import RunState
from mechbench_compute.protocol.serialize_params import serialize_params
from mechbench_compute.protocol.sort_edges import sort_edges
from mechbench_compute.protocol.store_result import store_result


class Pipeline:
    def _run_pipeline(self, spec: ProtocolSpec, on_progress=None,
                      secrets=None, resume=None) -> Any:
        from datetime import datetime

        import mechbench_schema as ms

        from mechbench_compute import __version__ as core_version
        from mechbench_compute import resume as resume_mod

        state = RunState(spec, resume)
        resolver = Resolver(
            bound_params=state.bound_params, secrets=secrets,
            on_download=self._on_download,
            on_download_bytes=self._on_download_bytes)
        progress = Progress(on_progress, len(state.order),
                            on_spool_item=self._on_spool_item)
        current = {"nid": ""}

        for pos, nid in enumerate(state.order):
            node = state.nodes[nid]
            try:
                block = lexicon.resolve(node["block"])
            except KeyError:
                raise ValueError(f"unknown block: {node['block']!r}") from None
            progress.start_node(pos, nid)
            params = resolver.resolve_node_params(node, block)
            in_edges = sort_edges(state.edges, nid)
            gathered = gather_inputs(state, nid, node,
                                     lexicon.BY_NAME.get(block), in_edges,
                                     resolver)
            if gathered is None:
                progress.bump()
                continue
            inputs, input_paths, inline_hashes = gathered
            current["nid"] = nid
            on_item = progress.open_items(nid)
            self._current = current
            from mechbench_compute.block_params import check_inputs, check_params
            check_params(block, serialize_params(params))
            inputs = check_inputs(block, inputs)
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
            try:
                state.results[nid] = self._run_node(
                    state, nid, block, inputs, params, secrets=secrets,
                    resolver=resolver, progress=progress, on_item=on_item,
                    on_checkpoint=on_checkpoint, input_paths=input_paths,
                    resume_kwargs=resume_kwargs)
            except MissingUpstream:
                raise
            except Exception as exc:  # noqa: BLE001
                state.failures[nid] = exc
                state.missing[nid] = {"reason": f"{type(exc).__name__}: {exc}",
                                      "source": [nid]}
                print(f"[graph] {nid} failed: {exc}")
                if self._on_node_done is not None:
                    self._on_node_done(nid, None, fingerprint)
                progress.bump()
                continue
            store_result(state, nid, node, block, params, in_edges,
                         inputs, resolver, fingerprint, self._on_node_kept)
            if self._on_node_done is not None and nid not in state.held:
                self._on_node_done(nid, state.node_paths.get(nid), fingerprint)
            if not progress.expanded:
                progress.bump()

        check_failures(state, resolver)
        payload = build_manifest(state, resolver.resolved)
        prov = ms.Provenance(
            created_at=datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            produced_by=ms.ToolInfo(tool="mechbench-runner",
                                    version=core_version),
            inputs=[],
            params_fingerprint=ms.fingerprint_params(
                {"graph": state.graph, "bindings": {}}),
            schema_version=ms.__version__,
        )
        return ms.Emitted(payload=payload, provenance=prov)
