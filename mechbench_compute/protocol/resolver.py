from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import dataflow, lexicon


class Resolver:
    def __init__(self, *, bound_params: Mapping[str, Any], secrets=None,
                 on_download=None, on_download_bytes=None) -> None:
        self.bound_params = bound_params
        self.secrets = secrets
        self.on_download = on_download
        self.on_download_bytes = on_download_bytes
        self.resolved: dict[str, dict] = {"objects": {}, "models": {}}

    def fetch_object(self, ref, want=None):
        from pathlib import Path

        from mechbench_compute import bench
        from mechbench_compute import tensors as tensors_mod

        fetched, meta = bench.fetch(ref, with_meta=True)
        got = (meta or {}).get("content_hash") or ""
        self.resolved["objects"][str(ref)] = got
        if want and not got.endswith(str(want)):
            raise ValueError(
                f"pinned object {ref!r} resolved to {got!r}, "
                f"expected sha256 {want!r}")
        payload = fetched.get("payload", fetched) if isinstance(fetched, dict) else fetched
        if tensors_mod.is_tensor(payload):
            if self.on_download is not None:
                self.on_download(str(ref), None)
            payload = tensors_mod.materialize(
                payload, str(ref), bench.get_file_chunks,
                Path.home() / ".mechbench" / "tensors",
                on_bytes=self.on_download_bytes)
        return payload

    def read_stored_inputs(self, node):
        found: list[str] = []

        def walk(v):
            if dataflow.is_param_ref(v):
                v = self.bound_params.get(v["$param"])
            if dataflow.is_object_ref(v):
                path = v["$ref"].get("bench")
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

    def resolve_value(self, v, keep_reference=False):
        if dataflow.is_param_ref(v):
            name = v["$param"]
            if name not in self.bound_params:
                raise ValueError(f"unbound param: {name!r}")
            return self.resolve_value(self.bound_params[name], keep_reference)
        if dataflow.is_object_ref(v):
            which, source = dataflow.source_of(v)
            if keep_reference:
                return dict(v["$ref"])
            if which == "bench":
                return self.fetch_object(source, v["$ref"].get("sha256"))
            if which == "hf_dataset":
                return self.resolve_hf_dataset(source)
            return self.resolve_hf_adapter(source)
        if isinstance(v, dict):
            return {k: self.resolve_value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self.resolve_value(x) for x in v]
        return v

    def resolve_hf_dataset(self, spec):
        from datasets import load_dataset

        hf_token = (self.secrets or {}).get("hf", {}).get("token")
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
        self.resolved.setdefault("datasets", {})[key] = {
            "repo": repo, "config": config, "split": split,
            "revision": revision,
            "fingerprint": getattr(ds, "_fingerprint", None),
            "rows_total": len(ds), "rows_used": n}
        return lexicon.collection("records/record", records)

    def resolve_hf_adapter(self, spec):
        from huggingface_hub import snapshot_download

        from mechbench_compute.peft import peft_import

        repo = spec["repo"]
        revision = spec.get("revision")
        kwargs = {"allow_patterns": ["adapter_*", "*.json"]}
        if revision:
            kwargs["revision"] = revision
        hf_token = (self.secrets or {}).get("hf", {}).get("token")
        if hf_token:
            kwargs["token"] = hf_token
        local = snapshot_download(repo, **kwargs)
        commit = local.rstrip("/").rsplit("/", 1)[-1]
        payload = peft_import(local)
        self.resolved.setdefault("adapters", {})[
            f"{repo}@{revision}" if revision else repo] = {
            "repo": repo, "revision": revision, "commit": commit,
            "target_modules": payload["lora"]["target_modules"]}
        return payload

    def record_model(self, ref):
        if not isinstance(ref, str) or ref in self.resolved["models"]:
            return
        from mechbench_compute.hub import (
            parse_model_ref,
            resolve_cached_revision,
        )
        try:
            repo, rev = parse_model_ref(ref)
            self.resolved["models"][ref] = {
                "repo": repo, "pinned": rev,
                "commit": resolve_cached_revision(repo, rev)}
        except Exception:  # noqa: BLE001
            self.resolved["models"][ref] = {"repo": ref, "pinned": None,
                                            "commit": None}

    def fetch_recording(self, label):
        from mechbench_compute import bench

        fetched, meta = bench.fetch(label, with_meta=True)
        self.resolved["objects"][str(label)] = (
            (meta or {}).get("content_hash") or ""
        )
        return (fetched.get("payload", fetched)
                if isinstance(fetched, dict) else fetched)

    def resolve_params(self, params, block=None):
        return {k: self.resolve_value(
                    v, block is not None and dataflow.wants_reference(block, k))
                for k, v in (params or {}).items()}

    def resolve_node_params(self, node, block):
        raw_params = node.get("params") or {}
        if (block in ("records/map", "records/fold")
                and isinstance(raw_params.get("body"), Mapping)):
            params = self.resolve_params(
                {k: v for k, v in raw_params.items() if k != "body"}, block)
            params["body"] = raw_params["body"]
        else:
            params = self.resolve_params(raw_params, block)
        if "model" in params:
            mval = params.get("model")
            if isinstance(mval, dict) or hasattr(mval, "adapter_labels"):
                from mechbench_compute import model_ref as model_ref_mod

                ref = model_ref_mod.resolve(mval, fetch=self.fetch_recording)
                params = {**params, "model": ref}
                if not ref.is_endpoint:
                    self.record_model(ref.base)
            else:
                self.record_model(mval)
        return params
