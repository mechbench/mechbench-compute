from __future__ import annotations

from mechbench_compute import Model


class ModelLoading:
    def _model_loaded(self, model_id) -> Model:
        if getattr(model_id, "is_endpoint", False):
            raise ValueError(
                f"{model_id.describe()} is a remote endpoint: this operation "
                "runs local weights and cannot use one. Use "
                "text/chat, which serves both.")
        if hasattr(model_id, "base"):
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
        if self._model is None or (model_id and model_id != self._model_id):
            self._model = Model.load(model_id, on_download=self._on_download,
                                     on_download_bytes=self._on_download_bytes)
            self._model_id = model_id
        return self._model

    @staticmethod
    def model_ref(model: Model) -> str | None:
        if getattr(model, "repo_id", None) and getattr(model, "revision", None):
            return f"{model.repo_id}@{model.revision}"
        return None

    def _run_model_block(self, fn, inputs, params, *args, **kwargs):
        mval = params.get("model")
        ref = mval if hasattr(mval, "adapter_payloads") else None
        model = self._model_loaded(mval)
        skipped: list[str] = []
        with self._adapter_fused(model, inputs, params, ref=ref, skipped=skipped):
            result = fn(inputs, params, *args, **kwargs)
        if skipped and isinstance(result, dict):
            result["adapter_skipped_modules"] = skipped
        arch = getattr(model, "arch", None)
        if isinstance(result, dict) and "arch" not in result and arch is not None:
            from mechbench_compute.blocks import arch_header

            result["arch"] = arch_header(arch)
        return result

    def _adapter_fused(self, model, inputs, params, ref=None, skipped=None):
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
        if self._on_download is not None:
            self._on_download(label, None)
        target = checkpoint.materialize(
            payload,
            lambda name: bench.get_file_chunks(f"{label}/{name}"),
            cache_root,
            on_bytes=self._on_download_bytes,
        )
        (target / ".label").write_text(label)
        memo[label] = target
        return target
