from __future__ import annotations

from typing import Any


class ModelLoading:
    def _model_loaded(self, model_id) -> Any:
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
            from mechbench_compute import Model

            self._model = Model.load(model_id, on_download=self._on_download,
                                     on_download_bytes=self._on_download_bytes)
            self._model_id = model_id
        return self._model

    def evict_model(self) -> None:
        self._model = None
        self._model_id = None

    @staticmethod
    def model_ref(model: Any) -> str | None:
        if getattr(model, "repo_id", None) and getattr(model, "revision", None):
            return f"{model.repo_id}@{model.revision}"
        return None

    def _run_model_block(self, fn, inputs, params, *args, **kwargs):
        mval = params.get("model")
        ref = mval if hasattr(mval, "adapter_payloads") else None
        model = self._model_loaded(mval)
        skipped: list[str] = []
        fused: list[dict] = []
        with self._adapter_fused(model, inputs, params, ref=ref, skipped=skipped, fused=fused):
            result = fn(inputs, params, *args, **kwargs)
        if skipped and isinstance(result, dict):
            result["adapter_skipped_modules"] = skipped
        if fused and isinstance(result, dict) and "fused" not in result:
            result["fused"] = fused
        arch = getattr(model, "arch", None)
        if isinstance(result, dict) and "arch" not in result and arch is not None:
            from mechbench_compute.blocks import arch_header

            result["arch"] = arch_header(arch)
        return result

    def _adapter_fused(self, model, inputs, params, ref=None, skipped=None, fused=None):
        import contextlib

        from mechbench_compute.adapters.is_operator import is_operator
        from mechbench_compute.lora import (
            fuse_adapter_stack,
            restore_adapter_stack,
        )

        @contextlib.contextmanager
        def _cm():
            payloads = list(ref.adapter_payloads) if ref is not None else []
            layers = list(ref.adapter_layers[:len(payloads)]) if ref is not None else []
            node_level = read_one_adapter(inputs.get("adapter"))
            if is_operator(node_level):
                raise ValueError(
                    "an operator attaches through the model reference: list it among the "
                    "reference's adapters. The `adapter` port takes a LoRA.")
            if node_level:
                payloads.append(node_level)
                layers.append(None)
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
                skipped=skipped, keys=model.architecture.adapter_keys, layers=layers)
            if node_level:
                model.node_adapter = handles[-1]
            if ref is not None and ref.adapter_payloads:
                model.fused_reference = ref
            if fused is not None:
                fused.extend(read_fused(ref if ref is not None and ref.adapter_payloads else None,
                                        node_level, override))
            try:
                yield handles
            finally:
                model.node_adapter = None
                model.fused_reference = None
                restore_adapter_stack(model.lm, handles)
        return _cm()

    def _reference_fused(self, model, ref, undo, name=None) -> None:
        labels = tuple(getattr(ref, "adapter_labels", ()) or ())
        if not labels:
            return
        who = name or "this operation"
        if len(ref.adapter_payloads) != len(labels):
            raise ValueError(
                f"{who}: the model reference {ref.describe()} carries the adapters "
                f"{', '.join(labels)}, and they reached the operation unresolved, so "
                "it cannot fuse them. Give the adapted model as the node's `model` "
                "parameter, which the executor resolves and fuses.")
        held = getattr(model, "fused_reference", None)
        if held is not None:
            if held == ref:
                return
            raise ValueError(
                f"{who}: the model already carries the adapters of {held.describe()}, "
                f"and the operation asked for {ref.describe()}; one node runs one "
                "adapted model")
        from mechbench_compute.lora import fuse_adapter_stack

        handles = fuse_adapter_stack(model.lm, list(ref.adapter_payloads),
                                     keys=model.architecture.adapter_keys,
                                     layers=ref.adapter_layers)
        model.fused_reference = ref
        undo.append((model, ref, handles))

    def _reference_restored(self, undo) -> list[dict]:
        from mechbench_compute.lora import restore_adapter_stack

        fused: list[dict] = []
        for model, ref, handles in reversed(undo):
            fused[:0] = read_fused(ref, None, None)
            if model is self._model:
                restore_adapter_stack(model.lm, handles)
            model.fused_reference = None
        undo.clear()
        return fused

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


def read_fused(ref, node_level, scale) -> list[dict]:
    out: list[dict] = ref.read_adapters() if getattr(ref, "adapter_labels", ()) else []
    if node_level:
        out.append({"port": "adapter", **({"scale": scale} if scale is not None else {})})
    return out


def read_one_adapter(value):
    from mechbench_compute import lexicon

    if not isinstance(value, dict) or value.get("kind") != lexicon.COLLECTION:
        return value
    items = lexicon.items_of(value)
    if len(items) != 1:
        raise ValueError(
            f"the `adapter` port takes one adapter, and this collection holds "
            f"{len(items)}; map over it with `records/map` to run once per adapter")
    return items[0]
