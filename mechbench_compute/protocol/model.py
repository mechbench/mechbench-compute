from __future__ import annotations

from typing import Any

from mechbench_compute.hub import check_hub_id
from mechbench_compute.model_ref import describe_layers
from mechbench_compute.protocol.model_cache_stale import ModelCacheStale
from mechbench_compute.protocol.model_fingerprint import mark_loaded, read_applied


class ModelLoading:
    def _model_loaded(self, model_id) -> Any:
        if getattr(model_id, "is_endpoint", False):
            raise ValueError(
                f"{model_id.describe()} is a remote endpoint: this operation "
                "runs local weights and cannot use one. Use "
                "text/chat, which serves both.")
        hub_only = True
        if hasattr(model_id, "base"):
            if getattr(model_id, "base_kind", None) == "bench":
                model_id, hub_only = str(self._materialize_checkpoint(model_id.base)), False
            else:
                model_id = model_id.base
        if not model_id:
            raise ValueError(
                "this operation did not declare a model. Every operation that "
                "runs one names it in its own params, so the result can say "
                "which weights produced it."
            )
        if hub_only:
            check_hub_id(str(model_id))
        from mechbench_compute.backends import backend_of, load_model_class

        backend = self._backend
        if (self._model is None or (model_id and model_id != self._model_id)
                or backend_of(self._model) != backend.name):
            self._model = load_model_class(backend).load(
                model_id, hub_only=hub_only, on_download=self._on_download,
                on_download_bytes=self._on_download_bytes)
            self._model_id = model_id
        if hasattr(self._model, "attention"):
            self._model.attention = self._attention
        mark_loaded(self._model, self._model_id)
        return self._model

    def evict_model(self) -> None:
        self._model = None
        self._model_id = None

    def _check_cached(self, model, requested, ctx=None) -> None:
        fingerprint = getattr(model, "fingerprint", None)
        if fingerprint is None or (ctx is not None and fingerprint.is_held_by(ctx)):
            return
        held = [a for a in fingerprint.applied if a.label is not None]
        loaded = fingerprint.to_wire()
        if fingerprint.holders:
            if held and held != read_applied(requested):
                self._refuse_stale(model, requested, ctx, loaded, (
                    f"the node it runs inside holds the model as "
                    f"{describe_model(fingerprint.base, fingerprint.applied)}"))
        elif fingerprint.applied:
            self._refuse_stale(model, requested, ctx, loaded, (
                f"the cached model is {describe_model(fingerprint.base, fingerprint.applied)}, "
                f"though no running node fused it"))
        else:
            changed = fingerprint.read_changed(model)
            if changed:
                self._refuse_stale(model, requested, ctx, {**loaded, "changed": changed}, (
                    f"the cached model is {fingerprint.base} with {len(changed)} "
                    f"weight{'s' if len(changed) != 1 else ''} changed since it was loaded "
                    f"({', '.join(changed[:3])}{f', and {len(changed) - 3} more' if len(changed) > 3 else ''})"))
        if ctx is not None:
            fingerprint.hold(ctx)

    def _refuse_stale(self, model, requested, ctx, loaded, finding) -> None:
        asked_applied = read_applied(requested)
        base = getattr(requested, "base", requested)
        asked = {"base": base, "adapters": [a.describe() for a in asked_applied]}
        who = getattr(ctx, "name", None) or "a node"
        if model is self._model:
            self.evict_model()
        raise ModelCacheStale(
            f"{who} asked for {describe_model(base, asked_applied)}, and {finding}; it does not "
            f"run on a model other than the one it asked for. The cache is dropped: the next "
            f"node that asks for the model loads it afresh.",
            loaded=loaded, asked=asked)

    def _release_models(self, ctx, failed: bool) -> list[dict]:
        try:
            fused = self._reference_restored(ctx.fused)
        except BaseException:
            failed = True
            raise
        finally:
            for model in ctx.models:
                fingerprint = getattr(model, "fingerprint", None)
                if fingerprint is not None:
                    fingerprint.release(ctx)
            if failed and ctx.models:
                self.evict_model()
        return fused

    def _read_hardware(self) -> dict[str, Any]:
        from mechbench_compute.backends import DEFAULT_BACKEND, backend_of
        from mechbench_compute.seeds import hardware_class

        info = hardware_class()
        if self._backend.name == DEFAULT_BACKEND:
            return info
        model = self._model if backend_of(self._model) == self._backend.name else None
        describe = getattr(model, "describe_hardware", None)
        return {**info, "backend": self._backend.name, **(describe() if describe else {}),
                "attention": sorted(self._attention)}

    @staticmethod
    def model_ref(model: Any) -> str | None:
        if getattr(model, "repo_id", None) and getattr(model, "revision", None):
            return f"{model.repo_id}@{model.revision}"
        return None

    def _run_model_block(self, fn, inputs, params, ctx=None, *args, **kwargs):
        mval = params.get("model")
        ref = mval if hasattr(mval, "adapter_payloads") else None
        model = self._model_loaded(mval)
        if ctx is None or not any(m is model for m in ctx.models):
            self._check_cached(model, mval, ctx)
            if ctx is not None:
                ctx.models.append(model)
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

        @contextlib.contextmanager
        def _cm():
            node_level = read_one_adapter(inputs.get("adapter"))
            if is_operator(node_level):
                raise ValueError(
                    "an operator attaches through the model reference: list it among the "
                    "reference's adapters. The `adapter` port takes a LoRA.")
            carried = ref is not None and bool(ref.adapter_payloads)
            held = carried and getattr(model, "fused_reference", None) == ref
            own = ref if carried and not held else None
            payloads = list(own.adapter_payloads) if own is not None else []
            layers = list(own.adapter_layers[:len(payloads)]) if own is not None else []
            override = (
                float(params["adapter_scale"])
                if node_level and "adapter_scale" in params
                else None
            )
            applied = read_applied(own, node_level, override)
            fingerprint = getattr(model, "fingerprint", None)
            if node_level:
                payloads.append(node_level)
                layers.append(None)
            if not payloads:
                if fused is not None and held:
                    fused.extend(fingerprint.read_fused() if fingerprint is not None
                                 else read_fused(ref, None, None))
                yield None
                return
            lora = load_lora_of(model)
            handles = lora.fuse_adapter_stack(
                model.lm, payloads, override,
                skip_missing=bool(params.get("adapter_skip_missing", False)),
                skipped=skipped, keys=model.architecture.adapter_keys, layers=layers)
            prior = (getattr(model, "node_adapter", None), getattr(model, "fused_reference", None))
            if node_level:
                model.node_adapter = handles[-1]
            if carried:
                model.fused_reference = ref
            if fingerprint is not None:
                fingerprint.push(applied)
            if fused is not None:
                fused.extend(fingerprint.read_fused() if fingerprint is not None
                             else read_fused(ref if carried else None, node_level, override))
            try:
                yield handles
            finally:
                model.node_adapter, model.fused_reference = prior
                lora.restore_adapter_stack(model.lm, handles)
                if fingerprint is not None:
                    fingerprint.drop(applied)
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
        applied = read_applied(ref)
        handles = load_lora_of(model).fuse_adapter_stack(
            model.lm, list(ref.adapter_payloads), keys=model.architecture.adapter_keys,
            layers=ref.adapter_layers)
        model.fused_reference = ref
        fingerprint = getattr(model, "fingerprint", None)
        if fingerprint is not None:
            fingerprint.push(applied)
        undo.append((model, handles, applied))

    def _reference_restored(self, undo) -> list[dict]:
        fused: list[dict] = []
        for model, handles, applied in reversed(undo):
            fused[:0] = [a.to_wire() for a in applied]
            load_lora_of(model).restore_adapter_stack(model.lm, handles)
            fingerprint = getattr(model, "fingerprint", None)
            if fingerprint is not None:
                fingerprint.drop(applied)
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


def describe_model(base, applied) -> str:
    named = [describe_applied(a) for a in applied]
    return f"{base} with {', '.join(named)} fused" if named else str(base)


def describe_applied(applied) -> str:
    if applied.label is None:
        return "an `adapter` port's adapter"
    if applied.layers is None:
        return applied.label
    return f"{applied.label} in {describe_layers(applied.layers)}"


def load_lora_of(model) -> Any:
    from mechbench_compute import backends

    return backends.load_lora(backends.find(backends.backend_of(model)))


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
