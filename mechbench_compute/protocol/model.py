"""Weights: loading them, and fusing adapters around a block.

One model is resident per executor and swapping ids reloads. A model
reference names a base — a hub repo, or a checkpoint on the bench, which
is materialized into a local directory here — and a stack of adapters,
which fuse around the block that runs and are restored afterwards.
"""

from __future__ import annotations

from mechbench_compute import Model


class ModelLoading:
    """ModelLoading: see this module's docstring."""

    def _model_loaded(self, model_id) -> Model:
        """The model an operation declared, loading it if it is not resident.

        Accepts a bare "repo[@rev]" string or a ModelRef, whose BASE is
        what loads here — adapters are the fuse layer's business, not the
        loader's.

        `model_id` is required, and there is no fallback to whatever
        happens to be resident: an operation that ran a model it did not
        name would produce a result whose recorded model is a claim
        rather than a fact.
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
            # bench base is translated to its materialized local directory
            # HERE — the one place every call site funnels through. A
            # second translation anywhere else would let a bench label
            # reach the hub as if it were a repo id.
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

    @staticmethod
    def model_ref(model: Model) -> str | None:
        """`repo@commit` for a loaded model — what a result should record."""
        if getattr(model, "repo_id", None) and getattr(model, "revision", None):
            return f"{model.repo_id}@{model.revision}"
        return None

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
        # writes, so a figure downstream can draw them without being
        # told: which layers attend globally, and where fresh keys and
        # values stop. One place, for every block.
        arch = getattr(model, "arch", None)
        if isinstance(result, dict) and "arch" not in result and arch is not None:
            from mechbench_compute.blocks import arch_header

            result["arch"] = arch_header(arch)
        return result

    def _adapter_fused(self, model, inputs, params, ref=None, skipped=None):
        """Context manager: fuse the model's adapter STACK, run the block,
        restore in reverse.

        Layering, not precedence: the ModelRef's adapters are part of
        what "the model" MEANS and fuse first, in order; an adapter
        arriving as data flow (input port `adapter`, or params.adapter)
        is the node's own operand and fuses last, on top. That is the
        successive-rounds composition: read on {B, [a1]} plus an edge
        from this graph's train node reads through a1 then the new
        round. params.adapter_scale applies to that last, node-level
        adapter only."""
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
        # holds, for `mechbench models` and the eviction report. The dot
        # prefix keeps it out of every safetensors glob.
        (target / ".label").write_text(label)
        memo[label] = target
        return target
