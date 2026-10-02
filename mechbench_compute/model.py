from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import mlx.core as mx
import numpy as np
from mlx_vlm.utils import get_model_path, load_config

from . import _arch, architectures, support
from .adapters.read_operator_hooks import read_operator_hooks
from .cache import ActivationCache
from .errors import InvalidHookName
from .hooks import HookFn, parse_hook_name
from .interventions import Intervention, compose
from .spans import add_to_span


def _peek_config(model_id: str) -> dict:
    try:
        path = get_model_path(model_id)
        if isinstance(path, tuple):
            path = path[0]
        return dict(load_config(path))
    except Exception:
        return {}


@dataclass(frozen=True)
class RunResult:
    logits: mx.array
    cache: ActivationCache

    @property
    def last_logits(self) -> mx.array:
        return self.logits[0, -1, :]

    def top_k(self, tokenizer, k: int = 5) -> list[tuple[str, float]]:
        probs_np = self.last_probs()
        top_idx = np.argsort(-probs_np)[:k]
        return [
            (tokenizer.decode([int(i)]), float(probs_np[i])) for i in top_idx
        ]

    def top1(self, tokenizer) -> tuple[int, str, float]:
        probs = self.last_probs()
        top_id = int(np.argmax(probs))
        return top_id, tokenizer.decode([top_id]), float(probs[top_id])

    def last_probs(self) -> np.ndarray:
        last = self.last_logits.astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        return np.array(probs)


class Model:
    def __init__(self, model, processor, arch: _arch.Arch | None = None,
                 architecture: support.Architecture | None = None):
        self._model = model
        self._processor = processor
        self.architecture = (architecture if architecture is not None
                             else architectures.for_model(model))
        self.arch = arch if arch is not None else self.architecture.arch_of(model)
        self.repo_id: str | None = None
        self.revision: str | None = None
        self.requested_ref: str | None = None
        self.node_adapter: dict | None = None
        self.fused_reference: Any = None

    @classmethod
    def load(cls, model_id: str, *,
             on_download: Callable[[str, str | None], None] | None = None,
             on_download_bytes: Callable[[int, int], None] | None = None) -> Model:
        from .hub import ensure_model

        requested = model_id
        from pathlib import Path as _Path

        fetch_started = time.perf_counter()
        if _Path(model_id).is_dir():
            repo_id, revision_sha, snapshot = (
                "local-checkpoint",
                _Path(model_id).name,
                _Path(model_id),
            )
        else:
            repo_id, revision_sha, snapshot = ensure_model(
                model_id, on_download=on_download, on_bytes=on_download_bytes)
        model_id = str(snapshot)
        load_started = time.perf_counter()
        add_to_span(download_seconds=load_started - fetch_started)

        config = _peek_config(model_id)
        refused = support.refusal(config)
        if refused is not None:
            raise NotImplementedError(
                f"mechbench-compute cannot load {requested!r}: {refused}.")
        architecture = architectures.for_type(config.get("model_type"))
        m, p = architecture.load(model_id)
        loaded = architectures.read_model_type(m)
        if loaded != architecture.model_type:
            raise NotImplementedError(
                f"mechbench-compute loaded {requested!r} as {architecture.model_type!r} "
                f"from its config.json, but the loaded model reports {loaded!r}.")
        model = cls(m, p, arch=architecture.arch_of(m, model_id), architecture=architecture)
        model.repo_id = repo_id
        model.revision = revision_sha
        model.requested_ref = requested
        add_to_span(model_load_seconds=time.perf_counter() - load_started)
        return model

    def unadapted(self):
        from .lora import unfused

        if self.node_adapter is None:
            raise ValueError("no adapter is fused on this model from a node's `adapter` port")
        return unfused(self.lm, self.node_adapter)

    @property
    def tokenizer(self):
        return getattr(self._processor, "tokenizer", self._processor)

    @property
    def lm(self):
        return self.architecture.lm(self._model)

    def prompt_cache(self):
        return self.architecture.prompt_cache(self._model)

    def trunk_hidden(self, input_ids: mx.array) -> mx.array:
        add_to_span(forwards=1, tokens_in=int(input_ids.size))
        h = self.lm.model(input_ids)
        return h[0] if isinstance(h, tuple) else h

    def head_logits(self, hidden: mx.array) -> mx.array:
        return self.architecture.head_logits(self._model, hidden)

    def tokenize(self, prompt: str, *, chat_template: bool = True) -> mx.array:
        return self.architecture.tokenize(self._model, self._processor, prompt,
                                          chat_template=chat_template)

    def make_ids(self, ids) -> mx.array:
        return mx.array([[int(t) for t in ids]], dtype=mx.int32)

    def run(
        self,
        input_ids: mx.array,
        *,
        hooks: dict[str, HookFn] | None = None,
        capture: list[str] | None = None,
        interventions: list[Intervention] | None = None,
        kv_cache=None,
    ) -> RunResult:
        final_hooks, final_capture = compose(
            interventions, hooks=hooks, capture=capture,
        )
        final_hooks = read_operator_hooks(self.lm, final_hooks)
        self._validate_hook_names(set(final_hooks.keys()) | set(final_capture))
        logits, cache = self.architecture.forward(
            self._model, input_ids, hooks=final_hooks, capture=final_capture,
            arch=self.arch, kv_cache=kv_cache,
        )
        add_to_span(forwards=1, bytes_captured=sum(
            int(cache[n].nbytes) for n in set(final_capture) if n in cache))
        return RunResult(logits=logits, cache=cache)

    def _validate_hook_names(self, names: Iterable[str]) -> None:
        from . import _arch as _arch_mod

        declared = architectures.for_type(self.arch.model_type)
        absent = declared.absent_points(self.arch) if declared is not None else {}
        for n in names:
            info = parse_hook_name(n, arch=self.arch)
            if not _arch_mod.family_supports(
                self.arch.model_type, info.point,
                layer_scoped=info.layer is not None,
            ):
                raise InvalidHookName(
                    f"{n} (not implemented by the {self.arch.model_type!r} forward)",
                    [x for x in self.arch.all_hook_names()
                     if _arch_mod.family_supports(
                         self.arch.model_type, x.split(".", 2)[-1] if x.startswith("blocks.") else x,
                         layer_scoped=x.startswith("blocks."))],
                )
            if info.point in absent:
                raise InvalidHookName(
                    f"{n} (absent: {absent[info.point]})",
                    [x for x in self.arch.all_hook_names()
                     if (x.split(".", 2)[-1] if x.startswith("blocks.") else x) not in absent])
            if (info.layer is not None
                    and info.point in _arch_mod.SHARED_LAYER_ABSENT_POINTS
                    and info.layer >= self.arch.first_kv_shared_layer):
                raise InvalidHookName(
                    f"{n} (layer {info.layer} is KV-shared: its keys arrive "
                    f"already normed and rotated from layer "
                    f"{self.arch.first_kv_shared_layer - 1} or earlier; hook "
                    f"'attn.k' there instead)",
                    self.arch.all_hook_names(),
                )

    def project_to_logits(self, residual: mx.array) -> mx.array:
        return self.architecture.project_to_logits(self._model, residual)

    def decoded_distribution(self, vector) -> np.ndarray:
        if isinstance(vector, np.ndarray):
            v = mx.array(vector[None, None, :], dtype=mx.bfloat16)
        elif vector.ndim == 1:
            v = vector[None, None, :]
        elif vector.ndim == 2:
            v = vector[None, :, :]
        else:
            v = vector
        logits = self.project_to_logits(v)
        last = logits[..., -1, :].astype(mx.float32)
        while last.ndim > 1:
            last = last[0]
        probs = mx.softmax(last)
        mx.eval(probs)
        return np.array(probs)
