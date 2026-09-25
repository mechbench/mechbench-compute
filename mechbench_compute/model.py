from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.gemma4.language import logit_softcap
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import get_model_path, load_config, prepare_inputs

from . import _arch, support
from ._forward import run_forward
from ._forward_gemma3 import run_forward_gemma3
from ._forward_llama import run_forward_llama
from ._forward_qwen import run_forward_qwen
from .cache import ActivationCache
from .errors import InvalidHookName
from .hooks import HookFn, parse_hook_name
from .interventions import Intervention, compose

_MLX_LM_FAMILIES: frozenset[str] = support.MLX_LM_MODEL_TYPES


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
    def __init__(self, model, processor, arch: _arch.Arch | None = None):
        self._model = model
        self._processor = processor
        self.arch = arch if arch is not None else _arch.Arch.from_mlx_model(model)
        self.repo_id: str | None = None
        self.revision: str | None = None
        self.requested_ref: str | None = None

    @classmethod
    def load(cls, model_id: str, *,
             on_download: Callable[[str, str | None], None] | None = None,
             on_download_bytes: Callable[[int, int], None] | None = None) -> Model:
        from .hub import ensure_model

        requested = model_id
        from pathlib import Path as _Path

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

        config = _peek_config(model_id)
        refused = support.refusal(config) if config.get("model_type") else None
        if refused is not None:
            raise NotImplementedError(
                f"mechbench-compute cannot load {requested!r}: {refused}.")
        # external: mlx-vlm — its text_only wrapper also loads these families, in a shape the mlx-lm forwards do not mirror
        if str(config.get("model_type") or "").lower() in _MLX_LM_FAMILIES:
            from mlx_lm import load as mlx_lm_load

            m, p = mlx_lm_load(model_id)
        else:
            try:
                m, p = load(model_id)
            except ValueError as exc:
                if "not supported" in str(exc):
                    from mlx_lm import load as mlx_lm_load

                    m, p = mlx_lm_load(model_id)
                else:
                    raise

        arch = _arch.Arch.from_mlx_model(m, model_id=model_id)
        if support.architecture(arch.model_type) is None:
            raise NotImplementedError(
                f"mechbench-compute's hook-aware forward path supports "
                f"{', '.join(a.name for a in support.ARCHITECTURES)}; loaded model "
                f"{model_id!r} reports model_type={arch.model_type!r}."
            )
        model = cls(m, p, arch=arch)
        model.repo_id = repo_id
        model.revision = revision_sha
        model.requested_ref = requested
        return model

    @property
    def tokenizer(self):
        return getattr(self._processor, "tokenizer", self._processor)

    @property
    def lm(self):
        if self.arch.model_type in ("qwen2", "llama"):
            return self._model
        return self._model.language_model

    def prompt_cache(self):
        if self.arch.model_type in ("qwen2", "llama"):
            from mlx_lm.models.cache import make_prompt_cache
            return make_prompt_cache(self._model)
        return self.lm.make_cache()

    def trunk_hidden(self, input_ids: mx.array) -> mx.array:
        h = self.lm.model(input_ids)
        return h[0] if isinstance(h, tuple) else h

    def head_logits(self, hidden: mx.array) -> mx.array:
        if self.arch.model_type in ("qwen2", "llama"):
            if self.lm.args.tie_word_embeddings:
                return self.lm.model.embed_tokens.as_linear(hidden)
            return self.lm.lm_head(hidden)
        lm = self.lm
        if self.arch.model_type == "gemma3":
            return lm.lm_head(hidden)
        logits = lm.model.embed_tokens.as_linear(hidden)
        if lm.final_logit_softcapping is not None:
            logits = logit_softcap(lm.final_logit_softcapping, logits)
        return logits

    def tokenize(self, prompt: str, *, chat_template: bool = True) -> mx.array:
        if self.arch.model_type in ("qwen2", "llama"):
            if chat_template:
                rendered = self._processor.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                rendered = prompt
            ids = self._processor.encode(rendered)
            return mx.array([ids], dtype=mx.int32)

        if not chat_template:
            tok = getattr(self._processor, "tokenizer", self._processor)
            ids = tok.encode(prompt)
            return mx.array([ids], dtype=mx.int32)

        add_special_tokens = getattr(self._processor, "chat_template", None) is None
        formatted = apply_chat_template(
            self._processor, self._model.config, prompt, num_images=0,
        )
        image_token_index = getattr(self._model.config, "image_token_index", None)
        inputs = prepare_inputs(
            self._processor,
            images=None,
            audio=None,
            prompts=formatted,
            image_token_index=image_token_index,
            resize_shape=None,
            add_special_tokens=add_special_tokens,
        )
        return inputs["input_ids"]

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
        self._validate_hook_names(set(final_hooks.keys()) | set(final_capture))
        forward = {
            "gemma3": run_forward_gemma3,
            "qwen2": run_forward_qwen,
            "llama": run_forward_llama,
        }.get(self.arch.model_type, run_forward)
        logits, cache = forward(
            self._model, input_ids, hooks=final_hooks, capture=final_capture,
            arch=self.arch, kv_cache=kv_cache,
        )
        return RunResult(logits=logits, cache=cache)

    def _validate_hook_names(self, names: Iterable[str]) -> None:
        from . import _arch as _arch_mod

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
        if self.arch.model_type in ("qwen2", "llama"):
            tm = self.lm.model
            h = tm.norm(residual)
            if self.lm.args.tie_word_embeddings:
                return tm.embed_tokens.as_linear(h)
            return self.lm.lm_head(h)

        lm = self.lm
        tm = lm.model
        h = tm.norm(residual)
        if self.arch.model_type == "gemma3":
            return lm.lm_head(h)
        logits = tm.embed_tokens.as_linear(h)
        if lm.final_logit_softcapping is not None:
            logits = logit_softcap(lm.final_logit_softcapping, logits)
        return logits

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
