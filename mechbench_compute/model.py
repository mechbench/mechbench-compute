"""Model: the user-facing wrapper around mlx-vlm's Gemma 4 E4B.

This is the entire surface a typical user touches. Load a model, tokenize a
prompt, run a forward pass with optional hooks and captures, inspect the
result. Lower-level types (HookInfo, ActivationCache, etc.) are exposed for
power users who want to write hook callbacks or post-process the cache, but
nothing else needs to be imported in a typical script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.gemma4.language import logit_softcap
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import get_model_path, load_config, prepare_inputs

from . import _arch
from ._forward import run_forward
from ._forward_gemma3 import run_forward_gemma3
from ._forward_llama import run_forward_llama
from ._forward_qwen import run_forward_qwen
from .cache import ActivationCache
from .hooks import HookFn, parse_hook_name
from .interventions import Intervention, compose

# Text-only families whose hook-aware forward (_forward_qwen / _forward_llama)
# mirrors mlx-lm's model structure. mlx-vlm 0.6.x added a `text_only` wrapper
# that now loads these too, but with an incompatible shape (no `.args`, empty
# text_config), so they must be routed to mlx-lm explicitly rather than letting
# mlx-vlm's `load` claim them.
_MLX_LM_FAMILIES: frozenset[str] = frozenset({"qwen2", "llama"})


def _peek_model_type(model_id: str) -> str:
    """Top-level `model_type` from a checkpoint's config.json, without loading
    weights — used to route text-only families to mlx-lm. Returns "" if the
    config can't be read (callers fall back to the normal load path)."""
    try:
        path = get_model_path(model_id)
        if isinstance(path, tuple):
            path = path[0]
        return (load_config(path).get("model_type") or "").lower()
    except Exception:
        return ""


@dataclass(frozen=True)
class RunResult:
    """Result of Model.run().

    Attributes:
        logits: [1, seq_len, vocab_size] bf16 tensor of next-token logits.
        cache: ActivationCache of any tensors named in capture=[]. Empty if
            no capture was requested.
    """

    logits: mx.array
    cache: ActivationCache

    @property
    def last_logits(self) -> mx.array:
        """Logits at the final sequence position. Shape [vocab_size], bf16."""
        return self.logits[0, -1, :]

    def top_k(self, tokenizer, k: int = 5) -> list[tuple[str, float]]:
        """Top-k next-token predictions at the final position.

        Returns a list of (decoded_token_string, probability) tuples sorted
        by descending probability.
        """
        probs_np = self.last_probs()
        top_idx = np.argsort(-probs_np)[:k]
        return [
            (tokenizer.decode([int(i)]), float(probs_np[i])) for i in top_idx
        ]

    def top1(self, tokenizer) -> tuple[int, str, float]:
        """Argmax of last_logits at the final position. Returns
        (token_id, decoded_token_string, probability).

        Use when you need both the token id (for indexing into other
        structures) and the human-readable decoded form. For a printable
        list of top-k predictions, use top_k instead.
        """
        probs = self.last_probs()
        top_id = int(np.argmax(probs))
        return top_id, tokenizer.decode([top_id]), float(probs[top_id])

    def last_probs(self) -> np.ndarray:
        """Softmax of last_logits at the final sequence position. Returns a
        numpy float32 vocabulary-sized probability distribution.

        Composes with vocab_concentration to ask 'how concentrated is the
        model's next-token distribution after this run?'.
        """
        last = self.last_logits.astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        return np.array(probs)


class Model:
    """A loaded Gemma 4 E4B model with the hook-aware forward pass.

    Construct via Model.load("mlx-community/gemma-4-E4B-it-bf16") — the constructor takes pre-loaded mlx-vlm
    objects and is intended for advanced use (e.g. tests that fixture a model).

    Example:
        from mechbench_compute import Model

        model = Model.load("mlx-community/gemma-4-E4B-it-bf16")
        ids = model.tokenize("Complete this sentence with one word: The Eiffel Tower is in")
        result = model.run(ids)
        for tok, p in result.top_k(model.tokenizer):
            print(f"{tok!r:20s} p={p:.4f}")
    """

    def __init__(self, model, processor, arch: _arch.Arch | None = None):
        self._model = model
        self._processor = processor
        self.arch = arch if arch is not None else _arch.Arch.from_mlx_model(model)
        # What this actually is, filled in by `load`. `revision` is the commit
        # the weights came from — the thing a run records, since the reference
        # asked for may have been a moving ref.
        self.repo_id: str | None = None
        self.revision: str | None = None
        self.requested_ref: str | None = None

    @classmethod
    def load(cls, model_id: str, *,
             on_download: "Callable[[str, str | None], None] | None" = None) -> "Model":
        """Load a model from the HuggingFace cache.

        The weights are installed if this machine does not have them, so a
        pinned reference works on a fresh machine and not only on one that
        already downloaded it.

        `model_id` is required and has no default. Which model to run is
        the caller's decision, not this layer's: a default here would let a
        result be recorded without saying which weights produced it, and
        would silently follow whatever a moving upstream ref points at.

        `model_id` may pin a revision as ``repo@revision`` (task
        000260): the pin resolves against the local cache and the model
        loads from that exact snapshot path; a pin that isn't cached
        (or is ambiguous) raises rather than silently loading main.

        Defaults to Gemma 4 E4B bf16. Any Gemma 4 family checkpoint that
        mlx-vlm can load works (E2B, E4B, future variants); per-model
        dimensions are read from the loaded config and bundled into
        `self.arch` (an `Arch` dataclass). Module-level constants like
        `N_LAYERS` continue to reflect the E4B defaults regardless of
        which variant was loaded — use `model.arch.n_layers` for code
        that should adapt.
        """
        # mlx-vlm covers Gemma 3 and Gemma 4 (multimodal-shaped models).
        # mlx-lm covers Qwen 2.x / Llama 3.x and other text-only families.
        # Route by the config's model_type: text-only families go straight to
        # mlx-lm (mlx-vlm 0.6.x would otherwise claim them via its `text_only`
        # wrapper, in a shape _forward_qwen/_forward_llama don't mirror). Other
        # families try mlx-vlm first, then fall back to mlx-lm on "not
        # supported" (covers checkpoints whose config we couldn't peek).
        from .hub import ensure_model

        # Install the weights if this machine does not have them, and learn
        # which commit they are. Loading always goes through the resolved
        # snapshot, so a run can record the revision it actually executed
        # rather than the reference it was handed.
        requested = model_id
        repo_id, revision_sha, snapshot = ensure_model(model_id, on_download=on_download)
        model_id = str(snapshot)

        if _peek_model_type(model_id) in _MLX_LM_FAMILIES:
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
        if arch.model_type not in ("gemma4", "gemma3", "qwen2", "llama"):
            raise NotImplementedError(
                f"mechbench-compute's hook-aware forward path supports Gemma 3, "
                f"Gemma 4, Qwen 2.x, and Llama 3.x; loaded model {model_id!r} "
                f"reports model_type={arch.model_type!r}. Other families "
                f"(Qwen 3.5 → 000202, DeepSeek V3 / Kimi-VL → 000203) are "
                f"tracked as future work."
            )
        model = cls(m, p, arch=arch)
        model.repo_id = repo_id
        model.revision = revision_sha
        model.requested_ref = requested
        return model

    @property
    def tokenizer(self):
        """The underlying tokenizer (processor.tokenizer or processor itself)."""
        return getattr(self._processor, "tokenizer", self._processor)

    @property
    def lm(self):
        """The text decoder, uniform across families: the module to train
        against (distill/lora) or call directly for plain forwards.

        For mlx-vlm families (Gemma 3/4) this is `language_model`; for
        mlx-lm families (Qwen 2.x, Llama 3.x) the loaded model itself.
        Either way the returned module exposes `.model.layers` and is
        callable on token ids (`lm(mx.array([ids]))`), returning logits
        (or an output object with `.logits`).
        """
        if self.arch.model_type in ("qwen2", "llama"):
            return self._model
        return self._model.language_model

    def prompt_cache(self):
        """A fresh, empty KV cache for `Model.lm`, family-forked: mlx-vlm
        language models build their own (`lm.make_cache()`, which knows the
        family's rotating/shared-KV layout); mlx-lm models use
        `mlx_lm.models.cache.make_prompt_cache`. Feed it via
        `lm(ids, cache=cache)`; see `distill.score_items_cached` for the
        prefix-reuse pattern built on it (task 000227)."""
        if self.arch.model_type in ("qwen2", "llama"):
            from mlx_lm.models.cache import make_prompt_cache
            return make_prompt_cache(self._model)
        return self.lm.make_cache()

    def trunk_hidden(self, input_ids: mx.array) -> mx.array:
        """Run the text decoder's embedding + layer stack only — no
        unembedding — returning the hidden states the family's head
        consumes: [B, S, D], post-final-norm in every supported family.

        With `head_logits` this splits the forward so callers can unembed
        only the rows they need (task 000227): the lm-head over a 262k
        vocab dominates full forwards whose outputs are read at 3–5
        positions. Verified per family: applying `head_logits` to this
        tensor reproduces `lm(input_ids)`'s logits bit-exactly. (Distinct
        from `project_to_logits`, which takes *pre*-norm residual-stream
        tensors from capture points and applies the final norm itself.)
        """
        h = self.lm.model(input_ids)
        return h[0] if isinstance(h, tuple) else h

    def head_logits(self, hidden: mx.array) -> mx.array:
        """Apply the family's unembedding (and softcap where the family
        has one) to post-norm hidden states from `trunk_hidden` — any
        leading shape ([B, S, D], [B, n, D], [n, D]).

        Note the tiling caveat that runs through the 0.5.x line: bf16
        matmuls tile by shape, so heading a sliced rows-block can differ
        from heading the full sequence at deep-tail rounding level. Same
        rows, same math; characterized envelope in `distill.score_items_fast`.
        """
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
        """Apply the chat template (default) and tokenize a prompt.

        Pass `chat_template=False` for base (non-instruct) models — they
        complete prompts as raw text and adding a chat wrapper distorts
        the next-token distribution. Instruct models (Gemma 3/4 -it,
        Qwen 2.5 -Instruct) want the default.

        Returns input_ids of shape [1, seq_len], int32.
        """
        if self.arch.model_type in ("qwen2", "llama"):
            # mlx-lm tokenizer wrapper exposes apply_chat_template directly.
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
    ) -> RunResult:
        """Run a forward pass with optional hook callbacks and tensor capture.

        Args:
            input_ids: Output of self.tokenize(prompt). Shape [1, seq_len].
            hooks: Raw hook dict. Map of hook-point name -> callback. Each
                callback receives (activation, HookInfo) and returns either
                None (leave unchanged) or a replacement mx.array of the same
                shape and dtype.
            capture: Raw capture list. Names of hook points whose post-hook
                activation should be saved into result.cache. Captures happen
                AFTER any user hook at the same point, so the recorded value
                reflects any modification.
            interventions: Declarative-intervention sugar. List of Intervention
                objects (e.g. Ablate.layer(14), Capture.attn_weights([23, 29]))
                that compile to hooks + capture. May be combined freely with
                raw hooks/capture; when several callbacks target the same
                point they chain in declaration order.

        Returns:
            RunResult(logits, cache).

        Raises:
            InvalidHookName: A name in hooks/capture/interventions isn't a
                known hook point. Error message includes typo suggestions.
            LayerIndexOutOfRange: A hook references a layer outside [0, 42).
        """
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
            arch=self.arch,
        )
        return RunResult(logits=logits, cache=cache)

    def _validate_hook_names(self, names: Iterable[str]) -> None:
        """Validate every name; raises on the first invalid one."""
        for n in names:
            parse_hook_name(n, arch=self.arch)

    def project_to_logits(self, residual: mx.array) -> mx.array:
        """Apply the model's final RMSNorm + tied unembed (+ optional softcap)
        to a residual-stream tensor.

        Used by logit-lens helpers (lens.py) and centroid decoding (geometry.py)
        to read off what an intermediate representation 'predicts' if the rest
        of the network were a no-op. Accepts any tensor whose last dimension
        is D_MODEL; the projection is applied along that axis.
        """
        if self.arch.model_type in ("qwen2", "llama"):
            # mlx-lm-loaded models share a shape: lm.model holds the
            # tower; tied vs untied unembed selected by
            # args.tie_word_embeddings.
            tm = self.lm.model
            h = tm.norm(residual)
            if self.lm.args.tie_word_embeddings:
                return tm.embed_tokens.as_linear(h)
            return self.lm.lm_head(h)

        lm = self.lm
        tm = lm.model
        h = tm.norm(residual)
        if self.arch.model_type == "gemma3":
            # Gemma 3 has a separate lm_head Linear (whose weights are
            # tied to embed_tokens at load time via the model's sanitize
            # hook). No final_logit_softcapping in the Gemma 3 line.
            return lm.lm_head(h)
        logits = tm.embed_tokens.as_linear(h)
        if lm.final_logit_softcapping is not None:
            logits = logit_softcap(lm.final_logit_softcapping, logits)
        return logits

    def decoded_distribution(self, vector) -> np.ndarray:
        """Project a single residual vector through the tied unembed, softmax,
        and return the full vocabulary-sized probability distribution as a
        numpy float32 array.

        Convenience wrapper around project_to_logits that handles the common
        case of "I have a 1D residual vector and I want its decoded
        probability distribution". Accepts:
          - mx.array of shape [D_MODEL]
          - mx.array of shape [1, D_MODEL] or [1, 1, D_MODEL]
          - np.ndarray of shape [D_MODEL] (will be converted to bf16 mx.array)

        Returns: np.ndarray of shape [vocab_size], dtype float32, summing to 1.

        For batches or per-position distributions, use project_to_logits
        directly and softmax yourself.
        """
        if isinstance(vector, np.ndarray):
            v = mx.array(vector[None, None, :], dtype=mx.bfloat16)
        elif vector.ndim == 1:
            v = vector[None, None, :]
        elif vector.ndim == 2:
            v = vector[None, :, :]
        else:
            v = vector
        logits = self.project_to_logits(v)
        # Reduce to [vocab] regardless of input shape: take [..., -1, :] then
        # flatten leading dims down to a single distribution.
        last = logits[..., -1, :].astype(mx.float32)
        # Flatten any leading batch dims by indexing from the front
        while last.ndim > 1:
            last = last[0]
        probs = mx.softmax(last)
        mx.eval(probs)
        return np.array(probs)
