"""Gemma 4 architectural facts.

The framework targets the Gemma 4 family — currently E4B (4B params, 42
layers) and E2B (2B params, 35 layers). Per-model dimensions are bundled
into an `Arch` dataclass that is read from the loaded model's HuggingFace
config at `Model.load()` time.

Module-level constants (`N_LAYERS`, `D_MODEL`, `GLOBAL_LAYERS`, etc.) remain
as the **E4B defaults** so existing experiment scripts that import them
keep working unchanged. New experiments should prefer `model.arch.<field>`,
which adapts automatically to whichever model variant was loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Hook-point names. These are the stable interface and don't depend on the
# specific model variant within the Gemma 4 family.
# ---------------------------------------------------------------------------

LAYER_HOOK_POINTS: tuple[str, ...] = (
    "resid_pre",          # layer input (== resid_post[i-1] except for i=0)
    "attn_out",           # attention branch contribution after o_proj + post_attention_layernorm
    "mlp_out",            # MLP branch contribution after post_feedforward_layernorm
    "gate_out",           # MatFormer per-layer-input side-channel contribution
    "resid_post",         # layer output after layer_scalar
    "attn.weights",       # post-softmax attention weights [B, n_heads, L, S_kv]
    "attn.per_head_out",  # weights @ values, before o_proj concat [B, n_heads, L, head_dim]
    "attn.q",             # per-head queries post-q_norm + post-RoPE [B, n_heads, L, head_dim]
    "attn.k",             # per-KV-head keys post-k_norm + post-RoPE [B, n_kv_heads, L_kv, head_dim]
    "attn.v",             # per-KV-head values post-v_norm [B, n_kv_heads, L_kv, head_dim]
    # --- task 000365: the rest of the grammar ---------------------------
    "attn.in_norm",       # input_layernorm output: what attention actually sees [B, L, D]
    "attn.q_pre_norm",    # q_proj output before q_norm [B, L, n_heads, head_dim]
    "attn.k_pre_norm",    # k_proj output before k_norm [B, L, n_kv_heads, head_dim] (absent on KV-shared layers)
    "attn.q_pre_rope",    # post-q_norm, pre-RoPE [B, n_heads, L, head_dim]
    "attn.k_pre_rope",    # post-k_norm, pre-RoPE [B, n_kv_heads, L_kv, head_dim] (absent on KV-shared layers)
    "attn.scores",        # pre-softmax scores, mask applied [B, n_heads, L, S_kv]
    "attn.o_in",          # per-head concat before o_proj [B, L, n_heads*head_dim]
    "mlp.in_norm",        # pre_feedforward_layernorm output: what the MLP sees [B, L, D]
    "mlp.gate",           # gate_proj output, pre-activation [B, L, F]
    "mlp.up",             # up_proj output [B, L, F]
    "mlp.act",            # gelu_approx(gate): the activation alone [B, L, F]
    "mlp.down_in",        # gelu(gate) * up: the neuron vector down_proj reads [B, L, F]
)

# Top-level (non-layer) hook points. Empty in v0; reserved for future
# expansion (e.g. embed.out, final_norm.out).
GLOBAL_HOOK_POINTS: tuple[str, ...] = (
    # The final RMSNorm's per-position scale, [B, S] float32 (000142):
    # what DLA's apply_ln divides by to make components sum to the
    # model's true logits.
    "final_norm.scale",
    # --- task 000365 ---
    "embed",              # token embeddings before layer 0 (and before the per-layer-input projection) [B, L, D]
    "final_norm",         # the final RMSNorm's output: what the unembedding reads [B, L, D]
    "logits",             # final logits after softcap [B, L, V] — logit-level surgery lives here
)

# Hook points that require manual attention computation. When any hook or
# capture targets one of these, the canonical forward switches from the fused
# scaled_dot_product_attention kernel to a manual softmax path that exposes
# the attention internals.
ATTN_INTERNAL_POINTS: frozenset[str] = frozenset({
    "attn.weights", "attn.per_head_out", "attn.q", "attn.k", "attn.v",
    "attn.q_pre_norm", "attn.k_pre_norm", "attn.q_pre_rope", "attn.k_pre_rope",
    "attn.scores", "attn.o_in",
})

# Hook points inside the MLP. Like the attention internals, targeting one
# switches THAT layer from the compiled geglu path to a manual, arithmetically
# identical path (bf16 rounding may differ at the last bit); other layers
# stay on the compiled path.
MLP_INTERNAL_POINTS: frozenset[str] = frozenset({
    "mlp.gate", "mlp.up", "mlp.act", "mlp.down_in",
})

# Points that do not EXIST on a KV-shared layer: its keys and values arrive
# from an earlier layer already normed and rotated, so there is no pre-norm
# or pre-RoPE key there to hook. (`attn.k` on a shared layer dispatches the
# shared keys, as before.)
SHARED_LAYER_ABSENT_POINTS: frozenset[str] = frozenset({
    "attn.k_pre_norm", "attn.k_pre_rope",
})

# The points each family's forward IMPLEMENTS. The canonical (Gemma 4)
# forward carries the whole grammar; the other forwards carry the original
# set until they are extended. A name that parses but is not implemented by
# the running family is refused (never silently uninvoked).
_LEGACY_LAYER_POINTS: frozenset[str] = frozenset({
    "resid_pre", "attn_out", "mlp_out", "resid_post",
    "attn.weights", "attn.per_head_out", "attn.q", "attn.k", "attn.v",
})
_LEGACY_GLOBAL_POINTS: frozenset[str] = frozenset({"final_norm.scale"})


def family_supports(model_type: str, point: str, *, layer_scoped: bool) -> bool:
    """Does `model_type`'s forward dispatch `point`?"""
    if model_type not in ("gemma3", "qwen2", "llama"):
        return True  # the canonical forward: everything
    if point == "gate_out":
        return False  # MatFormer side-channel: Gemma 4 only
    return point in (_LEGACY_LAYER_POINTS if layer_scoped else _LEGACY_GLOBAL_POINTS)


# ---------------------------------------------------------------------------
# Per-model architectural config.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arch:
    """Per-model architectural facts for one Gemma 4 variant.

    Built once at `Model.load()` from the loaded mlx-vlm model's text_config.
    Stored on the Model instance as `model.arch` and threaded through hook
    validation. Following TransformerLens 3.0's TransformerBridge pattern:
    the architecture adapter is one-per-family (Gemma 4); the per-variant
    dimensions live in this dataclass.

    Attributes:
        model_id: The HuggingFace repo id this Arch was derived from.
        n_layers: Total transformer-block count.
        d_model: Residual-stream dimensionality.
        n_heads: Query-side head count per attention block.
        n_kv_heads: KV-side head count (n_heads // n_kv_heads = GQA group size).
        vocab_size: Tokenizer / unembed vocabulary size.
        hidden_size_per_layer_input: MatFormer per-layer-input embedding width.
        global_layers: Layer indices using full ('full_attention') attention,
            in ascending order. The complement uses sliding-window attention.
        first_kv_shared_layer: Smallest layer index whose attention reuses
            the K/V tensors from an earlier non-shared layer. Layers
            [first_kv_shared_layer, n_layers) read from the cache rather
            than computing fresh K/V. The "last fresh-K/V global" layer
            (the project's L23 pivot for E4B; predicted L14 for E2B) is
            the largest global-layer index strictly less than this value.
    """

    model_id: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    hidden_size_per_layer_input: int
    global_layers: tuple[int, ...]
    first_kv_shared_layer: int
    # Family discriminator: "gemma4" or "gemma3". Drives forward-path
    # dispatch in `_forward.run_forward`. Defaults to "gemma4" because
    # every existing call site loads a Gemma 4 variant; Gemma 3 support
    # is being staged in under task 000192.
    model_type: str = "gemma4"

    @property
    def last_fresh_kv_global(self) -> int:
        """Largest global-layer index whose K/V are computed fresh (not shared).

        For E4B this is L23; for E2B it should be L14. The architectural
        pivot identified in essay section 21 lives at this layer.
        """
        fresh_globals = [g for g in self.global_layers
                         if g < self.first_kv_shared_layer]
        if not fresh_globals:
            raise ValueError(
                f"No fresh-K/V global layer in {self.model_id}: "
                f"globals={self.global_layers}, "
                f"first_kv_shared={self.first_kv_shared_layer}"
            )
        return max(fresh_globals)

    def layer_type(self, i: int) -> str:
        """Return 'full_attention' for global layers, 'sliding_attention' otherwise."""
        return "full_attention" if i in self.global_layers else "sliding_attention"

    def all_hook_names(self) -> list[str]:
        """Enumerate every valid hook-point name in this model.

        Length: n_layers * len(LAYER_HOOK_POINTS) + len(GLOBAL_HOOK_POINTS).
        """
        out = list(GLOBAL_HOOK_POINTS)
        for i in range(self.n_layers):
            for p in LAYER_HOOK_POINTS:
                out.append(f"blocks.{i}.{p}")
        return out

    @classmethod
    def from_mlx_model(cls, model: Any, model_id: str | None = None) -> "Arch":
        """Read architecture facts from a loaded model.

        Handles three sources:
          - mlx-vlm Gemma 4 (model.config.text_config with `layer_types`
            and optional `num_kv_shared_layers`).
          - mlx-vlm Gemma 3 (model.config.text_config with
            `sliding_window_pattern`; no KV-sharing).
          - mlx-lm Qwen 2.x (model.args ModelArgs dataclass; no hybrid
            attention, no KV-sharing).
        """
        # mlx-lm models expose `args` and don't have `language_model`.
        # Dispatch on that signature before touching mlx-vlm-specific paths.
        if hasattr(model, "args") and not hasattr(model, "language_model"):
            return cls._from_mlx_lm_args(model.args, model_id)

        cfg = model.config.text_config
        cfg_model_type = (getattr(cfg, "model_type", "") or "").lower()
        family = "gemma3" if cfg_model_type.startswith("gemma3") else "gemma4"

        if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
            # Gemma 4: layer_types is a per-layer list.
            layer_types = list(cfg.layer_types)
            n_layers = len(layer_types)
            global_layers = tuple(
                i for i, t in enumerate(layer_types) if t == "full_attention"
            )
        else:
            # Gemma 3: globals at i where (i+1) % sliding_window_pattern == 0,
            # i.e. the last layer of each group of `pattern` is global.
            n_layers = int(cfg.num_hidden_layers)
            pattern = int(
                getattr(cfg, "sliding_window_pattern", 6)
            )
            global_layers = tuple(
                i for i in range(n_layers)
                if (i + 1) % pattern == 0
            )

        # num_kv_shared_layers is the count of trailing layers that share
        # K/V from an earlier layer. Gemma 3 doesn't have this; default to 0
        # which makes first_kv_shared = n_layers (i.e. no KV-shared region).
        num_kv_shared = int(getattr(cfg, "num_kv_shared_layers", 0) or 0)
        first_kv_shared = n_layers - num_kv_shared

        return cls(
            model_id=model_id or getattr(cfg, "_name_or_path", "") or "",
            n_layers=n_layers,
            d_model=int(cfg.hidden_size),
            n_heads=int(cfg.num_attention_heads),
            n_kv_heads=int(cfg.num_key_value_heads),
            vocab_size=int(cfg.vocab_size),
            hidden_size_per_layer_input=int(
                getattr(cfg, "hidden_size_per_layer_input", 0) or 0
            ),
            global_layers=global_layers,
            first_kv_shared_layer=first_kv_shared,
            model_type=family,
        )

    @classmethod
    def _from_mlx_lm_args(cls, args, model_id: str | None) -> "Arch":
        """Adapter path for mlx-lm-loaded models. Currently supports
        Qwen 2.x (000201) and Llama 3.x (000208). Both lack KV-sharing
        and MatFormer side-channels; Llama 3.2 may have a hybrid
        attention pattern via `args.layer_types`."""
        n_layers = int(args.num_hidden_layers)
        family_raw = (getattr(args, "model_type", "") or "").lower()
        if family_raw not in ("qwen2", "llama"):
            raise NotImplementedError(
                f"mlx-lm model_type {family_raw!r} not yet supported. "
                f"Currently: qwen2 (000201), llama (000208). Other Qwen / "
                f"DeepSeek families tracked under 000202 / 000203."
            )

        # Llama 3.2 may declare a hybrid attention pattern via
        # args.layer_types; Llama 3.1 doesn't and is pure full-attention.
        # Qwen 2 has no layer_types attribute → all layers global.
        layer_types = getattr(args, "layer_types", None)
        if layer_types:
            global_layers = tuple(
                i for i, t in enumerate(layer_types) if t == "full_attention"
            )
        else:
            global_layers = tuple(range(n_layers))

        return cls(
            model_id=model_id or "",
            n_layers=n_layers,
            d_model=int(args.hidden_size),
            n_heads=int(args.num_attention_heads),
            n_kv_heads=int(args.num_key_value_heads),
            vocab_size=int(args.vocab_size),
            hidden_size_per_layer_input=0,
            global_layers=global_layers,
            first_kv_shared_layer=n_layers,
            model_type=family_raw,
        )


# ---------------------------------------------------------------------------
# E4B defaults — module-level aliases for backward compatibility.
#
# Existing experiments import these directly. New code should prefer
# `model.arch.<field>` so it adapts to whichever model variant was loaded.
# ---------------------------------------------------------------------------

E4B_DEFAULT = Arch(
    model_id="mlx-community/gemma-4-E4B-it-bf16",
    n_layers=42,
    d_model=2560,
    n_heads=8,
    n_kv_heads=2,
    vocab_size=262144,
    hidden_size_per_layer_input=256,
    global_layers=(5, 11, 17, 23, 29, 35, 41),
    first_kv_shared_layer=24,
)

N_LAYERS = E4B_DEFAULT.n_layers
D_MODEL = E4B_DEFAULT.d_model
N_HEADS = E4B_DEFAULT.n_heads
VOCAB_SIZE = E4B_DEFAULT.vocab_size
GLOBAL_LAYERS = E4B_DEFAULT.global_layers


def layer_type(i: int) -> str:
    """E4B-default version of Arch.layer_type. Use `model.arch.layer_type(i)`
    for variant-aware code."""
    return E4B_DEFAULT.layer_type(i)


def all_hook_names() -> list[str]:
    """E4B-default version of Arch.all_hook_names. Use
    `model.arch.all_hook_names()` for variant-aware code."""
    return E4B_DEFAULT.all_hook_names()
