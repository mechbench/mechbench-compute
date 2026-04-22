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

DEFAULT_MODEL_ID = "mlx-community/gemma-4-E4B-it-bf16"

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
)

# Top-level (non-layer) hook points. Empty in v0; reserved for future
# expansion (e.g. embed.out, final_norm.out).
GLOBAL_HOOK_POINTS: tuple[str, ...] = ()

# Hook points that require manual attention computation. When any hook or
# capture targets one of these, the canonical forward switches from the fused
# scaled_dot_product_attention kernel to a manual softmax path that exposes
# the attention internals.
ATTN_INTERNAL_POINTS: frozenset[str] = frozenset({
    "attn.weights", "attn.per_head_out", "attn.q", "attn.k", "attn.v",
})


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
        """Read architecture facts from a loaded mlx-vlm model.

        Reads model.config.text_config — which is the same data structure
        HuggingFace uses for these checkpoints. Works for any Gemma 4
        variant; raises if the config doesn't have the expected fields.
        """
        cfg = model.config.text_config
        layer_types = list(cfg.layer_types)
        n_layers = len(layer_types)
        global_layers = tuple(
            i for i, t in enumerate(layer_types) if t == "full_attention"
        )
        # num_kv_shared_layers is the count of trailing layers that share
        # K/V from an earlier layer. So the first shared layer index is
        # n_layers - num_kv_shared_layers.
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
        )


# ---------------------------------------------------------------------------
# E4B defaults — module-level aliases for backward compatibility.
#
# Existing experiments import these directly. New code should prefer
# `model.arch.<field>` so it adapts to whichever model variant was loaded.
# ---------------------------------------------------------------------------

E4B_DEFAULT = Arch(
    model_id=DEFAULT_MODEL_ID,
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
