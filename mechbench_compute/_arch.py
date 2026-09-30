from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LAYER_HOOK_POINTS: tuple[str, ...] = (
    "resid_pre",
    "attn_out",
    "mlp_out",
    "gate_out",
    "resid_post",
    "attn.weights",
    "attn.per_head_out",
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.in_norm",
    "attn.q_pre_norm",
    "attn.k_pre_norm",
    "attn.q_pre_rope",
    "attn.k_pre_rope",
    "attn.scores",
    "attn.o_in",
    "mlp.in_norm",
    "mlp.gate",
    "mlp.up",
    "mlp.act",
    "mlp.down_in",
)

GLOBAL_HOOK_POINTS: tuple[str, ...] = (
    "final_norm.scale",
    "embed",
    "final_norm",
    "logits",
)

ATTN_INTERNAL_POINTS: frozenset[str] = frozenset({
    "attn.weights", "attn.per_head_out", "attn.q", "attn.k", "attn.v",
    "attn.q_pre_norm", "attn.k_pre_norm", "attn.q_pre_rope", "attn.k_pre_rope",
    "attn.scores", "attn.o_in",
})

MLP_INTERNAL_POINTS: frozenset[str] = frozenset({
    "mlp.gate", "mlp.up", "mlp.act", "mlp.down_in",
})

SHARED_LAYER_ABSENT_POINTS: frozenset[str] = frozenset({
    "attn.k_pre_norm", "attn.k_pre_rope",
})

def family_supports(model_type: str, point: str, *, layer_scoped: bool) -> bool:
    from .architectures import for_type

    arch = for_type(model_type)
    if arch is None:
        return False
    return arch.supports(point, layer_scoped=layer_scoped)


@dataclass(frozen=True)
class Arch:
    model_id: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    hidden_size_per_layer_input: int
    global_layers: tuple[int, ...]
    first_kv_shared_layer: int
    model_type: str = "gemma4"
    n_global_kv_heads: int | None = None

    @property
    def last_fresh_kv_global(self) -> int:
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
        return "full_attention" if i in self.global_layers else "sliding_attention"

    def all_hook_names(self) -> list[str]:
        out = list(GLOBAL_HOOK_POINTS)
        for i in range(self.n_layers):
            for p in LAYER_HOOK_POINTS:
                out.append(f"blocks.{i}.{p}")
        return out

    @classmethod
    def from_mlx_model(cls, model: Any, model_id: str | None = None) -> "Arch":
        from .architectures import for_model

        return for_model(model).arch_of(model, model_id)


def read_arch_from_config(config: Mapping[str, Any], model_id: str = "") -> Arch:
    from .architectures import for_type

    top = str(config.get("model_type") or "").lower()
    declared = for_type(top)
    if declared is None:
        raise NotImplementedError(f"model_type {top!r} is not one compute loads")
    text = config.get("text_config")
    source = text if declared.loader == "mlx-vlm" and isinstance(text, Mapping) else config
    cfg = {**declared.config_defaults, **{k: v for k, v in source.items() if v is not None}}
    n_heads = int(cfg["num_attention_heads"])
    layer_types = cfg.get("layer_types")
    n_layers = len(layer_types) if layer_types else int(cfg["num_hidden_layers"])
    if layer_types:
        global_layers = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    elif declared.loader == "mlx-lm":
        global_layers = tuple(range(n_layers))
    else:
        pattern = int(cfg.get("sliding_window_pattern") or 6)
        global_layers = tuple(i for i in range(n_layers) if (i + 1) % pattern == 0)
    shared = int(cfg.get("num_kv_shared_layers") or 0)
    per_layer = int(cfg.get("hidden_size_per_layer_input") or 0)
    global_kv = cfg.get("num_global_key_value_heads") if cfg.get("attention_k_eq_v") else None
    return Arch(
        model_id=model_id,
        n_layers=n_layers,
        d_model=int(cfg["hidden_size"]),
        n_heads=n_heads,
        n_kv_heads=int(cfg.get("num_key_value_heads") or n_heads),
        vocab_size=int(cfg["vocab_size"]),
        hidden_size_per_layer_input=per_layer,
        global_layers=global_layers,
        first_kv_shared_layer=n_layers - shared,
        model_type=declared.model_type,
        n_global_kv_heads=None if global_kv is None else int(global_kv),
    )


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
    return E4B_DEFAULT.layer_type(i)


def all_hook_names() -> list[str]:
    return E4B_DEFAULT.all_hook_names()
