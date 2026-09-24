from __future__ import annotations

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

_LEGACY_LAYER_POINTS: frozenset[str] = frozenset({
    "resid_pre", "attn_out", "mlp_out", "resid_post",
    "attn.weights", "attn.per_head_out", "attn.q", "attn.k", "attn.v",
})
_LEGACY_GLOBAL_POINTS: frozenset[str] = frozenset({"final_norm.scale"})


def family_supports(model_type: str, point: str, *, layer_scoped: bool) -> bool:
    if model_type not in ("gemma3", "qwen2", "llama"):
        return True
    if point == "gate_out":
        return False
    return point in (_LEGACY_LAYER_POINTS if layer_scoped else _LEGACY_GLOBAL_POINTS)


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
        if hasattr(model, "args") and not hasattr(model, "language_model"):
            return cls._from_mlx_lm_args(model.args, model_id)

        cfg = model.config.text_config
        cfg_model_type = (getattr(cfg, "model_type", "") or "").lower()
        family = "gemma3" if cfg_model_type.startswith("gemma3") else "gemma4"

        if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
            layer_types = list(cfg.layer_types)
            n_layers = len(layer_types)
            global_layers = tuple(
                i for i, t in enumerate(layer_types) if t == "full_attention"
            )
        else:
            n_layers = int(cfg.num_hidden_layers)
            pattern = int(
                getattr(cfg, "sliding_window_pattern", 6)
            )
            global_layers = tuple(
                i for i in range(n_layers)
                if (i + 1) % pattern == 0
            )

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
        n_layers = int(args.num_hidden_layers)
        family_raw = (getattr(args, "model_type", "") or "").lower()
        if family_raw not in ("qwen2", "llama"):
            raise NotImplementedError(
                f"mlx-lm model_type {family_raw!r} is not supported; the "
                f"hook-aware forwards cover qwen2 and llama."
            )

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
