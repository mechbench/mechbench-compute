from __future__ import annotations

MODEL_SCOPE = "model."


RESIDUAL_SIDE: dict[str, tuple[str, str]] = {
    "q_proj": ("in", "attn.in_norm"),
    "k_proj": ("in", "attn.in_norm"),
    "v_proj": ("in", "attn.in_norm"),
    "o_proj": ("out", "attn_out"),
    "gate_proj": ("in", "mlp.in_norm"),
    "up_proj": ("in", "mlp.in_norm"),
    "down_proj": ("out", "mlp_out"),
}
