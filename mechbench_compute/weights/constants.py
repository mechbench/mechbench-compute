from __future__ import annotations

#: The scope every parameter name in an MLX model carries. A point may
#: be written with or without it; `layers.12.…` is what a person types
#: and what an adapter's keys use.
MODEL_SCOPE = "model."


#: Which side of a projection is the residual stream, and the point the
#: model reads or writes it at. A weight's singular vectors live in two
#: spaces — its input's and its output's — and only one of them is a
#: space anything else in the platform can talk about: `q_proj`'s rows
#: are questions asked OF the residual stream, `o_proj`'s columns are
#: what it writes INTO it. The other side is head or hidden space, where
#: a direction means nothing to an unembedding or to another layer.
RESIDUAL_SIDE: dict[str, tuple[str, str]] = {
    "q_proj": ("in", "attn.in_norm"),
    "k_proj": ("in", "attn.in_norm"),
    "v_proj": ("in", "attn.in_norm"),
    "o_proj": ("out", "attn_out"),
    "gate_proj": ("in", "mlp.in_norm"),
    "up_proj": ("in", "mlp.in_norm"),
    "down_proj": ("out", "mlp_out"),
}
