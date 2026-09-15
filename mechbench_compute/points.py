"""One vocabulary for the points of a forward pass.

A **point** is where an activation is read or edited: a residual
stream (`resid_pre`, `resid_post`), a sub-layer's output (`attn_out`,
`mlp_out`, `gate_out`), a place inside a block (`attn.q`, `mlp.act`, …),
or a whole-model point (`embed`, `final_norm`, `logits`). Every `point`
parameter, `space.point`, and a capture readout's hook names use these
names and nothing else.
"""

from __future__ import annotations

#: point -> (position axis, head axis, feature axis) of the tensor
#: dispatched at that point. Tensors are batch-first.
LAYOUT: dict[str, tuple[int, int | None, int]] = {
    # [B, L, D]
    "resid_pre": (1, None, 2), "resid_post": (1, None, 2), "attn_out": (1, None, 2),
    "mlp_out": (1, None, 2), "gate_out": (1, None, 2), "attn.in_norm": (1, None, 2),
    "mlp.in_norm": (1, None, 2), "embed": (1, None, 2), "final_norm": (1, None, 2),
    # [B, L, F]  (neurons on the feature axis)
    "mlp.gate": (1, None, 2), "mlp.up": (1, None, 2), "mlp.act": (1, None, 2),
    "mlp.down_in": (1, None, 2),
    # [B, L, V]
    "logits": (1, None, 2),
    # [B, L, n_heads*hd]
    "attn.o_in": (1, None, 2),
    # [B, n_heads, L, hd]
    "attn.q": (2, 1, 3), "attn.q_pre_rope": (2, 1, 3), "attn.per_head_out": (2, 1, 3),
    # [B, n_kv, L_kv, hd]
    "attn.k": (2, 1, 3), "attn.v": (2, 1, 3), "attn.k_pre_rope": (2, 1, 3),
    # [B, L, n_heads, hd]  (pre-transpose)
    "attn.q_pre_norm": (1, 2, 3), "attn.k_pre_norm": (1, 2, 3),
    # [B, n_heads, L, S]  (query positions on axis 2; keys are the feature axis)
    "attn.weights": (2, 1, 3), "attn.scores": (2, 1, 3),
}

POINTS: tuple[str, ...] = tuple(LAYOUT)

#: The two residual points, the ones a stream is read at.
RESIDUAL: tuple[str, ...] = ("resid_pre", "resid_post")

#: Points that take no `layers`: they occur once per forward pass.
WHOLE_MODEL: tuple[str, ...] = ("embed", "final_norm", "logits")

#: The spellings from before the vocabulary was one.
_RETIRED = {"post": "resid_post", "pre": "resid_pre"}

POINTS_DOC = "`" + "`, `".join(POINTS) + "`"


def normalize(point: str | None, *, default: str = "resid_post") -> str:
    """The one spelling of a point; `post`/`pre` are read as the residual
    points they always meant. An unknown name is refused with the list."""
    if point is None:
        return default
    p = str(point)
    p = _RETIRED.get(p, p)
    if p not in LAYOUT:
        raise ValueError(f"unknown point {point!r}: one of {POINTS_DOC}")
    return p


def residual(point: str | None, *, default: str = "resid_post") -> str:
    """A point that must be a residual stream."""
    p = normalize(point, default=default)
    if p not in RESIDUAL:
        raise ValueError(f"point {point!r} is not a residual stream: one of {RESIDUAL}")
    return p


def side(point: str) -> str:
    """`resid_post` → `post`: the short form the capture hooks are keyed by."""
    return residual(point).removeprefix("resid_")
