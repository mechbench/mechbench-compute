from __future__ import annotations

LAYOUT: dict[str, tuple[int, int | None, int]] = {
    "resid_pre": (1, None, 2), "resid_post": (1, None, 2), "attn_out": (1, None, 2),
    "mlp_out": (1, None, 2), "gate_out": (1, None, 2), "attn.in_norm": (1, None, 2),
    "mlp.in_norm": (1, None, 2), "embed": (1, None, 2), "final_norm": (1, None, 2),
    "mlp.gate": (1, None, 2), "mlp.up": (1, None, 2), "mlp.act": (1, None, 2),
    "mlp.down_in": (1, None, 2),
    "logits": (1, None, 2),
    "attn.o_in": (1, None, 2),
    "attn.q": (2, 1, 3), "attn.q_pre_rope": (2, 1, 3), "attn.per_head_out": (2, 1, 3),
    "attn.k": (2, 1, 3), "attn.v": (2, 1, 3), "attn.k_pre_rope": (2, 1, 3),
    "attn.q_pre_norm": (1, 2, 3), "attn.k_pre_norm": (1, 2, 3),
    "attn.weights": (2, 1, 3), "attn.scores": (2, 1, 3),
}

POINTS: tuple[str, ...] = tuple(LAYOUT)

RESIDUAL: tuple[str, ...] = ("resid_pre", "resid_post")

WHOLE_MODEL: tuple[str, ...] = ("embed", "final_norm", "logits")

_RETIRED = {"post": "resid_post", "pre": "resid_pre"}

POINTS_DOC = "`" + "`, `".join(POINTS) + "`"


def normalize(point: str | None, *, default: str = "resid_post") -> str:
    if point is None:
        return default
    p = str(point)
    p = _RETIRED.get(p, p)
    if p not in LAYOUT:
        raise ValueError(f"unknown point {point!r}: one of {POINTS_DOC}")
    return p


def residual(point: str | None, *, default: str = "resid_post") -> str:
    p = normalize(point, default=default)
    if p not in RESIDUAL:
        raise ValueError(f"point {point!r} is not a residual stream: one of {RESIDUAL}")
    return p


def side(point: str) -> str:
    return residual(point).removeprefix("resid_")
