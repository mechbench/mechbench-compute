"""Resume levels, the process-identity fingerprint, and training state
capture (epic 000320, task 000322).

A resumed job must be byte-identical to an uninterrupted one. That is
a promise each block makes at one of four levels:

  reproducible      every item is a pure function of (inputs, params,
                    item key, key-derived seed) — a partial set from
                    any attempt is reused as-is.
  exchangeable      items are independent draws from one fixed process
                    — a partial set may be topped up (not bit-identical;
                    the object records attempt provenance).
  state-restorable  items depend on all prior items but the full state
                    can be captured — resume from a checkpoint.
  restart           anything else, including `adaptive` blocks whose
                    next item depends on history: recompute the node.

The gate that matters more than the level is process identity: a
partial is reused only if the node's fingerprint — block, wire
params, upstream content hashes, compute version — matches the one
recorded when the partial was made. Process identity is binary; there
is no "close enough".
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

LEVELS: tuple[str, ...] = (
    "reproducible", "exchangeable", "state-restorable", "restart",
)

#: What each canonical block declares. Absent = `restart`. Item-level
#: resume is declared separately from the level: a pure block can be
#: reproducible and still not worth spooling item by item.
BLOCK_RESUME: dict[str, dict[str, Any]] = {
    "~canonical/ops/generate/1": {"level": "reproducible", "items": True},
    "~canonical/ops/decision-read/1": {"level": "reproducible", "items": True},
    "~canonical/ops/text/stats/1": {"level": "reproducible", "items": False},
    "~canonical/ops/eval/expectation/1": {"level": "reproducible", "items": False},
    "~canonical/ops/finetune/lora/1": {"level": "state-restorable", "items": False},
    "~canonical/ops/intervene/1": {"level": "reproducible", "items": True},
    "~canonical/ops/direction/vocab/1": {"level": "reproducible", "items": False},
    "~canonical/ops/direction/from-vectors/1": {"level": "reproducible", "items": False},
    "~canonical/ops/direction/from-pca/1": {"level": "reproducible", "items": False},
    "~canonical/ops/direction/add/1": {"level": "reproducible", "items": False},
    "~canonical/ops/direction/average/1": {"level": "reproducible", "items": False},
    "~canonical/ops/direction/orthogonalize/1": {"level": "reproducible", "items": False},
    "~canonical/ops/direction/normalize/1": {"level": "reproducible", "items": False},
    "~canonical/ops/direction/similarity/1": {"level": "reproducible", "items": False},
    "~canonical/ops/direction/project/1": {"level": "reproducible", "items": False},
}

_RANK = {"restart": 0, "exchangeable": 1, "state-restorable": 2, "reproducible": 2}


def resume_level(block: str) -> str:
    return BLOCK_RESUME.get(block, {}).get("level", "restart")


def item_resumable(block: str) -> bool:
    return bool(BLOCK_RESUME.get(block, {}).get("items", False))


def satisfies(offered: str, required: str) -> bool:
    """Does a block at `offered` meet a consumer's `require_resume`?
    `state-restorable` with full state capture is bit-identical, so it
    ranks with `reproducible`; `restart` recomputes and therefore
    satisfies any requirement (there is no partial to mistrust)."""
    if required not in _RANK or offered not in _RANK:
        raise ValueError(f"unknown resume level: {required!r} / {offered!r}")
    if offered == "restart":
        return True
    return _RANK[offered] >= _RANK[required]


def content_hash(value: Any) -> str:
    """sha256 of the canonical CBOR of a value — the same bytes the
    bench stores, so an upstream node's identity here equals its
    identity there."""
    from mechbench_schema import dump_canonical

    return hashlib.sha256(dump_canonical(value)).hexdigest()


def node_fingerprint(*, block: str, params: Mapping[str, Any],
                     input_hashes: list[str], core_version: str,
                     model: str = "") -> str:
    """The process identity of one node execution. Two attempts with
    the same fingerprint would compute the same items; partials cross
    attempts only under an equal fingerprint."""
    from mechbench_schema import dump_canonical

    body = {
        "block": block,
        "params": dict(params),
        "inputs": list(input_hashes),
        "compute": core_version,
        "model": model,
    }
    try:
        raw = dump_canonical(body)
    except Exception:  # noqa: BLE001 — a param that will not serialize
        raw = repr(sorted(body.items())).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


# --- training state ----------------------------------------------------------


def _mx_key() -> Any | None:
    """MLX's global RNG key, when the build exposes one. Nothing in
    the training loop draws from it (no dropout; Adam is
    deterministic) and the LoRA init it seeded is a trainable
    parameter the checkpoint carries — so its absence costs no
    bit-identity. It is captured when available for completeness."""
    import mlx.core as mx
    import numpy as np

    key = getattr(mx.random.state, "key", None)
    if key is None:
        return None
    try:
        return np.array(key)
    except Exception:  # noqa: BLE001
        return None


def _set_mx_key(value: Any) -> bool:
    import mlx.core as mx

    if value is None:
        return False
    try:
        mx.random.state.key = mx.array(value)
        return True
    except Exception:  # noqa: BLE001 — read-only state on this build
        return False


def capture_training_state(lm, opt, step: int, rng) -> dict[str, Any]:
    """Everything a step needs to continue exactly: trainable weights,
    optimizer state, the step counter, the numpy generator that samples
    sequences, and MLX's key when exposed. Arrays are numpy; the
    caller serializes (safetensors for weights, CBOR for the rest)."""
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    weights = {k: np.array(v) for k, v in tree_flatten(lm.trainable_parameters())}
    opt_state = {}
    for k, v in tree_flatten(opt.state):
        opt_state[k] = np.array(v) if isinstance(v, mx.array) else v
    return {
        "step": int(step),
        "weights": weights,
        "opt_state": opt_state,
        "np_rng": rng.bit_generator.state,
        "mx_key": _mx_key(),
    }


def restore_training_state(lm, opt, rng, state: Mapping[str, Any]) -> int:
    """Inverse of `capture_training_state`; returns the step to resume
    AFTER. `opt.init` runs first so the optimizer's tree exists to be
    overwritten."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    lm.update(tree_unflatten(
        [(k, mx.array(v)) for k, v in state["weights"].items()]))
    opt.init(lm.trainable_parameters())
    opt.state = tree_unflatten([
        (k, mx.array(v) if hasattr(v, "shape") else v)
        for k, v in state["opt_state"].items()
    ])
    rng.bit_generator.state = state["np_rng"]
    _set_mx_key(state.get("mx_key"))
    mx.eval(lm.trainable_parameters(), opt.state)
    return int(state["step"])
