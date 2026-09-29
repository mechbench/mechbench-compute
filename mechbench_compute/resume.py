from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

LEVELS: tuple[str, ...] = (
    "reproducible", "exchangeable", "state-restorable", "restart",
)

_RANK = {"restart": 0, "exchangeable": 1, "state-restorable": 2, "reproducible": 2}


def read_model_level(model: Any) -> str:
    provider = (model.get("provider") if isinstance(model, Mapping)
                else getattr(model, "provider", ""))
    return "exchangeable" if provider else "reproducible"


def read_body_level(params: Mapping[str, Any]) -> str:
    body = (params or {}).get("body") or {}
    nodes = body.get("nodes") if isinstance(body, Mapping) else None
    if not nodes:
        return "restart"
    from mechbench_compute import ops

    levels = []
    for n in nodes:
        if not isinstance(n, Mapping):
            continue
        block = _name(str(n.get("block", "")))
        levels.append("reproducible" if block in ops.find_standalone()
                      else resume_level(block, n.get("params") or {}, n.get("inputs") or {}))
    return min(levels, key=lambda lv: _RANK.get(lv, 0)) if levels else "restart"


def _name(block: str) -> str:
    from mechbench_compute import lexicon

    try:
        return lexicon.resolve(block, warn=False)
    except KeyError:
        return block


def _declared(block: str) -> Any:
    from mechbench_compute import ops

    return ops.find(_name(block))


def resume_level(block: str, params: Mapping[str, Any] | None = None,
                 inputs: Mapping[str, Any] | None = None) -> str:
    mod = _declared(block)
    if mod is None:
        return "restart"
    read = getattr(mod, "read_resume_level", None)
    if read is not None and params is not None:
        return read(params, inputs)
    return mod.OP.resume.level


def item_resumable(block: str) -> bool:
    mod = _declared(block)
    return bool(mod is not None and mod.OP.resume.items)


def satisfies(offered: str, required: str) -> bool:
    if required not in _RANK or offered not in _RANK:
        raise ValueError(f"unknown resume level: {required!r} / {offered!r}")
    if offered == "restart":
        return True
    return _RANK[offered] >= _RANK[required]


def content_hash(value: Any) -> str:
    from mechbench_schema import dump_canonical

    if isinstance(value, Mapping) and any(str(k).startswith("_") for k in value):
        value = {k: v for k, v in value.items() if not str(k).startswith("_")}
    return hashlib.sha256(dump_canonical(value)).hexdigest()


def node_fingerprint(*, block: str, params: Mapping[str, Any],
                     input_hashes: list[str], core_version: str,
                     model: str = "") -> str:
    from mechbench_schema import dump_canonical

    body = {
        "block": block,
        "params": {k: (v.to_wire() if hasattr(v, "to_wire") else v)
                   for k, v in params.items()},
        "inputs": list(input_hashes),
        "compute": core_version,
        "model": model,
    }
    try:
        raw = dump_canonical(body)
    except Exception as e:  # noqa: BLE001
        bad = [k for k, v in body["params"].items()
               if not _encodes(v)]
        raise TypeError(
            f"node fingerprint for {block!r}: params {bad or '?'} do not "
            f"canonical-encode ({type(e).__name__}: {e}); a block must "
            f"carry wire forms in its params, never live objects") from e
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _encodes(value: Any) -> bool:
    from mechbench_schema import dump_canonical

    try:
        dump_canonical(value)
        return True
    except Exception:  # noqa: BLE001
        return False


def _mx_key() -> Any | None:
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
    except Exception:  # noqa: BLE001
        return False


def capture_training_state(lm, opt, step: int, rng) -> dict[str, Any]:
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
