from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FUNCTIONS = ("affine", "polynomial", "mlp")

POSITIONS = ("last", "all")

FIELDS = ("layers", "point", "positions", "gate", "mask", "function", "degree", "width", "penalty")


def _is_index(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def read_operator_spec(value: Any, *, n_layers: int, d: int) -> dict[str, Any]:
    from mechbench_compute.lora import check_layers

    if not isinstance(value, Mapping):
        raise TypeError("an operator is an object: {layers, mask, function, …}")
    unknown = sorted(set(value) - set(FIELDS))
    if unknown:
        raise ValueError(f"an operator takes {', '.join(FIELDS)}; not {', '.join(unknown)}")
    if value.get("point", "resid_post") != "resid_post":
        raise ValueError(
            f"an operator acts on the residual stream after a layer, `resid_post`, not "
            f"{value['point']!r}: the stream before layer i is the one after layer i − 1")
    layers = value.get("layers")
    layers = [layers] if _is_index(layers) else layers
    if not (isinstance(layers, list) and layers and all(_is_index(i) for i in layers)
            and layers == sorted(set(layers))):
        raise ValueError(f"an operator's `layers` is a layer, or layers ascending and each once; "
                         f"got {value.get('layers')!r}")
    check_layers(layers, n_layers)
    gate = value.get("gate", False)
    if not isinstance(gate, bool):
        raise TypeError(f"an operator's `gate` is true or false, not {gate!r}")
    positions = value.get("positions", "all" if gate else "last")
    if positions not in POSITIONS or (gate and positions != "all"):
        raise ValueError(f"an operator acts at `last` (the last position of every forward pass) or "
                         f"`all`, and a gated one at `all`; not {positions!r}")
    mask = value.get("mask")
    if isinstance(mask, list):
        beyond = [i for i in mask if not (_is_index(i) and i < d)]
        if not mask or beyond or len(set(mask)) != len(mask):
            raise ValueError(f"MASK_OUT_OF_RANGE: an operator's dimensions are 0 through {d - 1}, "
                             f"each once; got {mask!r}")
    elif isinstance(mask, Mapping):
        rank = mask.get("rank")
        if set(mask) != {"rank"} or not (_is_index(rank) and 1 <= rank <= d):
            raise ValueError(f"a learned subspace is {{\"rank\": r}}, r from 1 to {d}; got {mask!r}")
    elif mask is not None:
        raise ValueError(f"an operator's `mask` is a list of dimensions, {{\"rank\": r}} or absent "
                         f"(every coordinate); not {mask!r}")
    function = value.get("function")
    if function not in FUNCTIONS:
        raise ValueError(f"an operator's `function` is one of {', '.join(FUNCTIONS)}; got {function!r}")
    spec: dict[str, Any] = {"point": "resid_post", "layers": layers, "positions": positions,
                            "gate": gate, "mask": mask, "function": function}
    for name, used in (("degree", "polynomial"), ("width", "mlp")):
        if value.get(name) is None:
            if function == used:
                spec[name] = 2 if name == "degree" else 16
            continue
        if function != used or not (_is_index(value[name]) and value[name] >= 1):
            raise ValueError(f"`{name}` is a whole number of at least 1, and belongs to `{used}`")
        spec[name] = value[name]
    penalty = value.get("penalty")
    if penalty is not None:
        l1 = penalty.get("l1") if isinstance(penalty, Mapping) else None
        if not (isinstance(penalty, Mapping) and set(penalty) == {"l1"}
                and isinstance(l1, (int, float)) and not isinstance(l1, bool) and l1 >= 0):
            raise ValueError(f"an operator's `penalty` is {{\"l1\": λ}}, λ at least 0; got {penalty!r}")
        spec["penalty"] = {"l1": float(l1)}
    return spec
