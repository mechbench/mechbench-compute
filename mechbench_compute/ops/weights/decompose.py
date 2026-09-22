from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.lexicon._base import Op, Output, P
from mechbench_compute.lexicon.model import ADAPTER
from mechbench_compute.weights.constants import RESIDUAL_SIDE
from mechbench_compute.weights.parse_parameter_coords import parse_parameter_coords
from mechbench_compute.weights.read_parameters import read_parameters
from mechbench_compute.weights.select_points import select_points

OP = Op(
    name="weights/decompose",
    requires="mlx-local",
    summary=(
        "A parameter's principal directions in the residual stream — what "
        "a projection reads, or what it writes — as directions the rest of "
        "the direction family can take."
    ),
    description="""\
A weight matrix has two spaces, its input's and its output's, and for the
attention and MLP projections exactly one of them is **the residual
stream** — the space the rest of the model, and every readout, talks
about. `q_proj`, `k_proj`, `v_proj`, `gate_proj` and `up_proj` READ the
residual stream, so their right singular vectors are the directions they
ask about; `o_proj` and `down_proj` WRITE it, so their left singular
vectors are the directions they contribute. The other side is head or
hidden space, where a direction means nothing to an unembedding or to
another layer, and this block refuses it by name.

What comes out is `direction/vector` — the same kind `direction/fit`
produces from activations — so everything that family does applies:
`direction/unembed` names the tokens a direction promotes,
`direction/project` measures activations against it,
`geometry/compare` measures two of them against each other. A weight's
direction and an activation's direction are comparable when they share a
space, which is what makes this the bridge between the two halves.

**What a principal direction is not.** σ₁ of a full weight matrix is the
dominant direction of the whole map — everything that module does to
everything it sees — and it is rarely a feature. Read through the
unembedding it usually names tokens in no order a person recognises.
The directions worth reading that way come from a matrix that was
trained to do ONE thing: an adapter's delta (`adapter/measure` with
`vectors`), or a difference between two checkpoints. Use this block for
what a module's geometry IS — rank, spread, how much of the map is one
direction — and for comparing modules across layers or models.

`points` is required: an SVD per tensor is not something to do to a
whole model by accident.
""",
    inputs=(ADAPTER,),
    output=Output('direction/vector', collection=True, doc="`top_k` items per module, ids `<parameter>#<k>`: the unit direction, `norm` its singular value, `space` naming the model, the layer and the point the module reads or writes (`attn.in_norm`, `attn_out`, `mlp.in_norm`, `mlp_out`), and `derivation` recording the module, the side and the index. The header's `decomposed` lists what was skipped and why."),
    params=(
        P("points", "list[string]",
          "Which parameters to decompose, named as the module tree names "
          "them; `*` stands for one segment. Required — an SVD per tensor "
          "is not something to do to a whole model by accident."),
        P("top_k", "int",
          "How many directions per module, largest singular value first.",
          4),
        P("side", "\"auto\" | \"in\" | \"out\"",
          "Which side to read. `auto` takes whichever side is the residual "
          "stream; naming one takes only the modules whose residual side is "
          "that, and says why it skipped the rest.",
          "auto"),
    ),
    example={"model": {"$param": "model"}, "points": ["layers.*.self_attn.o_proj"],
             "top_k": 2},
)


def run(ctx, inputs, params):
    """weights/decompose (task 000457): a parameter's principal
    directions in the residual stream, as directions the direction
    family can take."""

    model = ctx.model(params.get("model"))
    ref = params.get("model")
    return decompose_weights(
        model.lm, params,
        model_wire=ref.to_wire() if hasattr(ref, "to_wire") else ref)


def decompose_weights(lm: Any, params: Mapping[str, Any] | None = None,
                      *, model_wire: Any = None) -> dict[str, Any]:
    """`weights/decompose`: a parameter's principal directions, in the
    residual stream, as `direction/vector` items.

    The left singular vectors of `o_proj` are the directions that module
    WRITES into the residual stream, largest first; the right singular
    vectors of `q_proj` are the directions it READS. Either is a
    direction in the model's activation space, so everything the
    direction family does applies: `direction/unembed` names the tokens
    one promotes, `geometry/compare` measures two against each other.
    """
    params = dict(params or {})
    top_k = max(1, int(params.get("top_k", 4)))
    side_want = str(params.get("side", "auto"))
    tensors = read_parameters(lm)
    chosen = select_points(tensors, params.get("points", []))
    if not params.get("points"):
        raise ValueError(
            "weights/decompose needs `points`: an SVD per tensor is not "
            "something to do to a whole model by default. Name the modules "
            "— `layers.*.self_attn.o_proj`, `layers.12.mlp.down_proj`.")

    import mlx.core as mx

    items: list[dict[str, Any]] = []
    refused: list[str] = []
    for name in chosen:
        arr = np.array(tensors[name].astype(mx.float32), dtype=np.float32)
        if arr.ndim != 2:
            refused.append(f"{name} (not a matrix)")
            continue
        coords = parse_parameter_coords(name)
        proj = coords.get("projection")
        known = RESIDUAL_SIDE.get(str(proj))
        if known is None:
            refused.append(f"{name} (neither side is the residual stream)")
            continue
        side, point = known
        if side_want in ("in", "out") and side_want != side:
            refused.append(f"{name} (its {side_want} side is not the residual stream)")
            continue
        u, sv, vt = np.linalg.svd(arr, full_matrices=False)
        # `out` reads the left vectors (rows of the output space), `in`
        # the right ones (rows of the input space).
        basis = u.T if side == "out" else vt
        for k in range(min(top_k, basis.shape[0])):
            vec = basis[k]
            vec = vec / float(np.linalg.norm(vec))
            items.append({
                "id": f"{name}#{k}",
                "coords": {**coords, "index": k},
                "kind": "direction/vector",
                "space": {"model": model_wire if isinstance(model_wire, str)
                          else (model_wire or {}).get("base")
                          if isinstance(model_wire, Mapping) else None,
                          "layer": coords.get("layer"), "point": point,
                          "d": len(vec)},
                "vector": [float(x) for x in vec],
                "norm": float(sv[k]),
                "unit": True,
                "derivation": {"method": "weights/decompose", "module": name,
                               "side": side, "index": k,
                               "singular_value": float(sv[k]),
                               "spectral": float(sv[0])},
            })
    if not items:
        raise ValueError(
            "weights/decompose produced no direction: "
            + "; ".join(refused[:4])
            + ". A direction is only a direction in a space something "
            "reads — the residual stream. `q_proj`, `k_proj`, `v_proj`, "
            "`gate_proj` and `up_proj` read it; `o_proj` and `down_proj` "
            "write it.")

    from mechbench_compute.lexicon import kinds as K

    return K.collection(
        "direction/vector", items,
        model=model_wire,
        decomposed={"modules": len({it["derivation"]["module"] for it in items}),
                    "per_module": top_k, "skipped": refused},
        name=params.get("name"),
        description=params.get("description"),
    )
