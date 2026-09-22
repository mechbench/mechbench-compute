from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.weights.parse_parameter_coords import parse_parameter_coords
from mechbench_compute.weights.compute_effective_rank import compute_effective_rank
from mechbench_compute.weights.read_parameters import read_parameters
from mechbench_compute.weights.select_points import select_points

OP = Op(
    name="weights/capture",
    requires="mlx-local",
    summary=(
        "Read the model's own learned tensors — their shape, norm, "
        "sparsity and, on request, their spectrum and their values — with "
        "no prompt and no forward pass."
    ),
    description="""\
Every other readout runs the model. This one reads it: what the model IS,
rather than what it did on an input, so there is nothing for the reading
to be representative of and nothing to sample.

`points` names parameters the way the module tree does —
`layers.12.self_attn.q_proj`, `embed_tokens`,
`layers.*.mlp.down_proj`, `layers.*.post_attention_layernorm` — with
`*` standing for one segment. A point that names a module takes its
parameters; `"all"` takes every tensor the model has, which on a 4B
model is several hundred.

Each item carries the cheap facts always: `shape`, `frobenius`, `mean`,
`std`, `max_abs` (where an outlier channel shows) and `sparsity`. Two
are asked for, because both are expensive in their own way:

* `spectrum: k` computes an SVD per tensor and records the top `k`
  singular values, σ₁ and the effective rank — about a second for a
  2048×1536 matrix, so name the points you mean.
* `values: true` carries the tensor itself, and is refused past two
  million numbers across the selection: a 4B model's embedding table is
  four hundred million, and the reduced forms above are what this block
  is for.

An adapter on the node's `adapter` port is fused before the read, so the
parameters captured are the adapted ones — the base plus what training
wrote. To read what training wrote BY ITSELF, `adapter/measure` reads
the adapter's own deltas and needs no model at all.
""",
    inputs=(In("adapter", "adapter/lora",
               "A LoRA adapter to fuse on top of the model for this node only — "
               "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
               "reference, or a stored adapter. Fuses last, on top of any "
               "adapters the model reference itself carries; `adapter_scale` "
               "scales this one.",
               required=False),),
    output=Output('weights/parameter', collection=True, doc="One item per parameter tensor, id and `coords.module` naming it in the model's own tree: `shape`, `n`, `dtype`, `frobenius`, `mean`, `std`, `max_abs`, `sparsity`, plus `singular_values`/`spectral`/`effective_rank` under `spectrum` and `values` under `values`. The header carries the `model` and `captured` — the tensors read, the numbers that is, and how many the model has."),
    params=(
        P("points", "list[string] | \"all\"",
          "Which parameters to read, named as the module tree names them; "
          "`*` stands for one segment.",
          "all"),
        P("spectrum", "int",
          "Record this many singular values per tensor, with σ₁ and the "
          "effective rank. An SVD per tensor: name your points.",
          0),
        P("values", "bool",
          "Carry each tensor itself, flattened. Refused past two million "
          "numbers across the selection.",
          False),
    ),
    example={"model": {"$param": "model"}, "points": ["layers.*.mlp.down_proj"],
             "spectrum": 8},
)


def run(ctx, inputs, params):
    """weights/capture: the model's own parameters as data. No prompt,
    no forward pass — and an adapter on the port is
    fused first, so what is read is the model as this node has it."""

    model = ctx.model(params.get("model"))
    ref = params.get("model")
    return capture_weights(
        model.lm, params,
        model_wire=ref.to_wire() if hasattr(ref, "to_wire") else ref)


#: A capture refuses more values than this, as an activation capture
#: does: the reduced forms are the point, and one 262144×1536 embedding
#: table is 400M floats. `values: true` on a wide selection should fail
#: with a number, not fill a job's memory.
MAX_VALUES = 2_000_000


#: How many values a stats pass casts to float32 at once. The cast is
#: what costs memory (four bytes a number, on a model whose largest
#: tensor is 2.3 billion of them), so the reductions run in row blocks
#: of about this size.
_STAT_BLOCK_VALUES = 16_000_000


def measure_parameter(arr: Any, *, spectrum: int = 0) -> dict[str, Any]:
    """What a parameter is, as numbers.

    The cheap ones are computed WHERE THE TENSOR IS — four reductions on
    the GPU, five numbers back — because a 4B model is 4.6 billion of
    them and copying that to the host to take a mean costs half a minute
    for readings that are five floats each. The spectrum is the
    exception: an SVD is numpy's, and an SVD of a 2048×1536 matrix is
    about a second, so it happens only when asked for.
    """
    import mlx.core as mx

    a = arr if isinstance(arr, mx.array) else mx.array(np.asarray(arr))
    n = int(a.size)
    if n == 0:
        return {"frobenius": 0.0, "mean": 0.0, "std": 0.0,
                "max_abs": 0.0, "sparsity": 0.0}
    # In row blocks, for two reasons that both bite on a real model: the
    # float32 cast of a 262144×8960 embedding table is 9 GB held at once,
    # and MLX's shape dimensions are 32-bit, so its 2.3 billion elements
    # cannot be reshaped to a vector at all. Sums are accumulated in
    # float64 on the host, so a billion small numbers still add up.
    rows = a.shape[0] if a.ndim else 1
    block = max(1, min(rows, int(_STAT_BLOCK_VALUES // max(1, n // max(1, rows)))))
    total = sq = zeros = 0.0
    biggest = 0.0
    for start in range(0, rows, block):
        chunk = (a[start:start + block] if a.ndim else a).astype(mx.float32)
        s, s2, z, m = (mx.sum(chunk), mx.sum(chunk * chunk),
                       mx.sum(chunk == 0), mx.max(mx.abs(chunk)))
        mx.eval(s, s2, z, m)
        total += float(s)
        sq += float(s2)
        zeros += float(z)
        biggest = max(biggest, float(m))
    mean = total / n
    stats: dict[str, Any] = {
        "frobenius": float(np.sqrt(sq)),
        "mean": mean,
        "std": float(np.sqrt(max(0.0, sq / n - mean * mean))),
        "max_abs": biggest,
        "sparsity": zeros / n,
    }
    if spectrum and a.ndim == 2:
        sv = np.linalg.svd(np.array(a.astype(mx.float32)), compute_uv=False)
        stats["singular_values"] = [float(x) for x in sv[:spectrum]]
        stats["spectral"] = float(sv[0])
        stats["effective_rank"] = compute_effective_rank(sv)
    return stats


def capture_weights(lm: Any, params: Mapping[str, Any] | None = None,
                    *, model_wire: Any = None) -> dict[str, Any]:
    """`weights/capture`: parameters as objects — one item per tensor.

    No forward pass and no prompt: what is read is the model itself.
    """
    params = dict(params or {})
    spectrum = int(params.get("spectrum", 0) or 0)
    want_values = bool(params.get("values", False))
    tensors = read_parameters(lm)
    chosen = select_points(tensors, params.get("points", "all"))

    if want_values:
        total = sum(int(np.prod(tensors[n].shape)) for n in chosen)
        if total > MAX_VALUES:
            raise ValueError(
                f"`values: true` over {len(chosen)} parameters is "
                f"{total:,} values, past the {MAX_VALUES:,} ceiling. Name "
                f"fewer points, or leave the values off — the stats and the "
                f"spectrum are the reduced forms this block is for.")

    import mlx.core as mx

    items: list[dict[str, Any]] = []
    for name in chosen:
        tensor = tensors[name]
        item: dict[str, Any] = {
            "id": name,
            "coords": parse_parameter_coords(name),
            "kind": "weights/parameter",
            "shape": [int(d) for d in tensor.shape],
            "n": int(tensor.size),
            "dtype": str(tensor.dtype).replace("mlx.core.", ""),
            **measure_parameter(tensor, spectrum=spectrum),
        }
        if want_values:
            item["values"] = np.array(
                tensor.astype(mx.float32), dtype=np.float32).reshape(-1).tolist()
        items.append(item)

    from mechbench_compute.lexicon import kinds as K

    return K.collection(
        "weights/parameter", items,
        model=model_wire,
        captured={"parameters": len(items),
                  "values": sum(it["n"] for it in items),
                  "of": len(tensors)},
        name=params.get("name"),
        description=params.get("description"),
    )
