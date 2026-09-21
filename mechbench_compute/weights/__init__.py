"""Weight space: the model's parameters, and what training wrote to them.

Every readout the lexicon had until 0.81.2 reads an activation at a hook
point during a forward pass — what the model did on this input. A
parameter answers a different question: what the model IS, and what
training changed. Neither needs a prompt, a sample, or anything to be
representative of (weights-as-points, tasks 000457 and 000458).

Three readouts live here:

- **`weights/capture`** — the model's own tensors, named by the module
  tree (`layers.12.self_attn.q_proj.weight`). Shape, norm, sparsity and
  outlier magnitude always; the spectrum and the values only when asked,
  because both are expensive in their own way.
- **`weights/decompose`** — a parameter's principal directions IN THE
  RESIDUAL STREAM, as `direction/vector` items the direction family can
  take. Which side of a matrix is the residual stream is a per-module
  fact (`RESIDUAL_SIDE`), and a module where neither side is refuses.
- **`adapter/measure`** — an adapter's deltas, which are already the
  difference training made, and need neither the model nor its base.

**The r×r trick.** A LoRA delta is `ΔW = scale · B · A` with `B` out×r
and `A` r×in, r ≤ 8 — so ΔW is a 2560×2048 matrix of which at most 8
directions are non-zero. Forming it to take its norm or its spectrum
would be 5M floats per module, 70 modules per adapter, for information
that lives in an 8×8 matrix:

    B = Q_B R_B,  Aᵀ = Q_A R_A          (thin QR, r columns each)
    ΔW = scale · Q_B (R_B R_Aᵀ) Q_Aᵀ

`M = R_B R_Aᵀ` is r×r, its singular values ARE ΔW's (times |scale|),
and ΔW's left singular vectors are `Q_B U_M`. Everything here is exact
— not an estimate of a big matrix, but the small matrix that big one
was made of.

The numbers each module gets:

- **frobenius** — ‖ΔW‖_F: how much was written there at all.
- **spectral** — σ₁: how much of it is one direction.
- **singular_values** — the whole spectrum, r of them.
- **effective_rank** — exp(H(p)) over p = σ/Σσ (Roy & Vetterli): 1.0
  when the write is a single direction, r when it is spread evenly.
  A rank-8 adapter whose effective rank is 1.2 wrote a line, not a
  subspace, whatever its configuration said.
- **mass_share** — this module's ‖ΔW‖²_F over the adapter's total: the
  "where did training write" map, summing to 1 across the adapter.
- **vector** — the principal left-singular direction, in the module's
  OUTPUT space, when asked for. Two adapters' writes at the same
  module are comparable through it (`geometry/compare`), which is
  experiment 018's cosine question asked of the weights instead of the
  activations.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from mechbench_compute.weights.constants import RESIDUAL_SIDE, _SCOPE  # noqa: F401
from mechbench_compute.weights.coords_of import _coords_of  # noqa: F401
from mechbench_compute.weights.direction_of import _direction_of  # noqa: F401
from mechbench_compute.weights.edit_parameters import WEIGHT_OPS, edit_parameters  # noqa: F401
from mechbench_compute.weights.module_of import _module_of  # noqa: F401
from mechbench_compute.weights.parameter_names import parameter_names  # noqa: F401
from mechbench_compute.weights.pattern import _pattern  # noqa: F401
from mechbench_compute.weights.project_out import _project_out  # noqa: F401
from mechbench_compute.weights.restore_parameters import restore_parameters  # noqa: F401
from mechbench_compute.weights.select_points import select_points  # noqa: F401
from mechbench_compute.weights.truncate import _truncate  # noqa: F401

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


def parameter_stats(arr: Any, *, spectrum: int = 0) -> dict[str, Any]:
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
        stats["effective_rank"] = effective_rank(sv)
    return stats


def capture_weights(lm: Any, params: Mapping[str, Any] | None = None,
                    *, model_wire: Any = None) -> dict[str, Any]:
    """`weights/capture`: parameters as objects — one item per tensor.

    No forward pass and no prompt: what is read is the model itself.
    """
    params = dict(params or {})
    spectrum = int(params.get("spectrum", 0) or 0)
    want_values = bool(params.get("values", False))
    tensors = parameter_names(lm)
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
            "coords": _coords_of(name),
            "kind": "weights/parameter",
            "shape": [int(d) for d in tensor.shape],
            "n": int(tensor.size),
            "dtype": str(tensor.dtype).replace("mlx.core.", ""),
            **parameter_stats(tensor, spectrum=spectrum),
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
    tensors = parameter_names(lm)
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
        coords = _coords_of(name)
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


def adapter_pairs(payload: Mapping[str, Any]) -> dict[tuple[int, str, str], dict[str, np.ndarray]]:
    """`{(layer, container, projection): {"a": A, "b": B}}` from an
    adapter object's safetensors bytes."""
    import mlx.core as mx

    from mechbench_compute.lora import _KEY_RE, load_adapter

    data = payload.get("data")
    if data is None:
        raise ValueError(
            "this adapter object carries no `data`: an adapter is read from "
            "its safetensors bytes, and this one has none")
    if isinstance(data, str):
        # A JSON round-trip base64s the bytes; CBOR keeps them binary.
        import base64

        data = base64.b64decode(data)
    pairs: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)
        # Converted INSIDE the try: `mx.load` is lazy and its arrays are
        # backed by the file, so touching one after the unlink aborts the
        # process rather than raising.
        for key, w in load_adapter(path).items():
            m = _KEY_RE.match(key)
            if m is None:
                raise ValueError(f"unrecognized adapter key {key!r}")
            i, container, proj, ab = (int(m.group(1)), m.group(2),
                                      m.group(3), m.group(4))
            pairs.setdefault((i, container, proj), {})[ab] = np.array(
                w.astype(mx.float32), dtype=np.float64)
    finally:
        os.unlink(path)
    missing = [k for k, v in pairs.items() if set(v) != {"a", "b"}]
    if missing:
        where = ", ".join(f"{i}.{c}.{p}" for i, c, p in sorted(missing))
        raise ValueError(
            f"adapter is missing lora_a or lora_b for {where} — half a "
            "delta is not a delta")
    return pairs


def delta_spectrum(a: np.ndarray, b: np.ndarray,
                   scale: float) -> tuple[np.ndarray, np.ndarray]:
    """`(singular values, left singular vectors)` of `scale · B · A`,
    exactly, without forming it. See the module docstring."""
    qb, rb = np.linalg.qr(b)            # (out, r), (r, r)
    _qa, ra = np.linalg.qr(a.T)         # (in, r), (r, r)
    u, s, _vt = np.linalg.svd(rb @ ra.T)
    return np.abs(float(scale)) * s, qb @ u


def effective_rank(sv: Sequence[float] | np.ndarray) -> float:
    """exp(H(p)) over the normalised spectrum: 1 for a single
    direction, r for r equal ones, 0 for a delta that is all zero."""
    s = np.asarray(sv, dtype=np.float64)
    total = float(s.sum())
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _wanted(layer: int, container: str, proj: str, *,
            layers: Any, modules: Any) -> bool:
    if layers not in (None, "all") and int(layer) not in {int(x) for x in layers}:
        return False
    if modules not in (None, "all"):
        want = {str(m) for m in modules}
        if proj not in want and f"{container}.{proj}" not in want:
            return False
    return True


def measure_adapter(payload: Mapping[str, Any],
                    params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """`adapter/measure`: one `adapter/delta` item per (layer, module).

    Pure: the adapter's own bytes and nothing else. The header carries
    what the adapter says about itself, so a reader of the result knows
    which training wrote these numbers.
    """
    params = dict(params or {})
    cfg = dict(payload.get("lora") or {})
    rank = int(cfg.get("rank", 8))
    alpha = float(cfg.get("alpha", 16))
    scale = float(cfg.get("scale", alpha / rank if rank else 1.0))
    layers = params.get("layers", "all")
    modules = params.get("modules", "all")
    top_k = int(params.get("top_k", 4))
    want_vectors = bool(params.get("vectors", False))
    source = params.get("source")

    pairs = adapter_pairs(payload)
    chosen = {k: v for k, v in pairs.items() if _wanted(*k, layers=layers, modules=modules)}
    if not chosen:
        raise ValueError(
            f"no module of this adapter matches layers={layers!r} "
            f"modules={modules!r}; it carries "
            f"{', '.join(sorted({f'{c}.{p}' for _i, c, p in pairs}))} on layers "
            f"{min(i for i, _c, _p in pairs)}..{max(i for i, _c, _p in pairs)}")

    measured: list[tuple[tuple[int, str, str], np.ndarray, np.ndarray]] = []
    for key in sorted(chosen):
        ab = chosen[key]
        sv, u = delta_spectrum(ab["a"], ab["b"], scale)
        measured.append((key, sv, u))
    # The share is of what was MEASURED, and the header says so — a node
    # that read three layers must not imply those are the whole adapter.
    energy = np.array([float((sv ** 2).sum()) for _k, sv, _u in measured])
    total = float(energy.sum())

    items: list[dict[str, Any]] = []
    for ((layer, container, proj), sv, u), e in zip(measured, energy):
        # The module is the one in the model's own tree, layer included:
        # two layers' `q_proj` are two modules, and a comparison groups
        # on THIS, not on the projection they share.
        module = f"layers.{layer}.{container}.{proj}"
        coords: dict[str, Any] = {"layer": int(layer), "module": module,
                                  "projection": proj, "container": container}
        if source is not None:
            coords["adapter"] = str(source)
        item: dict[str, Any] = {
            "id": module,
            "coords": coords,
            "kind": "adapter/delta",
            "frobenius": float(np.sqrt(e)),
            "spectral": float(sv[0]) if len(sv) else 0.0,
            "singular_values": [float(x) for x in sv[:max(0, top_k)]],
            "effective_rank": effective_rank(sv),
            "mass_share": float(e / total) if total > 0 else 0.0,
            "rank": len(sv),
            "shape": [int(u.shape[0]), int(pairs[(layer, container, proj)]["a"].shape[1])],
        }
        if want_vectors and item["frobenius"] > 0:
            # A module training never wrote to has no principal
            # direction, and a zero vector is not one: it is left off,
            # and a comparison that needs it refuses by name rather than
            # returning the cosine of nothing.
            direction = u[:, 0]
            direction = direction / float(np.linalg.norm(direction))
            item["vector"] = [float(x) for x in direction]
            item["basis"] = {"module": module, "side": "out", "d": len(direction)}
        items.append(item)

    from mechbench_compute.lexicon import kinds as K

    return K.collection(
        "adapter/delta", items,
        base_model=payload.get("base_model"),
        trained_on=payload.get("trained_on"),
        lora={"rank": rank, "alpha": alpha, "scale": scale,
              "target_modules": cfg.get("target_modules")},
        measured={"modules": len(items),
                  "layers": sorted({int(i) for i, _c, _p in chosen}),
                  "frobenius": float(np.sqrt(total))},
        source=str(source) if source is not None else None,
        name=params.get("name"),
        description=params.get("description"),
    )


# --- a head's circuits, read from the weights alone (task 000610) -----------------


def head_circuits(model, params: Mapping[str, Any] | None = None,
                  on_item=None, on_start=None) -> dict[str, Any]:
    """What a head does, read from its weights and the vocabulary — no
    forward pass, no corpus (task 000610).

    `ov`: the top singular components of W_O·W_V, each an input
    direction (tokens that trigger it) paired with an output direction
    (tokens it then promotes). A copying head shows the same tokens on
    both sides.

    `qk`: the same for W_Qᵀ·W_K — the query tokens a component looks
    for, and the key tokens it matches. Positional terms are not in it:
    RoPE is applied to the activations, not the weights.

    `composition`: how much of what each earlier head WRITES lands in
    what this head READS, as Q-, K- or V-composition — which is how one
    names the heads that feed a head before patching any of them.
    """
    from mechbench_compute import head_weights as hw
    from mechbench_compute.lexicon import kinds as K

    params = dict(params or {})
    which = str(params.get("circuit", "ov"))
    if which not in ("ov", "qk", "composition"):
        raise ValueError(f"circuit is 'ov', 'qk' or 'composition', not {which!r}")
    n_layers, n_heads = model.arch.n_layers, model.arch.n_heads
    head = params.get("head")

    if which == "composition":
        if not isinstance(head, Mapping) or "layer" not in head:
            raise ValueError(
                'composition reads INTO one head: name it, `head: {"layer": 12, "index": 3}`')
        rows = hw.head_composition(
            model, int(head["layer"]), int(head.get("index", 0)),
            source_layers=params.get("layers"),
            kinds=tuple(params.get("kinds") or ("q", "k", "v")))
        if on_start:
            on_start(len(rows))
        if on_item:
            for _ in rows:
                on_item()
        return K.collection(
            "records/record", rows, circuit="composition",
            into={"layer": int(head["layer"]), "index": int(head.get("index", 0))},
            model=str(getattr(model, "model_id", "") or ""),
            description=(
                "Q-, K- and V-composition of each earlier head with this one: "
                "how much of what it writes lands in what this head reads."))

    if isinstance(head, Mapping):
        pairs = [(int(head["layer"]), int(head.get("index", 0)))]
    else:
        layers = ([int(x) for x in params["layers"]] if params.get("layers") is not None
                  else list(range(n_layers)))
        heads = ([int(x) for x in params["heads"]] if params.get("heads") is not None
                 else list(range(n_heads)))
        pairs = [(lay, h) for lay in layers for h in heads]
    for lay, h in pairs:
        if not 0 <= lay < n_layers or not 0 <= h < n_heads:
            raise ValueError(
                f"layer {lay} head {h} is outside this model ({n_layers} layers, "
                f"{n_heads} heads)")
    components = int(params.get("components", 3))
    top_k = int(params.get("top_k", 10))
    if on_start:
        on_start(len(pairs))

    def _tokens(pairs_):
        return [{"token": t, "score": round(float(v), 5)} for t, v in pairs_]

    analyse = hw.ov_circuit if which == "ov" else hw.qk_circuit
    rows = []
    for lay, h in pairs:
        got = analyse(model, lay, h, k_tokens=top_k, n_components=components)
        for comp in got.components:
            rows.append({
                "id": f"L{lay}H{h}:{comp.rank}",
                "coords": {"layer": lay, "head": h, "rank": comp.rank, "circuit": which},
                "strength": round(float(comp.strength), 5),
                # For `ov`: what it writes, and what triggers the write.
                # For `qk`: what the query looks for, and what the key
                # matches. The names say which side, not which meaning.
                "left": _tokens(comp.left_tokens),
                "right": _tokens(comp.right_tokens),
                "kv_group": got.kv_group,
            })
        if on_item:
            on_item()
    return K.collection(
        "records/record", rows, circuit=which, components=components, top_k=top_k,
        model=str(getattr(model, "model_id", "") or ""),
        description=(
            "The top singular components of each head's "
            + ("W_O·W_V: the tokens that trigger a write, and the tokens it writes."
               if which == "ov" else
               "W_Qᵀ·W_K: the query tokens a component looks for, and the keys it matches.")))
