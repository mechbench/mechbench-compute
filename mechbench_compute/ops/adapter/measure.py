from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.weights.compute_effective_rank import compute_effective_rank

OP = Op(
    name="adapter/measure",
    summary=(
        "Measure what training wrote, from the adapter itself: per layer and "
        "module, the norm, spectrum and effective rank of the delta, and its "
        "share of the adapter's mass — no model, no prompt, no forward pass."
    ),
    description="""\
An adapter IS a set of low-rank deltas, one per (layer, projection), and it
is already an object. Reading it answers a different question from any
capture: not what the model did on an input, but **what training changed**
— per module rather than per prompt, and in seconds rather than a forward
pass per condition.

Each item carries `frobenius` (how much was written there at all),
`spectral` and `singular_values` (how much of it is one direction),
`effective_rank` — exp(H(p)) over the normalised spectrum, so 1.0 means the
write is a line and `rank` means it fills the subspace it was given — and
`mass_share`, whose sum over the measured modules is 1: the "where did
training write" map.

The spectrum is **exact, not estimated**: `ΔW = scale · B · A` has at most
`rank` non-zero directions, so its singular values are those of an r×r
matrix built from the factors, and the delta itself is never formed.

With `vectors: true` each item also carries the principal left-singular
direction, unit length, in the module's output space. Two adapters
measured into one collection (`records/union`) are then compared by
`geometry/compare` with `by: "module"`: whether two training runs moved
the model the same way, answered in weight space rather than by capturing
what each does to a prompt.
""",
    inputs=(
        In("adapter", "adapter/lora",
           "The adapter to measure — from an `adapter/train` node or a stored "
           "one. Its bytes are read; the model it was trained on is not "
           "loaded."),
    ),
    output=Output('adapter/delta', collection=True, doc="One item per module, id and `coords.module` the module's own name in the model tree (`layers.12.self_attn.q_proj`) — two layers' `q_proj` are two modules, and `coords.projection` is what groups them: `frobenius`, `spectral`, `singular_values`, `effective_rank`, `mass_share`, `rank`, `shape`, and `vector`/`basis` when asked for. `coords` carry `layer`, `module`, `projection`, `container` and, when `source` is given, `adapter`. The header carries the adapter's `base_model`, `trained_on` and `lora` shape, and `measured` — the modules, layers and total norm the shares are shares of."),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to measure.",
          "all"),
        P("modules", "list[string] | \"all\"",
          "Which projections: `\"v_proj\"`, or `\"self_attn.v_proj\"` to "
          "disambiguate a name two containers share.",
          "all"),
        P("top_k", "int",
          "How many singular values to record per module, largest first.",
          4),
        P("vectors", "bool",
          "Also record each delta's principal direction, so two adapters can "
          "be compared at a module. One vector per module, the width of the "
          "module's output; a module training never wrote to has none, and "
          "carries no vector.",
          False),
        P("source", "string",
          "What to call this adapter, stamped on every item's "
          "`coords.adapter` — how a union of several stays readable.",
          None),
    ),
    example={"layers": [8, 12, 16], "vectors": True, "source": "zoo-cats"},
    example_inputs={"adapter": {"$ref": {"bench": "you/lab/adapter"}}},
)


def run(ctx, inputs, params):
    return _measure_adapter(inputs["adapter"], params)


def _measure_adapter(adapter: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    pass  # its imports now live in this file

    payload = adapter.get("payload", adapter) if isinstance(adapter, Mapping) else adapter
    return measure_adapter(payload, params)


def read_adapter_pairs(payload: Mapping[str, Any]) -> dict[tuple[int, str, str], dict[str, np.ndarray]]:
    """`{(layer, container, projection): {"a": A, "b": B}}` from an
    adapter object's safetensors bytes."""
    import mlx.core as mx

    from mechbench_compute.lora import KEY_RE, load_adapter

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
            m = KEY_RE.match(key)
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


def compute_delta_spectrum(a: np.ndarray, b: np.ndarray,
                           scale: float) -> tuple[np.ndarray, np.ndarray]:
    """`(singular values, left singular vectors)` of `scale · B · A`,
    exactly, without forming it. See the module docstring."""
    qb, rb = np.linalg.qr(b)            # (out, r), (r, r)
    _qa, ra = np.linalg.qr(a.T)         # (in, r), (r, r)
    u, s, _vt = np.linalg.svd(rb @ ra.T)
    return np.abs(float(scale)) * s, qb @ u


def _is_wanted(layer: int, container: str, proj: str, *,
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

    pairs = read_adapter_pairs(payload)
    chosen = {k: v for k, v in pairs.items() if _is_wanted(*k, layers=layers, modules=modules)}
    if not chosen:
        raise ValueError(
            f"no module of this adapter matches layers={layers!r} "
            f"modules={modules!r}; it carries "
            f"{', '.join(sorted({f'{c}.{p}' for _i, c, p in pairs}))} on layers "
            f"{min(i for i, _c, _p in pairs)}..{max(i for i, _c, _p in pairs)}")

    measured: list[tuple[tuple[int, str, str], np.ndarray, np.ndarray]] = []
    for key in sorted(chosen):
        ab = chosen[key]
        sv, u = compute_delta_spectrum(ab["a"], ab["b"], scale)
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
            "effective_rank": compute_effective_rank(sv),
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
