"""Weight space: what training WROTE, read from the weights themselves.

Every readout the lexicon had until now reads an activation at a hook
point during a forward pass — what the model did on this input. A
parameter answers a different question: what the model IS, and what
training changed. An adapter is the purest case of the second: it is
nothing but a set of low-rank deltas, one per (layer, projection), and
it is already an object on the bench. Reading it needs no model, no
prompt and no forward pass (task 000458, weights-as-points 000457).

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
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


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
