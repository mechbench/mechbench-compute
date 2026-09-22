"""Weight space: the model's parameters, and what training wrote to them.

Most readouts read an activation at a hook point during a forward pass
— what the model did on this input. A parameter answers a different
question: what the model IS, and what training changed. Neither needs a
prompt, a sample, or anything to be representative of.

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
  module are comparable through it (`geometry/compare`) — the cosine
  question asked of the weights instead of the activations.
"""

from __future__ import annotations

from mechbench_compute.weights.compute_effective_rank import compute_effective_rank  # noqa: F401
from mechbench_compute.weights.edit_parameters import edit_parameters  # noqa: F401
from mechbench_compute.weights.read_parameters import read_parameters  # noqa: F401
from mechbench_compute.weights.restore_parameters import restore_parameters  # noqa: F401
from mechbench_compute.weights.select_points import select_points  # noqa: F401

