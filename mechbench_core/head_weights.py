"""Static weight-level analysis of attention heads.

No forward passes, no hooks. Just read what each head reads, writes, and
emits by looking at its W_Q / W_K / W_V / W_O matrices and the tied
embedding. This is the cheapest kind of mech-interp analysis — pure
weight-space — and it produces a per-head interpretability map of the
whole network in seconds.

Gemma 4 E4B specifics:
  - 8 query heads, 2 KV heads per layer (Grouped Query Attention).
  - Local layers: head_dim = 256. Global layers (5,11,17,23,29,35,41):
    head_dim = 512.
  - Q head h maps to KV-group (h * n_kv_heads) // n_heads = h // 4.
  - Layers 24-41 have is_kv_shared_layer=True — their own stored W_K/W_V
    weights exist but at inference time K/V come from earlier layers. For
    weight-level analysis we just read the per-layer weights as stored;
    whether those weights are used in practice is a separate question.

Three analyses per head:
  - READ:   top tokens t by ||W_Q[h] @ E[t]||^2 — 'if the residual at the
             query position looks like this token, this head emits a large
             query.' Measures what the head's query projection picks up.
  - KEY:    top tokens t by ||W_K[kv_group(h)] @ E[t]||^2 — 'what tokens
             does this KV-head's key projection advertise.'
  - OV:     SVD of W_O[h-slice] @ W_V[kv_group(h)]. For each top singular
             component (u, σ, v): u projected through the unembed gives
             the 'output tokens this head writes' and v projected through
             the unembed gives 'input tokens whose attention would trigger
             this write.'

Caveats:
  - RoPE is applied to Q and K post-projection. It rotates them but
    preserves norms, so our READ and KEY scores are unchanged. But the
    QK dot product itself IS affected by RoPE (positional info enters
    there). The QK-circuit SVD analysis without RoPE gives a
    content-only view; the positional dimension lives in the rotation.
  - q_norm and k_norm are RMSNorm applied per-head post-projection. They
    normalize away the magnitude of each head's Q and K, so the absolute
    scores have no semantic meaning — only the ranking does.
  - Pre-attention input_layernorm (RMSNorm with scale) is applied to the
    residual before q_proj / k_proj / v_proj. We ignore it for
    simplicity. A more careful analysis would fold its learned scale into
    the W_Q/W_K/W_V matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mlx.core as mx
import numpy as np

from .interventions import Capture


@dataclass(frozen=True)
class HeadSpec:
    """Weight slices for one Q-head, plus its shared KV-group slices."""

    layer: int
    head: int          # Q-head index, 0..n_heads-1
    kv_group: int      # which KV-head this Q-head reads from, 0..n_kv_heads-1
    head_dim: int
    n_heads: int
    n_kv_heads: int
    W_Q: np.ndarray    # [head_dim, d_model]
    W_K: np.ndarray    # [head_dim, d_model]  (KV-group's weights)
    W_V: np.ndarray    # [head_dim, d_model]
    W_O: np.ndarray    # [d_model, head_dim]  (slice of o_proj for this Q-head)
    is_global: bool
    is_kv_shared: bool


@dataclass(frozen=True)
class CircuitComponent:
    """One top-K singular component of a head's QK or OV circuit."""

    rank: int                              # 0 = largest singular value
    strength: float                        # singular value
    left_tokens: list[tuple[str, float]]   # top tokens for the left singular vector
    right_tokens: list[tuple[str, float]]  # top tokens for the right singular vector


@dataclass(frozen=True)
class CircuitAnalysis:
    """Top singular components of a head's QK or OV circuit.

    For circuit_type='QK': left_tokens are 'query tokens this component
    looks for'; right_tokens are 'key tokens this component matches.'
    For circuit_type='OV': left_tokens are 'output tokens this component
    writes'; right_tokens are 'input tokens that trigger this write.'
    """

    circuit_type: str   # 'QK' or 'OV'
    layer: int
    head: int
    kv_group: int
    components: tuple[CircuitComponent, ...]


def _as_np_f32(mx_tensor: mx.array) -> np.ndarray:
    return np.array(mx_tensor.astype(mx.float32))


def _embed_matrix_f32(model) -> np.ndarray:
    """Vocab x d_model token embedding matrix as float32 numpy."""
    e = model._model.language_model.model.embed_tokens.weight
    return _as_np_f32(e)


def _unit_normalized_embed(model) -> np.ndarray:
    """Vocab x d_model matrix with each row L2-normalized to unit length.

    The model's pre-attention RMSNorm rescales each residual to roughly
    unit RMS before W_Q / W_K / W_V are applied. If we want the READ /
    KEY scores to reflect what the model actually picks up, we should
    normalize token embeddings the same way. Using the raw embeddings
    lets high-norm tokens (which are rare-token / code-snippet /
    unusual-unicode artifacts) dominate any ranking by ||W @ E[t]||, so
    unit-normalizing is the right default.
    """
    e = _embed_matrix_f32(model)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.clip(norms, 1e-12, None)


def get_head_spec(model, layer: int, head: int) -> HeadSpec:
    """Extract per-head Q, K, V, O weight slices for one Q-head."""
    block = model._model.language_model.model.layers[layer]
    attn = block.self_attn
    head_dim = int(attn.head_dim)
    n_heads = int(attn.n_heads)
    n_kv_heads = int(attn.n_kv_heads)
    kv_group = head * n_kv_heads // n_heads

    W_Q_full = _as_np_f32(attn.q_proj.weight)   # [n_heads*head_dim, d_model]
    W_K_full = _as_np_f32(attn.k_proj.weight)   # [n_kv_heads*head_dim, d_model]
    W_V_full = _as_np_f32(attn.v_proj.weight)
    W_O_full = _as_np_f32(attn.o_proj.weight)   # [d_model, n_heads*head_dim]

    W_Q = W_Q_full[head * head_dim:(head + 1) * head_dim, :]
    W_K = W_K_full[kv_group * head_dim:(kv_group + 1) * head_dim, :]
    W_V = W_V_full[kv_group * head_dim:(kv_group + 1) * head_dim, :]
    W_O = W_O_full[:, head * head_dim:(head + 1) * head_dim]

    return HeadSpec(
        layer=layer, head=head, kv_group=kv_group,
        head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O,
        is_global=(attn.layer_type == "full_attention"),
        is_kv_shared=bool(attn.is_kv_shared_layer),
    )


# ---------------------------------------------------------------------------
# Top-tokens-by-direction helpers
# ---------------------------------------------------------------------------


def _top_tokens_for_direction(
    model, direction: np.ndarray, k: int = 10,
    embed: Optional[np.ndarray] = None,
) -> list[tuple[str, float]]:
    """Project `direction` (d_model) through the token embedding and return
    top-k (decoded token string, raw inner product) pairs.

    Uses the raw embedding inner product (no final RMSNorm, no softcap).
    For 'decoded distribution as the model would see it,' use
    Model.decoded_distribution instead.
    """
    if embed is None:
        embed = _embed_matrix_f32(model)
    logits = embed @ direction                       # [vocab]
    top_idx = np.argsort(-logits)[:k]
    return [
        (model.tokenizer.decode([int(i)]), float(logits[int(i)]))
        for i in top_idx
    ]


def head_read_tokens(
    model, layer: int, head: int, k: int = 10,
    embed: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> list[tuple[str, float]]:
    """Top-k tokens t maximizing ||W_Q[h] @ E[t]||^2 (optionally unit-E[t]).

    Interpretation: these are the tokens whose embeddings, placed at the
    query position, produce the largest query vector for this head — i.e.
    the tokens this head 'reads' most strongly.

    Args:
        normalize: If True (default) each token's embedding is
            L2-normalized before the projection, matching what the model
            does via its pre-attention RMSNorm. If False, raw embeddings
            are used and high-magnitude-embedding tokens (unusual-unicode
            / rare / code-fragment tokens) dominate.

    Returns:
        list of (decoded_token, score). Scores have no absolute meaning
        across heads; only the ranking is informative.
    """
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _unit_normalized_embed(model) if normalize else _embed_matrix_f32(model)
    # Batched: [V, d_model] @ [d_model, head_dim] -> [V, head_dim]
    projected = embed @ spec.W_Q.T
    scores = (projected ** 2).sum(axis=1)
    top_idx = np.argsort(-scores)[:k]
    return [
        (model.tokenizer.decode([int(i)]), float(scores[int(i)]))
        for i in top_idx
    ]


def head_key_tokens(
    model, layer: int, head: int, k: int = 10,
    embed: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> list[tuple[str, float]]:
    """Top-k tokens t maximizing ||W_K[kv_group(h)] @ E[t]||^2.

    See head_read_tokens for the normalize argument. Because KV-group is
    shared among 4 Q-heads (Gemma 4 E4B GQA ratio 4:1), all 4 Q-heads
    within the same group produce identical head_key_tokens output.
    """
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _unit_normalized_embed(model) if normalize else _embed_matrix_f32(model)
    projected = embed @ spec.W_K.T
    scores = (projected ** 2).sum(axis=1)
    top_idx = np.argsort(-scores)[:k]
    return [
        (model.tokenizer.decode([int(i)]), float(scores[int(i)]))
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# Circuit SVD: QK and OV
# ---------------------------------------------------------------------------


def _truncated_svd_of_product(
    A: np.ndarray, B: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of M = A @ B without forming the full [m, n] product.

    A: [m, r], B: [r, n]. M has rank <= r. We compute QR of A, then SVD
    of the small (r, n) matrix R @ B. Much faster than direct SVD of
    [m, n] when r << m, n, which is our case (r = head_dim, m = n = d_model).

    Returns top-k components: U [m, k], S [k], Vt [k, n].
    """
    Q_A, R_A = np.linalg.qr(A)                     # Q_A: [m, r], R_A: [r, r]
    RB = R_A @ B                                    # [r, n]
    U_small, S, Vt = np.linalg.svd(RB, full_matrices=False)  # [r,r], [r], [r,n]
    U = Q_A @ U_small                               # [m, r]
    k = min(k, len(S))
    return U[:, :k], S[:k], Vt[:k, :]


def qk_circuit(
    model, layer: int, head: int,
    k_tokens: int = 10, n_components: int = 5,
    embed: Optional[np.ndarray] = None,
) -> CircuitAnalysis:
    """Top singular components of M_QK = W_Q[h].T @ W_K[kv_group(h)].

    For the i-th component (u_i, sigma_i, v_i):
      - u_i is a d_model query-direction: tokens near u_i (in embedding
        space) are the tokens this head MOST LOOKS FOR at the query
        position.
      - v_i is a d_model key-direction: tokens near v_i are the ones MOST
        ATTENDED TO when the query matches u_i.
      - sigma_i is the strength of this (query, key) pairing.
    """
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _embed_matrix_f32(model)
    # M_QK = W_Q.T @ W_K. Shape [d_model, head_dim] @ [head_dim, d_model].
    A = spec.W_Q.T   # [d_model, head_dim]
    B = spec.W_K     # [head_dim, d_model]
    U, S, Vt = _truncated_svd_of_product(A, B, n_components)
    components = tuple(
        CircuitComponent(
            rank=i, strength=float(S[i]),
            left_tokens=_top_tokens_for_direction(model, U[:, i], k_tokens, embed),
            right_tokens=_top_tokens_for_direction(model, Vt[i, :], k_tokens, embed),
        )
        for i in range(len(S))
    )
    return CircuitAnalysis(
        circuit_type="QK", layer=layer, head=head,
        kv_group=spec.kv_group, components=components,
    )


# ---------------------------------------------------------------------------
# Activation-level OV trajectories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionWrite:
    """Top tokens written by a head at one query position in one forward pass."""

    position: int
    query_token: str              # decoded token at this query position
    top_tokens: list[tuple[str, float]]  # (token, raw logit) pairs


def head_ov_position_writes(
    model, input_ids: mx.array, layer: int, head: int,
    *, k: int = 10, embed: Optional[np.ndarray] = None,
) -> list[PositionWrite]:
    """For each position p in the prompt, return the top-k tokens this head
    would write into the residual IF it attended fully to position p.

    This is the 'potential' read-out: what the head's OV circuit produces
    when pointed at each position's value vector. Independent of the actual
    attention pattern.

    Computation per position p:
      1. Capture attn.v — shape [1, n_kv_heads, L, head_dim].
      2. v_p = V[0, kv_group(head), p, :]  (the value this head's KV-group
         would emit from position p).
      3. Apply W_O[:, h_slice]: residual_delta = W_O_slice @ v_p  [d_model].
      4. Project through the tied embedding: logits = embed @ residual_delta.
      5. Take top-k tokens.

    Args:
        model, input_ids: as Model.run.
        layer, head: the Q-head to analyze. Pairs with kv_group = head // 4.
        k: number of top tokens to return per position.
        embed: optional cached embedding matrix; computed if None.

    Returns:
        list of PositionWrite, one per input position.
    """
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _embed_matrix_f32(model)

    result = model.run(input_ids, interventions=[Capture.values(layer)])
    V = np.array(result.cache[f"blocks.{layer}.attn.v"].astype(mx.float32))
    # V shape: [1, n_kv_heads, L, head_dim]. Pick our KV-group slice.
    V_group = V[0, spec.kv_group]  # [L, head_dim]

    # W_O[h_slice]: [d_model, head_dim]. residual_delta[p] = W_O @ V_group[p].
    writes = V_group @ spec.W_O.T  # [L, d_model]
    logits = writes @ embed.T       # [L, vocab]

    L = V_group.shape[0]
    out: list[PositionWrite] = []
    for p in range(L):
        top_idx = np.argsort(-logits[p])[:k]
        token_list = [
            (model.tokenizer.decode([int(i)]), float(logits[p, int(i)]))
            for i in top_idx
        ]
        query_tok = model.tokenizer.decode([int(input_ids[0, p])])
        out.append(PositionWrite(
            position=p, query_token=query_tok, top_tokens=token_list,
        ))
    return out


def head_ov_actual_writes(
    model, input_ids: mx.array, layer: int, head: int,
    *, k: int = 10, embed: Optional[np.ndarray] = None,
) -> list[PositionWrite]:
    """For each QUERY position q, return the top-k tokens this head actually
    writes into the residual at q during this forward pass (weighted by its
    attention pattern).

    Differs from head_ov_position_writes: uses the post-attention-weighting
    per_head_out tensor (the weighted sum softmax(QK) @ V) instead of raw V.
    This is the 'actual' token contribution at each query position, not the
    'potential' contribution if the head attended to each position fully.

    Computation per query position q:
      1. Capture attn.per_head_out — shape [1, n_heads, L, head_dim].
      2. ph_q = per_head_out[0, head, q, :]  (what this head emits at q).
      3. residual_delta = W_O[h_slice] @ ph_q  [d_model].
      4. logits = embed @ residual_delta; top-k tokens.

    The sum of these per-query deltas across all heads (plus other branches)
    equals the head's total contribution to the residual stream at each
    query position.
    """
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _embed_matrix_f32(model)

    result = model.run(input_ids, interventions=[Capture.per_head_out(layer)])
    PH = np.array(result.cache[f"blocks.{layer}.attn.per_head_out"].astype(mx.float32))
    # PH shape: [1, n_heads, L, head_dim]. Pick this head.
    PH_h = PH[0, head]  # [L, head_dim]

    writes = PH_h @ spec.W_O.T  # [L, d_model]
    logits = writes @ embed.T    # [L, vocab]

    L = PH_h.shape[0]
    out: list[PositionWrite] = []
    for q in range(L):
        top_idx = np.argsort(-logits[q])[:k]
        token_list = [
            (model.tokenizer.decode([int(i)]), float(logits[q, int(i)]))
            for i in top_idx
        ]
        query_tok = model.tokenizer.decode([int(input_ids[0, q])])
        out.append(PositionWrite(
            position=q, query_token=query_tok, top_tokens=token_list,
        ))
    return out


def ov_circuit(
    model, layer: int, head: int,
    k_tokens: int = 10, n_components: int = 5,
    embed: Optional[np.ndarray] = None,
) -> CircuitAnalysis:
    """Top singular components of M_OV = W_O[h-slice] @ W_V[kv_group(h)].

    For the i-th component (u_i, sigma_i, v_i):
      - v_i is a d_model INPUT direction: tokens near v_i at the attended
        position trigger this component's contribution to the output.
      - u_i is a d_model OUTPUT direction: tokens near u_i are what gets
        ADDED to the residual at the query position when this component
        fires.
      - sigma_i is the gain of this (input -> output) pairing.
    """
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _embed_matrix_f32(model)
    # M_OV = W_O @ W_V. Shape [d_model, head_dim] @ [head_dim, d_model].
    A = spec.W_O     # [d_model, head_dim]
    B = spec.W_V     # [head_dim, d_model]
    U, S, Vt = _truncated_svd_of_product(A, B, n_components)
    components = tuple(
        CircuitComponent(
            rank=i, strength=float(S[i]),
            left_tokens=_top_tokens_for_direction(model, U[:, i], k_tokens, embed),
            right_tokens=_top_tokens_for_direction(model, Vt[i, :], k_tokens, embed),
        )
        for i in range(len(S))
    )
    return CircuitAnalysis(
        circuit_type="OV", layer=layer, head=head,
        kv_group=spec.kv_group, components=components,
    )
