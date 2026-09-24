from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from .interventions import Capture


@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int
    kv_group: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    W_Q: np.ndarray
    W_K: np.ndarray
    W_V: np.ndarray
    W_O: np.ndarray
    is_global: bool
    is_kv_shared: bool
    use_k_eq_v: bool = False


@dataclass(frozen=True)
class CircuitComponent:
    rank: int
    strength: float
    left_tokens: list[tuple[str, float]]
    right_tokens: list[tuple[str, float]]


@dataclass(frozen=True)
class CircuitAnalysis:
    circuit_type: str
    layer: int
    head: int
    kv_group: int
    components: tuple[CircuitComponent, ...]


def _as_np_f32(mx_tensor: mx.array) -> np.ndarray:
    return np.array(mx_tensor.astype(mx.float32))


def _dense_weight_f32(module) -> np.ndarray:
    w = module.weight
    if hasattr(module, "scales"):
        w = mx.dequantize(
            w, module.scales, module.biases,
            group_size=module.group_size, bits=module.bits,
        )
    return _as_np_f32(w)


def _embed_matrix_f32(model) -> np.ndarray:
    return _dense_weight_f32(model._model.language_model.model.embed_tokens)


def _unit_normalized_embed(model) -> np.ndarray:
    e = _embed_matrix_f32(model)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.clip(norms, 1e-12, None)


def get_head_spec(model, layer: int, head: int) -> HeadSpec:
    block = model._model.language_model.model.layers[layer]
    attn = block.self_attn
    head_dim = int(attn.head_dim)
    n_heads = int(attn.n_heads)
    n_kv_heads = int(attn.n_kv_heads)
    kv_group = head * n_kv_heads // n_heads

    use_k_eq_v = bool(getattr(attn, "use_k_eq_v", False))
    W_Q_full = _dense_weight_f32(attn.q_proj)
    W_K_full = _dense_weight_f32(attn.k_proj)
    W_V_full = W_K_full if use_k_eq_v else _dense_weight_f32(attn.v_proj)
    W_O_full = _dense_weight_f32(attn.o_proj)

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
        use_k_eq_v=use_k_eq_v,
    )


def _top_tokens_for_direction(
    model, direction: np.ndarray, k: int = 10,
    embed: Optional[np.ndarray] = None,
) -> list[tuple[str, float]]:
    if embed is None:
        embed = _embed_matrix_f32(model)
    logits = embed @ direction
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
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _unit_normalized_embed(model) if normalize else _embed_matrix_f32(model)
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


def _truncated_svd_of_product(
    A: np.ndarray, B: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q_A, R_A = np.linalg.qr(A)
    RB = R_A @ B
    U_small, S, Vt = np.linalg.svd(RB, full_matrices=False)
    U = Q_A @ U_small
    k = min(k, len(S))
    return U[:, :k], S[:k], Vt[:k, :]


def qk_circuit(
    model, layer: int, head: int,
    k_tokens: int = 10, n_components: int = 5,
    embed: Optional[np.ndarray] = None,
) -> CircuitAnalysis:
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _embed_matrix_f32(model)
    A = spec.W_Q.T
    B = spec.W_K
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


@dataclass(frozen=True)
class PositionWrite:
    position: int
    query_token: str
    top_tokens: list[tuple[str, float]]


def head_ov_position_writes(
    model, input_ids: mx.array, layer: int, head: int,
    *, k: int = 10, embed: Optional[np.ndarray] = None,
) -> list[PositionWrite]:
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _embed_matrix_f32(model)

    result = model.run(input_ids, interventions=[Capture.values(layer)])
    V = np.array(result.cache[f"blocks.{layer}.attn.v"].astype(mx.float32))
    V_group = V[0, spec.kv_group]

    writes = V_group @ spec.W_O.T
    logits = writes @ embed.T

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
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _embed_matrix_f32(model)

    result = model.run(input_ids, interventions=[Capture.per_head_out(layer)])
    PH = np.array(result.cache[f"blocks.{layer}.attn.per_head_out"].astype(mx.float32))
    PH_h = PH[0, head]

    writes = PH_h @ spec.W_O.T
    logits = writes @ embed.T

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
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _embed_matrix_f32(model)
    A = spec.W_O
    B = spec.W_V
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


def _fro(a: np.ndarray) -> float:
    return float(np.sqrt(np.sum(a * a)))


def _fro_of_product(a: np.ndarray, b: np.ndarray) -> float:
    left = a.T @ a
    right = b @ b.T
    return float(np.sqrt(max(float(np.sum(left * right)), 0.0)))


def composition_score(reader: np.ndarray, spec_out: "HeadSpec") -> float:
    inner = reader @ spec_out.W_O
    numerator = _fro(inner @ spec_out.W_V)
    denominator = _fro(reader) * _fro_of_product(spec_out.W_O, spec_out.W_V)
    return float(numerator / denominator) if denominator > 0 else 0.0


def head_composition(
    model, layer: int, head: int, *,
    source_layers: Optional[Sequence[int]] = None,
    kinds: Sequence[str] = ("q", "k", "v"),
) -> list[dict]:
    unknown = [k for k in kinds if k not in ("q", "k", "v")]
    if unknown:
        raise ValueError(f"composition kinds are q, k, v — not {', '.join(unknown)}")
    layers = (list(range(layer)) if source_layers is None
              else [int(x) for x in source_layers])
    late = [x for x in layers if x >= layer]
    if late:
        raise ValueError(
            f"a head at layer {layer} can only compose with earlier layers; "
            f"{sorted(late)} are not earlier")
    dest = get_head_spec(model, layer, head)
    readers = {"q": dest.W_Q, "k": dest.W_K, "v": dest.W_V}
    rows: list[dict] = []
    for src_layer in layers:
        for src_head in range(dest.n_heads):
            src = get_head_spec(model, src_layer, src_head)
            for kind in kinds:
                rows.append({
                    "id": f"L{src_layer}H{src_head}->L{layer}H{head}:{kind}",
                    "coords": {"layer": src_layer, "head": src_head, "kind": kind,
                               "into_layer": layer, "into_head": head},
                    "score": round(composition_score(readers[kind], src), 5),
                })
    return rows
