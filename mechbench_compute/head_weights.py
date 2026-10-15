from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .arrays import read_f32


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
    kv_layer: int | None = None
    b_Q: np.ndarray | None = None
    b_K: np.ndarray | None = None
    b_V: np.ndarray | None = None


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


def read_dense_weight(module) -> np.ndarray:
    w = module.weight
    if hasattr(module, "scales"):
        from ._mlx import mx

        w = mx.dequantize(
            w, module.scales, module.biases,
            group_size=module.group_size, bits=module.bits,
        )
    return read_f32(w)


def read_bias(module) -> np.ndarray | None:
    b = getattr(module, "bias", None)
    return None if b is None else read_f32(b)


def find_kv_layer(layers: Sequence[Any], layer: int) -> int:
    attn = layers[layer].self_attn
    if not getattr(attn, "is_kv_shared_layer", False):
        return layer
    for i in range(layer - 1, -1, -1):
        source = layers[i].self_attn
        if (not getattr(source, "is_kv_shared_layer", False)
                and source.layer_type == attn.layer_type):
            return i
    raise ValueError(
        f"layer {layer} shares its keys and values, and no earlier {attn.layer_type} "
        f"layer computes them")


def slice_head(layer: int, head: int, *, n_heads: int, q_proj, k_proj, v_proj, o_proj,
               is_global: bool, is_kv_shared: bool, kv_layer: int) -> HeadSpec:
    W_Q_full = read_dense_weight(q_proj)
    W_K_full = read_dense_weight(k_proj)
    head_dim, rest = divmod(W_Q_full.shape[0], n_heads)
    n_kv_heads, kv_rest = divmod(W_K_full.shape[0], head_dim or 1)
    if rest or kv_rest or not n_kv_heads or n_heads % n_kv_heads:
        raise ValueError(
            f"layer {layer}: q_proj has {W_Q_full.shape[0]} rows and k_proj "
            f"{W_K_full.shape[0]}, which do not split into {n_heads} query heads "
            f"sharing whole key/value heads")
    use_k_eq_v = v_proj is None
    W_V_full = W_K_full if use_k_eq_v else read_dense_weight(v_proj)
    W_O_full = read_dense_weight(o_proj)
    kv_group = head * n_kv_heads // n_heads
    q_rows = slice(head * head_dim, (head + 1) * head_dim)
    kv_rows = slice(kv_group * head_dim, (kv_group + 1) * head_dim)
    b_Q, b_K = read_bias(q_proj), read_bias(k_proj)
    b_V = b_K if use_k_eq_v else read_bias(v_proj)
    return HeadSpec(
        layer=layer, head=head, kv_group=kv_group,
        head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        W_Q=W_Q_full[q_rows, :],
        W_K=W_K_full[kv_rows, :],
        W_V=W_V_full[kv_rows, :],
        W_O=W_O_full[:, q_rows],
        is_global=is_global,
        is_kv_shared=is_kv_shared,
        use_k_eq_v=use_k_eq_v,
        kv_layer=kv_layer,
        b_Q=None if b_Q is None else b_Q[q_rows],
        b_K=None if b_K is None else b_K[kv_rows],
        b_V=None if b_V is None else b_V[kv_rows],
    )


def make_head_weights(
    read_layers: Callable[[Any], Sequence[Any]],
    read_n_heads: Callable[[Any], int],
    is_global: Callable[[Any], bool],
) -> Callable[[Any, int, int], HeadSpec]:
    def head_weights(model, layer: int, head: int) -> HeadSpec:
        layers = read_layers(model)
        attn = layers[layer].self_attn
        kv_layer = find_kv_layer(layers, layer)
        kv = layers[kv_layer].self_attn
        return slice_head(
            layer, head, n_heads=int(read_n_heads(attn)),
            q_proj=attn.q_proj, k_proj=kv.k_proj, v_proj=getattr(kv, "v_proj", None),
            o_proj=attn.o_proj, is_global=bool(is_global(layers[layer])),
            is_kv_shared=kv_layer != layer, kv_layer=kv_layer)

    return head_weights


def read_embed_matrix(model) -> np.ndarray:
    return read_dense_weight(model.lm.model.embed_tokens)


def read_unembed_matrix(model) -> np.ndarray:
    head = getattr(model.lm, "lm_head", None)
    return read_embed_matrix(model) if head is None else read_dense_weight(head)


def _unit_normalized_embed(model) -> np.ndarray:
    e = read_embed_matrix(model)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.clip(norms, 1e-12, None)


def get_head_spec(model, layer: int, head: int) -> HeadSpec:
    return model.architecture.head_weights(model._model, layer, head)


def _top_tokens_for_direction(
    model, direction: np.ndarray, k: int = 10,
    embed: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    if embed is None:
        embed = read_embed_matrix(model)
    logits = embed @ direction
    top_idx = np.argsort(-logits)[:k]
    return [
        (model.tokenizer.decode([int(i)]), float(logits[int(i)]))
        for i in top_idx
    ]


def head_read_tokens(
    model, layer: int, head: int, k: int = 10,
    embed: np.ndarray | None = None,
    normalize: bool = True,
) -> list[tuple[str, float]]:
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _unit_normalized_embed(model) if normalize else read_embed_matrix(model)
    projected = embed @ spec.W_Q.T
    scores = (projected ** 2).sum(axis=1)
    top_idx = np.argsort(-scores)[:k]
    return [
        (model.tokenizer.decode([int(i)]), float(scores[int(i)]))
        for i in top_idx
    ]


def head_key_tokens(
    model, layer: int, head: int, k: int = 10,
    embed: np.ndarray | None = None,
    normalize: bool = True,
) -> list[tuple[str, float]]:
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = _unit_normalized_embed(model) if normalize else read_embed_matrix(model)
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
    embed: np.ndarray | None = None,
) -> CircuitAnalysis:
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = read_embed_matrix(model)
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
    model, input_ids: Any, layer: int, head: int,
    *, k: int = 10, unembed: np.ndarray | None = None,
) -> list[PositionWrite]:
    from .interventions import Capture

    spec = get_head_spec(model, layer, head)
    if unembed is None:
        unembed = read_unembed_matrix(model)

    result = model.run(input_ids, interventions=[Capture.values(layer)])
    V = read_f32(result.cache[f"blocks.{layer}.attn.v"])
    V_group = V[0, spec.kv_group]

    writes = V_group @ spec.W_O.T
    logits = writes @ unembed.T

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
    model, input_ids: Any, layer: int, head: int,
    *, k: int = 10, unembed: np.ndarray | None = None,
) -> list[PositionWrite]:
    from .interventions import Capture

    spec = get_head_spec(model, layer, head)
    if unembed is None:
        unembed = read_unembed_matrix(model)

    result = model.run(input_ids, interventions=[Capture.per_head_out(layer)])
    PH = read_f32(result.cache[f"blocks.{layer}.attn.per_head_out"])
    PH_h = PH[0, head]

    writes = PH_h @ spec.W_O.T
    logits = writes @ unembed.T

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
    embed: np.ndarray | None = None,
    unembed: np.ndarray | None = None,
) -> CircuitAnalysis:
    spec = get_head_spec(model, layer, head)
    if embed is None:
        embed = read_embed_matrix(model)
    if unembed is None:
        unembed = read_unembed_matrix(model)
    A = spec.W_O
    B = spec.W_V
    U, S, Vt = _truncated_svd_of_product(A, B, n_components)
    components = tuple(
        CircuitComponent(
            rank=i, strength=float(S[i]),
            left_tokens=_top_tokens_for_direction(model, U[:, i], k_tokens, unembed),
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


def composition_score(reader: np.ndarray, spec_out: HeadSpec) -> float:
    inner = reader @ spec_out.W_O
    numerator = _fro(inner @ spec_out.W_V)
    denominator = _fro(reader) * _fro_of_product(spec_out.W_O, spec_out.W_V)
    return float(numerator / denominator) if denominator > 0 else 0.0


def head_composition(
    model, layer: int, head: int, *,
    source_layers: Sequence[int] | None = None,
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
