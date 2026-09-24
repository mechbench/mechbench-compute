from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional

import mlx.core as mx
import numpy as np

from .interventions import Capture


def _find_subject_position(token_labels: list[str], subject: str) -> int:
    s = subject.lower()
    for i in range(len(token_labels) - 1, -1, -1):
        if s in token_labels[i].lower():
            return i
    for i in range(len(token_labels) - 1, -1, -1):
        t = token_labels[i].strip().lower()
        if len(t) >= 3 and t in s:
            return i
    raise ValueError(
        f"Subject substring {subject!r} not found in any token. "
        f"Tokens were: {token_labels}"
    )


def _resolve_position(
    model, validated_prompt, position: str
) -> int:
    if position == "final":
        return int(validated_prompt.input_ids.shape[1]) - 1
    if position == "subject":
        subj = validated_prompt.prompt.subject
        if subj is None:
            raise ValueError(
                f"position='subject' but Prompt.subject is None for: "
                f"{validated_prompt.prompt.text!r}"
            )
        token_labels = [
            model.tokenizer.decode([int(t)])
            for t in validated_prompt.input_ids[0]
        ]
        return _find_subject_position(token_labels, subj)
    raise ValueError(
        f"position must be 'subject' or 'final', got {position!r}"
    )


def fact_vectors(
    model,
    validated,
    layer: int,
    *,
    position: str = "subject",
) -> np.ndarray:
    n = len(validated)
    out = np.zeros((n, model.arch.d_model), dtype=np.float32)
    cap = Capture.residual(layer, point="post")
    key = f"blocks.{layer}.resid_post"
    for j, vp in enumerate(validated):
        result = model.run(vp.input_ids, interventions=[cap])
        pos = _resolve_position(model, vp, position)
        v = result.cache[key][0, pos, :].astype(mx.float32)
        mx.eval(v)
        out[j] = np.array(v)
    return out


def fact_vectors_at(
    model,
    validated,
    layers: Iterable[int],
    *,
    position: str = "subject",
    interventions: Iterable = (),
) -> dict[int, np.ndarray]:
    layers_list = list(layers)
    n = len(validated)
    out = {L: np.zeros((n, model.arch.d_model), dtype=np.float32) for L in layers_list}
    cap = Capture.residual(layers_list, point="post")
    extra = list(interventions)
    for j, vp in enumerate(validated):
        result = model.run(vp.input_ids, interventions=[cap, *extra])
        pos = _resolve_position(model, vp, position)
        for L in layers_list:
            v = result.cache[f"blocks.{L}.resid_post"][0, pos, :].astype(mx.float32)
            mx.eval(v)
            out[L][j] = np.array(v)
    return out


def fact_vectors_at_hook(
    model,
    validated,
    hook_point: str,
    *,
    head: Optional[int] = None,
    position: str = "subject",
    interventions: Iterable = (),
) -> np.ndarray:
    from .interventions import _Captures

    n = len(validated)
    cap = _Captures(names=(hook_point,))
    extra = list(interventions)
    out: Optional[np.ndarray] = None
    for j, vp in enumerate(validated):
        result = model.run(vp.input_ids, interventions=[cap, *extra])
        tensor = result.cache[hook_point].astype(mx.float32)
        pos = _resolve_position(model, vp, position)
        shape = tensor.shape
        if len(shape) == 3:
            v = tensor[0, pos, :]
        elif len(shape) == 4:
            if head is None:
                raise ValueError(
                    f"hook_point {hook_point!r} produces per-head tensor "
                    f"(shape {shape}); must supply `head=` kwarg."
                )
            v = tensor[0, head, pos, :]
        else:
            raise ValueError(
                f"Unexpected tensor shape {shape} for hook_point {hook_point!r}."
            )
        mx.eval(v)
        v_np = np.array(v)
        if out is None:
            out = np.zeros((n, v_np.shape[0]), dtype=np.float32)
        out[j] = v_np
    assert out is not None, "Empty ValidatedPromptSet"
    return out


def fact_vectors_pooled(
    model,
    validated,
    layers: Iterable[int],
    *,
    start: int = 0,
    end: int | None = None,
    interventions: Iterable = (),
) -> dict[int, np.ndarray]:
    layers_list = list(layers)
    n = len(validated)
    out = {L: np.zeros((n, model.arch.d_model), dtype=np.float32) for L in layers_list}
    cap = Capture.residual(layers_list, point="post")
    extra = list(interventions)
    for j, vp in enumerate(validated):
        result = model.run(vp.input_ids, interventions=[cap, *extra])
        seq_len = int(vp.input_ids.shape[1])
        s = start if start >= 0 else max(0, seq_len + start)
        e = seq_len if end is None else min(end, seq_len)
        if s >= e:
            s, e = seq_len - 1, seq_len
        for L in layers_list:
            resid = result.cache[f"blocks.{L}.resid_post"][0, s:e, :].astype(mx.float32)
            pooled = resid.mean(axis=0)
            mx.eval(pooled)
            out[L][j] = np.array(pooled)
    return out


def centroid_decode(
    model,
    vectors: np.ndarray,
    *,
    k: int = 10,
    mean_subtract: bool = True,
    overall_mean: Optional[np.ndarray] = None,
) -> list[tuple[str, float]]:
    centroid = vectors.mean(axis=0)
    if mean_subtract:
        if overall_mean is None:
            overall_mean = centroid
        centroid = centroid - overall_mean

    v = mx.array(centroid[None, None, :], dtype=mx.bfloat16)
    logits = model.project_to_logits(v)
    last = logits[0, 0, :].astype(mx.float32)
    probs = mx.softmax(last)
    mx.eval(probs)
    probs_np = np.array(probs)
    top_idx = np.argsort(-probs_np)[:k]
    return [
        (model.tokenizer.decode([int(i)]), float(probs_np[int(i)]))
        for i in top_idx
    ]


def orthogonalize_against(
    vectors: np.ndarray,
    baseline: np.ndarray,
    *,
    explain: float = 0.5,
) -> np.ndarray:
    if not 0.0 < explain <= 1.0:
        raise ValueError(f"explain must be in (0, 1], got {explain}")
    baseline = np.asarray(baseline, dtype=np.float64)
    baseline_centered = baseline - baseline.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(baseline_centered, full_matrices=False)
    var = S ** 2
    if var.sum() <= 0:
        return vectors.astype(np.float32, copy=True)
    cum = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum, explain) + 1)
    k = min(k, Vt.shape[0])
    P = Vt[:k]
    v = np.asarray(vectors, dtype=np.float64)
    projection = v @ P.T @ P
    return (v - projection).astype(np.float32)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-12, None)
    return normed @ normed.T


def intra_inter_separation(
    vectors: np.ndarray, labels: np.ndarray | list
) -> tuple[float, float, float]:
    labels = np.asarray(labels)
    sim = cosine_matrix(vectors)
    intra, inter = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels[i] == labels[j] else inter).append(sim[i, j])
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    return intra_mean, inter_mean, intra_mean - inter_mean


def cluster_purity(true_labels, pred_labels) -> float:
    n = len(true_labels)
    clusters = set(pred_labels)
    correct = 0
    for c in clusters:
        in_cluster = [t for t, p in zip(true_labels, pred_labels) if p == c]
        if in_cluster:
            most_common = max(in_cluster.count(label) for label in set(in_cluster))
            correct += most_common
    return correct / n if n else 0.0


def silhouette_cosine(vectors: np.ndarray, labels) -> float:
    from sklearn.metrics import silhouette_score
    labels_arr = np.asarray(labels)
    unique = list(dict.fromkeys(labels_arr.tolist()))
    int_labels = np.array([unique.index(l) for l in labels_arr])
    return float(silhouette_score(vectors, int_labels, metric="cosine"))


def nearest_neighbor_purity(
    vectors: np.ndarray, labels
) -> tuple[float, np.ndarray]:
    labels_arr = np.asarray(labels)
    sim = cosine_matrix(vectors)
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argmax(sim, axis=1)
    hits = labels_arr[nn_idx] == labels_arr
    return float(hits.mean()), hits


def iterate_clusters(
    vectors: np.ndarray,
    labels,
) -> Iterator[tuple[Any, np.ndarray, np.ndarray]]:
    labels_arr = np.asarray(labels)
    for label in dict.fromkeys(labels_arr.tolist()):
        mask = labels_arr == label
        yield label, vectors[mask], mask


def cohesion(vectors: np.ndarray) -> float:
    if len(vectors) == 0:
        return 0.0
    centroid = vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-12:
        return 0.0
    centroid_unit = centroid / centroid_norm
    member_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    member_units = vectors / np.clip(member_norms, 1e-12, None)
    sims = member_units @ centroid_unit
    return float(sims.mean())


@dataclass(frozen=True)
class VocabConcentration:
    top1: float
    top_k_mass: float
    k: int
    entropy_bits: float
    effective_vocab_size: float


def top_k_mass(probs: np.ndarray, k: int = 10) -> float:
    if probs.size == 0 or k <= 0:
        return 0.0
    k = min(k, probs.size)
    return float(np.partition(probs, -k)[-k:].sum())


def entropy_bits(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def effective_vocab_size(probs: np.ndarray) -> float:
    nats = entropy_bits(probs) * np.log(2)
    return float(np.exp(nats))


def vocab_concentration(probs: np.ndarray, k: int = 10) -> VocabConcentration:
    if probs.size == 0:
        return VocabConcentration(
            top1=0.0, top_k_mass=0.0, k=k,
            entropy_bits=0.0, effective_vocab_size=0.0,
        )
    top1 = float(probs.max())
    tk = top_k_mass(probs, k=k)
    h = entropy_bits(probs)
    ev = float(np.exp(h * np.log(2)))
    return VocabConcentration(
        top1=top1, top_k_mass=tk, k=k,
        entropy_bits=h, effective_vocab_size=ev,
    )
