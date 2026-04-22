"""Fact-vector extraction + geometric analyses.

Three layers of functionality:

1. Extraction: fact_vectors / fact_vectors_at run prompts through the model
   and pull the residual stream at chosen positions and layers.

2. Decoding: centroid_decode projects the (mean-subtracted) centroid of a
   group of vectors through the model's tied unembed to read off the
   relational concept that group represents (the technique from finding 11/12).

3. Pure-numpy stats: cosine_matrix, cluster_purity, silhouette_cosine,
   nearest_neighbor_purity, intra_inter_separation. No model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional

import mlx.core as mx
import numpy as np

from ._arch import D_MODEL
from .interventions import Capture


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _find_subject_position(token_labels: list[str], subject: str) -> int:
    """Last token whose decoded label contains the subject substring,
    or whose stripped form is itself contained in the subject (handles cases
    like German genitive 'Frankreichs' which tokenizes as ' Frankreich' + 's').
    """
    s = subject.lower()
    # Pass 1: exact substring match (subject is in token).
    for i in range(len(token_labels) - 1, -1, -1):
        if s in token_labels[i].lower():
            return i
    # Pass 2: reverse — non-trivial token stripped is contained in subject.
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
    """Resolve a position spec to a concrete index for one prompt."""
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
    """Extract residual-stream vectors at a chosen position for each prompt.

    Args:
        model: Model instance.
        validated: ValidatedPromptSet.
        layer: Layer index whose resid_post to extract.
        position: 'subject' (use Prompt.subject substring) or 'final'.

    Returns:
        np.ndarray of shape [n_prompts, D_MODEL], dtype float32.
    """
    n = len(validated)
    out = np.zeros((n, D_MODEL), dtype=np.float32)
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
    """Multi-layer fact-vector extraction in one model pass per prompt.

    Strictly more efficient than calling fact_vectors() once per layer: each
    prompt is run through the model exactly once, with all requested layers
    captured at once. Use when you need vectors at several depths.

    Args:
        model: Model instance.
        validated: ValidatedPromptSet.
        layers: Iterable of layer indices.
        position: 'subject' or 'final'.
        interventions: Optional extra interventions (e.g. an Ablate.layer or
            Ablate.side_channel) to run in the same forward pass as the
            residual capture. Useful for "what do the residuals look like
            with this layer disabled?" experiments. The capture happens AFTER
            any user intervention at the same hook point, so the recorded
            residuals reflect the post-intervention state.

    Returns:
        dict {layer: np.ndarray[n_prompts, D_MODEL]} in float32.
    """
    layers_list = list(layers)
    n = len(validated)
    out = {L: np.zeros((n, D_MODEL), dtype=np.float32) for L in layers_list}
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
    """Generic-hook-point analogue of fact_vectors_at.

    Captures the activation at `hook_point` and returns one vector per
    prompt at the chosen position. Works for:

      - residual-stream hooks ('blocks.N.resid_pre', 'blocks.N.resid_post'):
        head must be None; returned vectors are d_model long.
      - per-head hooks ('blocks.N.attn.q', '.per_head_out'):
        head must be a Q-head index 0..7; returned vectors are head_dim long.
      - per-KV-head hooks ('blocks.N.attn.k', 'blocks.N.attn.v'):
        head must be a KV-head index 0..1; returned vectors are head_dim long.
      - MLP / gate hooks ('blocks.N.mlp_out', 'blocks.N.gate_out'):
        head must be None; d_model long.

    Args:
        model: Model instance.
        validated: ValidatedPromptSet.
        hook_point: the fully-qualified hook name to capture.
        head: index along the per-head axis of the captured tensor (required
            for per-head hooks; ignored for residual/MLP/gate hooks).
        position: 'subject' or 'final'.
        interventions: optional additional interventions to compose.

    Returns:
        np.ndarray of shape [n_prompts, d] where d depends on the hook.
    """
    from .interventions import _Captures

    n = len(validated)
    cap = _Captures(names=(hook_point,))
    extra = list(interventions)
    out: Optional[np.ndarray] = None
    for j, vp in enumerate(validated):
        result = model.run(vp.input_ids, interventions=[cap, *extra])
        tensor = result.cache[hook_point].astype(mx.float32)
        pos = _resolve_position(model, vp, position)
        # Decide how to slice based on tensor shape:
        #   [1, L, d]                -> residual / MLP-style. vec = tensor[0, pos, :]
        #   [1, n_heads, L, head_dim] -> per-head. head required.
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
    """Residual-stream mean pooling over a range of token positions per prompt.

    Produces one vector per prompt per layer, by averaging resid_post over
    positions [start, end) for each prompt. Use when the concept you're
    probing for is distributed across a passage rather than localized at a
    single subject token — e.g. emotion content in a short story, register
    in a paragraph, style in an essay. This is the extraction pattern used
    by Anthropic's 'Emotion Concepts' work (transformer-circuits.pub, 2026),
    which averages over positions >= 50 within each story.

    Args:
        model: Model instance.
        validated: ValidatedPromptSet.
        layers: Iterable of layer indices.
        start: First token position to include. Negative is supported and
            interpreted relative to sequence length. If a prompt is shorter
            than `start`, we fall back to the final token position (so every
            prompt still contributes one vector rather than crashing).
        end: One past the last token position to include, or None for the
            end of each prompt. Clipped to each prompt's length.
        interventions: Optional extra interventions, same as fact_vectors_at.

    Returns:
        dict {layer: np.ndarray[n_prompts, D_MODEL]} in float32, where each
        row is the mean of resid_post over the requested position range for
        one prompt.
    """
    layers_list = list(layers)
    n = len(validated)
    out = {L: np.zeros((n, D_MODEL), dtype=np.float32) for L in layers_list}
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


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def centroid_decode(
    model,
    vectors: np.ndarray,
    *,
    k: int = 10,
    mean_subtract: bool = True,
    overall_mean: Optional[np.ndarray] = None,
) -> list[tuple[str, float]]:
    """Project the (mean-subtracted) centroid of `vectors` through the tied
    unembed, return top-k decoded tokens.

    The technique from finding 11/12: individual fact vectors at subject
    positions are vocab-opaque, but their averaged centroid (after removing
    the prompt-template common-mode via mean-subtraction) decodes to a
    multilingual cluster of tokens that names the relational frame.

    Args:
        model: Model instance.
        vectors: np.ndarray[n, D_MODEL] in float32. The vectors to centroid.
        k: How many top tokens to return.
        mean_subtract: If True, subtract `overall_mean` (or vectors.mean if
            overall_mean is None) from the centroid before projecting.
        overall_mean: Optional baseline mean to subtract. Pass the
            corpus-wide mean (over all categories) when decoding individual
            categories — using only this category's mean would zero out the
            very signal you're trying to decode. If None, defaults to
            vectors.mean(), which is correct only if `vectors` is the whole
            corpus.

    Returns:
        list of (decoded_token_string, probability) sorted by descending p.
    """
    centroid = vectors.mean(axis=0)
    if mean_subtract:
        if overall_mean is None:
            overall_mean = centroid  # degenerate, but documented above
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


# ---------------------------------------------------------------------------
# Vector transformations
# ---------------------------------------------------------------------------


def orthogonalize_against(
    vectors: np.ndarray,
    baseline: np.ndarray,
    *,
    explain: float = 0.5,
) -> np.ndarray:
    """Project out the top-variance directions of `baseline` from `vectors`.

    Fits PCA on `baseline`, keeps enough principal components to explain
    at least `explain` fraction of baseline's variance, and returns
    `vectors` with those directions subtracted out.

    Why: when building concept vectors (emotion probes, sentiment
    directions, register vectors, etc.) the raw centroids are contaminated
    by directions that are high-variance across ANY text, not specific to
    the concept of interest — sentence-start signals, punctuation,
    position-in-sequence artifacts. Computing PCs on an emotionally- or
    semantically-neutral baseline corpus and subtracting those directions
    yields a cleaner concept vector. This is the denoising step from
    Anthropic's 'Emotion Concepts' paper (transformer-circuits.pub, 2026).

    Mean-subtraction is the degenerate special case of this with a
    one-component baseline (just the grand mean direction).

    Args:
        vectors: np.ndarray of shape [n, d]. The concept vectors to clean.
        baseline: np.ndarray of shape [m, d]. The baseline corpus activations.
            Should be on comparable scale/layer to `vectors`.
        explain: float in (0, 1]. Keep PCs until cumulative explained
            variance reaches this fraction. Default 0.5 matches the paper.

    Returns:
        np.ndarray of same shape as `vectors`, with the baseline's
        top-variance subspace projected out.
    """
    if not 0.0 < explain <= 1.0:
        raise ValueError(f"explain must be in (0, 1], got {explain}")
    baseline = np.asarray(baseline, dtype=np.float64)
    baseline_centered = baseline - baseline.mean(axis=0, keepdims=True)
    # SVD of centered baseline: rows are observations, columns are features.
    # Singular vectors Vt[k] are the principal axes in feature space.
    _, S, Vt = np.linalg.svd(baseline_centered, full_matrices=False)
    var = S ** 2
    if var.sum() <= 0:
        return vectors.astype(np.float32, copy=True)
    cum = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum, explain) + 1)
    k = min(k, Vt.shape[0])
    P = Vt[:k]  # [k, d]
    v = np.asarray(vectors, dtype=np.float64)
    projection = v @ P.T @ P  # [n, d] — component of v in the baseline subspace
    return (v - projection).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure-numpy stats
# ---------------------------------------------------------------------------


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Returns [N, N]."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-12, None)
    return normed @ normed.T


def intra_inter_separation(
    vectors: np.ndarray, labels: np.ndarray | list
) -> tuple[float, float, float]:
    """Mean intra-category cosine, mean inter-category cosine, and their
    difference (the 'separation' score). Higher separation = more distinct
    categories.

    Returns (intra_mean, inter_mean, separation).
    """
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
    """Maximum achievable accuracy after relabeling clusters to match truth.

    For each predicted cluster, we 'relabel' it to whichever true label is
    most common in it; we then count how many points are correctly classified.
    Pure clusters (every member shares a true label) score 1.0; uniformly-
    mixed clusters score chance.
    """
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
    """Silhouette score (cosine metric) of the labeling. Range [-1, 1];
    higher = better-separated clusters. Wraps sklearn for the heavy lifting."""
    from sklearn.metrics import silhouette_score
    labels_arr = np.asarray(labels)
    # silhouette_score needs integer labels
    unique = list(dict.fromkeys(labels_arr.tolist()))
    int_labels = np.array([unique.index(l) for l in labels_arr])
    return float(silhouette_score(vectors, int_labels, metric="cosine"))


def nearest_neighbor_purity(
    vectors: np.ndarray, labels
) -> tuple[float, np.ndarray]:
    """For each vector, find its cosine-nearest neighbor (excluding self) and
    check whether that neighbor shares its label. Returns (hit_rate, hits)
    where hits[i] is True iff vectors[i]'s nearest neighbor matches.

    This is the cleanest 'is the geometry semantically organized' test —
    chance for a balanced k-class set is roughly 1/k.
    """
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
    """Yield (label, cluster_vectors, mask) for each unique label in `labels`,
    in first-seen order.

    Replaces the boilerplate:

        for label in dict.fromkeys(labels.tolist()):
            mask = labels == label
            cluster_vecs = vectors[mask]
            ...

    The mask is yielded alongside cluster_vectors because callers commonly
    need it for indexing into a parallel array (e.g., a PCA projection of
    the same vectors, where you want proj[mask] rather than vectors[mask]).
    """
    labels_arr = np.asarray(labels)
    for label in dict.fromkeys(labels_arr.tolist()):
        mask = labels_arr == label
        yield label, vectors[mask], mask


def cohesion(vectors: np.ndarray) -> float:
    """Mean cosine similarity between each vector and the cluster centroid.

    Measures how tightly a group of vectors is clustered around its own mean.
    Range [-1, 1]; higher = tighter cluster.

    Distinct from the mean intra-cluster pairwise cosine that you'd compute
    via cosine_matrix + masking: cohesion is member-to-centroid (n similarities)
    rather than member-to-member (n*(n-1)/2). Cohesion is generally >= the
    intra-cluster pairwise mean since the centroid is the maximum-likelihood
    'anchor' that minimizes the angular spread.
    """
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


# ---------------------------------------------------------------------------
# Vocabulary-space concentration
#
# Given a residual vector projected through the tied unembed and softmaxed,
# how concentrated is the resulting distribution? "Concentrated" can mean
# several things; these primitives give the standard framings, and
# vocab_concentration returns them all in a single structured result.
#
# The primitives work on ANY probability distribution, not just vocab ones —
# the "vocab" framing is about the typical use case (project a residual,
# softmax, ask how peaked the result is) but the math is generic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VocabConcentration:
    """Concentration metrics for a probability distribution (typically a
    vocabulary-space distribution produced by projecting a residual vector
    through the tied unembed and softmaxing).

    Attributes:
        top1: probability of the single most-likely token.
        top_k_mass: sum of the top-k probabilities. Use for robustness to
            ties and to capture how 'peaked' the distribution is at the high end.
        k: the k that top_k_mass was computed at (so the result is self-describing).
        entropy_bits: Shannon entropy in bits. Lower = more concentrated.
            Range [0, log2(vocab_size)]; for Gemma 4 E4B's 262,144-token vocab,
            uniform distribution = log2(262144) = 18.0 bits.
        effective_vocab_size: exp(entropy_in_nats). Roughly 'how many tokens
            worth of probability mass is the distribution effectively spread
            over'. A delta gives 1; a uniform distribution gives the vocab size.
    """

    top1: float
    top_k_mass: float
    k: int
    entropy_bits: float
    effective_vocab_size: float


def top_k_mass(probs: np.ndarray, k: int = 10) -> float:
    """Sum of the top-k probabilities in a discrete distribution."""
    if probs.size == 0 or k <= 0:
        return 0.0
    k = min(k, probs.size)
    # np.partition is O(n); avoids a full sort
    return float(np.partition(probs, -k)[-k:].sum())


def entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution, in bits.

    Lower = more concentrated. A delta distribution has entropy 0;
    a uniform distribution over N items has entropy log2(N).
    """
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def effective_vocab_size(probs: np.ndarray) -> float:
    """exp(entropy_in_nats). The 'how many tokens worth' the distribution
    is effectively spread over.

    A delta gives 1.0; a uniform over N items gives N. Equivalent to
    perplexity of a model whose distribution this is.
    """
    nats = entropy_bits(probs) * np.log(2)
    return float(np.exp(nats))


def vocab_concentration(probs: np.ndarray, k: int = 10) -> VocabConcentration:
    """Compute the standard concentration metrics for a probability
    distribution in one pass, returning a structured VocabConcentration.

    Use when you want all four metrics together (the common case for the
    "how concentrated is this decoded distribution" question). Use the
    individual primitives (top_k_mass / entropy_bits / effective_vocab_size)
    when you want just one.
    """
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
