"""Probe: a persistent, reusable concept vector.

A Probe bundles everything needed to apply a concept direction (emotion,
sentiment, register, any difference-of-means vector) to new text at the
right layer. Once constructed, a probe is a first-class object you can
save, share, compose, and apply across any corpus.

The canonical construction is `Probe.from_labeled_corpus`, which takes
per-concept vectors (the "positive" examples for each concept) plus a
neutral baseline corpus, and produces one probe per concept using the
difference-of-means + PC-orthogonalization recipe from Anthropic's
'Emotion Concepts' work (transformer-circuits.pub, 2026).

Example:
    from mechbench_core import Model, Probe, fact_vectors_pooled

    model = Model.load()
    emotion_vecs = fact_vectors_pooled(
        model, emotion_stories, layers=[28], start=20,
    )[28]  # [n_stories, d_model]
    neutral_vecs = fact_vectors_pooled(
        model, neutral_stories, layers=[28], start=20,
    )[28]

    # labeled_vectors: {emotion_name: rows of that emotion}
    by_emotion = {
        e: emotion_vecs[labels == e]
        for e in set(labels)
    }
    probes = Probe.from_labeled_corpus(
        by_emotion, neutral_vecs, layer=28,
    )

    # Score any new residuals:
    score = probes["happy"].score(new_residuals)  # [..., d_model] -> [...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import orthogonalize_against


@dataclass(frozen=True)
class Probe:
    """A concept vector with the metadata needed to apply it.

    Attributes:
        name: Human-readable label for the concept (e.g. 'happy').
        vec: Unit-normalized concept direction. Shape is [d] where d is
            the dimensionality of the stream this probe targets: d_model
            for residual-stream probes, head_dim for per-head Q/K/V
            probes, intermediate_size for MLP-activation probes, etc.
        layer: Layer index. For residual probes this is the layer the
            residual was extracted at. For per-head or per-stream probes,
            it is the layer index part of hook_point.
        baseline_mean: Vector subtracted from activations before
            projection, shape [d], float32. In the standard construction
            this is the grand mean across the positive corpus.
        orthogonalizer: Optional [k, d] array whose rows are orthonormal
            baseline-PC directions; these are projected out of
            activations before scoring, to remove non-concept common-mode
            variance. None if no orthogonalization was applied.
        hook_point: Optional fully-qualified hook-point name, e.g.
            'blocks.23.resid_post', 'blocks.14.attn.q', 'blocks.28.mlp_out'.
            Defaults to f'blocks.{layer}.resid_post' when not specified,
            which is the canonical residual-stream probe form. When set
            to a per-head stream ('.attn.q', '.attn.k', '.attn.v',
            '.attn.per_head_out'), `head` must also be set.
        head: For per-head streams, which head (Q-head for .attn.q and
            .attn.per_head_out; KV-head for .attn.k and .attn.v) this
            probe targets. None for residual-stream probes.
    """

    name: str
    vec: np.ndarray
    layer: int
    baseline_mean: np.ndarray
    orthogonalizer: Optional[np.ndarray] = None
    hook_point: Optional[str] = None
    head: Optional[int] = None

    @property
    def effective_hook_point(self) -> str:
        """The hook point this probe operates at. Falls back to the
        residual-post convention when hook_point is None."""
        return self.hook_point or f"blocks.{self.layer}.resid_post"

    def score(self, residuals: np.ndarray) -> np.ndarray:
        """Project residuals onto this probe and return a scalar per input.

        The full scoring pipeline:
          1. Subtract baseline_mean.
          2. Project out the orthogonalizer subspace (if any).
          3. Dot with the unit-normalized probe vector.

        Args:
            residuals: np.ndarray with last dim equal to d_model. Any
                leading shape is allowed; results are one scalar per leading
                element. A 2D [n_positions, d_model] in is an [n_positions]
                out; a 1D [d_model] in is a 0-D scalar out; a 3D
                [batch, seq, d_model] in is a [batch, seq] out.

        Returns:
            np.ndarray of shape residuals.shape[:-1], float32.
        """
        r = np.asarray(residuals, dtype=np.float32)
        if r.shape[-1] != self.vec.shape[-1]:
            raise ValueError(
                f"residuals last dim {r.shape[-1]} != probe d_model "
                f"{self.vec.shape[-1]}"
            )
        centered = r - self.baseline_mean
        if self.orthogonalizer is not None:
            P = self.orthogonalizer  # [k, d]
            centered = centered - centered @ P.T @ P
        return centered @ self.vec

    @classmethod
    def from_vector(
        cls,
        vec: np.ndarray,
        *,
        name: str,
        layer: int,
        baseline_mean: np.ndarray | None = None,
        orthogonalizer: np.ndarray | None = None,
        normalize: bool = True,
        hook_point: Optional[str] = None,
        head: Optional[int] = None,
    ) -> "Probe":
        """Build a Probe from a raw vector plus optional baseline/orthogonalizer.

        Use when you've computed a concept direction some other way
        (steering-vector arithmetic, trained classifier weight, etc.) and
        want to wrap it as a Probe for uniform downstream handling.
        """
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if normalize:
            n = float(np.linalg.norm(v))
            v = v / max(n, 1e-12)
        d = v.shape[0]
        bm = (
            np.zeros(d, dtype=np.float32)
            if baseline_mean is None
            else np.asarray(baseline_mean, dtype=np.float32).reshape(d)
        )
        return cls(name=name, vec=v, layer=layer,
                   baseline_mean=bm, orthogonalizer=orthogonalizer,
                   hook_point=hook_point, head=head)

    @classmethod
    def from_corpus(
        cls,
        positive: np.ndarray,
        baseline: np.ndarray,
        *,
        name: str,
        layer: int,
        explain: float = 0.5,
        orthogonalize: bool = True,
        hook_point: Optional[str] = None,
        head: Optional[int] = None,
    ) -> "Probe":
        """Difference-of-means probe from a single positive corpus vs a baseline.

        Computes (mean of positive) - (mean of baseline) as the concept
        direction, then (if `orthogonalize`) projects out the top-variance
        PCs of `baseline` explaining `explain` fraction of baseline variance.

        Args:
            positive: [n_pos, d_model] activations from prompts positively
                illustrating the concept.
            baseline: [n_base, d_model] activations from neutral or contrastive
                prompts. Used both as the baseline mean AND as the corpus for
                PC-orthogonalization.
            name: Name of the concept.
            layer: The residual layer these vectors came from.
            explain: Fraction of baseline variance to project out. Default 0.5.
            orthogonalize: If False, skip the PC step (baseline_mean is still
                applied). Useful for pedagogical comparisons.
        """
        pos = np.asarray(positive, dtype=np.float32)
        base = np.asarray(baseline, dtype=np.float32)
        mean_pos = pos.mean(axis=0)
        mean_base = base.mean(axis=0)
        raw_vec = mean_pos - mean_base
        if orthogonalize:
            clean_vec = orthogonalize_against(
                raw_vec[None, :], base, explain=explain
            )[0]
            Pmat = _baseline_pc_matrix(base, explain)
        else:
            clean_vec = raw_vec
            Pmat = None
        return cls.from_vector(
            clean_vec, name=name, layer=layer,
            baseline_mean=mean_base, orthogonalizer=Pmat,
            normalize=True, hook_point=hook_point, head=head,
        )

    @classmethod
    def from_labeled_corpus(
        cls,
        labeled: dict[str, np.ndarray],
        neutral: np.ndarray,
        *,
        layer: int,
        explain: float = 0.5,
        orthogonalize: bool = True,
        hook_point: Optional[str] = None,
        head: Optional[int] = None,
    ) -> dict[str, "Probe"]:
        """Build one probe per concept label from a labeled corpus.

        Follows the Anthropic 'Emotion Concepts' recipe: each concept's
        vector is its positive mean minus the GRAND MEAN across all labeled
        concepts (which captures prompt-template and domain common-mode
        shared by the labeled corpus). The neutral corpus is used ONLY to
        compute PC directions to orthogonalize against.

        Args:
            labeled: {concept_name: [n_k, d_model]} positive activations
                per concept.
            neutral: [n_neutral, d_model] activations from an emotionally/
                semantically neutral corpus, used for PC-orthogonalization.
            layer: Residual layer these came from.
            explain: Fraction of neutral variance to project out. Default 0.5.
            orthogonalize: If False, skip the PC step.

        Returns:
            dict {concept_name: Probe}.
        """
        labeled = {k: np.asarray(v, dtype=np.float32) for k, v in labeled.items()}
        # Grand mean across all labeled vectors, equally weighting concepts
        # (so a cohort with more samples doesn't dominate the mean):
        concept_means = np.stack([v.mean(axis=0) for v in labeled.values()])
        grand_mean = concept_means.mean(axis=0)
        Pmat = _baseline_pc_matrix(neutral, explain) if orthogonalize else None

        probes: dict[str, Probe] = {}
        for cname, cvecs in labeled.items():
            raw_vec = cvecs.mean(axis=0) - grand_mean
            if Pmat is not None:
                # Project out the PC subspace from the vector
                raw_vec = raw_vec - raw_vec @ Pmat.T @ Pmat
            probes[cname] = cls.from_vector(
                raw_vec, name=cname, layer=layer,
                baseline_mean=grand_mean, orthogonalizer=Pmat,
                normalize=True, hook_point=hook_point, head=head,
            )
        return probes


def _baseline_pc_matrix(baseline: np.ndarray, explain: float) -> np.ndarray:
    """Top-variance PCs of `baseline` explaining `explain` fraction.

    Returns an [k, d] array whose rows are orthonormal principal directions.
    Suitable for projecting out via `x - x @ P.T @ P`.
    """
    base = np.asarray(baseline, dtype=np.float64)
    centered = base - base.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    var = S ** 2
    if var.sum() <= 0:
        return np.zeros((0, base.shape[1]), dtype=np.float32)
    cum = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum, explain) + 1)
    k = min(k, Vt.shape[0])
    return Vt[:k].astype(np.float32)
