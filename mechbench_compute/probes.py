from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import orthogonalize_against


@dataclass(frozen=True)
class Probe:
    name: str
    vec: np.ndarray
    layer: int
    baseline_mean: np.ndarray
    orthogonalizer: Optional[np.ndarray] = None
    hook_point: Optional[str] = None
    head: Optional[int] = None

    @property
    def effective_hook_point(self) -> str:
        return self.hook_point or f"blocks.{self.layer}.resid_post"

    def score(self, residuals: np.ndarray) -> np.ndarray:
        r = np.asarray(residuals, dtype=np.float32)
        if r.shape[-1] != self.vec.shape[-1]:
            raise ValueError(
                f"residuals last dim {r.shape[-1]} != probe d_model "
                f"{self.vec.shape[-1]}"
            )
        centered = r - self.baseline_mean
        if self.orthogonalizer is not None:
            P = self.orthogonalizer
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
        labeled = {k: np.asarray(v, dtype=np.float32) for k, v in labeled.items()}
        concept_means = np.stack([v.mean(axis=0) for v in labeled.values()])
        grand_mean = concept_means.mean(axis=0)
        Pmat = _baseline_pc_matrix(neutral, explain) if orthogonalize else None

        probes: dict[str, Probe] = {}
        for cname, cvecs in labeled.items():
            raw_vec = cvecs.mean(axis=0) - grand_mean
            if Pmat is not None:
                raw_vec = raw_vec - raw_vec @ Pmat.T @ Pmat
            probes[cname] = cls.from_vector(
                raw_vec, name=cname, layer=layer,
                baseline_mean=grand_mean, orthogonalizer=Pmat,
                normalize=True, hook_point=hook_point, head=head,
            )
        return probes


def _baseline_pc_matrix(baseline: np.ndarray, explain: float) -> np.ndarray:
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
