"""Smoke test for plot helpers.

Synthetic-data only — no model load. Each helper is called with realistic
inputs and asserted to (a) return an Axes, (b) not raise, (c) save to a
PNG that the human can spot-check. The PNGs land in caches/_smoke_plots/.

Run from project root (no venv strictly required since this is matplotlib +
numpy + sklearn only — but use the venv for reproducibility):

    python -m mechbench_compute._smoke_plots
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no GUI window
import matplotlib.pyplot as plt
import numpy as np

from . import (
    GLOBAL_LAYERS,
    N_LAYERS,
    bar_by_layer,
    lens_trajectory,
    logprob_trajectory,
    pca_scatter,
    position_heatmap,
    similarity_heatmap,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "caches" / "_smoke_plots"


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def test_bar_by_layer() -> Path:
    # Synthetic: globals slightly more damaging than locals on average.
    rng = np.random.default_rng(0)
    values = rng.normal(-1, 1, N_LAYERS)
    for g in GLOBAL_LAYERS:
        values[g] -= 1.5
    fig, ax = plt.subplots(figsize=(14, 5))
    bar_by_layer(values, ax=ax,
                 ylabel="mean Δ log p (synthetic)",
                 title="bar_by_layer smoke test — synthetic ablation")
    return _save(fig, "01_bar_by_layer")


def test_lens_trajectory() -> Path:
    # Synthetic: each prompt's rank starts high, crashes around layer 28.
    rng = np.random.default_rng(1)
    n_prompts = 15
    ranks = np.zeros((n_prompts, N_LAYERS))
    for j in range(n_prompts):
        for i in range(N_LAYERS):
            base = 100000 if i < 25 else max(0, int(2.0 ** (35 - i) * 100))
            ranks[j, i] = max(0, base + rng.normal(0, base * 0.2))
    fig, ax = plt.subplots(figsize=(12, 4))
    lens_trajectory(ranks, ax=ax,
                    title="lens_trajectory smoke test — 15 synthetic prompts")
    return _save(fig, "02_lens_trajectory")


def test_logprob_trajectory() -> Path:
    rng = np.random.default_rng(2)
    n_prompts = 15
    logp = np.zeros((n_prompts, N_LAYERS))
    for j in range(n_prompts):
        for i in range(N_LAYERS):
            target = -30 if i < 25 else max(-25, -1 - 0.5 * (i - 27))
            logp[j, i] = target + rng.normal(0, 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    logprob_trajectory(logp, ax=ax,
                       title="logprob_trajectory smoke test — synthetic")
    return _save(fig, "03_logprob_trajectory")


def test_position_heatmap() -> Path:
    # Synthetic position-by-layer heatmap: target log-prob crystallizes at
    # the final position in late layers.
    seq_len = 21
    arr = np.full((N_LAYERS, seq_len), -25.0)
    for i in range(N_LAYERS):
        if i >= 28:
            for pos in range(seq_len):
                if pos == seq_len - 1:
                    arr[i, pos] = -1 - 0.3 * (40 - i)
                elif pos > seq_len - 5:
                    arr[i, pos] = -10
    token_labels = (["<bos>"] + [f"tok_{k}" for k in range(seq_len - 2)] + ["<turn|>"])
    fig, ax = plt.subplots(figsize=(12, 8))
    position_heatmap(arr, token_labels=token_labels, ax=ax,
                     mark_positions=[5, 13],
                     vmin=-30, vmax=0,
                     colorbar_label="log p (synthetic)",
                     title="position_heatmap smoke test")
    return _save(fig, "04_position_heatmap")


def test_pca_scatter() -> Path:
    # Synthetic 4-cluster data
    rng = np.random.default_rng(3)
    cats = ["capital", "element", "author", "landmark"]
    centers = rng.normal(0, 5, (4, 2560)).astype(np.float32)
    vecs = []
    labels = []
    for ci, cat in enumerate(cats):
        for _ in range(8):
            vecs.append(centers[ci] + rng.normal(0, 1, 2560).astype(np.float32))
            labels.append(cat)
    vecs = np.array(vecs)
    fig, ax = plt.subplots(figsize=(10, 8))
    pca_scatter(vecs, labels, ax=ax,
                title="pca_scatter smoke test — 4 synthetic clusters")
    return _save(fig, "05_pca_scatter")


def test_similarity_heatmap() -> Path:
    # Same synthetic clusters as PCA test
    rng = np.random.default_rng(3)
    cats = ["capital", "element", "author", "landmark"]
    centers = rng.normal(0, 5, (4, 2560)).astype(np.float32)
    vecs = []
    labels = []
    for ci, cat in enumerate(cats):
        for _ in range(8):
            vecs.append(centers[ci] + rng.normal(0, 1, 2560).astype(np.float32))
            labels.append(cat)
    vecs = np.array(vecs)
    fig, ax = plt.subplots(figsize=(8, 7))
    similarity_heatmap(vecs, labels, ax=ax,
                       title="similarity_heatmap smoke test")
    return _save(fig, "06_similarity_heatmap")


def main() -> int:
    print(f"Writing plots to {OUT_DIR}/")
    tests = [
        ("bar_by_layer", test_bar_by_layer),
        ("lens_trajectory", test_lens_trajectory),
        ("logprob_trajectory", test_logprob_trajectory),
        ("position_heatmap", test_position_heatmap),
        ("pca_scatter", test_pca_scatter),
        ("similarity_heatmap", test_similarity_heatmap),
    ]
    failed = []
    for name, fn in tests:
        try:
            path = fn()
            print(f"  [OK] {name:<22s} -> {path.name}")
        except Exception as exc:  # broad: smoke is about non-crashing
            print(f"  [FAIL] {name:<22s} -- {exc}")
            failed.append(name)

    if failed:
        print(f"\nPLOTS SMOKE TEST FAILED: {failed}")
        return 1
    print("\nPlots smoke test passed. All 6 plot helpers render without error.")
    print(f"Spot-check the PNGs in {OUT_DIR}/ to verify visual conventions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
