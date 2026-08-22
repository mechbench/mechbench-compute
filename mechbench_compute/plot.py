"""Matplotlib plot helpers with project conventions baked in.

Ten helpers covering the recurring chart styles in this project's
experiments:

  bar_by_layer            per-layer bar chart with red=global / blue=local
                          conventions (step_02/03/04)
  lens_trajectory         per-layer rank curves with optional geometric-mean
                          aggregation (step_01)
  logprob_trajectory      per-layer log-probability curves (step_01 variant)
  position_heatmap        [layer x position] heatmap with subject and global-
                          layer markers (step_08/09)
  pca_scatter             2D PCA projection colored by category (step_10/12/13)
  similarity_heatmap      pairwise cosine, reordered block-diagonal (step_10/12)
  head_heatmap            [n_layers x n_heads] per-head metric heatmap with
                          global-layer markers (step_26/28/29)
  probe_diagonal_heatmap  true x predicted aggregated scoring grid with
                          per-cell text annotations (step_21/23)
  grouped_row_heatmap     per-row heatmap with rows grouped by category +
                          horizontal boundary lines (step_21/23)
  intensity_curve         multi-line plot against a scalar parameter with
                          target/antipode emphasis (step_24/25)
  leaderboard_bar         ranked horizontal bar chart with text labels and
                          global/local color coding (step_26)

API contract:
  - Inputs are numpy arrays.
  - Each function accepts ax=... for composition; if None, a new figure+axes
    is created and the new Axes returned.
  - Functions return the Axes (never call plt.show() or fig.savefig()).
  - The conventions (colors, GLOBAL_LAYERS dashed lines, log-scale ticks for
    rank) are defaults; every one is overridable via keyword.

Optional layer; every plot can still be hand-rolled in matplotlib.
"""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from ._arch import GLOBAL_LAYERS, N_LAYERS, layer_type

# Default colors used across the existing experiments.
COLOR_GLOBAL = "#d62728"  # red
COLOR_LOCAL = "#1f77b4"   # blue
COLOR_AGGREGATE = "#d62728"  # red, for the bold mean line

# Distinct 12-category palette used in step_12/13.
DEFAULT_CATEGORY_COLORS: dict[str, str] = {
    "capital": "#e41a1c", "element": "#377eb8", "author": "#4daf4a",
    "landmark": "#ff7f00", "opposite": "#984ea3", "past_tense": "#a65628",
    "plural": "#f781bf", "french": "#ffff33", "profession": "#999999",
    "animal_home": "#66c2a5", "color_mix": "#8dd3c7", "math": "#fb8072",
}


def _ensure_axes(ax: Optional[Axes], **figkwargs) -> Axes:
    if ax is None:
        _, ax = plt.subplots(**figkwargs)
    return ax


def _color_for(label: str, color_map: dict[str, str]) -> str:
    """Pick a color, falling back to DEFAULT_CATEGORY_COLORS, then matplotlib's
    cycler (deterministic via hash)."""
    if label in color_map:
        return color_map[label]
    if label in DEFAULT_CATEGORY_COLORS:
        return DEFAULT_CATEGORY_COLORS[label]
    # Fallback: deterministic palette index from label hash
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[hash(label) % len(cycle)]


# ---------------------------------------------------------------------------
# bar_by_layer
# ---------------------------------------------------------------------------


def bar_by_layer(
    values: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    color_global: str = COLOR_GLOBAL,
    color_local: str = COLOR_LOCAL,
    show_global_lines: bool = False,
    show_legend: bool = True,
    xticks_step: int = 3,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple[float, float] = (14, 5),
) -> Axes:
    """Per-layer bar chart with red=global / blue=local convention.

    Args:
        values: 1D array of length N_LAYERS (or shorter; bars indexed 0..len-1).
        ax: existing Axes to draw on; creates a new figure if None.
        color_global / color_local: bar colors for the two layer types.
        show_global_lines: draw vertical dashed lines at every global-layer
            index (often noisy when there are also bars; default off).
        show_legend: include a (global, local) color legend in the lower-left.
        xticks_step: tick every N layers (default 3).
        title / ylabel: optional axis labels.
    """
    ax = _ensure_axes(ax, figsize=figsize)
    n = len(values)
    colors = [color_global if layer_type(i) == "full_attention" else color_local
              for i in range(n)]
    ax.bar(range(n), values, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(0, n, xticks_step))
    ax.set_xlabel("layer index")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    if show_global_lines:
        for g in GLOBAL_LAYERS:
            if g < n:
                ax.axvline(g, color="#999999", linestyle="--",
                           linewidth=0.7, alpha=0.4)

    if show_legend:
        ax.legend(
            handles=[Patch(color=color_global, label="global attention"),
                     Patch(color=color_local, label="local (sliding window)")],
            loc="lower left",
        )
    return ax


# ---------------------------------------------------------------------------
# lens_trajectory  (rank version + logprob version)
# ---------------------------------------------------------------------------


def lens_trajectory(
    ranks: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    individuals: bool = True,
    aggregate: bool = True,
    aggregate_label: Optional[str] = None,
    show_global_lines: bool = True,
    log_scale: bool = True,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 4),
) -> Axes:
    """Per-layer rank trajectory plot.

    Args:
        ranks: array of shape [n_prompts, n_layers] for batch, or [n_layers]
            for a single prompt.
        individuals: draw one thin line per prompt (no-op if 1D).
        aggregate: draw a bold geometric-mean line over prompts (no-op if 1D).
        aggregate_label: legend label for the aggregate line.
        log_scale: y-axis on log scale (rank=0 is replaced by 0.5 for
            display so it doesn't hit -inf in log space).
        show_global_lines: vertical dashed lines at GLOBAL_LAYERS.
    """
    ax = _ensure_axes(ax, figsize=figsize)
    ranks = np.asarray(ranks)
    if ranks.ndim == 1:
        ranks = ranks[None, :]
    n_prompts, n_layers = ranks.shape
    layers_x = np.arange(n_layers)

    def _safe(arr):
        return np.where(arr < 0.5, 0.5, arr) if log_scale else arr

    if individuals and n_prompts > 1:
        for j in range(n_prompts):
            ax.plot(layers_x, _safe(ranks[j]), color=COLOR_LOCAL,
                    alpha=0.2, linewidth=0.8)

    if aggregate and n_prompts > 1:
        log_rank = np.log(ranks + 1)
        geomean = np.exp(np.mean(log_rank, axis=0)) - 1
        label = aggregate_label or f"geometric mean (n={n_prompts})"
        ax.plot(layers_x, _safe(geomean), color=COLOR_AGGREGATE,
                linewidth=2.5, label=label)
    elif n_prompts == 1:
        ax.plot(layers_x, _safe(ranks[0]), color=COLOR_AGGREGATE,
                linewidth=2.0, label="rank")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("layer index")
    ax.set_ylabel("rank of target token" + (" (log)" if log_scale else ""))
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    if show_global_lines:
        for g in GLOBAL_LAYERS:
            ax.axvline(g, color="#999999", linestyle="--",
                       linewidth=0.7, alpha=0.6)

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right")
    return ax


def logprob_trajectory(
    logprobs: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    individuals: bool = True,
    aggregate: bool = True,
    aggregate_label: Optional[str] = None,
    show_global_lines: bool = True,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 4),
) -> Axes:
    """Per-layer log-probability trajectory plot.

    Same shape conventions as lens_trajectory; aggregate is arithmetic mean
    (log p is already in log space)."""
    ax = _ensure_axes(ax, figsize=figsize)
    arr = np.asarray(logprobs)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_prompts, n_layers = arr.shape
    layers_x = np.arange(n_layers)

    if individuals and n_prompts > 1:
        for j in range(n_prompts):
            ax.plot(layers_x, arr[j], color="#2ca02c",
                    alpha=0.2, linewidth=0.8)

    if aggregate and n_prompts > 1:
        mean = arr.mean(axis=0)
        label = aggregate_label or f"mean (n={n_prompts})"
        ax.plot(layers_x, mean, color=COLOR_AGGREGATE,
                linewidth=2.5, label=label)
    elif n_prompts == 1:
        ax.plot(layers_x, arr[0], color=COLOR_AGGREGATE,
                linewidth=2.0, label="log p")

    ax.set_xlabel("layer index")
    ax.set_ylabel("log p(target)")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    if show_global_lines:
        for g in GLOBAL_LAYERS:
            ax.axvline(g, color="#999999", linestyle="--",
                       linewidth=0.7, alpha=0.6)

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="lower right")
    return ax


# ---------------------------------------------------------------------------
# position_heatmap
# ---------------------------------------------------------------------------


def position_heatmap(
    values: np.ndarray,
    token_labels: Optional[list[str]] = None,
    *,
    ax: Optional[Axes] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mark_positions: Iterable[int] = (),
    mark_layers: Iterable[int] = GLOBAL_LAYERS,
    colorbar: bool = True,
    colorbar_label: str = "",
    log_scale: bool = False,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 8),
) -> Axes:
    """Heatmap of values[layer, position] with optional position/layer markers.

    Args:
        values: 2D array shape [n_layers, seq_len].
        token_labels: per-position labels for the x-axis ticks.
        cmap: matplotlib colormap name (default RdYlGn for log p; pass
            'RdYlGn_r' for rank).
        vmin/vmax: color scale bounds. If None, set from data percentiles.
        mark_positions: vertical red-dashed lines at these positions
            (e.g. subject-entity tokens).
        mark_layers: horizontal gray-dotted lines at these layers
            (default: GLOBAL_LAYERS).
        log_scale: pass values through log10(x+1) before plotting (for rank).
    """
    ax = _ensure_axes(ax, figsize=figsize)
    arr = np.asarray(values)
    if log_scale:
        arr = np.log10(arr + 1)
    if vmin is None:
        vmin = float(np.nanpercentile(arr, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 99))

    im = ax.imshow(arr, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, origin="lower",
                   interpolation="nearest")
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    if token_labels is not None:
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=70, ha="right", fontsize=6)
    if title:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6, label=colorbar_label)

    for p in mark_positions:
        ax.axvline(p, color="red", linewidth=1.5, alpha=0.7, linestyle="--")
    for g in mark_layers:
        ax.axhline(g, color="gray", linewidth=0.5, alpha=0.5, linestyle=":")

    return ax


# ---------------------------------------------------------------------------
# pca_scatter
# ---------------------------------------------------------------------------


def pca_scatter(
    vectors: np.ndarray,
    labels,
    *,
    ax: Optional[Axes] = None,
    color_map: Optional[dict[str, str]] = None,
    show_legend: bool = True,
    show_variance: bool = True,
    n_components: int = 2,
    seed: int = 42,
    s: int = 60,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8),
) -> Axes:
    """2D PCA projection colored by category.

    Args:
        vectors: [N, D] array of vectors to project.
        labels: [N] sequence of category labels (string or hashable).
        color_map: optional {label -> color}. Falls back to
            DEFAULT_CATEGORY_COLORS, then to a deterministic hash-based pick
            from matplotlib's color cycler.
        show_variance: append explained-variance percentage to the title.
    """
    from sklearn.decomposition import PCA  # imported lazily

    ax = _ensure_axes(ax, figsize=figsize)
    color_map = color_map or {}
    labels = np.asarray(labels)

    pca = PCA(n_components=n_components, random_state=seed)
    proj = pca.fit_transform(vectors)
    cats = list(dict.fromkeys(labels.tolist()))

    for cat in cats:
        mask = labels == cat
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=_color_for(cat, color_map),
            label=f"{cat} ({int(mask.sum())})",
            s=s, alpha=0.85, edgecolors="black", linewidths=0.5,
        )

    suffix = ""
    if show_variance:
        suffix = f" (PCA variance: {pca.explained_variance_ratio_.sum():.1%})"
    ax.set_title((title or "PCA scatter") + suffix, fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="best", fontsize=8, ncol=max(1, len(cats) // 6))
    return ax


# ---------------------------------------------------------------------------
# similarity_heatmap
# ---------------------------------------------------------------------------


def similarity_heatmap(
    vectors: np.ndarray,
    labels,
    *,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    vmin: float = -1,
    vmax: float = 1,
    show_category_labels: bool = True,
    show_boundary_lines: bool = True,
    colorbar: bool = True,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8),
) -> Axes:
    """Pairwise cosine heatmap, reordered so members of the same category
    sit in contiguous blocks (block-diagonal layout).

    Args:
        vectors: [N, D].
        labels: [N] categorical.
        cmap, vmin, vmax: passed to imshow.
        show_category_labels: place category names along both axes at the
            block midpoints.
        show_boundary_lines: black lines between category blocks.
    """
    from .geometry import cosine_matrix  # avoid circular import at module load

    ax = _ensure_axes(ax, figsize=figsize)
    labels = np.asarray(labels)
    cats = list(dict.fromkeys(labels.tolist()))

    order = np.argsort([cats.index(l) for l in labels])
    sim = cosine_matrix(vectors)
    sim_ord = sim[np.ix_(order, order)]
    labels_ord = labels[order]

    im = ax.imshow(sim_ord, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    if show_boundary_lines or show_category_labels:
        boundaries = [0]
        prev = None
        for i, lab in enumerate(labels_ord):
            if lab != prev:
                if prev is not None and show_boundary_lines:
                    ax.axhline(i - 0.5, color="black", linewidth=0.8)
                    ax.axvline(i - 0.5, color="black", linewidth=0.8)
                if prev is not None:
                    boundaries.append(i)
                prev = lab
        boundaries.append(len(labels_ord))

        if show_category_labels:
            mids = [(boundaries[k] + boundaries[k + 1]) / 2
                    for k in range(len(boundaries) - 1)]
            ax.set_yticks(mids)
            ax.set_yticklabels(cats, fontsize=9)
            ax.set_xticks(mids)
            ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=9)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=11)
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8)

    return ax


# ---------------------------------------------------------------------------
# head_heatmap
# ---------------------------------------------------------------------------


def head_heatmap(
    values: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    diverging: bool = True,
    mark_global_layers: bool = True,
    layer_label: str = "layer",
    head_label: str = "head",
    yticks_step: int = 3,
    title: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: str = "",
    figsize: tuple[float, float] = (5, 9),
) -> Axes:
    """Per-(layer, head) metric heatmap.

    Args:
        values: 2D array shape [n_layers, n_heads] (n_heads can be 8 for
            Q-heads or 2 for KV-heads in Gemma 4 E4B).
        diverging: if True, the colormap is symmetric around zero (vmin/vmax
            set to ±max-abs unless overridden). Use False for non-negative
            metrics like accuracy.
        mark_global_layers: draw small red ticks at GLOBAL_LAYERS rows.
        title / colorbar_label: optional labels.

    Used by step_26 (rank-0 OV singular values), step_28 (Q/K silhouette),
    step_29 (Q/K/V silhouette and accuracy).
    """
    ax = _ensure_axes(ax, figsize=figsize)
    arr = np.asarray(values)
    n_layers, n_heads = arr.shape

    if diverging:
        if vmin is None or vmax is None:
            m = float(np.abs(arr).max())
            vmin = -m if vmin is None else vmin
            vmax = m if vmax is None else vmax
    else:
        if vmin is None:
            vmin = float(arr.min())
        if vmax is None:
            vmax = float(arr.max())

    im = ax.imshow(arr, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel(head_label)
    ax.set_ylabel(layer_label)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(0, n_layers, yticks_step))

    if mark_global_layers:
        # Small red ticks on the left edge marking each global layer.
        for g in GLOBAL_LAYERS:
            if g < n_layers:
                ax.axhline(g, color="red", linewidth=0.4, alpha=0.4,
                           xmin=-0.02, xmax=0.0)

    if title:
        ax.set_title(title, fontsize=10)
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)

    return ax


# ---------------------------------------------------------------------------
# probe_diagonal_heatmap
# ---------------------------------------------------------------------------


def probe_diagonal_heatmap(
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
    annotation_format: str = "+.2f",
    annotation_threshold: float = 0.5,
    title: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: str = "score",
    figsize: tuple[float, float] = (8, 6),
) -> Axes:
    """Aggregated probe-vs-target scoring heatmap (e.g. emotion probes scored
    on emotion training corpus).

    Args:
        values: [len(row_labels), len(col_labels)] score matrix. Rows
            typically = true category, columns = probe under test.
        row_labels / col_labels: axis tick labels.
        annotate: write each cell value on the heatmap.
        annotation_format: format-string for cell text (e.g. '+.2f', '.3f').
        annotation_threshold: when |value| / |values|.max() exceeds this
            fraction, the annotation is drawn white instead of black for
            readability against dark cells.

    Used by step_21 (emotion-probe self-consistency), step_23 (implicit-
    scenario validation).
    """
    ax = _ensure_axes(ax, figsize=figsize)
    arr = np.asarray(values)
    vmax_abs = float(np.abs(arr).max()) if arr.size > 0 else 1.0

    im = ax.imshow(arr, aspect="auto", cmap=cmap,
                   vmin=-vmax_abs, vmax=vmax_abs, interpolation="nearest")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                color = ("white"
                         if abs(v) > vmax_abs * annotation_threshold
                         else "black")
                ax.text(j, i, format(v, annotation_format),
                        ha="center", va="center",
                        color=color, fontsize=8)

    if title:
        ax.set_title(title, fontsize=11)
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)

    return ax


# ---------------------------------------------------------------------------
# grouped_row_heatmap
# ---------------------------------------------------------------------------


def grouped_row_heatmap(
    values: np.ndarray,
    row_groups: np.ndarray,
    *,
    col_labels: Optional[list[str]] = None,
    group_order: Optional[list[str]] = None,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    boundary_color: str = "black",
    boundary_width: float = 0.8,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: str = "",
    figsize: tuple[float, float] = (8, 10),
) -> Axes:
    """Per-row heatmap with rows visually grouped by category + boundary lines.

    Rows get re-ordered so members of the same group sit contiguously, and
    thin horizontal lines are drawn between groups. The y-axis is unlabeled
    (too many rows) but the grouping structure is visually obvious.

    Args:
        values: [n_rows, n_cols] score matrix.
        row_groups: [n_rows] sequence of group labels (strings).
        col_labels: optional column-axis labels.
        group_order: optional canonical ordering of groups. If None, groups
            are ordered by first appearance in row_groups.

    Used by step_21 and step_23 for per-passage probe-score heatmaps
    where passages are visually grouped by their true emotion.
    """
    ax = _ensure_axes(ax, figsize=figsize)
    arr = np.asarray(values)
    groups = np.asarray(row_groups)

    if group_order is None:
        group_order = list(dict.fromkeys(groups.tolist()))

    # Sort rows by group order
    order = np.argsort([group_order.index(g) for g in groups])
    arr_ord = arr[order]
    groups_ord = groups[order]

    vmax_abs = float(np.abs(arr).max()) if arr.size > 0 else 1.0
    if vmin is None:
        vmin = -vmax_abs
    if vmax is None:
        vmax = vmax_abs

    im = ax.imshow(arr_ord, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks([])

    # Boundaries between groups
    cur = groups_ord[0] if len(groups_ord) else None
    for y in range(1, len(groups_ord)):
        if groups_ord[y] != cur:
            ax.axhline(y - 0.5, color=boundary_color, linewidth=boundary_width)
            cur = groups_ord[y]

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)

    return ax


# ---------------------------------------------------------------------------
# intensity_curve
# ---------------------------------------------------------------------------


def intensity_curve(
    levels,
    scores: np.ndarray,
    series_names: list[str],
    *,
    ax: Optional[Axes] = None,
    target_up: Optional[str] = None,
    target_down: Optional[str] = None,
    colors: Optional[dict[str, str]] = None,
    log_x: bool = True,
    xlabel: Optional[str] = None,
    ylabel: str = "probe score",
    title: Optional[str] = None,
    marker: str = "o",
    bold_linewidth: float = 3.0,
    thin_linewidth: float = 1.3,
    bold_alpha: float = 1.0,
    thin_alpha: float = 0.55,
    figsize: tuple[float, float] = (7, 5),
) -> Axes:
    """Multi-series line plot against a scalar parameter, with optional
    target/antipode emphasis.

    Args:
        levels: iterable of x-axis scalar values (e.g. Tylenol dose in mg).
        scores: [n_levels, n_series] array of per-series y values.
        series_names: list of n_series labels (used for legend + to match
            target_up / target_down).
        target_up: name of the series expected to rise with x; drawn bold.
        target_down: name of the series expected to fall with x; also bold.
        colors: {series_name: color}. Falls back to matplotlib's default
            cycle when absent.
        log_x: x-axis log scale (default True; most intensity axes are log).

    Used by step_24 (four intensity axes) and step_25 (side-by-side
    intensity comparison between probe-sets).
    """
    ax = _ensure_axes(ax, figsize=figsize)
    arr = np.asarray(scores)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for j, name in enumerate(series_names):
        is_up = name == target_up
        is_dn = name == target_down
        is_emph = is_up or is_dn
        lw = bold_linewidth if is_emph else thin_linewidth
        alpha = bold_alpha if is_emph else thin_alpha
        label = name
        if is_up:
            label = f"{name} (target up)"
        elif is_dn:
            label = f"{name} (antipode down)"
        c = (colors or {}).get(name, cycle[j % len(cycle)])
        ax.plot(levels, arr[:, j],
                marker=marker, linewidth=lw, alpha=alpha,
                color=c, label=label)

    if log_x:
        ax.set_xscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    return ax


# ---------------------------------------------------------------------------
# leaderboard_bar
# ---------------------------------------------------------------------------


def leaderboard_bar(
    items: list[tuple[str, float]],
    *,
    ax: Optional[Axes] = None,
    color_groups: Optional[list[str]] = None,
    color_global: str = COLOR_GLOBAL,
    color_local: str = COLOR_LOCAL,
    default_color: str = "#888888",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    figsize: tuple[float, float] = (8, 9),
) -> Axes:
    """Ranked horizontal bar chart with text labels per bar.

    Args:
        items: list of (label, value) tuples, already sorted by value.
        color_groups: optional list of same length as items, one of
            'global' / 'local' / None per bar; sets bar color.
        title / xlabel: optional labels.

    Used by step_26's top-20 heads by OV rank-0 sigma.
    """
    ax = _ensure_axes(ax, figsize=figsize)
    n = len(items)
    ys = np.arange(n)
    values = [v for _, v in items]

    if color_groups:
        colors = [
            color_global if g == "global"
            else color_local if g == "local"
            else default_color
            for g in color_groups
        ]
    else:
        colors = [default_color] * n

    ax.barh(ys, values, color=colors)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in items], fontsize=8)
    ax.invert_yaxis()
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    return ax
