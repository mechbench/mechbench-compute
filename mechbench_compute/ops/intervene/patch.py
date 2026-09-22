from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.read_pair import read_pair
from mechbench_compute.interp.render_text import render_text
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interp.resolve_target import resolve_target
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="intervene/patch",
    requires="mlx-local",
    summary=(
        "Causal tracing: run a clean and a corrupted prompt, patch the clean "
        "activations into the corrupted run one (layer, position) at a time, "
        "and map where the clean answer's probability comes back."
    ),
    description="""\
Each record is a pair: prompt `a` (clean) where the model gets the answer,
and prompt `b` (corrupted, the same length in tokens) where it does not — the
older field names `clean` and `corrupt` are still read. The block runs
the clean prompt once, capturing the residual stream at every layer, and the
corrupt prompt once for a baseline. Then for every (layer, position) it runs
the corrupt prompt again with that one activation replaced by the clean
one, and records how much of the target's probability is recovered.

The result is a heat map over (layer, position). Bright cells — where a
single patch restores the answer — are where the fact is carried.

Pairs whose prompts tokenize to different lengths cannot be aligned and are
reported as errors rather than silently shifted.

### Exact, or estimated

`method: "exact"` (the default) runs one forward pass per cell: a 42-layer
model on a 40-token prompt is 1,680 passes per pair. `method:
"attribution"` — attribution patching (Nanda 2023; Syed, Rager & Conmy
2023) — estimates every cell from ONE forward and ONE backward pass over
the corrupt prompt: the gradient of the metric with respect to each
activation, dotted with the clean activation minus the corrupt one. The
grid has the same shape and the same sign, so the two compare cell for
cell with `records/subtract`; the header's `method` says which ran.

An attribution is a first-order estimate, and the difference shows in
two ways. Where the metric saturates — a log-probability near zero, a
probability near one — the exact trace is a step (every cell that flips
the answer scores the full recovery) and the estimate is graded; and a
cell whose patch flips the answer outright is a large, nonlinear effect
the estimate reads at a fraction of its size. What survives is the
ranking: on Gemma 4 E2B, a capital-city pair under `metric: "logit"`
(the raw logit, the most nearly linear — the usual choice with this
method) puts eight of the exact trace's top ten cells in the estimate's
top ten, with Spearman 0.86 over the cells that matter. Use it to find
the cells worth patching, then patch them exactly. Attribution also reads
`attn_out` and `mlp_out`, the sublayer outputs, where an exact trace reads
only the residual stream.
""",
    inputs=(
        In("records", "records/pair",
           "Pairs, each with prompt strings `a` and `b`, and optionally its "
           "own `tracked`.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('intervene/trace', collection=True, doc="One grid per record over axes `[layer, position]`: `measures.recovery` is the change in the target's `metric` from the `b` baseline when the `a` activation is patched in — measured under `method: \"exact\"`, estimated at first order under `\"attribution\"` — and `measures.share` is the same cell as a fraction of the `value_a`−`value_b` gap, 0 the corrupt run and 1 the clean one, so pairs with different gaps read on one scale (absent when a pair has no gap); `tokens` are prompt `b`'s; `target`, `metric`, `value_a` and `value_b` (the metric on each prompt) ride along. A pair that could not be aligned has `error` and empty measures. The header carries `method`, `point`, `metric`, `layers`."),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to run over.",
          "all"),
        P("method", "string",
          "`\"exact\"`: one forward pass per (layer, position), the patch "
          "itself. `\"attribution\"`: every cell estimated from one forward "
          "and one backward pass — a ranking of where to patch.",
          "exact", choices=("exact", "attribution")),
        P("metric", "string",
          "What is recovered: `\"logprob\"` (the target's log-probability — "
          "registers recovery at any probability mass), `\"prob\"` (raw "
          "probability — only registers when the clean prompt puts real "
          "mass on the target) or `\"logit\"` (the raw logit — the usual "
          "choice with `attribution`, being the most nearly linear).",
          "logprob", choices=("logprob", "prob", "logit")),
        P("point", "string",
          "The point patched: `\"resid_post\"` (after the layer) or "
          "`\"resid_pre\"` (before it); under `attribution` also "
          "`\"attn_out\"` and `\"mlp_out\"`.",
          "resid_post", choices=("resid_post", "resid_pre", "attn_out", "mlp_out"), value="point"),
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is "
          "the target — the clean answer whose recovery is traced; defaults to "
          "the clean prompt's top‑1. Each is tokenized as a continuation of the "
          "rendered prompt: with a leading space after a raw prompt, without one "
          "after a chat template's assistant prefix. A record's own `tracked` "
          "takes precedence; with none named, the model's own top-1 prediction "
          "for that prompt is the target, and a target that differs from it is "
          "reported beside it.",
          None),
    ),
    example={
        "model": {"$param": "model"},
        "tracked": {"answer": " Paris"},
        "metric": "logprob",
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/pairs"}}},
)


def run(ctx, inputs, params):
    """Replace an activation with the one another run had at the same
    place, and measure how much of the answer comes back.

    Causal tracing: where a clean run and a corrupted one differ, the
    activation whose restoration recovers the prediction is where the
    information was being carried. Unlike ablation this localizes
    content rather than participation.
    """

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return patch_trace(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


def patch_trace(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Causal tracing: run CLEAN capturing every layer, run
    CORRUPT for the baseline, then patch the clean residual into the
    corrupt run one (layer, position) at a time and measure how much of
    the clean answer's probability comes back. The map localizes WHERE
    the fact lives. Progress ticks per layer row (a full row of
    positions is one unit)."""
    from mechbench_compute.interventions import Patch

    method = str(params.get("method", "exact"))
    if method not in ("exact", "attribution"):
        raise ValueError(f"unknown method {method!r}: 'exact' or 'attribution'")
    if method == "exact":
        point = hookpoints.residual(params.get("point"))
    else:
        point = hookpoints.normalize(params.get("point"))
        if point not in _ATTRIBUTION_POINTS:
            raise ValueError(
                f"attribution reads {', '.join(_ATTRIBUTION_POINTS)}, not {point!r}")
    metric = str(params.get("metric", "logprob"))
    if metric not in ("logprob", "prob", "logit"):
        raise ValueError(f"unknown metric {metric!r}: 'logprob', 'prob' or 'logit'")
    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("patch/trace needs at least one pair")
    if on_start:
        # An exact trace ticks per layer row; an attribution ticks once
        # per pair — its rows arrive together.
        on_start(len(records) * (len(layers) if method == "exact" else 1))

    cap = (Capture.residual(layers, point=hookpoints.side(point)) if method == "exact"
           else Capture.at([f"blocks.{layer}.{point}" for layer in layers]))
    pairs: list[dict[str, Any]] = []
    for record in records:
        clean, corrupt = read_pair(record)
        ids_clean = render_text(model, record, clean)
        ids_corrupt = render_text(model, record, corrupt)
        n_clean = int(np.array(ids_clean).shape[-1])
        n_corrupt = int(np.array(ids_corrupt).shape[-1])
        if n_clean != n_corrupt:
            pairs.append(S.grid(
                record.get("id"), ["layer", "position"], {},
                coords=record.get("coords"),
                error=(f"prompts tokenize to different lengths ({n_clean} vs "
                       f"{n_corrupt}) — positions cannot align under patching")))
            if on_item:
                for _ in layers:
                    on_item()
            continue
        clean_run = model.run(ids_clean, interventions=[cap])
        clean_lp = read_last_logp(clean_run.logits)
        tok, _ = resolve_target(model, record, params, clean_lp)
        # 'prob' only registers when the clean prompt puts real mass on
        # the target, so an explicitly named rare target measures nothing
        # in prob space; 'logprob' registers recovery at ANY mass.
        # 'logit' is the raw logit, which is what a first-order estimate
        # is most nearly linear in.
        def read(lp: np.ndarray, logits: np.ndarray | None = None, tok: int = tok) -> float:
            if metric == "logit":
                return float(logits[tok])
            return (float(np.exp(lp[tok])) if metric == "prob"
                    else float(lp[tok]))

        def read_run(res) -> float:
            return read(read_last_logp(res.logits), _read_last_logits(res.logits))

        p_clean_in_clean = read_run(clean_run)
        corrupt_run = model.run(ids_corrupt)
        baseline = read_run(corrupt_run)

        seq = n_corrupt
        if method == "attribution":
            recovery = _compute_attribution_grid(
                model, ids_corrupt, layers, point, clean_run.cache, tok, metric)
            if on_item:
                on_item()
        else:
            recovery = []
            for layer in layers:
                row: list[float] = []
                for pos in range(seq):
                    patch = Patch.position(
                        layer=layer, position=pos, source=clean_run.cache,
                        point=point)
                    row.append(round(read_run(model.run(ids_corrupt, interventions=[patch]))
                                     - baseline, 5))
                recovery.append(row)
                if on_item:
                    on_item()
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids_corrupt).reshape(-1)]
        # The same grid as a share of the clean-corrupt gap: 0 is the
        # corrupt run, 1 is the clean one, so pairs with different gaps
        # read on one scale. A pair whose two prompts score alike has
        # no gap to share and the measure is left out.
        gap = p_clean_in_clean - baseline
        measures = {"recovery": recovery}
        if abs(gap) > 1e-9:
            measures["share"] = [[round(v / gap, 4) for v in row] for row in recovery]
        pairs.append(S.grid(
            record.get("id"), ["layer", "position"], measures,
            tokens=tokens, coords=record.get("coords"),
            target=S.token(model.tokenizer, tok), metric=metric,
            value_a=round(p_clean_in_clean, 5), value_b=round(baseline, 5)))
    return load_kinds().collection(
        "intervene/trace", pairs,
        point=point,
        metric=metric,
        method=method,
        layers=layers,
        description=(
            "Activation patching: the clean answer's metric recovered when "
            "the clean activation is patched into the corrupt run at each "
            "(layer, position). Bright cells are where the fact lives."
            if method == "exact" else
            "Attribution patching: the recovery at every (layer, position) "
            "estimated at first order from one backward pass — the gradient "
            "of the metric on the corrupt run, dotted with the clean minus "
            "corrupt activation. A ranking of where to patch, not the patch."
        ),
    )


#: The points an attribution reads: every layer-scoped activation with a
#: (batch, position, feature) layout, which is what a delta is added to.
_ATTRIBUTION_POINTS = ("resid_post", "resid_pre", "attn_out", "mlp_out")


def _read_last_logits(logits: mx.array) -> np.ndarray:
    row = logits[0, -1, :].astype(mx.float32)
    mx.eval(row)
    return np.array(row)


def _compute_attribution_grid(model, ids_corrupt, layers: Sequence[int], point: str,
                              clean_cache, tok: int, metric: str) -> list[list[float]]:
    """Attribution patching (Nanda 2023; Syed, Rager & Conmy 2023): the
    recovery at every (layer, position) at once, from ONE forward and
    ONE backward pass over the corrupt prompt.

    Every named activation gets a zero delta added by a hook; the
    metric's gradient with respect to each delta is its gradient with
    respect to the activation; and the patch's effect is estimated at
    first order as that gradient dotted with (clean − corrupt). An exact
    trace runs one forward per cell; this runs two for the grid, which
    is what makes a circuit search over thousands of cells affordable —
    and a first-order estimate, which is why it ranks cells rather than
    measures them (a saturating metric is where it overshoots most;
    `logit` is the most nearly linear).
    """
    names = [f"blocks.{layer}.{point}" for layer in layers]
    # Shapes come from the clean capture, never assumed.
    deltas = {n: mx.zeros(clean_cache[n].shape, dtype=mx.float32) for n in names}

    def objective(ds):
        hooks = {n: (lambda act, info, d=ds[n]: act + d.astype(act.dtype)) for n in names}
        res = model.run(ids_corrupt, hooks=hooks)
        row = res.logits[0, -1, :].astype(mx.float32)
        if metric == "logit":
            return row[tok]
        lp = (row - mx.logsumexp(row))[tok]
        return mx.exp(lp) if metric == "prob" else lp

    grads = mx.grad(objective)(deltas)
    mx.eval(*grads.values())
    # The corrupt activations at the same points, for the difference:
    # captured on their own pass so the differentiated one stays a
    # function of the deltas alone.
    corrupt_cache = model.run(ids_corrupt, capture=names).cache
    grid: list[list[float]] = []
    for n in names:
        diff = clean_cache[n].astype(mx.float32) - corrupt_cache[n].astype(mx.float32)
        cell = mx.sum(grads[n] * diff, axis=-1)[0]      # [positions]
        mx.eval(cell)
        grid.append([round(float(x), 5) for x in np.array(cell)])
    return grid
