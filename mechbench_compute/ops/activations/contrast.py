from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.read_pair import read_pair
from mechbench_compute.interp.render_text import render_text
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P

_PAIRS = In("records", "records/pair",
            "Pairs, each with prompt strings `a` and `b`.", many=True)


OP = Op(
    name="activations/contrast",
    requires="mlx-local",
    summary=(
        "Run two prompts that differ in one place and measure, at every "
        "(layer, position), how far their residual streams have drifted "
        "apart — where a one-token change ripples."
    ),
    description="""\
Each record is a matched pair `a`/`b` that must tokenize to the same length.
Both prompts are run with the residual stream captured at every requested
layer, and for each (layer, position) the block reports 1 − cosine
similarity between the two streams: 0 where they are identical, rising where
the change has propagated.

Unequal-length pairs are reported as errors, not aligned by guesswork.
""",
    inputs=(_PAIRS, In("adapter", "adapter/lora",
                       "A LoRA adapter to fuse on top of the model for this node only — "
                       "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
                       "reference, or a stored adapter. Fuses last, on top of any "
                       "adapters the model reference itself carries; `adapter_scale` "
                       "scales this one.",
                       required=False)),
    output=Output('activations/divergence', collection=True, doc="One grid per record over axes `[layer, position]`: `measures.divergence` is 1 − cosine; `tokens` are prompt `a`'s. An unaligned pair has `error` and empty measures."),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to run over.",
          "all"),
        P("point", "string",
          "Which residual stream to read: `\"resid_post\"` (after each "
          "layer) or `\"resid_pre\"` (before it).",
          "resid_post", choices=("resid_post", "resid_pre"), value="point"),
    ),
    example={
        "model": {"$param": "model"},
        "layers": "all",
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/pairs"}}},
)


def run(ctx, inputs, params):
    """For a matched pair of prompts, how far apart the residual
    streams run at every (layer, position).

    The map that answers "where do these two inputs stop being
    processed the same way?" — the first thing to look at when two
    conditions behave differently and nobody knows yet where the
    difference begins.
    """

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return measure_residual_divergence(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


def measure_residual_divergence(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Matched-pair divergence: run prompts `a` and `b`, and per
    (layer, position) report 1 − cosine of the residual streams. The
    map shows WHERE a one-token change ripples (000050/000052).

    Pairs must tokenize to equal lengths — that is what 'matched'
    means; unequal pairs are reported as errors, not silently aligned.
    """
    point = hookpoints.residual(params.get("point"))
    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("residuals/divergence needs at least one pair")
    if on_start:
        on_start(len(records) * 2)

    cap = Capture.residual(layers, point=hookpoints.side(point))
    pairs: list[dict[str, Any]] = []
    for record in records:
        a, b = read_pair(record)
        ids_a = render_text(model, record, a)
        ids_b = render_text(model, record, b)
        len_a = int(np.array(ids_a).shape[-1])
        len_b = int(np.array(ids_b).shape[-1])
        if len_a != len_b:
            pairs.append(S.grid(
                record.get("id"), ["layer", "position"], {},
                coords=record.get("coords"),
                error=(f"prompts tokenize to different lengths ({len_a} vs "
                       f"{len_b}) — a matched pair must match")))
            if on_item:
                on_item()
                on_item()
            continue
        run_a = model.run(ids_a, interventions=[cap])
        if on_item:
            on_item()
        run_b = model.run(ids_b, interventions=[cap])
        if on_item:
            on_item()
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids_a).reshape(-1)]
        matrix: list[list[float]] = []
        for layer in layers:
            va = np.array(run_a.cache[f"blocks.{layer}.{point}"][0]
                          .astype(mx.float32))
            vb = np.array(run_b.cache[f"blocks.{layer}.{point}"][0]
                          .astype(mx.float32))
            na = np.linalg.norm(va, axis=-1)
            nb = np.linalg.norm(vb, axis=-1)
            cos = (va * vb).sum(axis=-1) / np.maximum(na * nb, 1e-9)
            matrix.append([round(float(1.0 - c), 5) for c in cos])
        pairs.append(S.grid(
            record.get("id"), ["layer", "position"], {"divergence": matrix},
            tokens=tokens, coords=record.get("coords")))
    return load_kinds().collection(
        "activations/divergence", pairs,
        point=point,
        layers=layers,
        description=(
            "1 − cosine similarity of the two residual streams per "
            "(layer, position). 0 = identical; the map shows where a "
            "one-token change ripples."
        ),
    )
