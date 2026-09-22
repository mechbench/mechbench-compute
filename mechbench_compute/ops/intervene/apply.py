from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.intervene.cell import Cell
from mechbench_compute.intervene.compile import compile
from mechbench_compute.intervene.spec_error import SpecError
from mechbench_compute.intervene.spec_intervention import SpecIntervention
from mechbench_compute.intervene.read_spec_items import read_spec_items
from mechbench_compute.intervene.sweep_as_run import sweep_as_run
from mechbench_compute.intervene.sweep_cells import sweep_cells
from mechbench_compute.intervene.serialize_spec import serialize_spec
from mechbench_compute.lexicon._base import In, Op, Otherwise, Output, P

OP = Op(
    name="intervene/apply",
    requires="mlx-local",
    summary=(
        "Edit a model's activations at chosen points during the forward "
        "pass — zero them, patch them from another run, add or remove a "
        "direction, scale them — and read out what changed in the next-token "
        "distribution or in the activations themselves."
    ),
    description="""\
An intervention is a list of **spec items**. Each item names a **point** in
the forward pass (a residual stream, an attention or MLP output, a head's
query vector, …), which **layers** and token **positions** it applies to, and
an **op** to perform there. All items are applied together in one forward
pass over each record, and the result is read out either as a **decision** —
the next-token distribution at the last position — or as a **capture** — the
raw activations at named points, which come out as an `activations/vector`
collection exactly as `activations/capture` would emit them, `factor` on each.

A `sweep` runs the whole spec at several strengths — and at several
layers, heads, positions or neurons: each axis is a list of values the
items' field of that name takes in turn, several axes form a cartesian
product, and every axis becomes a coordinate on the rows it produced. By
default a strength‑0 **control** is added, one run, so every record has a
baseline to compare against. The output has one row per record per sweep
cell. A `records/map` over a corpus of integers is no longer how a layer
sweep is written.

### Spec items

```json
{
  "point": "resid_post",
  "layers": [14],
  "positions": "last",
  "op": "add",
  "strength": 4.0,
  "direction": {"$ref": {"bench": "you/lab/direction"}}
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `point` | string | `"resid_post"` | Where in the forward pass to act. Residual points: `resid_pre`, `resid_post`, `attn_out`, `mlp_out`. Inside a block: `attn.in_norm`, `attn.q`, `attn.k`, `attn.v`, `attn.q_pre_rope`, `attn.k_pre_rope`, `attn.q_pre_norm`, `attn.k_pre_norm`, `attn.scores`, `attn.weights`, `attn.per_head_out`, `attn.o_in`, `mlp.in_norm`, `mlp.gate`, `mlp.up`, `mlp.act`, `mlp.down_in`, `gate_out`. Whole-model points, which take no `layers`: `embed`, `final_norm`, `logits`. |
| `layers` | list[int] \\| `"all"` | `"all"` | Which layers the item applies to. |
| `positions` | selector | `"last"` | Which token positions: `"last"`, `"all"`, a list of indices (negative from the end), `{"tokens": ["lighthouse"]}` (every position whose token matches), `{"range": [2, 6]}`, `{"after": n}`, `"subject"` or `"generated"`. |
| `heads` | list[int] | all | Restrict the item to these attention heads, at a point that has a head axis (`attn.q`, `attn.k`, `attn.v`, `attn.scores`, `attn.weights`, `attn.per_head_out`, and the pre-norm/pre-rope variants). |
| `neurons` | list[int] | all | Restrict the item to these indices along the feature axis — MLP neurons at `mlp.act`, residual dimensions at `resid_post`, vocabulary entries at `logits`. |
| `op` | string | `"zero"` | What to do there — see the table below. |
| `strength` | float | `1.0` | The item's magnitude: the coefficient for `add`, the factor for `scale`, the bound for `clamp`, the angle in radians for `rotate`. Multiplied by each sweep factor. |
| `direction` | direction | — | The direction for `add`, `project_out`, `clamp`, `rotate` and (optionally) `patch`. May instead arrive on the node's `direction` port, which fills every item that names none. |
| `direction2` | direction | — | The second axis of the plane for `rotate`. |
| `source` | collection | — | A collection of `activations/vector` — a capture, intervened or not — supplying replacement activations for `mean`, `resample` and `patch`; items are matched to the item's layer (and point). May instead arrive on the node's `source` port. |
| `row` | object | — | For `patch`: which row of `source` to write in, e.g. `{"index": 0}`. |
| `condition` | object | — | Apply the item only at positions whose activation projects onto a direction above (or below) a threshold: `{"direction": …, "threshold": 0.0, "above": true}`. |
| `except` | bool | `false` | Invert the sets this item names: every layer, head or neuron BUT these. |
| `from` | object | — | For `patch`: `{"layer": 8, "point": "resid_post"}` — where the row is READ, when that differs from where it is written. |
| `pattern` | object | — | At `attn.scores`/`attn.weights`: `{"from": selector, "to": selector}` — the attention edge, source to destination. |
| `renormalize` | bool | `true` | At `attn.weights`: rescale a row that lost mass so it sums to one again. |
| `seed` | int | the block's `seed` | The seed `resample` draws with. |

The ops:

| `op` | Effect at the selected positions | Needs |
|---|---|---|
| `zero` | Set the activation to zero. | — |
| `mean` | Replace it with the mean of the `source` rows. | `source` |
| `resample` | Replace it with a row drawn at random from `source`. | `source` |
| `patch` | Replace it with one specific `source` row, or with `direction` itself. | `source` + `row`, or `direction` |
| `add` | Add `strength × direction`. | `direction` |
| `scale` | Multiply it by `strength`. | — |
| `clamp` | Clip its component along `direction` into [−\|`strength`\|, \|`strength`\|]. | `direction` |
| `project_out` | Remove its component along `direction`. | `direction` |
| `rotate` | Rotate it by `strength` radians in the plane of `direction` and `direction2`. | `direction`, `direction2` |

The older `intervene/ablate-layers` and `intervene/ablate-heads` and `intervene/steer` operations are special cases of this
grammar.

### Ablating the complement

`except: true` inverts the sets an item names. `{"heads": [3, 7],
"layers": [23], "except": true}` zeroes every head of layer 23 BUT 3 and 7:
where the direct ablation asks whether the named components are necessary
(faithfulness), its complement asks whether they are sufficient
(completeness), and a circuit claim wants both. It needs a set to invert,
and a whole-model point (`embed`, `logits`) has none.

### Cutting an attention edge

`positions` selects where an intervention acts; at `attn.scores` and
`attn.weights` a **`pattern`** selects the EDGE — `{"from": {"tokens":
["Paris"]}, "to": "last"}` is what the last position reads from the
subject, and zeroing it is the direct test of an attention-mediated
story. At `attn.scores`, `op: "zero"` writes −∞, because a score of zero
is a score and not a cut; at `attn.weights` it writes zero and
renormalises the rows that lost mass (`renormalize: false` leaves them
short, which is the ablation some papers mean).

### Reading one layer into another

A `patch` reads its row from the layer it writes to. `from: {"layer": 8}`
reads there instead, which is the patchscope: take a hidden state from one
prompt's layer 8 and write it into an explanation prompt's layer 20, then
see what the model says about it.

### Editing a weight instead of an activation

A spec item that names a **`parameter`** rather than a `point` edits the
model itself — `{"parameter": "layers.12.self_attn.o_proj", "op":
"project_out", "direction": {"$ref": {"bench": "you/lab/axis"}}}`. The two kinds compose
in one spec, and differ in scope: an activation edit lasts for one
forward pass, a weight edit for the node. The tensor is changed, every
record runs against the changed model, and the original is reinstalled
afterwards — the tensor itself, kept and put back, never a subtraction
that would not round-trip in bf16.

A sweep re-applies the weight edits at each factor, so `{"strength":
[0.5, 1.0]}` on a `project_out` removes half the direction and then all
of it. Factor 0 is the unedited model, which is the control the readout
compares against.

| Weight `op` | Effect | Needs |
|---|---|---|
| `zero` | Set the tensor to zero — the module stops contributing. | — |
| `scale` | Multiply it by `strength`. | — |
| `project_out` | Remove a direction from the side of the matrix that faces the residual stream: what the module WRITES (`o_proj`, `down_proj`) or what it READS (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`). `strength` 1.0 removes it entirely. `side` names the side where it is not implied. | `direction` |
| `truncate` | Keep the top `rank` singular directions and drop the rest — how much of the module survives being low-rank. | `rank` |

`parameter` takes the same names `weights/capture` does, `*` included:
one item can zero every layer's `o_proj`.
""",
    inputs=(
        In("records", "records/record",
           "The prompts to run, one forward pass each; a record's prompt is "
           "its `user`, `prompt` or `text` field. A record may also carry its "
           "own `tracked`, which takes precedence over the param of the same "
           "name.", many=True),
        In("intervention", "intervene/spec",
           "An intervention declared as an object — its `items` are spec "
           "items in the grammar `intervene/apply` documents — applied "
           "during every forward pass this node runs. A `direction` or "
           "`source` an item needs arrives on the port of that name. "
           "Where the node also has an inline `spec` or `intervention` "
           "param, the param wins when both are given.",
           required=False),
        In("direction", "direction/vector",
           "A direction that fills any spec item without one.", required=False),
        In("source", "activations/vector | intervene/readout",
           "A collection of `activations/vector` — a capture, intervened or "
           "not — that fills any `mean`/`resample`/`patch` item without one. "
           "A capture readout stored before 0.110.0 is read too.",
           many=True, required=False),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('intervene/readout', collection=True,
                  doc='For a decision readout, one item per record per factor: `id`, `coords`, `factor`, `entropy_bits`, `top` (the most likely next tokens, each `{token, p, logp}`) and `tracked` (name → `{token, p, logp}` for the tokens asked about). A capture readout is a capture: an `activations/vector` collection with one item per record per factor per hook point — `id`, `coords`, `factor`, `position`, `token`, `space` (at most 4096 values) — the shape `activations/capture` emits, so `geometry/compare`, `direction/regress` and another intervention\'s `source` read it unchanged. Either way the header carries `spec` (the list as run, with directions and sources replaced by their provenance), `weights` (the parameter edits, when any — so a reader knows the model was not the one on the shelf), `sweep` (the factors, including `0.0` when a control was added) and `readout`.',
                  otherwise=(Otherwise("activations/vector", collection=True, param="readout.type", equals="capture"),)),
    params=(
        P("spec", "list[object]",
          "The intervention items, applied together in one forward pass per "
          "record; the fields are described under *Spec items* above. Or the "
          "items arrive as an `intervene/spec` object on the `intervention` "
          "port; one or the other is required.", None, fields=(
              P("point", "string", "Where in the forward pass to act.", "resid_post", value="point"),
              P("parameter", "string",
                "Edit this weight instead of an activation, named as the module tree names it; "
                "`*` stands for one segment.",
                None),
              P("layers", "int | list[int] | \"all\"", "Which layers the item applies to.", "all"),
              P("positions", "selector", "Which token positions.", "last"),
              P("heads", "int | list[int]", "Only these attention heads, at a point with a head axis.", None),
              P("neurons", "int | list[int]", "Only these indices along the feature axis.", None),
              P("op", "string", "What to do there — the table above lists each op and what it needs.", "zero",
                choices=("zero", "mean", "resample", "patch", "add", "scale", "clamp", "project_out",
                         "rotate", "truncate")),
              P("strength", "float", "The item's magnitude, multiplied by each sweep factor.", 1.0),
              P("direction", "json",
                "The direction, usually a stored one (`{\"$ref\": …}`); or it arrives on the node's `direction` port.", None),
              P("direction2", "json", "For `rotate`: the second axis of the plane.", None),
              P("source", "json",
                "For `mean`, `resample` and `patch`: the replacement activations, or they arrive on the "
                "node's `source` port.",
                None),
              P("row", "object", "For `patch`: which row of `source` to write in.", None,
                fields=(P("index", "int", "The row's index.", 0),)),
              P("condition", "object",
                "Act only where the activation projects onto a direction above (or below) a threshold.", None,
                fields=(
                    P("direction", "json", "The direction projected onto."),
                    P("threshold", "float", "The projection's threshold.", 0.0),
                    P("above", "bool", "Act above the threshold; `false` acts below it.", True),
                )),
              P("except", "bool",
                "Invert the sets this item names — every layer, head or neuron BUT "
                "those — which measures a circuit's completeness where the direct "
                "ablation measures its faithfulness.",
                False),
              P("from", "object",
                "For `patch`: where the row is read, when that is not where it is "
                "written — the patchscope's move.", None, fields=(
                    P("layer", "int", "The source layer."),
                    P("point", "string", "The source point; the item's own by default.", None, value="point"),
                )),
              P("pattern", "object",
                "At `attn.scores` or `attn.weights`: the attention EDGE to act on — "
                "which source positions the selected destinations may attend to.",
                None, fields=(
                    P("from", "selector", "The source (key) positions."),
                    P("to", "selector", "The destination (query) positions; the item's `positions` by default.", None),
                )),
              P("renormalize", "bool",
                "At `attn.weights`, after zeroing: rescale the rows that lost mass so "
                "they sum to one again. The rows that lost none are left as they are.",
                True),
              P("seed", "int", "The seed `resample` draws with; the node's `seed` by default.", None),
              P("sweep_over", "list[string]",
                "Which of the node's sweep axes vary THIS item — `[\"layers\"]` sweeps "
                "this item's layers and leaves its strength alone. Every axis, by "
                "default.",
                None, choices=("strength", "layers", "heads", "positions", "neurons")),
              P("side", "string",
                "For a weight's `project_out`: the side facing the residual stream, where the module's "
                "name does not imply it.",
                None, choices=("in", "out")),
              P("rank", "int", "For a weight's `truncate`: how many singular directions to keep.", None),
          )),
        P("sweep", "object",
          "The axes to vary, each a list of values the spec items' field of "
          "that name takes in turn: `{\"strength\": [0.5, 1.0, 2.0]}` scales "
          "every item's `strength`; `{\"layers\": [0, 1, 2]}` runs the spec at "
          "each layer; `heads`, `positions` and `neurons` likewise. Several "
          "axes are a cartesian product, run with `strength` outermost, and "
          "each becomes a coordinate on every row — `factor`, `layer`, `head`, "
          "`position`, `neuron` — so a sweep is summarised, compared and "
          "plotted on the axis it varied.",
          {"strength": [1.0]}, fields=(
              P("strength", "list[float]", "The factors; `0` is the untouched model.", [1.0]),
              P("layers", "list[json]",
                "The layers, one cell each: `[0, 1, 2]`, or `[[0,1],[2,3]]` for "
                "groups. The coordinate is the layer, or `0+1` for a group.", None),
              P("heads", "list[json]", "The attention heads, one cell each.", None),
              P("positions", "list[selector]", "The position selectors, one cell each.", None),
              P("neurons", "list[json]", "The feature indices, one cell each.", None),
          )),
        P("control", "bool",
          "Add a factor‑0 run — the model untouched — to the sweep, so every "
          "record has a baseline (`factor: 0.0`). One run however many axes "
          "the sweep has: an unintervened pass does not depend on the layer "
          "the intervention would have named. Set `false` when the sweep "
          "already contains a strength of `0` or no baseline is wanted.",
          True),
        P("readout", "object",
          "What to read after the intervened forward pass. "
          "`{\"type\": \"decision\"}` records the next-token distribution at "
          "the last position. `{\"type\": \"capture\", \"points\": "
          "[\"blocks.14.resid_post\"], \"position\": \"last\"}` records the "
          "activation vectors at the named hook points instead — `position` "
          "is a selector naming one position — and the result is then an "
          "`activations/vector` collection, which any reader of a capture "
          "(including another intervention's `source`) takes. `readout.top_k` "
          "overrides the `top_k` param.",
          {"type": "decision"}, fields=(
              P("type", "string", "`decision` reads the next-token distribution; `capture` reads activations.",
                "decision", choices=("decision", "capture")),
              P("kind", "string", "The older spelling of `type`; read when `type` is absent.", None,
                choices=("decision", "capture")),
              P("top_k", "int", "For `decision`: overrides the node's `top_k`.", None),
              P("points", "list[string]",
                "For `capture`: the hook points to read, such as `blocks.14.resid_post`.", None),
              P("position", "selector", "For `capture`: the one position to read.", "last"),
          )),
        P("top_k", "int",
          "How many of the most likely next tokens to record per row in a "
          "decision readout.",
          5),
        P("tracked", "map[string, string]",
          "Tokens to follow by name: `{\"yes\": \" Yes\", \"no\": \" No\"}` "
          "records each one's probability and log-probability under "
          "`tracked.<name>`. Tokenized as a continuation, so include the "
          "leading space where the model would. A record's own `tracked` "
          "field takes precedence.",
          None),
    ),
    example={
        "model": {"$param": "model"},
        "spec": [{
            "point": "resid_post", "layers": [14], "positions": "last",
            "op": "add", "strength": 4.0,
            "direction": {"$ref": {"bench": "you/lab/direction"}},
        }],
        "sweep": {"strength": [0.5, 1.0, 2.0]},
        "tracked": {"answer": " Paris"},
        "top_k": 10,
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    """intervene/apply (task 000366): the declarative
    points × operations grammar with a decision or capture readout.
    Items are (record, sweep cell); spooled items are reused in
    canonical order under a matching fingerprint."""

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    reuse = dict(ctx.resume_items or {})
    emitted: dict[str, Any] = {}

    def _on_item(key, row):
        emitted[key] = row
        if ctx.on_item:
            ctx.on_item(key, row, key in reuse)

    out = run_intervene(model, records, params, inputs=inputs,
                            on_item=_on_item, on_start=ctx.on_start)
    if reuse:
        # Reproducible: a spooled row IS the row this loop produced.
        out["items"] = [reuse.get(f"{r['id']}:{r.get('cell') or r.get('factor')}", r)
                        for r in out["items"]]
    return out


def _parse_hook_name(name: str) -> tuple[int | None, str]:
    """`blocks.14.resid_post` → (14, "resid_post"); a whole-model point
    → (None, name)."""
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) >= 3 and parts[1].isdigit():
        return int(parts[1]), ".".join(parts[2:])
    return None, name


def _walk_cells(records: Sequence[Mapping[str, Any]], cells: Sequence[Cell],
                weight_items: Sequence[Mapping[str, Any]], model):
    """(record, cell) pairs, with any weight edits in scope.

    Without weight items this is the loop it always was: record outer,
    cell inner. With them the strength goes outside, because a weight
    edit is applied once for every record that runs under it — and the
    edit is undone before the next strength, and before the generator
    returns, whatever happens in between. A run that left a model edited
    would poison every later node in the job, which is the failure this
    `finally` exists for. `sweep_cells` orders strength outermost, so
    the cells sharing one are contiguous and each edit is made once.
    """
    if not weight_items:
        for record in records:
            for cell in cells:
                yield record, cell
        return

    from mechbench_compute import weights as weights_mod

    for factor in dict.fromkeys(c.factor for c in cells):
        handle = ([] if factor == 0.0
                  else weights_mod.edit_parameters(model.lm, weight_items, factor))
        try:
            for record in records:
                for cell in cells:
                    if cell.factor == factor:
                        yield record, cell
        finally:
            weights_mod.restore_parameters(model.lm, handle)


def run_intervene(model, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
        on_item: Callable | None = None,
        on_start: Callable[[int], None] | None = None) -> dict[str, Any]:
    from mechbench_compute.distill import render
    from mechbench_compute.interp import read_last_logp

    inputs = inputs or {}
    items = read_spec_items(params.get("spec"), inputs)
    if not items:
        raise SpecError("intervene needs a non-empty `spec` list, or an "
                        "intervene/spec on the `intervention` port")
    compiled = compile(model, items, inputs=inputs, seed=int(params.get("seed", 0)))
    specs, weight_items, filled = compiled.specs, compiled.weight_items, compiled.filled
    cells = sweep_cells(params)
    readout = dict(params.get("readout") or {"type": "decision"})
    rk = str(readout.get("type") or readout.get("kind") or "decision")
    if rk not in ("decision", "capture"):
        raise SpecError(f"unknown readout kind {rk!r}")
    top_k = int(readout.get("top_k", params.get("top_k", 5)))
    if not records:
        raise SpecError("intervene needs at least one record")
    if on_start:
        on_start(len(records) * len(cells))

    from mechbench_compute.interp import collect_tracked_ids
    from mechbench_compute.lexicon import kinds as K

    mid = S.model_id_of(model)
    rows: list[dict[str, Any]] = []
    # A weight edit is applied once per sweep factor, not once per
    # record: the tensor is the same for every record that runs under it,
    # and an SVD per record would be absurd. So the factor is the outer
    # loop when there are weight items, and the record loop is the same
    # body either way.
    for record, cell in _walk_cells(records, cells, weight_items, model):
            ids = render(model, record).array
            flat = [int(t) for t in np.array(ids).reshape(-1)]
            tokens = [model.tokenizer.decode([t]) for t in flat]
            tracked = collect_tracked_ids(model, record, tracked=params.get("tracked"))
            factor = cell.factor
            key = f"{record.get('id')}:{cell.key}"
            # The cell's axes ride on every row it produces: `factor` as
            # it always has, and a coordinate per other swept axis, so a
            # layer sweep groups on `layer` (000602).
            coords = {**record.get("coords", {}), **cell.coords}
            named = {"cell": cell.label} if cell.label else {}
            if factor == 0.0:
                ivs: list[Any] = []
            else:
                ivs = [SpecIntervention(compiled.at(cell), tokens, record)]
            if rk == "decision":
                res = model.run(ids, interventions=ivs)
                lp = read_last_logp(res.logits)
                row: dict[str, Any] = {
                    "id": record.get("id"), "coords": dict(coords),
                    "factor": factor, **named,
                    **S.distribution(lp, model.tokenizer, top_k=top_k, tracked=tracked),
                }
            else:
                points = [str(p) for p in readout.get("points", [])]
                if not points:
                    raise SpecError("capture readout needs `points`")
                res = model.run(ids, interventions=ivs, capture=points)
                L = len(flat)
                pidx = POS.one(readout.get("position", "last"), L, tokens=tokens,
                               record=record, prompt_len=L)
                # A capture under intervention IS a capture: one
                # `activations/vector` per hook point, the shape
                # `activations/capture` emits, so whatever reads a
                # capture — `geometry/compare`, `direction/regress`,
                # another intervention's `source` — reads this one too.
                # `factor` rides on each vector, since a sweep's rows
                # differ only by it (000599).
                for p in points:
                    t = res.cache[p]
                    v = t[0, pidx] if t.ndim == 3 else t[0]
                    # bf16 has no numpy buffer protocol: cast first.
                    arr = np.array(v.astype(mx.float32)).reshape(-1)[:4096]
                    cl, cp = _parse_hook_name(p)
                    row = S.vector(
                        arr, S.space(model=mid, layer=cl, point=cp, d=int(arr.size)),
                        id=record.get("id"), coords=dict(coords),
                        factor=factor, position=pidx, **named,
                        token=S.token(model.tokenizer, flat[pidx]))
                    rows.append(row)
                    if on_item:
                        on_item(f"{key}:{p}", row)
                continue
            rows.append(row)
            if on_item:
                on_item(key, row)
    # The weight edits ride in the header beside the activation spec, so
    # a reader of the result knows the model was not the one on the shelf
    # (task 000457: "the manifest says so").
    weights_wire = [dict(it) for it in weight_items] or None
    what = []
    if specs:
        what.append(f"{len(specs)} activation intervention(s) per forward")
    if weight_items:
        what.append(f"{len(weight_items)} weight edit(s) for the run, "
                    f"restored after")
    return K.collection(
        "activations/vector" if rk == "capture" else "intervene/readout", rows,
        spec=serialize_spec(filled),
        weights=weights_wire,
        sweep=sweep_as_run(params.get("sweep") or {}, cells),
        readout=rk,
        **({"model": mid, "position": str(readout.get("position", "last")),
            "points": [str(p) for p in readout.get("points", [])]} if rk == "capture" else {}),
        description=(
            f"{'; '.join(what)}. Factor 0 is the control; strengths scale "
            f"with the sweep factor."),
    )
