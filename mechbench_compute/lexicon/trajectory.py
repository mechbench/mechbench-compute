"""Ops over **trajectories**: a residual vector followed along an axis.

The logit lens computes one position's vector at every layer; a story
trace computes one layer's vector at every position. Both are the same
object read along a different axis, and a trajectory record makes that
explicit:

* `axis: "layers"` — one position's vector at every captured layer:
  where in *depth* a commitment forms.
* `axis: "positions"` — one layer's vector at every position along a
  sequence: where in a *text* it forms.

A trajectory is `rows` of `{id, label, step, layer, position, token,
norm, vector}`, with `step` indexing the axis. A projected trajectory
carries `coord` (the scalar coordinate along a direction) instead of
`vector`, which is what makes a corpus-scale trace small enough to
store.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Op, P

_TRAJ_IN = (
    "`trajectory` (by edge, or the `trajectory` param) — a `trajectory` or "
    "`trajectory_projection` record."
)

CAPTURE = Op(
    name="trajectory/capture",
    summary=(
        "Follow the residual stream along an axis — one position through "
        "every layer, or one layer along every position of a text — and "
        "record the vector (or its coordinate along a direction) at each "
        "step."
    ),
    description="""\
One forward pass per record. With `axis: "layers"` the block reads the
vector at one `position` after each of `layers`; step *k* is the *k*-th
layer. With `axis: "positions"` it reads one `layer`'s vector at each of the
chosen `positions`; step *k* is the *k*-th position read, which for
`"generated"` is the *k*-th generated token.

A corpus-scale trajectory is large (200 stories × 160 steps × the model
width), so there are three ways to keep it an object:

* `max_steps` — stop after that many steps per record.
* `reduce: "mean"` — one pooled vector per record over the selected
  `steps` window: what an outcome axis is fit on.
* `project` — a direction: read the scalar coordinate along it at capture
  time and emit no vectors at all. The trace itself, as numbers.

**Replay.** A record generated at trace fidelity carries the exact token ids
it was generated as and where generation began. `replay: "auto"` uses those
when present and tokenizes the text otherwise; re-tokenizing a generated
story can shift a token boundary, and the trace is the ground truth of what
the model saw.
""",
    inputs="""\
`records` — the texts, each a record with a prompt (`user`, `prompt` or
`text`) or a `trace`, and optionally `label`, `coords`, and a `subject`
when `position` is `"subject"`.

`project` (optional, by edge) — a direction record, the same as the
`project` param.
""",
    emits="""\
One `trajectory` record (or `trajectory_projection` when `project` was
given): `axis`, `point`, `layers`, `position`/`positions`, `d_model`,
`replay` (`"trace"`, `"text"` or `"mixed"`), and `rows` — per record per
step, `{id, label, step, layer, position, token, norm, vector}`, with
`coord` in place of `vector` when projected, `vocab_top` when asked for,
and `n_pooled` plus `steps` when reduced.
""",
    params=(
        P("axis", "string",
          "`\"layers\"`: one position through every layer. `\"positions\"`: "
          "one layer along the sequence.",
          "layers"),
        P("layers", "list[int] | \"all\"",
          "For `axis: \"layers\"`, the layers to step through.",
          "all"),
        P("layer", "int",
          "For `axis: \"positions\"`, the one layer to read along the "
          "sequence. (A one-element `layers` list is accepted too.)",
          None),
        P("position", "\"final\" | \"subject\" | int",
          "For `axis: \"layers\"`, which token to follow through the layers.",
          "final"),
        P("positions", "\"generated\" | \"all\" | object",
          "For `axis: \"positions\"`, which positions to step along: "
          "`\"generated\"` (from where generation began — the story, not the "
          "prompt), `\"all\"`, `{\"range\": [a, b]}` or `{\"after\": n}`.",
          "generated"),
        P("steps", "object",
          "A window of steps, counted from the trajectory's own start: "
          "`{\"range\": [5, 30]}` keeps steps 5 … 29. With `reduce`, the "
          "window the pooled vector is taken over.",
          None),
        P("max_steps", "int",
          "Stop after this many steps per record.",
          None),
        P("reduce", "string",
          "`\"mean\"`: emit one vector per record — the mean over the "
          "selected steps — instead of one per step.",
          None),
        P("project", "direction",
          "Read each step's scalar coordinate along this direction at "
          "capture time and emit `coord` instead of `vector`.",
          None),
        P("point", "string",
          "Which residual to read: `\"post\"` (after each layer) or `\"pre\"`.",
          "post"),
        P("replay", "string",
          "`\"auto\"`: use the record's stored token ids when it has a "
          "trace, else tokenize its text. `\"trace\"`: require the trace. "
          "`\"text\"`: always tokenize the text.",
          "auto"),
        P("vocab_top", "int",
          "Also record each step's top tokens through the unembedding "
          "(`vocab_top`), a lens reading per step. `0` records none.",
          0),
        P("label_coord", "string",
          "Which key of a record's `coords` (or `metadata`) to use as its "
          "`label` when it has no `label` field.",
          None),
        P("label_field", "string",
          "Which top-level field of a record to use as its `label` when it "
          "has neither `label` nor the coordinate — what `text/stats` "
          "writes when it annotates.",
          None),
        P("template", "string",
          "How a record's text is tokenized when it has no trace: "
          "`\"raw\"` or `\"chat\"`.",
          "raw"),
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$stories"},
        "axis": "positions",
        "layer": 12,
        "positions": "generated",
        "project": {"$fetch": "$outcome_axis"},
        "label_coord": "genre",
    },
)

PROJECT = Op(
    name="trajectory/project",
    summary=(
        "Project every step of a trajectory onto a direction — the trace of "
        "a text, or the funnel of a prompt, as one number per step."
    ),
    description="""\
Each row's vector becomes its dot product with the unit direction, recorded
as `coord`; the vectors are dropped unless `keep_vectors` is set. The
direction's layer is recorded but not enforced: projecting a layers-axis
trajectory onto a single-layer direction is the funnel read against one
axis, which is a legitimate question.
""",
    inputs=(
        f"{_TRAJ_IN} `direction` (by edge, or the `direction` param) — a "
        "direction record of the trajectory's width."
    ),
    emits=(
        "A `trajectory_projection` record: the input's fields, `direction` "
        "(layer, point, method and labels of the direction used), and `rows` "
        "with `coord`."
    ),
    params=(
        P("trajectory", "record",
          "The trajectory, when it does not arrive by edge.",
          None),
        P("direction", "direction",
          "The direction to project onto, when it does not arrive by edge.",
          None),
        P("keep_vectors", "bool",
          "Keep each row's `vector` beside its `coord`.",
          False),
    ),
    example={
        "trajectory": {"$fetch": "$trajectory"},
        "direction": {"$fetch": "$axis"},
    },
)

COMPARE = Op(
    name="trajectory/compare",
    summary=(
        "Compare two trajectories step by step — cosine, angle and norm "
        "ratio at each step, and the step at which they diverge."
    ),
    description="""\
Rows are paired by (`id`, `step`) — the same texts under two models, or two
conditions — or by `step` alone (`pair_by: "step"`) for two single-item
trajectories. For each pair the block reports the cosine between the two
vectors, the angle in degrees, both norms and their ratio. Across pairs it
reports the mean cosine per step and the **divergence step**: the first
step at which the mean cosine falls below `threshold`.

Both trajectories must have the same axis and carry vectors (not
projections).
""",
    inputs="`a` and `b` (by edge, or the params of the same names) — two `trajectory` records.",
    emits="""\
One `trajectory_comparison` record: `divergence_step`, `min_cosine_step`,
`per_step` (`{step, mean_cosine, n}`), `n_pairs`, and `rows` per pair with
`cosine`, `angle_deg`, `norm_a`, `norm_b`, `norm_ratio`.
""",
    params=(
        P("a", "record", "The first trajectory, when it does not arrive by edge.", None),
        P("b", "record", "The second trajectory, when it does not arrive by edge.", None),
        P("pair_by", "string",
          "`\"id\"`: pair rows with the same record id and step. `\"step\"`: "
          "pair by step alone.",
          "id"),
        P("threshold", "float",
          "The mean cosine below which the trajectories count as diverged.",
          0.9),
    ),
    example={"a": {"$fetch": "$base_traj"}, "b": {"$fetch": "$tuned_traj"}, "threshold": 0.9},
)

AGGREGATE = Op(
    name="trajectory/aggregate",
    summary=(
        "Group a trajectory's rows and reduce them — a mean trajectory with "
        "spread per step, one value per group over a window, or per-group "
        "mean vectors that `direction/from-vectors` can read directly."
    ),
    description="""\
Rows are grouped `by` a field (`label` by default, or `id`, or any field on
the rows), optionally restricted to a `steps` window, and reduced `as`:

* `"per_step"` — for each (group, step): with vectors, the mean vector, its
  norm, the mean norm of the members, and `spread` (mean cosine of members
  to the mean); with coordinates, mean and std.
* `"window"` — one value per group over every step in the window: with
  coordinates, mean/std/n (a story's late-window commitment, with
  `by: "id"`); with vectors, the mean vector.
* `"vectors"` — one `residual_vectors` row per group, the mean over its
  members and steps in the window, labelled by the group. This is the
  shape `direction/from-vectors` reads, so an outcome axis — "the
  lighthouse-story mean minus the other-story mean over tokens 5 … 30" —
  is this block followed by that one.
""",
    inputs=_TRAJ_IN,
    emits="""\
For `per_step` and `window`: a `trajectory` (vectors) or
`trajectory_summary` (coordinates) record with `aggregated: {by, as, steps}`
and `rows` per group (and per step). For `vectors`: a `residual_vectors`
record with one labelled row per group.
""",
    params=(
        P("trajectory", "record",
          "The trajectory, when it does not arrive by edge.",
          None),
        P("by", "string",
          "The row field to group on: `\"label\"`, `\"id\"`, or any field "
          "present on the rows.",
          "label"),
        P("as", "string",
          "The reduction: `\"per_step\"`, `\"window\"` or `\"vectors\"` — "
          "see above.",
          "per_step"),
        P("steps", "\"all\" | object",
          "Which steps take part: `\"all\"` or `{\"range\": [a, b]}`.",
          "all"),
    ),
    example={
        "trajectory": {"$fetch": "$trajectory"},
        "by": "label",
        "as": "vectors",
        "steps": {"range": [5, 30]},
    },
)

OPS: tuple[Op, ...] = (CAPTURE, PROJECT, COMPARE, AGGREGATE)
