"""Ops over **trajectories**: a residual vector followed along an axis.

The logit lens computes one position's vector at every layer; a story
trace computes one layer's vector at every position. Both are the same
object read along a different axis, and a trajectory record makes that
explicit:

* `axis: "layers"` — one position's vector at every captured layer:
  where in *depth* a commitment forms.
* `axis: "positions"` — one layer's vector at every position along a
  sequence: where in a *text* it forms.

A trajectory is a collection of `trajectory/point` — vector items
(`id`, `coords`, `space`, `vector`, `norm`) with `step`, `position` and
the `token` read — `step` indexing the axis. A projected trajectory is a
collection of `activations/coordinate`: `coord` (the scalar coordinate
along a direction) in place of the vector, which is what makes a
corpus-scale trace small enough to store.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Emits, In, Op, P
from mechbench_compute.lexicon.model import _POSITIONS_DOC, _RESIDUAL_POINT, ADAPTER

_TRAJECTORY = In("trajectory", "trajectory/point",
                 "A trajectory: a collection of points with vectors.", many=True)

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
* `pool` — one vector per record, the reduction over a window of its
  steps (`{"reduce": "mean", "over": {"range": [5, 30]}}`): what an
  outcome axis is fit on.
* `project` — a direction: read the scalar coordinate along it at capture
  time and emit no vectors at all. The trace itself, as numbers.

Every item carries the record's `coords`, which is what
`trajectory/aggregate` groups on. A measurement a record carries as a
field (what `text/stats` writes) becomes a coordinate through
`records/rename` — `{"opening": "coords.opening"}` — before the capture.

**Replay.** A record generated at trace fidelity carries the exact token ids
it was generated as and where generation began. `replay: "auto"` uses those
when present and tokenizes the text otherwise; re-tokenizing a generated
story can shift a token boundary, and the trace is the ground truth of what
the model saw.
""",
    inputs=(
        In("records", "records/record",
           "The texts, each a record with a prompt (`user`, `prompt` or "
           "`text`) or a `trace`, and optionally `coords` and a `subject` "
           "when `position` is `\"subject\"`. A document collection is read "
           "the same way.", many=True),
        In("project", "direction/vector",
           "A direction: read each step's scalar coordinate along it at "
           "capture time and emit coordinates instead of vectors.",
           required=False),
        ADAPTER,
    ),
    emits=Emits('trajectory/point', collection=True, doc='One item per record per step: `{id, coords, space, step, position, token, norm, vector}`, with `vocab` (a distribution) when asked for, and `n_pooled` plus `steps` when reduced. With `project`, a collection of `activations/coordinate` instead: `coord` and the direction\'s identity in place of the vector, `projected: true` in the header. The header carries `axis`, `point`, `layers`, `position`/`positions`, `d_model` and `replay` (`"trace"`, `"text"` or `"mixed"`).'),
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
        P("position", "selector",
          f"For `axis: \"layers\"`, which token to follow through the layers: "
          f"{_POSITIONS_DOC}, resolving to one position.",
          "last"),
        P("positions", "selector",
          f"For `axis: \"positions\"`, which positions to step along: "
          f"{_POSITIONS_DOC} — `\"generated\"` is the story, not the prompt.",
          "generated"),
        P("max_steps", "int",
          "Stop after this many steps per record.",
          None),
        P("pool", "object",
          "Emit one vector per record — the reduction over a window of its "
          "steps — instead of one per step: `{\"reduce\": \"mean\" | \"max\", "
          "\"over\": <selector>}`, `over` counting steps from the trajectory's "
          "own start, so `{\"range\": [5, 30]}` is steps 5 … 29.",
          None),
        _RESIDUAL_POINT,
        P("replay", "string",
          "`\"auto\"`: use the record's stored token ids when it has a "
          "trace, else tokenize its text. `\"trace\"`: require the trace. "
          "`\"text\"`: always tokenize the text.",
          "auto"),
        P("vocab_top", "int",
          "Also record each step's distribution through the unembedding "
          "(`vocab`, with this many top tokens), a lens reading per step. "
          "`0` records none.",
          0),
    ),
    example={
        "model": "$model",
        "axis": "positions",
        "layer": 12,
        "positions": "generated",
    },
    example_inputs={"records": {"$fetch": "$stories"}, "project": {"$fetch": "$outcome_axis"}},
)

PROJECT = Op(
    name="trajectory/project",
    summary=(
        "Project every step of a trajectory onto a direction — the trace of "
        "a text, or the funnel of a prompt, as one number per step."
    ),
    description="""\
Each point's vector becomes its dot product with the unit direction,
recorded as `coord` on an `activations/coordinate` item that keeps the
point's `step`, `position`, `token` and `space`; the vectors are dropped
unless `keep_vectors` is set. The direction's layer is recorded but not
enforced: projecting a layers-axis trajectory onto a single-layer direction
is the funnel read against one axis, which is a legitimate question.
""",
    inputs=(
        _TRAJECTORY,
        In("direction", "direction/vector",
           "A direction of the trajectory's width."),
    ),
    emits=(
        Emits('activations/coordinate', collection=True, doc="One coordinate per input point: `id`, `coords`, `space`, `step`, `position`, `token`, the `direction`'s identity and `coord` (the vector kept as well under `keep_vectors`); the header repeats the input's, with `projected: true`.")
    ),
    params=(
        P("keep_vectors", "bool",
          "Keep each row's `vector` beside its `coord`.",
          False),
    ),
    example={"keep_vectors": False},
    example_inputs={
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
    inputs=(
        In("a", "trajectory/point", "The first trajectory.", many=True),
        In("b", "trajectory/point", "The second trajectory.", many=True),
    ),
    emits=Emits('trajectory/comparison', collection=True, doc='One item per pair: `cosine`, `angle_deg`, `norm_a`, `norm_b`, `norm_ratio`. The header carries `divergence_step`, `min_cosine_step`, `per_step` (`{step, mean_cosine, n}`) and `n_pairs`.'),
    params=(
        P("pair_by", "string",
          "`\"id\"`: pair rows with the same record id and step. `\"step\"`: "
          "pair by step alone.",
          "id"),
        P("threshold", "float",
          "The mean cosine below which the trajectories count as diverged.",
          0.9),
    ),
    example={"threshold": 0.9},
    example_inputs={"a": {"$fetch": "$base_traj"}, "b": {"$fetch": "$tuned_traj"}},
)

AGGREGATE = Op(
    name="trajectory/aggregate",
    summary=(
        "Group a trajectory's rows and reduce them — a mean trajectory with "
        "spread per step, one value per group over a window, or per-group "
        "mean vectors that `direction/from-vectors` can read directly."
    ),
    description="""\
Items are grouped `by` a coordinate (`label` by default — which also reads
the older `label` field — or any coordinate such as `genre`), by `id`, or by
a field on the items, optionally restricted to a `steps` window, and
reduced `as`:

* `"per_step"` — for each (group, step): with vectors, the mean vector, its
  norm, the mean norm of the members, and `spread` (mean cosine of members
  to the mean); with coordinates, mean and std.
* `"window"` — one value per group over every step in the window: with
  coordinates, mean/std/n (a story's late-window commitment, with
  `by: "id"`); with vectors, the mean vector.
* `"vectors"` — one `activations/vector` per group, the mean over its
  members and steps in the window, with the group as its coordinate on the
  `by` axis. This is the shape `direction/from-vectors` reads, so an
  outcome axis — "the lighthouse-story mean minus the other-story mean over
  tokens 5 … 30" — is this block followed by that one (with `axis` set to
  the same `by`).
""",
    inputs=(
        In("trajectory", "trajectory/point | activations/coordinate",
           "A trajectory of points, or a projected one of coordinates, read "
           "the same way.", many=True),
    ),
    emits=Emits('trajectory/summary', collection=True, doc="For `per_step` and `window`: one item per group (and per step), with a mean `vector` or a mean `coord` as the input had; the header repeats the input's and adds `aggregated: {by, as, steps}`. For `vectors`: a collection of `activations/vector` instead, one item per group with the group on the `by` coordinate."),
    params=(
        P("by", "string",
          "What to group on: a coordinate (`\"label\"`, `\"genre\"`), "
          "`\"id\"`, or a field present on the items.",
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
        "by": "label",
        "as": "vectors",
        "steps": {"range": [5, 30]},
    },
    example_inputs={"trajectory": {"$fetch": "$trajectory"}},
)

OPS: tuple[Op, ...] = (CAPTURE, PROJECT, COMPARE, AGGREGATE)
