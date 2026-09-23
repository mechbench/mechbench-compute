from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="eval/benchmark",
    requires="mlx-local",
    summary=(
        "Run standard benchmark tasks from the lm-evaluation-harness against "
        "the model — through mechbench's own model, so pinned revisions and "
        "fused adapters count."
    ),
    description="""\
Each named task is evaluated by the harness with the bound model wrapped as
its backend, so anything the platform knows how to load — a pinned
revision, a stacked adapter, a merged checkpoint — is what gets measured.
Every (task, metric) the harness reports becomes one row, stamped with
`variant`, ready for `records/union` and `records/subtract` against another run.

The harness version is recorded on the table: prompt templates change
between its releases, so the version is part of the measurement.
""",
    inputs=(In("adapter", "adapter/lora",
               "A LoRA adapter to fuse on top of the model for this node only — "
               "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
               "reference, or a stored adapter. Fuses last, on top of any "
               "adapters the model reference itself carries; `adapter_scale` "
               "scales this one.",
               required=False),),
    output=(
        Output('records/table', collection=False, doc='One row per (task, metric): `task`, `metric`, `variant`, `value`, `stderr`, `n`.')
    ),
    params=(
        P("tasks", "list[string]",
          "The harness task names to run, e.g. `[\"hellaswag\", "
          "\"arc_easy\"]`."),
        P("limit", "int",
          "Evaluate only this many examples per task — for a quick look. "
          "By default every example.",
          None),
        P("num_fewshot", "int",
          "How many in-context examples to give. By default each task's "
          "own setting.",
          None),
        P("variant", "string",
          "A label for which model was measured, stamped on every row.",
          "base"),
    ),
    example={"model": {"$param": "model"}, "tasks": ["hellaswag", "arc_easy"], "limit": 200, "variant": "base"},
)


def run(ctx, inputs, params):
    """eval/benchmark — the lm-eval-harness bridge: run standard
    benchmark tasks against the bound model THROUGH OUR OWN Model
    (lm_bridge.MechbenchLM), so revision pinning, VLM-shaped
    checkpoints, and adapter fusion (the standard `adapter` input port)
    all come free. Publishes a metric table whose rows carry (task,
    metric, variant) coords — composable straight into
    `records/union` and `records/subtract` for base-vs-adapter deltas.

    Harness versions ride in the description: prompt templates
    change across lm-eval releases, so the version IS part of the
    measurement.
    """
    import lm_eval

    pass
    from mechbench_compute.lm_bridge import MechbenchLM

    tasks = list(params.get("tasks") or [])
    if not tasks:
        raise ValueError("eval/benchmark needs params.tasks (list of "
                         "lm-eval task names)")
    limit = params.get("limit")
    num_fewshot = params.get("num_fewshot")
    variant = str(params.get("variant", "base"))
    seed = int(params.get("seed", 0))

    model = ctx.model(params.get("model"))
    wrapper = MechbenchLM(model)

    if ctx.on_start:
        ctx.on_start(len(tasks))
    results_all: dict = {}
    nsamples_all: dict = {}
    for i, task in enumerate(tasks):
        out = lm_eval.simple_evaluate(
            model=wrapper,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=None,
            fewshot_random_seed=seed,
        )
        results_all.update(out["results"])
        nsamples_all.update(out.get("n-samples") or {})
        if ctx.on_item:
            ctx.on_item()

    recs = build_metric_records(results_all, nsamples_all,
                                variant=variant)
    columns = [{"name": "id", "dtype": "string"},
               {"name": "task", "dtype": "string"},
               {"name": "metric", "dtype": "string"},
               {"name": "variant", "dtype": "string"},
               {"name": "value", "dtype": "number"},
               {"name": "stderr", "dtype": "number"},
               {"name": "n", "dtype": "number"}]
    rows = [{"id": r["id"], **r["coords"],
             "value": r.get("value"), "stderr": r.get("stderr"),
             "n": r.get("n"), "coords": r["coords"]}
            for r in recs]
    import mlx_lm

    return {"kind": "records/table",
            "name": params.get("name", f"suite-{variant}"),
            "description": (
                f"lm-eval {lm_eval.__version__} via MechbenchLM "
                f"(mlx_lm {mlx_lm.__version__} unused for load); "
                f"tasks={','.join(tasks)} limit={limit} "
                f"fewshot={num_fewshot}"),
            "row_axis": "task-metric",
            "columns": columns,
            "rows": rows}


def build_metric_records(results: Mapping[str, Any],
                         n_samples: Mapping[str, Any] | None = None,
                         variant: str = "base") -> list[dict[str, Any]]:
    """Shape lm-eval-harness `results` (task -> {"acc,none": v,
    "acc_stderr,none": s, ...}) into coord-carrying records:
    one record per (task, metric) with value/stderr/n and coords
    {task, metric, variant} — composable straight into `records/union`
    and `records/subtract` for base-vs-adapter deltas."""
    out: list[dict[str, Any]] = []
    for task in sorted(results):
        metrics = results[task]
        ns = (n_samples or {}).get(task) or {}
        n = ns.get("effective", ns.get("original"))
        stderrs = {}
        values = {}
        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            name = key.split(",", 1)[0]
            if name in ("sample_len",):  # harness bookkeeping, not a metric
                continue
            if name.endswith("_stderr"):
                stderrs[name[: -len("_stderr")]] = float(val)
            else:
                values[name] = float(val)
        for name in sorted(values):
            rec: dict[str, Any] = {
                "id": f"{task}:{name}:{variant}",
                "coords": {"task": task, "metric": name,
                           "variant": variant},
                "value": values[name],
            }
            if name in stderrs:
                rec["stderr"] = stderrs[name]
            if n is not None:
                rec["n"] = int(n)
            out.append(rec)
    return out
