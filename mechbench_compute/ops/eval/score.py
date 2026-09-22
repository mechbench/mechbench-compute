from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="eval/score",
    summary=(
        "Score prediction and reference fields on a record stream with any "
        "metric from the Hugging Face `evaluate` hub — accuracy, exact "
        "match, F1, BLEU — rather than reimplementing it."
    ),
    description="""\
The named metric is loaded from the hub and computed over every record's
`prediction` against its `reference`. Each numeric value the metric
returns becomes one row, stamped with `variant` so that a base run and an
adapter run union into one table for `records/subtract`. The metric library's
version is recorded on the table, because metric definitions change across
releases.
""",
    inputs=(
        In("records", "records/record",
           "Records carrying a `prediction` and a `reference`.", many=True),
    ),
    output=Output('records/table', collection=False, doc='One row per value the metric returned: `metric`, `variant`, `value`, `n`.'),
    params=(
        P("metric", "string",
          "The hub metric's name: `\"accuracy\"`, `\"exact_match\"`, "
          "`\"f1\"`, `\"bleu\"`, `\"rouge\"`, …"),
        P("kwargs", "map[string, json]",
          "Extra keyword arguments for the metric's `compute` — e.g. "
          "`{\"average\": \"macro\"}` for F1.",
          None),
        P("variant", "string",
          "A label for which model produced the predictions, stamped on "
          "every row as a coordinate.",
          "base"),
    ),
    example={"metric": "exact_match", "variant": "adapter"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/answers"}}},
)


def run(ctx, inputs, params):
    """eval/score — the HuggingFace `evaluate` metric layer: score
    prediction/reference fields on a record stream with any hub metric
    (accuracy, exact_match, f1, bleu, ...) instead of reimplementing
    them."""
    import evaluate

    metric_name = params.get("metric")
    if not metric_name:
        raise ValueError("eval/score needs params.metric")
    pf = "prediction"
    rf = "reference"
    variant = str(params.get("variant", "base"))
    recs = lexicon.items_of(inputs["records"])
    preds = [r.get(pf) for r in recs]
    refs = [r.get(rf) for r in recs]
    metric = evaluate.load(metric_name)
    out = metric.compute(predictions=preds, references=refs,
                         **(params.get("kwargs") or {}))
    rows = []
    for k in sorted(out or {}):
        v = out[k]
        if isinstance(v, (int, float)):
            rows.append({"id": f"{metric_name}:{k}:{variant}",
                         "metric": k, "variant": variant,
                         "value": float(v), "n": len(recs),
                         "coords": {"metric": k,
                                    "variant": variant}})
    return {"kind": "records/table",
            "name": params.get("name", f"hf-{metric_name}"),
            "description": (
                f"huggingface evaluate {evaluate.__version__}: "
                f"{metric_name} over {len(recs)} records "
                f"({pf} vs {rf})"),
            "row_axis": "metric",
            "columns": [
                {"name": "id", "dtype": "string"},
                {"name": "metric", "dtype": "string"},
                {"name": "variant", "dtype": "string"},
                {"name": "value", "dtype": "number"},
                {"name": "n", "dtype": "number"}],
            "rows": rows}
