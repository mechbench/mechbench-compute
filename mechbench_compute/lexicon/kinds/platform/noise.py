from __future__ import annotations

from mechbench_compute.lexicon._base import Draw, Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "platform/noise",
    "How far one numeric field of one operation moves across seeds and machines: the noise floor a replication is judged against.",
    doc="Written by `records/measure-noise` from runs of one protocol on several seeds, several "
        "machines, or both: one record per (architecture, operation, field, dtype, machine class). "
        "For every record key the runs share, the field's values are set side by side (a list of "
        "numbers element by element); `spread` is the largest difference between the smallest and "
        "the largest of them over every key and element, and `relative_spread` the largest such "
        "difference divided by the size of the value it was taken at. `records/diff` reads a "
        "collection of these on its `noise` port and reports each difference in units of the floor. "
        "Runs on one seed and several machines measure what the machines alone do; runs on several "
        "seeds of a sampling op measure the sampling as well, a wider floor.",
    extends="records/record",
    fields={
        "architecture": F("string", "The checkpoint's architecture, its config.json `model_type` (`gemma4`)."),
        "model": F("string", "The checkpoint as `repo@revision`, or the repo alone; empty when the runs "
                   "did not name one."),
        "operation": F("string", "The operation whose output was measured (`text/generate`)."),
        "field": F("string", "The field's path in the operation's records, dotted as `records/diff` "
                   "names it (`metadata.logprob`); `*` in a name matches within it."),
        "dtype": F("string", "The weights' dtype or quantization (`bfloat16`, `4bit`)."),
        "machine_class": F("string", "The class of machine the floor holds for (`apple-silicon`)."),
        "spread": F("number", "The largest absolute difference observed between runs at one key."),
        "relative_spread": F("number", "The largest difference between runs at one key, over the "
                             "size of the value there (the larger magnitude); 0 where every value "
                             "was 0."),
        "n": F("integer", "How many values were compared: runs × keys × elements."),
        "records": F("integer", "How many record keys the runs shared and the field was measured at."),
        "machines": F("array", "The machines the runs were taken on, one per distinct machine.",
                      items={"type": "string"}),
        "seeds": F("array", "The seeds the runs were taken at, one per distinct seed; empty when "
                   "the runs did not say.", items={"type": "integer"}),
    },
    required=("id", "architecture", "operation", "field", "dtype", "machine_class", "spread",
              "relative_spread", "n", "records", "machines", "seeds"),
    key=("architecture", "operation", "field", "dtype", "machine_class"),
    platform=True,
    header={
        "compute": "The compute version that measured the floor.",
        "runs": "One entry per run measured, in order: `{machine, seed, records}`.",
        "matched_by": "What matched a record in one run to its counterparts in the others.",
        "excluded": "The field patterns left out, the moving fields among them.",
        "not_numeric": "The fields whose values differed between runs but are not numbers of one "
                       "shape, with how many keys they differed at; they have no floor.",
    },
    speak="{operation} {field} on {architecture} {dtype}: ±{spread} ({relative_spread} relative) "
          "over {n} values",
    draw=Draw(mark="bar", encoding={"x": "field", "y": "spread", "series": "operation"}),
)
