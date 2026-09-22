"""Ops over **records**: making them, filtering, joining, summarising and
presenting them. None of these touch a model; they are the plumbing
between the ops that do.

A record is a JSON object with an `id`, usually a `coords` object (the
experimental condition it belongs to — `{"genre": "noir", "seed": 3}`),
and whatever fields the op that made it wrote. Most of the ops here read
records from an edge onto their `records` port.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import In, Op

_RECORDS = In("records", "collection | records/table",
              "The records to work on: any collection of items — records, "
              "decision reads, vectors, verdicts, tree summaries — since every "
              "item has an id and its fields; a table's rows are read as records.",
              many=True)


OPS: tuple[Op, ...] = (
    
    
    
    
)
