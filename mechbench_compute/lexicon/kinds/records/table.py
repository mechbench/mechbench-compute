from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "records/table",
    "A presentation table: named, typed columns and rows of scalars. Singular; its rows are typed by its columns, not by a kind.",
    fields={
        "name": F("string", "A label for the table."),
        "description": F("string", "Free text beside the name."),
        "row_axis": F("string", "What one row stands for."),
        "columns": F("array", "`{name, dtype}` per column, in order.", items={"type": "object"}),
        "rows": F("array", "One object per row, keyed by column name.", items={"type": "object"}),
        "n_missing": F("integer", "Records skipped for lacking the summarised field, when any were."),
    },
    required=("columns", "rows"),
    renderer={"primitive": "table", "field_map": {"rows": "rows"}},
    doc="A table is for reading, not for further computation: its rows are plain objects typed by `columns`, "
        "not items of a kind, so nothing downstream reads a table but a chart and a person. `records/tabulate` "
        "makes one from any collection (coordinates become the leading columns), and `records/group` summarises records into one "
        "directly.",
)
