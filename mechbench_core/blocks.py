"""Stdlib building blocks (epic 000258, arc B): the pure blocks —
Grid and Template — plus the block registry the pipeline executor
resolves op refs against.

Design rules these implement:

- **Grid**: axes -> records with coordinates. An axis either
  enumerates values or samples them from a seeded generator. Records
  carry {id, coords, values}: `coords` are the axis value KEYS
  (structured, for GroupBy/PairedDelta — never parsed from the id),
  `values` the substitution payloads (which carry their own
  whitespace; there is no hidden joining logic).
- **Range-splitting invariance**: sampled axes derive one rng per
  instance from (seed, index), so generate(seed, 0, 1000) equals
  generate(seed, 0, 100) + generate(seed, 100, 900). Incremental
  dataset growth is the same node run over a later range.
- **Template**: records x named string templates -> the same records
  with instantiated string fields. Source-agnostic: records may come
  from a Grid or from any record stream (dataset rows, later).
  No block in this module knows what a "prompt" is.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping

# The original ai-randomness noise charset, reproduced exactly.
SEED_CHARS = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "1234567890!@#$%^&*-_=+`~[]{}\\|;'\"/?.>,<")


def _sample_value(gen: Mapping[str, Any], index: int) -> str:
    """One sampled axis value, deterministic in (seed, index) alone —
    the range-splitting guarantee."""
    kind = gen["kind"]
    size = int(gen["size"])
    rng = random.Random(f"{gen.get('seed', 0)}:{index}")
    if kind == "noise":
        body = "".join(rng.choice(SEED_CHARS) for _ in range(size))
    elif kind == "words":
        words = gen.get("word_list") or []
        if not words:
            raise ValueError("words generator needs word_list")
        body = " ".join(rng.choice(words) for _ in range(size))
    else:
        raise ValueError(f"unknown generator kind: {kind!r}")
    wrap = gen.get("wrap", "{x}")
    return wrap.replace("{x}", body)


def _axis_values(axis: Mapping[str, Any]) -> list[dict[str, str]]:
    """Materialize an axis to [{key, value}]."""
    if "values" in axis:
        return [{"key": v["key"], "value": v.get("value", v["key"])}
                for v in axis["values"]]
    if "sampled" in axis:
        gen = axis["sampled"]
        start = int(gen.get("start", 0))
        count = int(gen["count"])
        prefix = gen.get("key_prefix") or f"{gen['kind']}-{gen['size']}"
        return [{"key": f"{prefix}-i{i}", "value": _sample_value(gen, i)}
                for i in range(start, start + count)]
    raise ValueError(f"axis {axis.get('name')!r} has neither values nor sampled")


def grid(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cross-product of axes into coordinate-carrying records."""
    axes = params.get("axes") or []
    records: list[dict[str, Any]] = [
        {"id": "", "coords": {}, "values": {}}
    ]
    for axis in axes:
        name = axis["name"]
        vals = _axis_values(axis)
        nxt = []
        for rec in records:
            for v in vals:
                nxt.append({
                    "id": "-".join(
                        [p for p in [rec["id"], v["key"]] if p]),
                    "coords": {**rec["coords"], name: v["key"]},
                    "values": {**rec["values"], name: v["value"]},
                })
        records = nxt
    for i, rec in enumerate(records):
        if not rec["id"]:
            rec["id"] = f"record-{i}"
    return records


def template(records: list[dict[str, Any]],
             params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Instantiate named string templates against each record's values.
    `{axis-name}` placeholders substitute; everything else is verbatim."""
    templates: Mapping[str, str] = params.get("templates") or {}
    out = []
    for rec in records:
        fields = {}
        for fname, tmpl in templates.items():
            s = str(tmpl)
            for axis, value in rec.get("values", {}).items():
                s = s.replace("{" + axis + "}", str(value))
            fields[fname] = s
        out.append({"id": rec["id"], "coords": dict(rec.get("coords", {})),
                    **fields})
    return out


# --- registry ---------------------------------------------------------------

# Op ref -> callable. Pure blocks take (inputs, params); model blocks
# are registered by the executor host (the runner), which owns model
# lifecycle. Descriptor objects at ~canonical/ops/... arrive with the
# 000248 registry arc; until then this in-code table is the resolver.
PURE_BLOCKS: dict[str, Callable[..., Any]] = {
    "~canonical/ops/grid/1":
        lambda inputs, params: grid(params),
    "~canonical/ops/template/1":
        lambda inputs, params: template(inputs["records"], params),
}
