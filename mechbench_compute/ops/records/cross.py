from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.lexicon._base import Op, Output, P

_COORDS_TYPE = "map[string, string | float]"


#: One factor of a crossed design: enumerated levels, sampled ones, or both.
_FACTOR_FIELDS = (
    P("name", "string", "The factor's name: the coordinate every record carries it under."),
    P("levels", "list[object]", "The enumerated levels.", None,
      fields=(
          P("key", "string", "The level's key: its coordinate value, and part of each record's id."),
          P("value", "string", "The level's text, for `records/fill`; the key by default.", None),
          P("coords", _COORDS_TYPE, "Further coordinates the level stamps on its records.", None),
      )),
    P("sampled", "object | list[object]",
      "Levels drawn by a generator, or several generators; each value depends only on the "
      "generator's `seed` and its index.",
      None,
      fields=(
          P("type", "string",
            "`noise` draws random strings; `words` draws word sequences. One of `type` or "
            "`kind` is required.",
            None, choices=("noise", "words")),
          P("kind", "string", "The older spelling of `type`; read when `type` is absent.", None,
            choices=("noise", "words")),
          P("size", "int", "How long each value is: characters for `noise`, words for `words`."),
          P("count", "int", "How many values to draw."),
          P("start", "int", "The first index, so a later node can extend the set.", 0),
          P("seed", "int | string", "The generator's own seed (not the node's).", 0),
          P("word_list", "list[string] | object",
            "For `words`: the words drawn from — inline, or a stored word list by reference.", None,
            fields=(P("words", "list[string]", "The words."),), stored="text/word-list"),
          P("wrap", "string", "A template each value is framed in, `{x}` for the value.", "{x}"),
          P("key_prefix", "string", "The level keys' prefix; `<type>-<size>` by default.", None),
          P("kind_coord", "string",
            "The coordinate naming the generator; `<factor name>_kind` by default.", None),
          P("coords", _COORDS_TYPE, "Further coordinates stamped on the sampled records.", None),
      )),
)


_FACTORS_DESC = """\
Each factor has a `name` and its **levels**: either enumerated —
`{"levels": [{"key": "noir", "value": "a noir story"}, …]}` — or **sampled**
by a generator — `{"sampled": {"type": "noise", "size": 12, "count": 5}}`
(random strings) or `{"type": "words", "size": 3, "count": 5, "word_list":
[…]}` (random word sequences) — or both. A generator's values depend only on
(`seed`, index), so `start`/`count` can grow the set later without changing
what exists; a `wrap` template such as `"Seed: {x}"` frames each value, and a
`key_prefix` names the level keys.

The output is one record per combination: `id` joins the level keys
(`noir-seed-2`), `coords` maps each factor name to its level key (plus any
`coords` the level or generator attached — generators stamp
`<name>_kind`), and `values` maps each factor name to its level's text,
ready for `records/fill`.
"""


OP = Op(
    name="records/cross",
    summary=(
        "Make one record per combination of experimental factors — the "
        "fully-crossed design, with each record carrying its coordinates."
    ),
    description=_FACTORS_DESC,
    inputs=(),
    output=Output('records/record', collection=True, doc='One record per combination: `{id, coords, values}`.'),
    params=(
        P("factors", "list[object]",
          "The factors to cross, each `{name, levels?, sampled?}` as "
          "described above. At least one is needed for a non-trivial "
          "design.",
          None, fields=_FACTOR_FIELDS),
        P("axes", "list[object]",
          "The older name for `factors`; read when `factors` is absent.",
          None, fields=_FACTOR_FIELDS),
    ),
    example={
        "factors": [
            {"name": "genre", "levels": [
                {"key": "noir", "value": "a noir detective story"},
                {"key": "fable", "value": "a fable with a moral"},
            ]},
            {"name": "seed", "sampled": {"type": "noise", "size": 8, "count": 4}},
        ],
    },
)


def run(ctx, inputs, params):
    return build_collection(cross_factors(params))


# The noise charset, fixed: a change here changes every sampled value
# a protocol has drawn from it.
SEED_CHARS = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "1234567890!@#$%^&*-_=+`~[]{}\\|;'\"/?.>,<")


def _sample_value(gen: Mapping[str, Any], index: int) -> str:
    """One sampled axis value, deterministic in (seed, index) alone —
    the range-splitting guarantee."""
    kind = gen.get("type") or gen["kind"]
    size = int(gen["size"])
    rng = random.Random(f"{gen.get('seed', 0)}:{index}")
    if kind == "noise":
        body = "".join(rng.choice(SEED_CHARS) for _ in range(size))
    elif kind == "words":
        # word_list: an inline list, or a fetched word_list object
        # payload ({kind: "word_list", words: [...]}) — the executor's
        # {"$ref": source} resolution hands the payload through whole.
        raw = gen.get("word_list") or []
        words = raw.get("words") if isinstance(raw, Mapping) else raw
        if not words:
            raise ValueError(
                "words generator needs word_list (inline list or a "
                "fetched word_list object)")
        body = " ".join(rng.choice(words) for _ in range(size))
    else:
        raise ValueError(f"unknown generator kind: {kind!r}")
    wrap = gen.get("wrap", "{x}")
    return wrap.replace("{x}", body)


def _materialize_levels(factor: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Materialize a factor to its [{key, value, coords}] levels. A
    factor may carry enumerated `levels`, `sampled` generators (one or
    a list), or both (enumerated first). Levels and generators may
    attach extra `coords` merged into each record (generators stamp a
    `<name>_kind` coordinate by default, so analysis groups by
    generator type, never by parsing keys). `values` is an accepted
    spelling of `levels`."""
    name = factor.get("name", "")
    out: list[dict[str, Any]] = []
    enumerated = factor.get("levels", factor.get("values"))
    if enumerated:
        out += [{"key": v["key"], "value": v.get("value", v["key"]),
                 "coords": dict(v.get("coords", {}))}
                for v in enumerated]
    sampled = factor.get("sampled")
    if sampled:
        gens = sampled if isinstance(sampled, list) else [sampled]
        for gen in gens:
            start = int(gen.get("start", 0))
            count = int(gen["count"])
            prefix = gen.get("key_prefix") or f"{gen.get('type') or gen['kind']}-{gen['size']}"
            kind_coord = gen.get("kind_coord", f"{name}_kind")
            extra = {kind_coord: prefix, **dict(gen.get("coords", {}))}
            out += [{"key": f"{prefix}-i{i}",
                     "value": _sample_value(gen, i),
                     "coords": dict(extra)}
                    for i in range(start, start + count)]
    if not out:
        raise ValueError(
            f"factor {factor.get('name')!r} has neither levels nor sampled")
    return out


def cross_factors(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fully crossed factors: the Cartesian product of every factor's
    levels, as coordinate-carrying records. `axes` is an accepted
    spelling of `factors`."""
    factors = params.get("factors", params.get("axes")) or []
    records: list[dict[str, Any]] = [
        {"id": "", "coords": {}, "values": {}}
    ]
    for axis in factors:
        name = axis["name"]
        vals = _materialize_levels(axis)
        nxt = []
        for rec in records:
            for v in vals:
                nxt.append({
                    "id": "-".join(
                        [p for p in [rec["id"], v["key"]] if p]),
                    "coords": {**rec["coords"], name: v["key"],
                               **v.get("coords", {})},
                    "values": {**rec["values"], name: v["value"]},
                })
        records = nxt
    for i, rec in enumerate(records):
        if not rec["id"]:
            rec["id"] = f"record-{i}"
    return records
