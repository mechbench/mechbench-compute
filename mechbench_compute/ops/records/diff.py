from __future__ import annotations

import fnmatch
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

#: Fields whose values move on every run whatever the computation did:
#: when it happened, how long the call took, the ids a provider or the
#: platform minted, the rate-limit counters, where a job's results sit,
#: and the compute version with its `+src` stamp. Each is a dotted path
#: matched against the end of a field's path, and one naming a mapping
#: excludes everything under it.
MOVING = (
    "created_at", "updated_at", "started_at", "finished_at", "timestamp",
    "latency_ms", "throttled_seconds", "attempts",
    "response_id", "request_id", "job_id", "jobId", "run_id",
    "rate_limits", "result_path", "resultPath",
    "produced_by.version", "provenance.inputs",
)

RELATIONS = ("added", "removed", "extends", "truncates", "changed", "any")

OP = Op(
    name="records/diff",
    summary=(
        "Compare two collections record by record, matched by key: whether "
        "they are identical, the records only one side has, and for every "
        "record that differs, which fields moved, from what to what, and by "
        "how much."
    ),
    description="""\
The comparison a re-run needs: the same protocol run twice, before and
after a change, and the question whether anything moved. Records are
matched by **key**, never by position — `id` by default, or named
coordinates (`key: ["prompt", "sample"]`) when the two runs number their
records differently but share a design. A key held by two records on one
side is refused: the fix is to key on more coordinates.

Each record's fields are compared by path (`metadata.sampling.ended`); a
list is one value. A field present on one side only is `added` or
`removed`; a string the other side's string begins with is `extends` (a
text that ran on further) or `truncates`; anything else that differs is
`changed`, and a number that changed carries its `delta`, b minus a. A
kind written in a retired spelling is the same kind as its current name
(`~canonical/kinds/text` and `text/document`), and is not a difference. A
record is `identical`, `extends` (every difference is an extension),
`differs`, or on one side only (`only_a`, `only_b`).

**What is left out.** Fields that move on every run whatever the
computation did — timestamps, latencies, provider and job ids, rate-limit
counters, result paths, the compute version and its `+src` stamp — are
excluded by default and listed in the answer, with how many records they
differed on, so nothing is hidden silently. `exclude` adds patterns;
`exclude_moving: false` compares everything. `fields` narrows the
comparison to the fields named, and what is under them.

**What may differ.** `allow` states which differences are expected, and
`holds` says whether every difference was: a rule names a field, the
relation allowed, and optionally `when`, values the record on side `a`
must have for the rule to apply. A regeneration with a larger token
allowance is `fields: ["text"]` and `allow: [{field: "text", relation:
"extends", when: {"metadata.sampling.ended": "max_tokens"}}]`: every
story that ended on its own is identical, and every story that was cut
now runs on from where it stopped.

**By group.** `by` names coordinates to summarise within: for each
combination, each side's count, and the mean and total of every numeric
field on each side with the difference of the means — a corpus's rates,
side by side.

The two sides arrive on ports `a` (the earlier, or the reference) and `b`.
To compare two stored runs, reference their nodes' results; the node then
cites both. The command line's `run diff` is this comparison, run where
the command runs.
""",
    inputs=(
        In("a", "collection | records/table",
           "The first side: the earlier run's records, or the reference."),
        In("b", "collection | records/table",
           "The second side, compared against the first."),
    ),
    output=Output(
        "records/record", collection=True,
        doc="One record per key that is not identical on both sides: `id`, "
            "`coords` (the key's values), `status` (`extends`, `differs`, "
            "`only_a`, `only_b`), and `fields`, each differing path's "
            "`relation`, `a`, `b` and, for a number, `delta`; with `allow`, "
            "whether the record's differences were `allowed`. The header's "
            "`diff` has the verdicts (`identical`, the payloads equal in "
            "full and in order; `equivalent`, nothing differs once matched "
            "and excluded; `holds`), the counts by status, each field's "
            "counts by relation with the largest and mean delta, the header "
            "fields that differ, what was excluded and what it hid, and the "
            "groups under `by`."),
    params=(
        P("key", "\"id\" | list[string]",
          "What matches a record to its counterpart: the record ids, or the "
          "named coordinates.",
          "id"),
        P("fields", "list[string]",
          "Compare only these fields, and what is under them. None compares "
          "every field.",
          None),
        P("exclude", "list[string]",
          "Fields to leave out as well, by the end of their path "
          "(`reasoning`, `metadata.call.usage`); `*` matches within a name.",
          None),
        P("exclude_moving", "bool",
          "Leave out the fields that move on every run whatever the "
          "computation did: timestamps, latencies, ids, rate limits, result "
          "paths, the compute version.",
          True),
        P("allow", "list[map[string, json]]",
          "Differences that are expected, each `{field, relation, when?}`: "
          "`relation` is one of added, removed, extends, truncates, changed "
          "or any (or a list of them); `when` maps a field to the value (or "
          "values) the `a` record must have for the rule to apply. `holds` "
          "is true when no record is on one side only and every difference "
          "is allowed.",
          None),
        P("by", "list[string]",
          "Coordinates to summarise within: each side's count, and the mean "
          "and total of every numeric field, per combination.",
          None),
    ),
    example={"key": ["prompt", "sample"], "fields": ["text"],
             "allow": [{"field": "text", "relation": "extends",
                        "when": {"metadata.sampling.ended": "max_tokens"}}]},
    example_inputs={"a": {"$ref": {"bench": "you/lab/results/j_before/gen"}},
                    "b": {"$ref": {"bench": "you/lab/results/j_after/gen"}}},
)


def run(ctx, inputs, params):
    return diff_collections(inputs["a"], inputs["b"], params)


def diff_collections(a: Any, b: Any, params: Mapping[str, Any], *,
                     provenance: tuple[Any, Any] | None = None) -> dict[str, Any]:
    """`records/diff`: two collections compared record by record, matched
    by key. `provenance`, the two sides' envelopes' provenance when the
    caller fetched them, is compared as the header is."""
    key = params.get("key") or "id"
    key = [key] if isinstance(key, str) else list(key)
    selected = list(params.get("fields") or [])
    patterns = (list(MOVING) if params.get("exclude_moving", True) else []) \
        + list(params.get("exclude") or [])
    rules = [_check_rule(r) for r in (params.get("allow") or [])]
    by = list(params.get("by") or [])

    items_a, items_b = read_items(a), read_items(b)
    side_a, side_b = _index_by_key(items_a, key, "a"), _index_by_key(items_b, key, "b")

    hidden: dict[str, int] = {}
    fields: dict[str, dict[str, Any]] = {}
    counts = {"identical": 0, "extends": 0, "differs": 0, "only_a": 0, "only_b": 0}
    out, disallowed = [], 0
    for k in sorted(set(side_a) | set(side_b), key=_sort_key):
        ra, rb = side_a.get(k), side_b.get(k)
        head = {"id": (ra or rb).get("id"), **({} if key == ["id"] else
                                               {"coords": dict(zip(key, k))})}
        if ra is None or rb is None:
            status = "only_b" if ra is None else "only_a"
            counts[status] += 1
            out.append({**head, "status": status})
            continue
        changes = _compare(_flatten(ra), _flatten(rb), selected, patterns, hidden)
        if not changes:
            counts["identical"] += 1
            continue
        status = "extends" if all(c["relation"] == "extends" for c in changes.values()) \
            else "differs"
        counts[status] += 1
        for path, c in changes.items():
            _count_change(fields.setdefault(path, {"records": 0, "relations": {}}), c)
        row = {**head, "status": status, "fields": changes}
        if rules:
            flat_a = _flatten(ra)
            row["allowed"] = all(_is_allowed(p, c, flat_a, rules) for p, c in changes.items())
            disallowed += not row["allowed"]
        out.append(row)

    one_sided = counts["only_a"] + counts["only_b"]
    equivalent = one_sided == 0 and counts["extends"] == counts["differs"] == 0
    summary: dict[str, Any] = {
        "identical": _encode_canonical(a) == _encode_canonical(b),
        "equivalent": equivalent,
        "holds": one_sided == 0 and (disallowed == 0 if rules else equivalent),
        "key": key,
        "fields": selected or None,
        "a": {"records": len(items_a)},
        "b": {"records": len(items_b)},
        "records": counts,
        "changed_fields": {p: _finish(f) for p, f in sorted(fields.items())},
        "header": _compare(_flatten(_read_header(a)), _flatten(_read_header(b)),
                           [], patterns, hidden),
        "excluded": patterns,
        "excluded_differ": dict(sorted(hidden.items())),
    }
    if provenance is not None:
        summary["provenance"] = _compare(_flatten({"provenance": provenance[0] or {}}),
                                         _flatten({"provenance": provenance[1] or {}}),
                                         [], patterns, hidden)
        summary["excluded_differ"] = dict(sorted(hidden.items()))
    if rules:
        summary["allow"] = rules
        summary["disallowed"] = disallowed
    if by:
        summary["groups"] = _summarize_groups(items_a, items_b, by, selected, patterns)
    return build_collection(out, diff=summary)


def _check_rule(rule: Any) -> dict[str, Any]:
    if not isinstance(rule, Mapping) or not rule.get("field"):
        raise ValueError(f"diff: an allow rule is {{field, relation, when?}}, not {rule!r}")
    rel = rule.get("relation", "any")
    rels = [rel] if isinstance(rel, str) else list(rel)
    bad = [r for r in rels if r not in RELATIONS]
    if bad:
        raise ValueError(f"diff: relation {bad[0]!r} is not one of {', '.join(RELATIONS)}")
    out = {"field": rule["field"], "relation": rels}
    if rule.get("when"):
        out["when"] = dict(rule["when"])
    return out


def _read_key(record: Mapping[str, Any], name: str) -> Any:
    if name == "id":
        return record.get("id")
    for where in (record.get("coords"), record, (record.get("metadata") or {}).get("coords")):
        if isinstance(where, Mapping) and name in where:
            return where[name]
    return None


def _index_by_key(items: Sequence[Mapping[str, Any]], key: list[str],
               side: str) -> dict[tuple, Mapping[str, Any]]:
    out: dict[tuple, Mapping[str, Any]] = {}
    for it in items:
        k = tuple(_read_key(it, n) for n in key)
        if any(v is None for v in k):
            missing = [n for n, v in zip(key, k) if v is None]
            raise ValueError(f"diff: record {it.get('id')!r} on side {side} has no "
                             f"{', '.join(missing)} to key on")
        if k in out:
            raise ValueError(f"diff: two records on side {side} at "
                             f"{dict(zip(key, k))}; key on more coordinates")
        out[k] = it
    return out


def _sort_key(k: tuple) -> tuple:
    return tuple((0, v, "") if isinstance(v, (int, float)) and not isinstance(v, bool)
                 else (1, 0, json.dumps(v, sort_keys=True, default=str)) for v in k)


def _flatten(d: Any, pre: str = "") -> dict[str, Any]:
    if not isinstance(d, Mapping):
        return {}
    out: dict[str, Any] = {}
    for f, v in d.items():
        if isinstance(v, Mapping) and v:
            out.update(_flatten(v, f"{pre}{f}."))
        else:
            out[f"{pre}{f}"] = v
    return out


def _read_header(x: Any) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        return {}
    return {k: v for k, v in x.items() if k not in ("items", "rows")}


def _matches(path: str, pattern: str) -> bool:
    parts = path.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        if fnmatch.fnmatchcase(prefix, pattern) or fnmatch.fnmatchcase(prefix, "*." + pattern):
            return True
    return False


def _is_selected(path: str, selected: list[str]) -> bool:
    return not selected or any(path == s or path.startswith(s + ".") for s in selected)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_same(x: Any, y: Any) -> bool:
    if _is_number(x) and _is_number(y):
        return x == y or (isinstance(x, float) and isinstance(y, float)
                          and math.isnan(x) and math.isnan(y))
    return type(x) is type(y) and _encode_canonical(x) == _encode_canonical(y)


def _is_same_kind(path: str, x: Any, y: Any) -> bool:
    """A kind named by its retired spelling and by its current name is one
    kind (`~canonical/kinds/text` is `text/document`)."""
    if path.rsplit(".", 1)[-1] not in ("kind", "item_kind"):
        return False
    if not (isinstance(x, str) and isinstance(y, str)):
        return False
    from mechbench_compute.lexicon.kinds import resolve_kind

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return resolve_kind(x) == resolve_kind(y)
    except KeyError:
        return False


def _encode_canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str, separators=(",", ":"))


def _compare(fa: Mapping[str, Any], fb: Mapping[str, Any], selected: list[str],
             patterns: list[str], hidden: dict[str, int]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(set(fa) | set(fb)):
        if not _is_selected(path, selected):
            continue
        in_a, in_b = path in fa, path in fb
        if in_a and in_b and (_is_same(fa[path], fb[path])
                              or _is_same_kind(path, fa[path], fb[path])):
            continue
        if any(_matches(path, p) for p in patterns):
            hidden[path] = hidden.get(path, 0) + 1
            continue
        if not in_a:
            out[path] = {"relation": "added", "b": fb[path]}
        elif not in_b:
            out[path] = {"relation": "removed", "a": fa[path]}
        else:
            va, vb = fa[path], fb[path]
            c: dict[str, Any] = {"relation": "changed", "a": va, "b": vb}
            if isinstance(va, str) and isinstance(vb, str):
                if vb.startswith(va):
                    c["relation"] = "extends"
                elif va.startswith(vb):
                    c["relation"] = "truncates"
            elif _is_number(va) and _is_number(vb):
                c["delta"] = vb - va
            out[path] = c
    return out


def _is_allowed(path: str, change: Mapping[str, Any], flat_a: Mapping[str, Any],
                rules: list[dict[str, Any]]) -> bool:
    for r in rules:
        if not _matches(path, r["field"]):
            continue
        if "any" not in r["relation"] and change["relation"] not in r["relation"]:
            continue
        when = r.get("when") or {}
        if all(_is_among(flat_a.get(f), v) for f, v in when.items()):
            return True
    return False


def _is_among(value: Any, wanted: Any) -> bool:
    options = wanted if isinstance(wanted, list) else [wanted]
    return any(value is not None and _is_same(value, w) for w in options)


def _count_change(f: dict[str, Any], change: Mapping[str, Any]) -> None:
    f["records"] += 1
    rel = change["relation"]
    f["relations"][rel] = f["relations"].get(rel, 0) + 1
    if "delta" in change:
        f.setdefault("deltas", []).append(change["delta"])


def _finish(f: dict[str, Any]) -> dict[str, Any]:
    deltas = f.pop("deltas", None)
    if deltas:
        f["max_abs_delta"] = max(abs(d) for d in deltas)
        f["mean_delta"] = sum(deltas) / len(deltas)
    return f


def _is_coordinate(path: str) -> bool:
    return path.startswith(("coords.", "metadata.coords."))


def _summarize_groups(items_a: Sequence[Mapping[str, Any]], items_b: Sequence[Mapping[str, Any]],
                      by: list[str], selected: list[str],
                      patterns: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, dict[str, list[float]]]] = {}
    sizes: dict[tuple, dict[str, int]] = {}
    for side, items in (("a", items_a), ("b", items_b)):
        for it in items:
            g = tuple(_read_key(it, n) for n in by)
            sizes.setdefault(g, {"a": 0, "b": 0})[side] += 1
            for path, v in _flatten(it).items():
                if (_is_number(v) and _is_selected(path, selected)
                        and not _is_coordinate(path)
                        and not any(_matches(path, p) for p in patterns)):
                    groups.setdefault(g, {}).setdefault(path, {"a": [], "b": []})[side].append(v)
    rows = []
    for g in sorted(sizes, key=_sort_key):
        numeric = {}
        for path, vals in sorted((groups.get(g) or {}).items()):
            row: dict[str, Any] = {}
            for side in ("a", "b"):
                if vals[side]:
                    row[f"mean_{side}"] = sum(vals[side]) / len(vals[side])
                    row[f"sum_{side}"] = sum(vals[side])
            if "mean_a" in row and "mean_b" in row:
                row["diff"] = row["mean_b"] - row["mean_a"]
            numeric[path] = row
        rows.append({"coords": dict(zip(by, g)), "n_a": sizes[g]["a"],
                     "n_b": sizes[g]["b"], "fields": numeric})
    return rows
