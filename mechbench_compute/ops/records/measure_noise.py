from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.api import (
    MOVING_FIELDS,
    In,
    Op,
    Output,
    P,
    Resume,
    collection,
    compute_version,
    flatten_record,
    index_by_key,
    is_field_match,
    items_of,
    read_numbers,
)

OP = Op(
    name="records/measure-noise",
    resume=Resume("restart"),
    summary=(
        "Measure the noise floor: how far each numeric field of one "
        "operation's records moves across runs that differ only in the seed "
        "or the machine."
    ),
    description="""\
A result taken on one Mac and replicated on another rarely agrees to the
last bit. bf16 kernels tile differently from chip to chip, so a logprob,
an activation or a score moves in its last places while nothing about the
experiment changed. This operation measures how far, so that a later
comparison can say "within the floor" instead of listing that noise as
findings.

Wire one edge per run onto `runs`: the same node's result from the same
protocol, run on different machines, at different seeds, or both. Records
are matched across runs by **key** (`id` by default, or named
coordinates, as `records/diff` matches them). For every key the runs
share, each numeric field's values are set side by side (a list of
numbers element by element), and the floor for the field is the widest
difference seen at any key: `spread` absolutely, `relative_spread` over
the size of the value.

The result is a `platform/noise` collection, one record per field, keyed
by (architecture, operation, field, dtype, machine class). Those five are
params, because a result collection does not know them; `machines` and
`seeds` say, in edge order, where and at which seed each run was taken.
Union the collections several nodes produce into one object, publish it,
and hand it to `records/diff` on its `noise` port.

**What it measures.** Runs at one seed on several machines measure what
the machines alone do: the floor a replication on another machine is
judged against. Runs at several seeds of a sampling operation measure the
sampling as well, and a text that differs between runs has no floor at
all: its fields are listed under `not_numeric` in the header, with how
many keys they differed at. Fields that move on every run whatever the
computation did (timestamps, latencies, ids) are left out, as
`records/diff` leaves them out, and so are the coordinates.
""",
    inputs=(
        In("runs", "collection",
           "One edge per run: the same node's result from each run, in the "
           "order `machines` and `seeds` name them.",
           many=True, variadic=True, min_edges=2),
    ),
    output=Output(
        "platform/noise", collection=True,
        doc="One record per numeric field: `architecture`, `model`, "
            "`operation`, `field`, `dtype`, `machine_class`, the floor as "
            "`spread` and `relative_spread`, `n` values compared over "
            "`records` keys, and the distinct `machines` and `seeds`. The "
            "header carries `compute`, `runs` (machine, seed and record count "
            "of each), `matched_by`, `excluded` and `not_numeric`."),
    params=(
        P("architecture", "string",
          "The checkpoint's architecture, its config.json `model_type` "
          "(`gemma4`)."),
        P("operation", "string",
          "The operation whose records these are (`text/generate`)."),
        P("machines", "list[string]",
          "Where each run was taken, one per edge in edge order (`Apple M4 "
          "Max`, `Apple M2 Ultra`)."),
        P("seeds", "list[int]",
          "The seed of each run, one per edge in edge order. None when the "
          "runs share their seed.",
          None),
        P("dtype", "string", "The weights' dtype or quantization.", "bfloat16"),
        P("machine_class", "string",
          "The class of machine the floor holds for.", "apple-silicon"),
        P("checkpoint", "string",
          "The checkpoint, as `repo@revision`, recorded as each record's "
          "`model`. None reads the runs' `model` header when they agree.",
          None),
        P("key", "\"id\" | list[string]",
          "What matches a record to its counterparts in the other runs: the "
          "record ids, or the named coordinates.",
          "id"),
        P("fields", "list[string]",
          "Measure only these fields, and what is under them. None measures "
          "every numeric field.",
          None),
        P("exclude", "list[string]",
          "Fields to leave out as well, by the end of their path; `*` "
          "matches within a name.",
          None),
        P("exclude_moving", "bool",
          "Leave out the fields that move on every run whatever the "
          "computation did: timestamps, latencies, ids, rate limits.",
          True),
    ),
    example={"architecture": "gemma4", "operation": "logits/read",
             "machines": ["Apple M4 Max", "Apple M2 Ultra"]},
    example_inputs={"runs": {"$ref": {"bench": "you/lab/results/j_home/read"}}},
)


def run(ctx, inputs, params):
    edges = inputs.get("runs") or []
    if isinstance(edges, Mapping):
        edges = [{"value": edges}]
    return measure_noise([e.get("value") if isinstance(e, Mapping) and "value" in e else e
                          for e in edges], params)


def measure_noise(runs: Sequence[Any], params: Mapping[str, Any]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("records/measure-noise needs at least two runs: one edge per run "
                         "on `runs`. With one there is nothing to compare.")
    machines = [str(m) for m in (params.get("machines") or [])]
    seeds = params.get("seeds")
    if len(machines) != len(runs):
        raise ValueError(f"`machines` names {len(machines)} machines for {len(runs)} runs; "
                         "name the machine of every run, in edge order")
    if seeds is not None and len(seeds) != len(runs):
        raise ValueError(f"`seeds` has {len(seeds)} entries for {len(runs)} runs")
    key = params.get("key") or "id"
    key = [key] if isinstance(key, str) else list(key)
    selected = list(params.get("fields") or [])
    patterns = (list(MOVING_FIELDS) if params.get("exclude_moving", True) else []) \
        + list(params.get("exclude") or [])

    indexed = [index_by_key(items_of(r), key, f"run {i}", "measure-noise")
               for i, r in enumerate(runs)]
    floors: dict[str, dict[str, float]] = {}
    not_numeric: dict[str, int] = {}
    for k in sorted({k for ix in indexed for k in ix}, key=repr):
        present = [flatten_record(ix[k]) for ix in indexed if k in ix]
        if len(present) < 2:
            continue
        for path in sorted({p for f in present for p in f}):
            if not _is_measured(path, key, selected, patterns):
                continue
            _measure_field(path, [f.get(path) for f in present], floors, not_numeric)

    header_models = {r.get("model") for r in runs if isinstance(r, Mapping)}
    model = params.get("checkpoint")
    if model is None:
        model = next(iter(header_models)) if len(header_models) == 1 else None
    distinct_seeds = sorted({int(s) for s in seeds if s is not None}) if seeds else []
    items = [_build_record(path, f, params, str(model or ""), sorted(set(machines)),
                           distinct_seeds)
             for path, f in sorted(floors.items())]
    return collection(
        "platform/noise", items,
        compute=compute_version,
        runs=[{"machine": machines[i], "seed": seeds[i] if seeds else None,
               "records": len(indexed[i])} for i in range(len(runs))],
        matched_by=key,
        excluded=patterns,
        not_numeric=dict(sorted(not_numeric.items())),
    )


def _is_measured(path: str, key: list[str], selected: list[str], patterns: list[str]) -> bool:
    if path.startswith(("coords.", "metadata.coords.")) or path in key or path == "id":
        return False
    if selected and not any(path == s or path.startswith(s + ".") for s in selected):
        return False
    return not any(is_field_match(path, p) for p in patterns)


def _measure_field(path: str, values: list[Any], floors: dict[str, dict[str, float]],
                   not_numeric: dict[str, int]) -> None:
    numbers = [read_numbers(v) for v in values]
    if any(n is None for n in numbers) or len({len(n) for n in numbers}) != 1:
        if any(v != values[0] for v in values[1:]):
            not_numeric[path] = not_numeric.get(path, 0) + 1
        return
    f = floors.setdefault(path, {"spread": 0.0, "relative_spread": 0.0, "n": 0, "records": 0})
    for column in zip(*numbers):
        lo, hi = min(column), max(column)
        size = max(abs(lo), abs(hi))
        f["spread"] = max(f["spread"], hi - lo)
        if size > 0:
            f["relative_spread"] = max(f["relative_spread"], (hi - lo) / size)
    f["n"] += sum(len(n) for n in numbers)
    f["records"] += 1


def _build_record(path: str, f: Mapping[str, float], params: Mapping[str, Any], model: str,
                  machines: list[str], seeds: list[int]) -> dict[str, Any]:
    architecture = str(params["architecture"])
    operation = str(params["operation"])
    dtype = str(params.get("dtype") or "bfloat16")
    machine_class = str(params.get("machine_class") or "apple-silicon")
    return {"id": ":".join((architecture, operation, path, dtype, machine_class)),
            "architecture": architecture, "model": model, "operation": operation,
            "field": path, "dtype": dtype, "machine_class": machine_class,
            "spread": f["spread"], "relative_spread": f["relative_spread"],
            "n": int(f["n"]), "records": int(f["records"]),
            "machines": machines, "seeds": seeds}
