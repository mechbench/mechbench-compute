from __future__ import annotations

import re

from mechbench_compute.lexicon._base import (
    COLLECTION,
    KIND_ROOT,
    NEEDS,
    REQUIRED,
    ROOT,
    WILDCARD,
    Draw,
    Emits,
    Output,
    Family,
    In,
    Kind,
    Op,
    Otherwise,
    P,
    Param,
    Port,
    Resume,
    Value,
    display_name,
    name_of_title,
    title,
)
from mechbench_compute.lexicon.common import COMMON
from mechbench_compute.lexicon.families import BY_FAMILY, FAMILIES
from mechbench_compute.lexicon.values import BY_VALUE, VALUES
from mechbench_compute.lexicon.kinds import (
    RetiredKindName,
    BY_KIND,
    KIND_ALIASES,
    KINDS,
    ancestry,
    canonical_collection,
    canonical_kind_path,
    collection,
    item_kind_of,
    items_of,
    resolve_kind,
    satisfies,
)

from mechbench_compute.lexicon.address import is_extension_spelling
from mechbench_compute.lexicon.views import read_registry, view_by_name, view_ops

OPS = view_ops()

BY_NAME = view_by_name()

RETIRED: dict[str, str] = {
    "factor-cross": "records/cross",
    "grid": "records/cross",
    "template": "records/fill",
    "select": "records/select",
    "union": "records/union",
    "paired-delta": "records/subtract",
    "group-stats": "records/summarize",
    "reduce/sum": "records/total",
    "reduce/top-k": "records/rank",
    "reduce/histogram": "records/bin",
    "table/from-records": "records/tabulate",
    "viz/spec": "records/plot",
    "generate": "text/generate",
    "chat": "text/chat",
    "score": "text/score",
    "tokenize/stats": "text/tokenize",
    "judge": "eval/judge",
    "eval/hf-metric": "eval/score",
    "decision-read": "logits/read",
    "lens/positions": "logits/scan",
    "lens-trajectory": "logits/read-layers",
    "attribution/logits": "logits/attribute",
    "residuals/vectors": "activations/capture",
    "residuals/divergence": "activations/contrast",
    "attention/patterns": "activations/capture-attention",
    "vectors/similarity": "geometry/compare",
    "vectors/mst": "geometry/span",
    "intervene": "intervene/apply",
    "ablate/layers": "intervene/ablate-layers",
    "ablate/heads": "intervene/ablate-heads",
    "steer/inject": "intervene/steer",
    "patch/trace": "intervene/patch",
    "finetune/lora": "adapter/train",
    "merge": "adapter/merge",
    "hf/push-adapter": "adapter/publish",
    "tools/bench-lookup": "tools/lookup",
    "direction/similarity": "geometry/compare",
    "records/template": "records/fill",
    "records/delta": "records/subtract",
    "records/stats": "records/summarize",
    "records/top-k": "records/rank",
    "records/histogram": "records/bin",
    "records/table": "records/tabulate",
    "records/sum": "records/total",
    "records/chart": "records/plot",
    "text/stats": "text/measure",
    "eval/expectation": "eval/expect",
    "eval/metric": "eval/score",
    "eval/suite": "eval/benchmark",
    "logits/decision": "logits/read",
    "logits/funnel": "logits/read-layers",
    "logits/lens": "logits/scan",
    "logits/attribution": "logits/attribute",
    "activations/vectors": "activations/capture",
    "activations/divergence": "activations/contrast",
    "activations/attention": "activations/capture-attention",
    "geometry/similarity": "geometry/compare",
    "geometry/mst": "geometry/span",
    "intervene/layers": "intervene/ablate-layers",
    "intervene/heads": "intervene/ablate-heads",
    "intervene/trace": "intervene/patch",
    "direction/from-vectors": "direction/fit",
    "direction/from-pca": "direction/decompose",
    "direction/vocab": "direction/unembed",
}

ALIASES_REMOVED_IN = "0.82.0"

REPLACED_IN = "0.156.0"

REPLACED: dict[str, str] = {
    "records/select": 'records/filter (where: `coords.prompt == "flash"`), and records/derive (keep: [...]) for its fields',
    "records/rename": "records/derive (fields: {new: old}, drop: [old])",
    "records/count": "records/group (aggregates: {\"k, n, rate, lo, hi\": \"wilson(cond)\"})",
    "records/summarize": "records/group (aggregates: {n: count(), mean: mean(x), median: median(x), "
                         "\"lo, hi\": bootstrap_mean(x)})",
    "records/contrast": "records/group (aggregates: {\"diff, lo, hi\": paired_difference(x, on, a, b, paired)})",
    "records/correlate": "records/group (aggregates: {\"rho, lo, hi\": spearman(x, y, level=0.95)})",
    "records/total": "records/group (aggregates: {n: count(), sum: sum(x)})",
    "records/subtract": "records/filter the baseline, records/join it on the matching coordinates, and "
                        "records/derive the difference",
    "records/fill": "records/derive (templates: {user: \"Write a story about {values.animal}.\"})",
    "records/relabel": "records/derive (fields: {coords.x: '{\"a\": \"its label\"}.get(coords.x, coords.x)'})",
    "records/lookup": "records/derive (fields: {coords.x: header.table[coords.i]})",
    "records/rank": "records/sort (by: [\"-x\"], limit: k)",
    "records/bin": "records/derive the bin (floor((x - lo) / width)), then records/group with count()",
}

_VERSION_TAIL = re.compile(r"/\d+$")


def canonical_path(name: str) -> str:
    return name if name.startswith("~") or is_extension_spelling(name) else f"{ROOT}{name}"


def is_canonical(block: str) -> bool:
    try:
        resolve(block, warn=False)
    except KeyError:
        return False
    return True


def resolve(block: str, *, warn: bool = True) -> str:
    return read_registry().resolve(block).name


def explain_unknown(block: str) -> str:
    return read_registry().explain(block)


def explain_core(block: str) -> str:
    s = block.strip()
    if s.startswith(ROOT):
        s = s[len(ROOT):]
    bare = _VERSION_TAIL.sub("", s)
    replaced = REPLACED.get(RETIRED.get(bare, bare))
    if replaced is not None:
        return (f"unknown block {block!r}: it was retired in mechbench-compute {REPLACED_IN}, "
                f"when the records operations became expressions. Write {replaced}.")
    target = RETIRED.get(bare) or (bare if bare in BY_NAME else None)
    if target is not None:
        return (f"unknown block {block!r}: that spelling was retired in "
                f"mechbench-compute {ALIASES_REMOVED_IN}. Write {target!r}.")
    return (f"unknown block {block!r}: no operation has that name. "
            f"Operations are listed at https://docs.mechbench.ai/ops/.")


__all__ = [
    "ALIASES_REMOVED_IN", "BY_FAMILY", "BY_NAME", "BY_VALUE", "COMMON",
    "FAMILIES", "OPS", "REPLACED", "REPLACED_IN", "REQUIRED", "RETIRED", "ROOT", "VALUES", "WILDCARD",
    "Draw", "Family", "In", "NEEDS", "Op", "Otherwise", "P", "Param", "Port", "Resume", "RetiredKindName", "Value",
    "ancestry", "canonical_path", "explain_core", "explain_unknown", "is_canonical",
    "display_name", "name_of_title", "resolve", "satisfies", "title",
]
