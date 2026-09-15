"""The operation lexicon: every canonical op, described for the person
who will use it.

One declaration, three consumers. `check_params` refuses a parameter
that is not declared here; `tests/test_block_params.py` proves the
declaration equals what the code reads, in both directions; and the
documentation site renders these entries as the operation reference.
Because all three read the same source, a documented parameter the
block does not read fails a test, and a parameter the block reads but
nobody documented fails the same test.

Naming (docs/LEXICON.md in the meta repo): an op is spelled bare,
`records/select`, exactly two levels, no version segment. Its stored
identity is `~canonical/ops/records/select`; `canonical_path` makes
one from the other and `resolve` accepts any spelling a protocol may
carry — bare, stored, with the retired `/1`, or one of the names from
before the 2026-09 renames — and returns the bare name, warning once
per retired spelling.

Entries are grouped by family in the sibling modules and assembled here.
"""

from __future__ import annotations

import re
import warnings

from mechbench_compute.lexicon import (
    direction,
    external,
    model,
    records,
    trajectory,
)
from mechbench_compute.lexicon._base import (
    COLLECTION,
    KIND_ROOT,
    REQUIRED,
    ROOT,
    WILDCARD,
    Emits,
    Family,
    In,
    Kind,
    Op,
    P,
    Param,
    Port,
    Value,
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

OPS: tuple[Op, ...] = tuple(
    sorted(
        (*model.OPS, *direction.OPS, *trajectory.OPS, *records.OPS, *external.OPS),
        key=lambda op: op.name,
    )
)

BY_NAME: dict[str, Op] = {op.name: op for op in OPS}

#: The names retired on 2026-09-14 (task 000495), old bare name -> new.
#: Each resolves with a deprecation warning until the release named in
#: `ALIASES_REMOVED_IN`, then refuses. `grid` was an alias already.
ALIASES: dict[str, str] = {
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
    "conversation": "text/converse",
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
    # Retired 2026-09-15 (metrics on kinds): a cosine between directions
    # is `geometry/compare` over a collection of them — `records/union`
    # the directions first. The alias lands the protocol on the new op,
    # whose port check then names the port to wire.
    "direction/similarity": "geometry/compare",
    # Retired 2026-09-16: operations are verbs. The twenty-eight names
    # below — fifteen of them the names of the kinds they emit — became
    # imperatives; the kinds keep the nouns.
    "records/template": "records/fill",
    "records/delta": "records/subtract",
    "records/stats": "records/summarize",
    "records/top-k": "records/rank",
    "records/histogram": "records/bin",
    "records/table": "records/tabulate",
    "records/sum": "records/total",
    "records/chart": "records/plot",
    "text/conversation": "text/converse",
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

#: The compute release that drops the aliases — both tables, the
#: 2026-09-14 family renames and the 2026-09-16 verbs. Named 0.77.0
#: when the first table was introduced (0.75.0); moved to 0.80.0 in
#: 0.77.0 because the protocols stored on the bench still spell the old
#: names and their migration is a scheduled task; moved to 0.82.0 in
#: 0.80.0 so one release removes both.
ALIASES_REMOVED_IN = "0.82.0"

_VERSION_TAIL = re.compile(r"/\d+$")
_warned: set[str] = set()


class RetiredOpName(DeprecationWarning):
    """A protocol spelled an op by a name that has been renamed."""


class RetiredParam(DeprecationWarning):
    """A protocol gave an input under `params` — where it lived before
    ports were typed — rather than under the node's `inputs`."""


def canonical_path(name: str) -> str:
    """The stored identity of a bare op name: `~canonical/ops/<name>`."""
    return name if name.startswith("~") else f"{ROOT}{name}"


def is_canonical(block: str) -> bool:
    """Whether a block string names (or aliases) a canonical op, as
    opposed to a user's or an extension's."""
    try:
        resolve(block, warn=False)
    except KeyError:
        return False
    return True


def resolve(block: str, *, warn: bool = True) -> str:
    """The bare name of the op a protocol's block string means.

    Accepts the bare name, the stored path, either with the retired
    version segment, and any name from `ALIASES`. Returns the bare name.
    A retired spelling warns once per process (`RetiredOpName`), naming
    the replacement and the release that will refuse it.

    A string that is none of these — a user op `owner/project/ops/x`,
    or a typo — raises `KeyError` with the string, so the executor can
    say "unknown block" and an extension resolver can try next.
    """
    s = block.strip()
    stored = s.startswith(ROOT)
    if stored:
        s = s[len(ROOT):]
    versioned = bool(_VERSION_TAIL.search(s))
    if versioned:
        s = _VERSION_TAIL.sub("", s)
    retired = ALIASES.get(s)
    if retired is not None:
        target = retired
    elif s in BY_NAME:
        target = s
    else:
        raise KeyError(block)
    if warn and (retired is not None or versioned) and block not in _warned:
        _warned.add(block)
        warnings.warn(
            f"{block!r} is a retired spelling of the operation {target!r}; "
            f"write {target!r}. Retired names resolve until mechbench-compute "
            f"{ALIASES_REMOVED_IN}, then refuse.",
            RetiredOpName, stacklevel=2)
    return target


__all__ = [
    "ALIASES", "ALIASES_REMOVED_IN", "BY_FAMILY", "BY_NAME", "BY_VALUE", "COMMON",
    "FAMILIES", "OPS", "REQUIRED", "ROOT", "VALUES", "WILDCARD", "Family", "In", "Op",
    "P", "Param", "Port", "RetiredKindName", "RetiredOpName", "Value", "ancestry",
    "canonical_path", "is_canonical", "resolve", "satisfies",
]
