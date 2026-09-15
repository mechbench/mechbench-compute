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

#: Names that are no longer operations, old -> what replaced it.
#:
#: These DO NOT RESOLVE (task 000512, compute 0.82.0). The table is kept
#: for one purpose: so a protocol that still spells one of them is
#: refused with the name to write instead, rather than with "unknown
#: block" and a shrug. The first table was introduced in 0.75.0 with a
#: removal scheduled for 0.77.0, moved twice — to 0.80.0, then to 0.82.0
#: so that one release would drop the 2026-09-14 family renames and the
#: 2026-09-16 verbs together, after the protocols stored on the bench
#: had been migrated (they were, on 2026-09-16; a dry run on 2026-09-16
#: found nothing left to rewrite).
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

#: The release the aliases were dropped in. Kept as a fact, not a
#: schedule: a message that says when a spelling stopped working is
#: worth more than one that says it stopped.
ALIASES_REMOVED_IN = "0.82.0"

_VERSION_TAIL = re.compile(r"/\d+$")


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

    Accepts the bare name and the stored path (`~canonical/ops/<name>`).
    Returns the bare name.

    A string that is neither — a user op `owner/project/ops/x`, a name
    retired before 0.82.0, or a typo — raises `KeyError` with the
    string, so the executor can refuse it by name and an extension
    resolver can try next. `explain_unknown` turns that into a sentence
    that names the current spelling where there is one.

    `warn` is accepted and ignored: nothing resolves with a warning any
    more, and a caller that passed `warn=False` to silence one should
    not have to change.
    """
    s = block.strip()
    if s.startswith(ROOT):
        s = s[len(ROOT):]
    if s in BY_NAME:
        return s
    raise KeyError(block)


def explain_unknown(block: str) -> str:
    """Why this block string names no operation — as a sentence.

    A retired spelling, a version segment (`…/1`, which stored protocols
    carried until the 2026-09-16 migration), or something else entirely;
    each gets the answer that helps, and the first two name what to
    write instead.
    """
    s = block.strip()
    if s.startswith(ROOT):
        s = s[len(ROOT):]
    bare = _VERSION_TAIL.sub("", s)
    target = RETIRED.get(bare) or (bare if bare in BY_NAME else None)
    if target is not None:
        return (f"unknown block {block!r}: that spelling was retired in "
                f"mechbench-compute {ALIASES_REMOVED_IN}. Write {target!r}.")
    return (f"unknown block {block!r}: no operation has that name. "
            f"Operations are listed at https://docs.mechbench.ai/ops/.")


__all__ = [
    "ALIASES_REMOVED_IN", "BY_FAMILY", "BY_NAME", "BY_VALUE", "COMMON",
    "FAMILIES", "OPS", "REQUIRED", "RETIRED", "ROOT", "VALUES", "WILDCARD",
    "Family", "In", "Op", "P", "Param", "Port", "RetiredKindName", "Value",
    "ancestry", "canonical_path", "explain_unknown", "is_canonical", "resolve",
    "satisfies",
]
