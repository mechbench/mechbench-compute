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
    In,
    Kind,
    Op,
    P,
    Param,
    Port,
)
from mechbench_compute.lexicon.common import COMMON
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
    "template": "records/template",
    "select": "records/select",
    "union": "records/union",
    "paired-delta": "records/delta",
    "group-stats": "records/stats",
    "reduce/sum": "records/sum",
    "reduce/top-k": "records/top-k",
    "reduce/histogram": "records/histogram",
    "table/from-records": "records/table",
    "viz/spec": "records/chart",
    "generate": "text/generate",
    "chat": "text/chat",
    "conversation": "text/conversation",
    "score": "text/score",
    "tokenize/stats": "text/tokenize",
    "judge": "eval/judge",
    "eval/hf-metric": "eval/metric",
    "decision-read": "logits/decision",
    "lens/positions": "logits/lens",
    "lens-trajectory": "logits/funnel",
    "attribution/logits": "logits/attribution",
    "residuals/vectors": "activations/vectors",
    "residuals/divergence": "activations/divergence",
    "attention/patterns": "activations/attention",
    "vectors/similarity": "geometry/similarity",
    "vectors/mst": "geometry/mst",
    "intervene": "intervene/apply",
    "ablate/layers": "intervene/layers",
    "ablate/heads": "intervene/heads",
    "steer/inject": "intervene/steer",
    "patch/trace": "intervene/trace",
    "finetune/lora": "adapter/train",
    "merge": "adapter/merge",
    "hf/push-adapter": "adapter/publish",
    "tools/bench-lookup": "tools/lookup",
}

#: The compute release that drops the aliases. Named 0.77.0 when they
#: were introduced (0.75.0); moved to 0.80.0 in 0.77.0 because the
#: protocols stored on the bench still spell the old names and their
#: migration is a scheduled task, not a side effect of a release.
ALIASES_REMOVED_IN = "0.80.0"

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
    "ALIASES", "ALIASES_REMOVED_IN", "BY_NAME", "COMMON", "OPS", "REQUIRED",
    "ROOT", "WILDCARD", "In", "Op", "P", "Param", "Port", "RetiredKindName",
    "RetiredOpName", "ancestry", "canonical_path", "is_canonical", "resolve",
    "satisfies",
]
