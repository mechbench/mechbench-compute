"""What params a block accepts — so asking for something it cannot do
fails loudly.

Six variety jobs once declared `center: true`, all six succeeded, and
all six were uncentered: the executing runner's block predated the
parameter and simply never read it. A protocol asked for a
measurement, the platform said done, and returned a different
measurement.

**Every canonical op is declared**, in `mechbench_compute.lexicon`, and
`tests/test_block_params.py` fails when a new op arrives undeclared.
The tables here are derived from that lexicon: the lexicon is where a
parameter's type, default and meaning live, beside its name, and this
module is the runtime check that reads the names.

The forward-compatibility argument cuts toward strictness. A NEW
protocol running against an OLD block is precisely the case that bit
us, and "this runner's `vectors/mst` does not accept `center`" is
strictly better than a quiet wrong number.

**Both directions are bugs.** An incomplete declaration falsely refuses
a param the block does read. An over-declaration accepts one it never
reads — which is the original failure wearing a different hat: a
protocol could set it and be ignored. The test asserts the declaration
EQUALS what the code reads, following `params` across modules and into
packages to find out.

A block absent from `ACCEPTED` is still unchecked at runtime — an
extension's op is nobody's business but its own — but no canonical op
may be absent, and the test enforces that.
"""

from __future__ import annotations

from collections.abc import Mapping

from mechbench_compute import lexicon
from mechbench_compute.lexicon import BY_NAME
from mechbench_compute.lexicon import COMMON as _COMMON_PARAMS

#: Accepted by every block: the wiring, not the operation. Documented
#: in `lexicon.common`.
COMMON: frozenset[str] = frozenset(p.name for p in _COMMON_PARAMS)

#: op name -> the params it reads, beyond COMMON.
ACCEPTED: dict[str, frozenset[str]] = {
    name: op.param_names for name, op in BY_NAME.items()
}


def check_params(block: str, params: Mapping[str, object]) -> None:
    """Raise if `block` was handed a param it does not read.

    `block` may be spelled any way `lexicon.resolve` accepts. The message
    names the parameter AND the block, because the useful question when
    this fires is 'does this runner's copy of that block know about
    it?' — usually the answer is that the runner is old.
    """
    try:
        block = lexicon.resolve(block, warn=False)
    except KeyError:
        return  # not canonical: an extension's op checks its own params
    accepted = ACCEPTED.get(block)
    if accepted is None:
        return
    # Underscore keys are the executor's own injections (a block
    # runner, a resume handle), never something a protocol declared.
    unknown = sorted(k for k in set(params) - accepted - COMMON
                     if not k.startswith("_"))
    if not unknown:
        return
    known = ", ".join(sorted(accepted | COMMON))
    raise ValueError(
        f"{block} does not accept {', '.join(repr(u) for u in unknown)}. "
        f"If the protocol is newer than this runner, the runner's copy of "
        f"the block predates the parameter — check its compute version. "
        f"Accepted here: {known}."
    )
