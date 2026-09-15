"""What params and inputs a block accepts — so asking for something it
cannot do fails loudly.

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

**Inputs are checked the same way.** An op declares its ports — name,
kind, whether a collection, whether required — and `check_inputs`
refuses an edge onto a port the op does not have, a required port with
nothing on it, and a value whose kind does not satisfy the port's. The
same test proves the declared ports equal the ones the code reads.

A block absent from `ACCEPTED` is still unchecked at runtime — an
extension's op is nobody's business but its own — but no canonical op
may be absent, and the test enforces that.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import lexicon
from mechbench_compute.lexicon import BY_NAME
from mechbench_compute.lexicon import COMMON as _COMMON_PARAMS
from mechbench_compute.lexicon import kinds as K

#: Accepted by every block: the wiring, not the operation. Documented
#: in `lexicon.common`.
COMMON: frozenset[str] = frozenset(p.name for p in _COMMON_PARAMS)

#: op name -> the params it reads, beyond COMMON.
ACCEPTED: dict[str, frozenset[str]] = {
    name: op.param_names for name, op in BY_NAME.items()
}

#: op name -> the ports it reads.
PORTS: dict[str, frozenset[str]] = {
    name: op.port_names for name, op in BY_NAME.items()
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
    op = BY_NAME[block]
    ported = [u for u in unknown if op.port(u) is not None]
    hint = ""
    if ported:
        hint = (f" {', '.join(repr(u) for u in ported)} "
                f"{'is an input port' if len(ported) == 1 else 'are input ports'}"
                f" of {block}: wire an edge onto it, or give the value under "
                f"the node's `inputs`, not its `params`.")
    known = ", ".join(sorted(accepted | COMMON))
    raise ValueError(
        f"{block} does not accept {', '.join(repr(u) for u in unknown)}.{hint} "
        f"If the protocol is newer than this runner, the runner's copy of "
        f"the block predates the parameter — check its compute version. "
        f"Accepted here: {known}."
    )


def _kind_of(value: Any) -> tuple[str | None, bool]:
    """(bare kind name, is-a-collection) of a value arriving on a port,
    however it is spelled: a `collection`, a legacy plural object, a
    singular kinded object, or a bare list (a collection of
    `records/record`, by convention). `(None, …)` for a value that
    carries no kind at all — an older stored object, or a literal — which
    is accepted as it is, since there is no name to refuse it by."""
    if isinstance(value, list):
        # A list of kinded objects (directions fetched one by one) is a
        # collection of that kind; a list of plain objects is records; a
        # list of strings, or of references not yet resolved, says
        # nothing about its kind.
        first = value[0] if value else None
        if isinstance(first, Mapping) and isinstance(first.get("kind"), str):
            try:
                return K.resolve_kind(first["kind"], warn=False)[0], True
            except KeyError:
                return first["kind"], True
        return None, True
    if not isinstance(value, Mapping):
        return None, False
    k = value.get("kind")
    if not isinstance(k, str):
        return None, False
    ik = K.item_kind_of(value)
    if ik is not None:
        return ik, True
    try:
        name, plural = K.resolve_kind(k, warn=False)
    except KeyError:
        # A kind the registry does not know — an extension's, or a
        # string an older author wrote — is no name to refuse by.
        return None, False
    return name, plural


def check_inputs(block: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse what the op cannot take on its ports, and return the inputs
    with a bare list wrapped as the collection it stands for.

    Three refusals, each naming the op, the port and the kind: a port the
    op does not declare; a required port with nothing on it; a value
    whose kind does not satisfy the port's (by `extends`). A value
    carrying no kind — an older stored object, an inline literal object —
    passes: there is no name to refuse it by, and the block reads it as
    it always did.

    An op that is not canonical is not second-guessed.
    """
    try:
        block = lexicon.resolve(block, warn=False)
    except KeyError:
        return dict(inputs)
    op = BY_NAME.get(block)
    if op is None:
        return dict(inputs)
    out: dict[str, Any] = {}
    for name, value in inputs.items():
        port = op.port(name)
        if port is None:
            known = ", ".join(sorted(p.name for p in op.inputs)) or "none"
            raise ValueError(
                f"{block} has no input port {name!r}. Its ports: {known}.")
        if value is None:
            continue
        actual, plural = _kind_of(value)
        if actual is not None and not any(K.satisfies(actual, d) for d in port.kinds):
            what = "a collection of " if plural else ""
            raise ValueError(
                f"{block} port {name!r} takes {'a collection of ' if port.many else ''}"
                f"`{port.kind}`, but was wired {what}`{actual}`.")
        if isinstance(value, list) and port.many:
            item_kind = actual if actual in K.BY_KIND else port.kinds[0]
            if item_kind == K.COLLECTION:
                item_kind = "records/record"
            value = K.collection(item_kind, value)
        out[name] = value
    for port in op.inputs:
        if not port.required:
            continue
        if port.wildcard:
            if not out:
                raise ValueError(
                    f"{block} needs at least one input edge "
                    f"({'a collection of ' if port.many else ''}`{port.kind}` "
                    f"on a port of your naming).")
            continue
        if out.get(port.name) is None:
            raise ValueError(
                f"{block} needs an input on its {port.name!r} port "
                f"({'a collection of ' if port.many else ''}`{port.kind}`): "
                f"wire an edge onto it, or give it under the node's `inputs`.")
    return out
