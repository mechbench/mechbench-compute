from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import lexicon
from mechbench_compute.lexicon import BY_NAME
from mechbench_compute.lexicon import COMMON as _COMMON_PARAMS
from mechbench_compute.lexicon import kinds as K

COMMON: frozenset[str] = frozenset(p.name for p in _COMMON_PARAMS)

ACCEPTED: dict[str, frozenset[str]] = {
    name: op.param_names for name, op in BY_NAME.items()
}

PORTS: dict[str, frozenset[str]] = {
    name: op.port_names for name, op in BY_NAME.items()
}


def check_params(block: str, params: Mapping[str, object]) -> None:
    try:
        block = lexicon.resolve(block, warn=False)
    except KeyError:
        return
    accepted = ACCEPTED.get(block)
    if accepted is None:
        return
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
    if isinstance(value, list):
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
        return None, False
    return name, plural


def check_inputs(block: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
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
        if port.variadic and isinstance(value, list) and all(
                isinstance(v, Mapping) and "value" in v and ("node" in v or "input" in v)
                for v in value):
            for entry in value:
                actual, _plural = _kind_of(entry["value"])
                if actual is not None and not any(
                        K.satisfies(actual, d) for d in port.kinds):
                    source = (f"node {entry['node']!r}" if "node" in entry
                              else f"input {entry['input']!r}")
                    raise ValueError(
                        f"{block} port {name!r} takes `{port.kind}`, but the "
                        f"edge from {source} carries `{actual}`.")
            out[name] = value
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
