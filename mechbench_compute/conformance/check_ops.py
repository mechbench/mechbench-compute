from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any

from mechbench_compute.conformance.finding import ERROR, Finding
from mechbench_compute.conformance.manifest import Manifest, read_kinds_of, read_outputs
from mechbench_compute.conformance.names import VERBS
from mechbench_compute.lexicon._base import COLLECTION, WILDCARD, is_need

INTERNAL = [
    re.compile(r"\b0\d{5}\b"),
    re.compile(r"\b(?:task|tasks|epic|epics)\s+\d", re.IGNORECASE),
    re.compile(r"\bstep\s+\d{2}\b", re.IGNORECASE),
    re.compile(r"\bexperiment\s+0\d{2}\b", re.IGNORECASE),
    re.compile(r"\bmechbench-experiments\b", re.IGNORECASE),
]

PORT_NAME = re.compile(r"[a-z][a-z0-9_]*")

OP_NAME = re.compile(r"[a-z0-9-]+/[a-z0-9-]+")

INPUT_AS_PARAM = re.compile(r"by edge,? or (the|by) (the )?param", re.IGNORECASE)


def read_common() -> frozenset[str]:
    from mechbench_compute.lexicon.common import COMMON

    return frozenset(p.name for p in COMMON)


def read_texts(op: Mapping[str, Any]) -> list[tuple[str, str]]:
    out = [("summary", str(op.get("summary") or "")), ("description", str(op.get("description") or ""))]
    out += [(where, str(o.get("doc") or "")) for where, o in read_outputs(op)]
    out += [(f"port {p.get('name')}.doc", str(p.get("doc") or "")) for p in op.get("inputs") or ()]
    out += [(f"param {p.get('name')}.doc", str(p.get("doc") or "")) for p in op.get("params") or ()]
    out += [(f"param {p.get('name')}.type", str(p.get("type") or "")) for p in op.get("params") or ()]
    return out


def find_idioms(at: str, texts: list[tuple[str, str]]) -> Iterator[Finding]:
    for where, text in texts:
        for pat in INTERNAL:
            m = pat.search(text)
            if m is not None:
                yield Finding("HOUSE_IDIOM", at, f"{where}: {m.group(0)!r} is a reference nobody outside "
                              "this repo can follow; rewrite it as what it means")


def check_complete(op: Mapping[str, Any], name: str) -> Iterator[Finding]:
    if not str(op.get("summary") or "").strip():
        yield Finding("NO_SUMMARY", name, "no summary")
    if not str(op.get("description") or "").strip():
        yield Finding("NO_DESCRIPTION", name, "no description")
    outputs = read_outputs(op)
    if not outputs:
        yield Finding("NO_OUTPUT", name, "emits nothing")
    for where, o in outputs:
        if not str(o.get("doc") or "").strip():
            yield Finding("NO_OUTPUT_DOC", name, f"{where} says nothing about what it produces")
    params = list(op.get("params") or ())
    for p in params:
        at = f"{name}.{p.get('name')}"
        if not str(p.get("type") or "").strip():
            yield Finding("PARAM_UNTYPED", at, "no type")
        if not str(p.get("doc") or "").strip():
            yield Finding("PARAM_UNDOCUMENTED", at, "no description")
    names = [p.get("name") for p in params]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        yield Finding("PARAM_DUPLICATE", f"{name}.{dup}", "declared twice")
    ports = [p.get("name") for p in op.get("inputs") or ()]
    for dup in sorted({n for n in ports if ports.count(n) > 1}):
        yield Finding("PORT_DUPLICATE", f"{name}:{dup}", "declared twice")
    for p in op.get("inputs") or ():
        if not str(p.get("doc") or "").strip():
            yield Finding("PORT_UNDOCUMENTED", f"{name}:{p.get('name')}", "no description")


def check_ports(op: Mapping[str, Any], name: str, m: Manifest) -> Iterator[Finding]:
    for p in op.get("inputs") or ():
        at = f"{name}:{p.get('name')}"
        for k in read_kinds_of(p.get("kind")):
            if m.find_kind(k) is None:
                yield Finding("PORT_KIND_UNKNOWN", at, f"unknown kind {k!r}")
        if p.get("name") != WILDCARD and not PORT_NAME.fullmatch(str(p.get("name"))):
            yield Finding("PORT_NAME_INVALID", at, f"port name {p.get('name')!r} is not [a-z][a-z0-9_]*")


def check_params_and_ports(op: Mapping[str, Any], name: str) -> Iterator[Finding]:
    params = {p.get("name"): p for p in op.get("params") or ()}
    ports = {p.get("name") for p in op.get("inputs") or ()}
    both = sorted(set(params) & ports)
    if both:
        yield Finding("PARAM_NAMES_PORT", name, f"{both} declared as both a param and a port")
    for where, text in read_texts(op):
        if INPUT_AS_PARAM.search(text):
            yield Finding("PARAM_NAMES_PORT", name, f"{where}: an input is described as a param")
    for pname, p in params.items():
        at = f"{name}.{pname}"
        expression = "expression" in str(p.get("type") or "")
        reads = list(p.get("reads") or ())
        if not set(reads) <= ports:
            yield Finding("READS_UNKNOWN", at, f"reads undeclared ports {reads}")
        if p.get("replaces") is not None and p["replaces"] not in params:
            yield Finding("REPLACES_UNKNOWN", at, f"replaces {p['replaces']!r}, which is not declared")
        if (reads or p.get("replaces")) and not expression:
            yield Finding("READS_NOT_EXPRESSION", at, "reads and replaces are for expressions")
        if expression and len(op.get("inputs") or ()) > 1 and not reads:
            yield Finding("READS_UNSAID", at, "an expression on an op with several ports declares the ports it reads")


def check_summary(op: Mapping[str, Any], name: str) -> Iterator[Finding]:
    s = str(op.get("summary") or "").strip()
    if not s:
        return
    if s.startswith("~canonical/"):
        yield Finding("SUMMARY_RESTATES_REF", name, "the summary restates the ref")
    if s[-1] not in ".?!":
        yield Finding("SUMMARY_NOT_SENTENCE", name, "the summary should end as a sentence does")
    if len(s) >= 400:
        yield Finding("SUMMARY_TOO_LONG", name, f"the summary is {len(s)} characters: a paragraph, not a sentence")


def read_value_kind(value: Any, m: Manifest) -> str | None:
    if isinstance(value, list):
        first = value[0] if value else None
        return first.get("kind") if isinstance(first, Mapping) and isinstance(first.get("kind"), str) else None
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        return None
    if value["kind"] == COLLECTION and value.get("item_kind"):
        return str(value["item_kind"])
    return value["kind"] if m.find_kind(value["kind"]) is not None else None


def check_example_accepted(op: Mapping[str, Any], name: str, m: Manifest) -> Iterator[Finding]:
    example = op.get("example")
    if example is None:
        return
    params = {p.get("name") for p in op.get("params") or ()}
    ports = {p.get("name"): p for p in op.get("inputs") or ()}
    wildcard = ports.get(WILDCARD)
    unknown = sorted(k for k in set(example) - params - read_common() if not str(k).startswith("_"))
    for k in unknown:
        hint = " (it is an input port: give it under example_inputs)" if k in ports else ""
        yield Finding("EXAMPLE_REFUSED", name, f"the example's params name {k!r}, which {name} does not accept{hint}")
    inputs = dict(op.get("example_inputs") or {})
    for port, value in inputs.items():
        p = ports.get(port) or wildcard
        if p is None:
            yield Finding("EXAMPLE_REFUSED", name, f"the example wires {port!r}, which is not a port")
            continue
        actual = read_value_kind(value, m)
        if actual is not None and m.find_kind(actual) is not None and not any(
                m.satisfies(actual, k) for k in read_kinds_of(p.get("kind"))):
            yield Finding("EXAMPLE_REFUSED", name, f"port {port!r} takes `{p.get('kind')}`, and the example "
                          f"wires `{actual}`")
    for port, p in ports.items():
        if not p.get("required", True):
            continue
        if port == WILDCARD:
            if not inputs:
                yield Finding("EXAMPLE_REFUSED", name, "the example wires no input, and at least one is required")
        elif inputs.get(port) is None:
            yield Finding("EXAMPLE_REFUSED", name, f"the example wires no {port!r}, which is required")


def check_name(op: Mapping[str, Any], name: str, m: Manifest) -> Iterator[Finding]:
    if not OP_NAME.fullmatch(name) or name.endswith(tuple(f"/{d}" for d in "0123456789")):
        yield Finding("NAME_INVALID", name, "an operation's name is two levels, family/leaf, with no version")
        return
    leaf = name.split("/", 1)[1]
    if leaf.split("-")[0] not in VERBS:
        yield Finding("OP_NOT_VERB", name, f"{leaf!r} does not start with a verb; an operation's leaf is one")
    if name in m.own or name in m.known:
        yield Finding("OP_NAMES_KIND", name, "an operation shares a kind's name")


def check_shape(op: Mapping[str, Any], name: str) -> Iterator[Finding]:
    common = read_common()
    for p in op.get("params") or ():
        if p.get("name") in common:
            yield Finding("PARAM_REDECLARES_COMMON", f"{name}.{p.get('name')}", "redeclares a common param")
        if ("default" in p) == bool(p.get("required")):
            yield Finding("NOT_JSON", f"{name}.{p.get('name')}", "a param has a default exactly when it is not required")
    for need in op.get("needs") or ():
        if not is_need(str(need)):
            yield Finding("NEED_UNKNOWN", name, f"{need!r} is not a need; the vocabulary is in CAPABILITY.md")
    try:
        json.dumps(op)
    except (TypeError, ValueError) as e:
        yield Finding("NOT_JSON", name, f"the declaration has no JSON form: {e}")


def check_op(op: Mapping[str, Any], m: Manifest) -> Iterator[Finding]:
    name = str(op.get("name") or "")
    yield from check_complete(op, name)
    yield from check_ports(op, name, m)
    yield from check_params_and_ports(op, name)
    yield from check_summary(op, name)
    yield from find_idioms(name, read_texts(op))
    yield from check_example_accepted(op, name, m)
    yield from check_name(op, name, m)
    yield from check_shape(op, name)


def check_ops(m: Manifest) -> Iterator[Finding]:
    names = [str(op.get("name") or "") for op in m.ops]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        yield Finding("NAME_DUPLICATE", dup, "two operations share this name", ERROR)
    for op in m.ops:
        yield from check_op(op, m)
