from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from mechbench_compute.conformance.finding import Finding
from mechbench_compute.conformance.manifest import Manifest, read_outputs
from mechbench_compute.lexicon._base import TypeNode, parse_type, type_words


def read_values() -> dict[str, Any]:
    from mechbench_compute.lexicon.values import BY_VALUE

    return BY_VALUE


def walk_params(params: Any, where: str) -> Iterator[tuple[str, dict[str, Any]]]:
    for p in params or ():
        at = f"{where}.{p.get('name')}"
        yield at, p
        yield from walk_params(p.get("fields"), at)


def check_param(at: str, p: Mapping[str, Any]) -> Iterator[Finding]:
    try:
        words = type_words(str(p.get("type") or ""))
    except ValueError as e:
        yield Finding("TYPE_INVALID", at, str(e))
        return
    values = read_values()
    fields = list(p.get("fields") or ())
    shape = values.get(p["value"]) if p.get("value") else None
    if "object" in words:
        if not fields and not (shape and shape.fields):
            yield Finding("OBJECT_UNDECLARED", at, "an `object` with no declared fields: declare them "
                          "(`fields=`), name the shared value it is (`value=`), or type it `json` if it is open on purpose")
    elif fields:
        yield Finding("FIELDS_INVALID", at, "declares fields but its type has no `object`")
    names = [f.get("name") for f in fields]
    if len(names) != len(set(names)):
        yield Finding("FIELDS_INVALID", at, "a field is declared twice")
    choices = list(p.get("choices") or ())
    if p.get("value"):
        if shape is None:
            yield Finding("VALUE_INVALID", at, f"value {p['value']!r} is not a declared value")
        elif not shape.grammar:
            yield Finding("VALUE_INVALID", at, f"{p['value']} is a stored field's shape, not a parameter grammar")
        elif shape.choices:
            if "string" not in words:
                yield Finding("VALUE_INVALID", at, f"{p['value']} is a vocabulary; the type has no string")
            outside = sorted(set(choices) - set(shape.choices))
            if outside:
                yield Finding("CHOICES_INVALID", at, f"choices {outside} are not {p['value']} words")
    if choices:
        if "string" not in words:
            yield Finding("CHOICES_INVALID", at, "choices on a type with no `string`")
        if len(choices) != len(set(choices)):
            yield Finding("CHOICES_INVALID", at, "a choice is listed twice")
        d = p.get("default")
        if isinstance(d, str) and d not in choices:
            yield Finding("CHOICES_INVALID", at, f"default {d!r} is not one of the choices")
        if isinstance(d, list) and not all(x in choices for x in d if isinstance(x, str)):
            yield Finding("CHOICES_INVALID", at, f"default {d!r} names a word outside the choices")


def is_binding(v: Any) -> bool:
    return isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("$param", "$ref")


def read_fields(p: Mapping[str, Any]) -> list[dict[str, Any]]:
    if p.get("fields"):
        return list(p["fields"])
    v = read_values().get(p["value"]) if p.get("value") else None
    if v is None or not v.fields:
        return []
    return [{"name": n, "type": s["type"], "doc": s["description"], "required": n in v.required,
             "choices": list(s.get("choices", ()))} for n, s in v.fields.items()]


def read_choices(p: Mapping[str, Any]) -> tuple[str, ...]:
    if p.get("choices"):
        return tuple(p["choices"])
    v = read_values().get(p["value"]) if p.get("value") else None
    return v.choices if v is not None else ()


def is_selector(v: Any) -> bool:
    if isinstance(v, str):
        return True
    if isinstance(v, list):
        return all(isinstance(i, int) and not isinstance(i, bool) for i in v)
    return isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("tokens", "range", "after")


def check_conforms(v: Any, alts: tuple[TypeNode, ...], p: Mapping[str, Any], where: str) -> list[str]:
    if is_binding(v):
        return []
    reasons: list[str] = []
    for a in alts:
        why = check_fits(v, a, p, where)
        if not why:
            return []
        reasons += why
    return reasons or [f"{where}: {v!r} is not {p.get('type')}"]


def check_word(v: Any, w: str, p: Mapping[str, Any], where: str) -> list[str] | None:
    if w in ("expression", "template"):
        if not isinstance(v, str):
            return None
        from mechbench_compute.expr.engine import ExprError, load_engine
        try:
            if w == "expression":
                load_engine().check(v.removeprefix("-") if str(p.get("type")).startswith("list") else v)
            else:
                load_engine().render(v, [{}])
        except ExprError as e:
            return [f"{where}: {v!r} is not a {w}: {e.detail}"]
        return []
    fits = {
        "int": lambda: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda: isinstance(v, (int, float)) and not isinstance(v, bool),
        "bool": lambda: isinstance(v, bool),
        "null": lambda: v is None,
        "json": lambda: True,
        "model": lambda: isinstance(v, (str, dict)),
        "selector": lambda: is_selector(v),
    }.get(w)
    if fits is None:
        return None
    return [] if fits() else None


def check_fits(v: Any, a: TypeNode, p: Mapping[str, Any], where: str) -> list[str]:
    no = [f"{where}: {v!r} is not {p.get('type')}"]
    if a.form == "literal":
        return [] if v == a.word else no
    if a.form == "list":
        if not isinstance(v, list):
            return no
        return [r for i, x in enumerate(v) for r in check_conforms(x, a.of, p, f"{where}[{i}]")]
    if a.form == "map":
        if not isinstance(v, dict):
            return no
        return [r for k, x in v.items() for r in check_conforms(x, a.of, p, f"{where}.{k}")]
    w = a.word
    if w == "string":
        if not isinstance(v, str):
            return no
        c = read_choices(p)
        return [] if not c or v in c else [f"{where}: {v!r} is not one of {c}"]
    if w == "object":
        if not isinstance(v, dict):
            return no
        fs = {f["name"]: f for f in read_fields(p)}
        out = [f"{where}: {k!r} is not a declared field" for k in v if k not in fs]
        out += [f"{where}: required field {n!r} missing" for n, f in fs.items()
                if f.get("required") and n not in v]
        for k, x in v.items():
            if k in fs:
                out += check_conforms(x, parse_type(fs[k]["type"]), fs[k], f"{where}.{k}")
        return out
    got = check_word(v, w, p, where)
    return no if got is None else got


def check_example_types(op: Mapping[str, Any], name: str, common: list[dict[str, Any]]) -> Iterator[Finding]:
    example = op.get("example")
    if example is None:
        return
    by = {p["name"]: p for p in (*(op.get("params") or ()), *common)}
    for k, v in example.items():
        if k not in by:
            continue
        try:
            alts = parse_type(str(by[k].get("type") or ""))
        except ValueError:
            continue
        for reason in check_conforms(v, alts, by[k], f"{name}.{k}"):
            yield Finding("EXAMPLE_TYPE_MISMATCH", name, reason)


def check_otherwise(op: Mapping[str, Any], name: str, m: Manifest) -> Iterator[Finding]:
    ports = {p.get("name") for p in op.get("inputs") or ()}
    for where, out in read_outputs(op):
        for o in out.get("otherwise") or ():
            kind, when = o.get("kind"), o.get("when") or {}
            if m.find_kind(str(kind)) is None:
                yield Finding("OTHERWISE_INVALID", name, f"{kind} is not a declared kind")
            if ("port" in when) == ("param" in when and when.get("param") is not None):
                yield Finding("OTHERWISE_INVALID", name, "name a param or a port, not both")
            elif when.get("port") is not None:
                if when["port"] not in ports:
                    yield Finding("OTHERWISE_INVALID", name, f"no port {when['port']!r}")
            else:
                fields, p = list(op.get("params") or ()), None
                for key in str(when["param"]).split("."):
                    p = next((f for f in fields if f.get("name") == key), None)
                    if p is None:
                        break
                    fields = list(p.get("fields") or ())
                if p is None:
                    yield Finding("OTHERWISE_INVALID", name, f"no param {when['param']!r}")
                elif p.get("choices") and when.get("equals") not in p["choices"]:
                    yield Finding("OTHERWISE_INVALID", name, f"{when.get('equals')!r} is not a {when['param']} choice")
            if f"`{kind}`" not in str(out.get("doc") or ""):
                yield Finding("OTHERWISE_INVALID", name, f"the {where} prose does not name `{kind}`")


def check_emits(op: Mapping[str, Any], name: str, m: Manifest) -> Iterator[Finding]:
    for where, out in read_outputs(op):
        kind = m.find_kind(str(out.get("kind")))
        if kind is None:
            yield Finding("OUTPUT_KIND_UNKNOWN", name, f"{where} produces undeclared {out.get('kind')!r}")
        elif out.get("collection") and not kind.get("key"):
            yield Finding("OUTPUT_NOT_COLLECTABLE", name, f"{where} is a collection of {out.get('kind')}, "
                          "which declares no key")


def check_types(m: Manifest, common: list[dict[str, Any]]) -> Iterator[Finding]:
    for op in m.ops:
        name = str(op.get("name") or "")
        for at, p in walk_params(op.get("params"), name):
            yield from check_param(at, p)
        yield from check_example_types(op, name, common)
        yield from check_otherwise(op, name, m)
        yield from check_emits(op, name, m)
