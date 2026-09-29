from __future__ import annotations

import re
from dataclasses import dataclass

OP_ROOT = "~canonical/ops/"

KIND_ROOT = "~canonical/kinds/"

COLLECTION = "collection"

CATEGORY = {"ops": "op", "kinds": "kind", "marks": "mark",
            "extensions": "extension", "architectures": "architecture"}

PLURAL = {thing: category for category, thing in CATEGORY.items()}

SEGMENT = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
_NAME = re.compile(rf"^{SEGMENT}/{SEGMENT}$")
_MARK = re.compile(rf"^{SEGMENT}$")
_HANDLE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")
_MODEL_TYPE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_COUNTER = re.compile(r"^[1-9]\d{0,8}$")
_PIN = re.compile(r"^sha256:[0-9a-f]{64}$")


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Address:
    thing: str
    name: str
    owner: str | None = None
    project: str | None = None
    version: int | None = None
    pin: str | None = None

    @property
    def is_core(self) -> bool:
        return self.owner is None

    @property
    def scope(self) -> str | None:
        return None if self.is_core else f"{self.owner}/{self.project}"

    @property
    def bare(self) -> str:
        return self.name if self.is_core else f"{self.scope}/{PLURAL[self.thing]}/{self.name}"

    @property
    def family(self) -> str | None:
        return self.name.split("/", 1)[0] if "/" in self.name else None

    def with_pin(self, pin: str) -> Address:
        return Address(self.thing, self.name, self.owner, self.project, None, pin)

    def __str__(self) -> str:
        if self.pin is not None:
            return f"{self.bare}@{self.pin}"
        if self.version is not None:
            return f"{self.bare}@{self.version}"
        return self.bare


def split_suffix(text: str) -> tuple[str, int | None, str | None]:
    base, at, tail = text.partition("@")
    if not at:
        return base, None, None
    if "@" in tail:
        raise AddressError(f"{text!r} is not an address: it has more than one @")
    if _COUNTER.match(tail):
        return base, int(tail), None
    if _PIN.match(tail):
        return base, None, tail
    raise AddressError(
        f"{text!r} is not an address: after @ comes a version (a counter, @3) or a pin (@sha256:<64 hex>)")


def parse(text: str, expect: str = "op") -> Address:
    def fail(why: str) -> AddressError:
        return AddressError(f"{text!r} is not an address: {why}")

    base, counter, pin = split_suffix(text.strip())
    versioned = counter is not None or pin is not None
    segs = base.split("/")
    if len(segs) == 1:
        if expect == "kind" and base == COLLECTION and not versioned:
            return Address("kind", COLLECTION)
        if not _MARK.match(base):
            raise fail("a single name is a core mark, name@version")
        if counter is None:
            raise fail("a mark is always named with its version: name@1")
        return Address("mark", base, version=counter)
    if segs[0] == "~canonical":
        thing = {"ops": "op", "kinds": "kind"}.get(segs[1])
        name = "/".join(segs[2:])
        if thing is None or not _NAME.match(name):
            if thing is not None and name.split("/")[-1].isdigit():
                raise fail("a version is never a path segment; write it after @")
            raise fail("a ~canonical address is ~canonical/ops/family/leaf or ~canonical/kinds/family/leaf")
        if versioned:
            raise fail("a core operation or kind is not versioned by @")
        return Address(thing, name)
    if len(segs) == 2:
        if not _NAME.match(base):
            raise fail("a core name is family/leaf, lowercase words joined by hyphens")
        if versioned:
            raise fail("a core operation or kind is not versioned by @")
        return Address(expect, base)
    owner, project, category, *rest = segs
    thing = CATEGORY.get(category)
    if thing is None and len(segs) == 3 and segs[2].isdigit() and _NAME.match("/".join(segs[:2])):
        raise fail("a version is never a path segment; write it after @")
    if not _HANDLE.match(owner):
        raise fail(f"{owner!r} is not an owner's handle")
    if not _HANDLE.match(project):
        raise fail(f"{project!r} is not a project")
    if thing is None:
        raise fail(f"the third segment says what it is: {', '.join(CATEGORY)}")
    if any(s.isdigit() for s in rest):
        raise fail("a version is never a path segment; write it after @")
    name = "/".join(rest)
    if thing in ("op", "kind"):
        if not _NAME.match(name):
            raise fail(f"after {category}/ comes family/leaf, two levels")
        if thing == "kind" and versioned:
            raise fail("a kind is versioned by its extension, not by @")
        return Address(thing, name, owner, project, counter, pin)
    if len(rest) != 1:
        raise fail(f"{category}/ is followed by one name")
    if thing == "extension":
        if not _HANDLE.match(name):
            raise fail("an extension is owner/project/extensions/<name>")
        return Address(thing, name, owner, project, counter, pin)
    if thing == "mark":
        if not _MARK.match(name):
            raise fail("a mark is owner/project/marks/<name>@<n>")
        if counter is None:
            raise fail("a mark is always named with its version: …/marks/<name>@1")
        return Address(thing, name, owner, project, counter)
    if not _MODEL_TYPE.match(name):
        raise fail("an architecture is owner/project/architectures/<model_type>")
    if versioned:
        raise fail("an architecture is versioned by its extension, not by @")
    return Address(thing, name, owner, project)


def parse_as(text: str, thing: str) -> Address:
    got = parse(text, "kind" if thing == "kind" else "op")
    if got.thing != thing:
        raise AddressError(f"{text!r} names {'an' if got.thing[0] in 'aeiou' else 'a'} {got.thing}, not {'an' if thing[0] in 'aeiou' else 'a'} {thing}")
    return got


def parse_op(text: str) -> Address:
    return parse_as(text, "op")


def parse_kind(text: str) -> Address:
    return parse_as(text, "kind")


def parse_mark(text: str) -> Address:
    return parse_as(text, "mark")


def parse_extension(text: str) -> Address:
    return parse_as(text, "extension")


def is_extension_spelling(text: str) -> bool:
    segs = str(text).strip().split("@", 1)[0].split("/")
    return len(segs) >= 3 and segs[2] in CATEGORY and not segs[0].startswith("~")


def address_op(scope: str, name: str) -> str:
    return f"{scope}/ops/{name}"


def address_kind(scope: str, name: str) -> str:
    return f"{scope}/kinds/{name}"


def address_extension(scope: str, name: str) -> str:
    return f"{scope}/extensions/{name}"
