from __future__ import annotations

import pytest

from mechbench_compute.lexicon.address import (
    AddressError,
    is_extension_spelling,
    parse,
    parse_extension,
    parse_kind,
    parse_mark,
    parse_op,
)

PIN = "sha256:" + "ab" * 32


@pytest.mark.parametrize("text, thing, bare, version, pin, core", [
    ("records/filter", "op", "records/filter", None, None, True),
    ("~canonical/ops/records/filter", "op", "records/filter", None, None, True),
    ("alice/extras/ops/geometry/align", "op", "alice/extras/ops/geometry/align", None, None, False),
    ("alice/extras/ops/geometry/align@2", "op", "alice/extras/ops/geometry/align", 2, None, False),
    (f"alice/extras/ops/geometry/align@{PIN}", "op", "alice/extras/ops/geometry/align", None, PIN, False),
    ("alice/extras/extensions/interp-extras", "extension", "alice/extras/extensions/interp-extras", None, None, False),
    ("alice/extras/extensions/interp-extras@3", "extension", "alice/extras/extensions/interp-extras", 3, None, False),
    (f"alice/extras/extensions/interp-extras@{PIN}", "extension", "alice/extras/extensions/interp-extras",
     None, PIN, False),
    ("alice/extras/kinds/geometry/alignment", "kind", "alice/extras/kinds/geometry/alignment", None, None, False),
    ("alice/extras/marks/circuit@1", "mark", "alice/extras/marks/circuit", 1, None, False),
    ("heat@1", "mark", "heat", 1, None, True),
    ("alice/extras/architectures/qwen3", "architecture", "alice/extras/architectures/qwen3", None, None, False),
])
def test_every_row_of_the_address_table_reads(text, thing, bare, version, pin, core):
    got = parse(text)
    assert (got.thing, got.bare, got.version, got.pin, got.is_core) == (thing, bare, version, pin, core)


@pytest.mark.parametrize("text", [
    "records/filter",
    "alice/extras/ops/geometry/align@2",
    f"alice/extras/ops/geometry/align@{PIN}",
    "alice/extras/extensions/interp-extras@3",
    "alice/extras/marks/circuit@1",
    "heat@1",
])
def test_an_address_prints_as_it_was_written(text):
    assert str(parse(text)) == text


def test_a_two_level_name_is_what_the_reader_expects():
    assert parse_kind("records/record").thing == "kind"
    assert parse_kind("~canonical/kinds/records/record").bare == "records/record"
    assert parse_kind("collection").bare == "collection"
    assert parse_op("records/filter").family == "records"


@pytest.mark.parametrize("text, why", [
    ("records/filter/2", "never a path segment"),
    ("~canonical/ops/records/filter/2", "never a path segment"),
    ("alice/extras/ops/geometry/align/2", "never a path segment"),
    ("alice/extras/extensions/interp-extras/3", "never a path segment"),
    ("records/filter@2", "not versioned by @"),
    ("alice/extras/ops/geometry/align@v2", "a counter"),
    ("alice/extras/ops/geometry/align@sha256:abc", "a counter"),
    ("alice/extras/ops/geometry/align@2@3", "more than one @"),
    ("alice/extras/marks/circuit", "always named with its version"),
    ("heat", "always named with its version"),
    ("alice/extras/kinds/geometry/alignment@2", "versioned by its extension"),
    ("alice/extras/architectures/qwen3@2", "versioned by its extension"),
    ("alice/extras/widgets/thing", "the third segment"),
    ("Alice/extras/ops/geometry/align", "owner's handle"),
    ("alice/extras/ops/geometry", "two levels"),
    ("Records/Filter", "family/leaf"),
])
def test_what_is_not_an_address_is_refused_with_its_reason(text, why):
    with pytest.raises(AddressError, match=why):
        parse(text)


def test_a_reader_for_one_thing_refuses_another():
    assert parse_extension("alice/extras/extensions/interp-extras").name == "interp-extras"
    assert parse_mark("alice/extras/marks/circuit@1").scope == "alice/extras"
    with pytest.raises(AddressError, match="names an extension, not an op"):
        parse_op("alice/extras/extensions/interp-extras")


def test_a_pin_replaces_a_counter():
    got = parse("alice/extras/ops/geometry/align@2").with_pin(PIN)
    assert str(got) == f"alice/extras/ops/geometry/align@{PIN}"


def test_only_an_owner_and_project_spelling_is_an_extension_spelling():
    assert is_extension_spelling("alice/extras/ops/geometry/align@2")
    assert not is_extension_spelling("records/filter")
    assert not is_extension_spelling("~canonical/ops/records/filter")
    assert not is_extension_spelling("records/filter/2")
