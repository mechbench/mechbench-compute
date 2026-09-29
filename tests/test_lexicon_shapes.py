from __future__ import annotations

from collections.abc import Iterator

import pytest

from mechbench_compute import lexicon
from mechbench_compute.conformance import check_core, format_findings, select_findings
from mechbench_compute.lexicon import OPS, Op
from mechbench_compute.lexicon._base import Param


def _walk(params: tuple[Param, ...], where: str) -> Iterator[tuple[str, Param]]:
    for p in params:
        yield f"{where}.{p.name}", p
        yield from _walk(p.fields, f"{where}.{p.name}")


def _all_params() -> list[tuple[str, Param]]:
    out = list(_walk(lexicon.COMMON, "common"))
    for op in OPS:
        out += list(_walk(op.params, op.name))
    return out


ALL = _all_params()


def assert_clean(codes: tuple[str, ...], *, subject: str | None = None, at: str | None = None) -> None:
    hits = select_findings(check_core(), codes, subject=subject, at=at)
    assert not hits, format_findings(hits)


@pytest.mark.parametrize(("where", "p"), ALL, ids=[w for w, _ in ALL])
def test_the_type_is_in_the_grammar(where: str, p: Param) -> None:
    assert_clean(("TYPE_INVALID",), at=where)


@pytest.mark.parametrize(("where", "p"), ALL, ids=[w for w, _ in ALL])
def test_every_object_is_declared(where: str, p: Param) -> None:
    assert_clean(("OBJECT_UNDECLARED", "FIELDS_INVALID"), at=where)


@pytest.mark.parametrize(("where", "p"), ALL, ids=[w for w, _ in ALL])
def test_value_and_choices_are_well_formed(where: str, p: Param) -> None:
    assert_clean(("VALUE_INVALID", "CHOICES_INVALID"), at=where)


def test_every_closed_set_is_the_codes_own() -> None:
    assert_clean(("CLOSED_SET_UNENFORCED",))


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_the_example_conforms_to_the_declared_types(op: Op) -> None:
    if op.example is None:
        pytest.skip("no example written")
    assert_clean(("EXAMPLE_TYPE_MISMATCH",), subject=op.name)


def test_shapes_reach_the_published_dict() -> None:
    import json

    capture = lexicon.BY_NAME["trajectory/capture"].to_dict()
    by = {p["name"]: p for p in capture["params"]}
    assert by["axis"]["choices"] == ["layers", "positions"]
    assert by["pool"]["value"] == "pool"
    lora = {p["name"]: p for p in lexicon.BY_NAME["adapter/train"].to_dict()["params"]}["lora"]
    assert [f["name"] for f in lora["fields"]][:2] == ["rank", "alpha"]
    json.dumps([op.to_dict() for op in OPS])


@pytest.mark.parametrize("op", [op for op in OPS if op.output and op.output.otherwise], ids=lambda op: op.name)
def test_what_an_op_emits_instead_is_declared_against_the_node(op: Op) -> None:
    assert_clean(("OTHERWISE_INVALID",), subject=op.name)


@pytest.mark.parametrize("op", OPS, ids=[op.name for op in OPS])
def test_every_need_is_in_the_vocabulary(op: Op) -> None:
    assert_clean(("NEED_UNKNOWN",), subject=op.name)


def test_a_need_outside_the_vocabulary_is_refused() -> None:
    from mechbench_compute.lexicon._base import Output, Resume

    with pytest.raises(ValueError, match="not needs"):
        Op("x/y", "s", "d", (), needs=frozenset({"gpu"}))
    with pytest.raises(ValueError, match="not needs"):
        Op("x/y", "s", "d", (), needs=frozenset({"network:"}))
    with pytest.raises(ValueError, match="not both"):
        Op("x/y", "s", "d", (), output=Output("records/record"),
           outputs={"a": Output("records/record")})
    with pytest.raises(TypeError, match="not an Output"):
        Op("x/y", "s", "d", (), outputs={"a": "records/record"})
    with pytest.raises(ValueError, match="resume level"):
        Op("x/y", "s", "d", (), resume=Resume("sometimes"))
    with pytest.raises(TypeError):
        Op("x/y", "s", "d", (), requires="pure")


@pytest.mark.parametrize(("needs", "requires"), [
    (set(), "pure"),
    ({"secrets", "executor.sub"}, "pure"),
    ({"model.forward"}, "mlx-local"),
    ({"model.backward"}, "mlx-local"),
    ({"provider.embed"}, "remote"),
    ({"network:huggingface.co"}, "remote"),
    ({"model.sample", "provider.chat"}, "by-model"),
    ({"model.forward", "network:example.com"}, "by-model"),
])
def test_requires_is_derived_from_needs(needs: set[str], requires: str) -> None:
    assert Op("x/y", "s", "d", (), needs=frozenset(needs)).requires == requires


def test_the_manifest_fields_reach_the_published_dict() -> None:
    from mechbench_compute.lexicon._base import Output, Resume

    op = Op("x/y", "s", "d", (), outputs={"a": Output("records/record")},
            needs=frozenset({"model.forward", "secrets"}), resume=Resume("restart", items=True),
            deterministic=False, min_compute="0.170.0")
    d = op.to_dict()
    assert d["needs"] == ["model.forward", "secrets"]
    assert d["requires"] == "mlx-local"
    assert d["resume"] == {"level": "restart", "items": True}
    assert d["deterministic"] is False
    assert d["min_compute"] == "0.170.0"
    assert d["outputs"]["a"]["kind"] == "records/record"
    bare = Op("x/y", "s", "d", ()).to_dict()
    assert "outputs" not in bare and "min_compute" not in bare
    assert bare["resume"] == {"level": "reproducible", "items": False}
