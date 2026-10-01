from __future__ import annotations

import re

import pytest

from mechbench_compute import lexicon
from mechbench_compute.block_params import COMMON, accepted, check_inputs, check_params, ports
from mechbench_compute.conformance import check_core, format_findings, select_findings
from mechbench_compute.conformance.check_ops import INTERNAL
from mechbench_compute.lexicon import BY_NAME, OPS, Op

NOT_VERBS = frozenset({"activations/examples", "eval/benchmark", "intervene/path", "records/jq",
                       "records/python", "tools/calc", "weights/circuit"})

ALLOWED: dict[str, frozenset[str]] = {
    op.name: frozenset({"NO_OUTPUT"} if op.family == "tools" else ())
    | frozenset({"OP_NOT_VERB"} if op.name in NOT_VERBS else ())
    for op in OPS
}


def assert_clean(codes: tuple[str, ...], subject: str | None = None) -> None:
    hits = select_findings(check_core(), codes, subject=subject, allowed=ALLOWED)
    assert not hits, format_findings(hits)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_entry_is_complete(op: Op) -> None:
    assert_clean(("NO_SUMMARY", "NO_DESCRIPTION", "NO_OUTPUT", "NO_OUTPUT_DOC", "PARAM_UNTYPED",
                  "PARAM_UNDOCUMENTED", "PARAM_DUPLICATE", "PORT_DUPLICATE", "PORT_UNDOCUMENTED"), op.name)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_every_port_names_a_declared_kind(op: Op) -> None:
    assert_clean(("PORT_KIND_UNKNOWN", "PORT_NAME_INVALID"), op.name)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_no_param_names_a_port(op: Op) -> None:
    assert_clean(("PARAM_NAMES_PORT",), op.name)
    for retired in ("user_field", "system_field", "prefill_field", "answer_field",
                    "prediction_field", "reference_field", "messages_field",
                    "label_field", "label_coord", "pairwise_fields", "collection_path"):
        assert retired not in op.param_names, f"{op.name} still declares {retired}"


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_an_expression_on_several_ports_says_which_it_reads(op: Op) -> None:
    assert_clean(("READS_UNKNOWN", "REPLACES_UNKNOWN", "READS_NOT_EXPRESSION", "READS_UNSAID"), op.name)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_summary_is_one_sentence_for_a_stranger(op: Op) -> None:
    assert_clean(("SUMMARY_RESTATES_REF", "SUMMARY_NOT_SENTENCE", "SUMMARY_TOO_LONG"), op.name)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_no_house_idioms_in_published_text(op: Op) -> None:
    assert_clean(("HOUSE_IDIOM",), op.name)


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_example_would_be_accepted(op: Op) -> None:
    if op.example is None:
        pytest.skip("no example written")
    assert_clean(("EXAMPLE_REFUSED",), op.name)
    check_params(op.name, op.example)
    check_inputs(op.name, dict(op.example_inputs or {}))


def test_common_params_are_documented() -> None:
    assert_clean(("PARAM_UNDOCUMENTED", "HOUSE_IDIOM"), "common")
    assert COMMON == frozenset(p.name for p in lexicon.COMMON)


def test_block_params_is_derived_from_the_lexicon() -> None:
    for name, op in BY_NAME.items():
        assert accepted(name) == op.param_names
        assert ports(name) == op.port_names
    assert accepted("no/such-op") is None
    assert_clean(("PARAM_REDECLARES_COMMON",))


def test_names_are_two_level_bare_and_unique() -> None:
    assert_clean(("NAME_INVALID", "NAME_DUPLICATE"))
    families: dict[str, int] = {}
    for op in OPS:
        families[op.family] = families.get(op.family, 0) + 1
    lonely = sorted(f for f, n in families.items() if n < 2)
    assert not lonely, f"families of one: {lonely}"
    assert set(families) == {
        "records", "text", "eval", "logits", "activations", "geometry",
        "intervene", "direction", "dictionary", "trajectory", "adapter", "weights", "tools",
    }


def test_operations_are_verbs_and_never_share_a_kinds_name() -> None:
    from mechbench_compute.lexicon import kinds as K

    assert_clean(("OP_NAMES_KIND", "OP_NOT_VERB"))
    assert NOT_VERBS <= set(BY_NAME), sorted(NOT_VERBS - set(BY_NAME))
    assert lexicon.resolve("logits/read", warn=False) == "logits/read"
    assert lexicon.resolve("records/tabulate", warn=False) == "records/tabulate"
    assert lexicon.resolve("geometry/compare", warn=False) == "geometry/compare"
    assert "logits/decision" in K.BY_KIND and "records/table" in K.BY_KIND


def test_to_dict_is_json_shaped() -> None:
    assert_clean(("NOT_JSON",))


def _publishable(where: str, text: str) -> None:
    for pat in INTERNAL:
        m = pat.search(text)
        assert m is None, f"{where}: {m.group(0)!r} is a reference nobody outside this repo can follow"


def test_every_family_is_declared_and_described() -> None:
    from mechbench_compute.lexicon import kinds as K

    used = {op.family for op in OPS} | {k.family for k in K.KINDS if k.name != K.COLLECTION}
    declared = {f.name for f in lexicon.FAMILIES}
    assert used == declared, f"undeclared: {sorted(used - declared)}; empty: {sorted(declared - used)}"
    for f in lexicon.FAMILIES:
        s = f.summary.strip()
        assert s and s[-1] in ".?!" and len(s) < 400, f"family {f.name}: summary"
        assert f.doc.strip(), f"family {f.name}: no doc"
        _publishable(f"family {f.name}", f.summary + "\n" + f.doc)
        assert lexicon.BY_FAMILY[f.name] is f
        import json
        json.dumps(f.to_dict())


def test_every_value_type_is_described() -> None:
    import json

    from mechbench_compute import points
    from mechbench_compute.lexicon import values as V

    for v in lexicon.VALUES:
        s = v.summary.strip()
        assert s and s[-1] in ".?!" and len(s) < 400, f"value {v.name}: summary"
        assert v.doc.strip(), f"value {v.name}: no doc"
        assert re.fullmatch(r"[a-z]+", v.name), v.name
        for f, spec in v.fields.items():
            assert spec.get("type") and spec.get("description"), f"value {v.name}.{f}: type and description"
        for r in v.required:
            assert r in v.fields, f"value {v.name}: required {r!r} is not a field"
        _publishable(f"value {v.name}", "\n".join([v.summary, v.doc, *(f["description"] for f in v.fields.values())]))
        json.dumps(v.to_dict())
    assert V.DOCUMENTED_POINTS == frozenset(points.POINTS)
    for name in points.POINTS:
        assert f"`{name}`" in V.BY_VALUE["point"].doc, f"point {name} is not on the page"
    assert {v.name for v in lexicon.VALUES if v.grammar} == {"position", "pool", "point"}


class TestTheNameAndItsRendering:
    def test_it_reads_as_a_family_and_a_verb(self) -> None:
        assert lexicon.title("records/cross") == "Records :: Cross"
        assert lexicon.title("logits/read-layers") == "Logits :: Read Layers"
        assert (lexicon.title("activations/capture-attention")
                == "Activations :: Capture Attention")

    def test_every_name_in_the_lexicon_reads_back(self) -> None:
        for name in (*lexicon.BY_NAME, *(k.name for k in lexicon.KINDS)):
            assert lexicon.name_of_title(lexicon.title(name)) == name, name

    def test_it_takes_a_stored_path_as_readily_as_a_name(self) -> None:
        assert lexicon.title("~canonical/ops/records/cross") == "Records :: Cross"
        assert lexicon.title("~canonical/ops/records/cross/1") == "Records :: Cross"

    def test_the_identity_is_the_path_and_the_name_is_bare(self) -> None:
        assert lexicon.canonical_path("records/cross") == "~canonical/ops/records/cross"
        assert lexicon.display_name("~canonical/ops/records/cross") == "records/cross"
        assert lexicon.title("records/cross") == "Records :: Cross"

    def test_it_needs_no_table_of_its_own(self) -> None:
        assert lexicon.title("weights/decompose") == "Weights :: Decompose"
        assert lexicon.title("nothing/here-yet") == "Nothing :: Here Yet"


class TestWhatAnOperationNeeds:
    def test_needs_are_declared(self) -> None:
        assert_clean(("NEED_UNDECLARED",))

    def test_no_declared_need_goes_unused(self) -> None:
        assert_clean(("NEED_UNUSED",))

    def test_nothing_reaches_past_the_needs(self) -> None:
        from mechbench_compute import ops

        assert ops.REACHING_PAST_NEEDS == {}, (
            "every ctx member an operation touches is named by a need it declares")

    def test_no_op_reaches_into_the_executor(self) -> None:
        import pathlib

        import mechbench_compute

        root = pathlib.Path(mechbench_compute.__file__).parent
        reaching = [str(f.relative_to(root)) for d in ("ops", "live")
                    for f in sorted((root / d).rglob("*.py"))
                    if "ctx.executor." in f.read_text() or "type(ctx.executor)" in f.read_text()]
        assert not reaching, (
            f"{reaching} reach into the executor; ask ctx for it: sub, provider, "
            "memo, materialize, evict_model")

    def test_the_fused_set_is_the_one_before_the_boundary(self) -> None:
        from mechbench_compute import ops

        before = {name for name, mod in ops.load_modules().items()
                  if mod.OP.requires == "mlx-local" and mod.OP.port("adapter") is not None}
        assert {name for name in ops.load_modules() if ops.fuses_adapter(name)} == before
        assert {name for name in ops.load_modules() if ops.fuses_adapter_locally(name)} == {"text/chat"}

    def test_derived_requires_match_the_wire_form(self) -> None:
        import json
        import pathlib

        fixture = pathlib.Path(__file__).parent / "fixtures" / "declarations_before_needs.json"
        before = json.loads(fixture.read_text())
        assert {op.name: op.requires for op in lexicon.OPS} == before["requires"]

    def test_resume_is_declared_as_it_was_tabled(self) -> None:
        import json
        import pathlib

        from mechbench_compute import resume

        fixture = pathlib.Path(__file__).parent / "fixtures" / "declarations_before_needs.json"
        before = json.loads(fixture.read_text())
        assert {op.name: {"level": resume.resume_level(op.name), "items": resume.item_resumable(op.name)}
                for op in lexicon.OPS} == before["resume"]

    def test_the_standalone_set_is_the_one_before_needs(self) -> None:
        import json
        import pathlib

        from mechbench_compute import ops

        fixture = pathlib.Path(__file__).parent / "fixtures" / "declarations_before_needs.json"
        assert sorted(ops.find_standalone()) == json.loads(fixture.read_text())["standalone"]

    def test_an_undeclared_member_raises_at_run_time(self) -> None:
        from mechbench_compute import ops

        op = lexicon.Op("x/y", "s", "d", ())
        ctx = ops.Context.for_op(op, executor=object(), secrets={"hf": {}})
        with pytest.raises(ops.NeedNotDeclared, match="x/y did not declare"):
            ctx.model("m")
        with pytest.raises(ops.NeedNotDeclared):
            ctx.executor._limiter
        with pytest.raises(ops.NeedNotDeclared):
            (ctx.secrets or {}).get("hf")

    def test_a_declared_member_is_lent(self) -> None:
        from mechbench_compute import ops

        op = lexicon.Op("x/y", "s", "d", (), needs=frozenset({"executor.sub", "secrets", "model.forward"}))
        executor = object()
        ctx = ops.Context.for_op(op, executor=executor, secrets={"hf": {}}, loaded="m")
        assert ctx.executor is executor and ctx.secrets == {"hf": {}} and ctx.model("r") == "m"

    def test_the_walk_sees_through_a_helper(self, tmp_path, monkeypatch) -> None:
        import importlib

        from mechbench_compute import ops

        (tmp_path / "walked_op.py").write_text(
            "def _load(ctx, ref):\n"
            "    return ctx.model(ref)\n"
            "\n"
            "def _unused(ctx):\n"
            "    return ctx.executor\n"
            "\n"
            "def run(ctx, inputs, params):\n"
            "    getattr(ctx, 'run_params', None)\n"
            "    return _load(ctx, params['model'])\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        mod = importlib.import_module("walked_op")
        assert ops.read_context_uses(mod) == {"model", "run_params"}

    def test_every_operation_declares_one_of_the_four(self) -> None:
        for op in lexicon.OPS:
            assert op.requires in ("pure", "mlx-local", "remote", "by-model"), op.name

    def test_the_operations_that_run_either_side_are_the_ones_that_chat_with_a_provider(self) -> None:
        either = {op.name for op in lexicon.OPS if op.requires == "by-model"}
        assert either == {op.name for op in lexicon.OPS if "provider.chat" in op.needs}
        assert either == {"text/chat", "eval/judge"}

    def test_the_pure_registry_is_pure(self) -> None:
        from mechbench_compute import ops

        for name in ops.find_standalone():
            if name in lexicon.BY_NAME:
                assert not lexicon.BY_NAME[name].needs, name

    def test_only_publishing_and_loading_a_dictionary_need_the_network_without_a_model(self) -> None:
        assert {op.name for op in lexicon.OPS if op.requires == "remote"} == {"adapter/publish", "dictionary/load"}
