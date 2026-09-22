"""The lexicon is the operation reference: every canonical op described,
every parameter typed and explained, in language a stranger can read.

`test_block_params.py` proves the declared NAMES equal what the code
reads. This file proves the rest of the declaration is fit to publish:
nothing is blank, every example would pass `check_params`, and no entry
leans on a house idiom — a task number, an experiment step, an epic —
that resolves to nothing outside this repository.
"""

from __future__ import annotations

import re

import pytest

from mechbench_compute import lexicon
from mechbench_compute.block_params import ACCEPTED, COMMON, check_inputs, check_params
from mechbench_compute.lexicon import BY_NAME, OPS, Op

# What must never appear in published text: a bare six-digit id, a
# tracker reference, a numbered step or experiment. Every one of these
# is an argument to us and an opaque string to a reader of the
# documentation site.
INTERNAL = [
    re.compile(r"\b0\d{5}\b"),
    re.compile(r"\b(?:task|tasks|epic|epics)\s+\d", re.I),
    re.compile(r"\bstep\s+\d{2}\b", re.I),
    re.compile(r"\bexperiment\s+0\d{2}\b", re.I),
    re.compile(r"\bmechbench-experiments\b", re.I),
]


def _texts(op: Op) -> list[tuple[str, str]]:
    out = [("summary", op.summary), ("description", op.description),
           ("output", op.output.doc if op.output else "")]
    out += [(f"port {p.name}.doc", p.doc) for p in op.inputs]
    out += [(f"param {p.name}.doc", p.doc) for p in op.params]
    out += [(f"param {p.name}.type", p.type) for p in op.params]
    return out


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_entry_is_complete(op: Op) -> None:
    assert op.summary.strip(), f"{op.name}: no summary"
    assert op.description.strip(), f"{op.name}: no description"
    if op.output is not None:
        assert op.output.doc.strip(), f"{op.name}: says nothing about what it produces"
    else:
        assert op.family == "tools", f"{op.name}: emits nothing and is not a tool"
    for p in op.params:
        assert p.type.strip(), f"{op.name}.{p.name}: no type"
        assert p.doc.strip(), f"{op.name}.{p.name}: no description"
    names = [p.name for p in op.params]
    assert len(names) == len(set(names)), f"{op.name}: duplicate params"
    ports = [p.name for p in op.inputs]
    assert len(ports) == len(set(ports)), f"{op.name}: duplicate ports"
    for p in op.inputs:
        assert p.doc.strip(), f"{op.name} port {p.name}: no description"


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_every_port_names_a_declared_kind(op: Op) -> None:
    """A port's kind is a name in the kinds registry, so the docs can link
    it and `check_inputs` can walk its ancestry."""
    for p in op.inputs:
        for k in p.kinds:
            assert k in lexicon.BY_KIND, f"{op.name} port {p.name}: unknown kind {k!r}"
        assert p.name == lexicon.WILDCARD or re.fullmatch(r"[a-z][a-z0-9_]*", p.name), (
            f"{op.name}: port name {p.name!r}")


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_no_param_names_a_port(op: Op) -> None:
    """An input is a port, never a param: the two would be the same value
    spelled twice, and `check_params` could not tell an unwired port from
    a typo. The sentence "by edge, or the param" is the smell."""
    assert not (op.param_names & op.port_names), (
        f"{op.name}: {sorted(op.param_names & op.port_names)} declared as both")
    for where, text in _texts(op):
        assert not re.search(r"by edge,? or (the|by) (the )?param", text, re.I), (
            f"{op.name} {where}: an input is described as a param")
    for retired in ("user_field", "system_field", "prefill_field", "answer_field",
                    "prediction_field", "reference_field", "messages_field",
                    "label_field", "label_coord", "pairwise_fields", "collection_path"):
        assert retired not in op.param_names, f"{op.name} still declares {retired}"


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_summary_is_one_sentence_for_a_stranger(op: Op) -> None:
    s = op.summary.strip()
    # One sentence: ends once, and does not restate the block ref the
    # page already shows.
    assert not s.startswith("~canonical/"), f"{op.name}: summary restates the ref"
    assert s[-1] in ".?!", f"{op.name}: summary should end as a sentence does"
    assert len(s) < 400, f"{op.name}: summary is a paragraph, not a sentence"


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_no_house_idioms_in_published_text(op: Op) -> None:
    for where, text in _texts(op):
        for pat in INTERNAL:
            m = pat.search(text)
            assert m is None, (
                f"{op.name} {where}: {m.group(0)!r} is a reference nobody "
                f"outside this repo can follow — rewrite it as what it means")


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_example_would_be_accepted(op: Op) -> None:
    if op.example is None:
        pytest.skip("no example written")
    # The example must not name a param the block refuses — the exact
    # mistake the documentation exists to prevent — and its inputs must
    # land on ports the block has. A `{"$fetch": …}` carries no kind
    # until it resolves, so the example's inputs are checked by port
    # name, with every required port present.
    check_params(op.name, op.example)
    inputs = dict(op.example_inputs or {})
    for p in op.inputs:
        if p.required and not p.wildcard and p.name not in inputs:
            assert p.name in inputs, f"{op.name}: example wires no {p.name!r}, which is required"
    check_inputs(op.name, inputs)


def test_common_params_are_documented() -> None:
    for p in lexicon.COMMON:
        assert p.type.strip() and p.doc.strip(), f"common param {p.name} undocumented"
        for pat in INTERNAL:
            assert not pat.search(p.doc), f"common param {p.name}: house idiom in doc"
    assert COMMON == frozenset(p.name for p in lexicon.COMMON)


def test_block_params_is_derived_from_the_lexicon() -> None:
    """One source. `block_params` must not carry its own table."""
    assert set(ACCEPTED) == set(BY_NAME)
    for name, op in BY_NAME.items():
        assert ACCEPTED[name] == op.param_names
    # A param cannot be both common and op-specific: the table would
    # accept it twice and the docs would describe it twice.
    for op in OPS:
        assert not (op.param_names & COMMON), f"{op.name} redeclares a common param"


def test_names_are_two_level_bare_and_unique() -> None:
    """docs/LEXICON.md §1: `family/op`, exactly two levels, no root, no
    version segment; §2: every family has at least two members."""
    names = [op.name for op in OPS]
    assert len(names) == len(set(names)), "two ops share a name"
    for name in names:
        assert re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", name), name
        assert not name.endswith(tuple(f"/{d}" for d in "0123456789")), name
    families: dict[str, int] = {}
    for op in OPS:
        families[op.family] = families.get(op.family, 0) + 1
    lonely = sorted(f for f, n in families.items() if n < 2)
    assert not lonely, f"families of one: {lonely}"
    assert set(families) == {
        "records", "text", "eval", "logits", "activations", "geometry",
        "intervene", "direction", "trajectory", "adapter", "weights", "tools",
    }


def test_operations_are_verbs_and_never_share_a_kinds_name() -> None:
    """docs/LEXICON.md §1: an operation's leaf is an imperative verb and
    a kind's is a noun, so the two vocabularies never meet on a name —
    `logits/read` emits `logits/decision`, `records/tabulate` a
    `records/table`. The grammar itself is not machine-checkable; its
    consequence is, and this holds it."""
    from mechbench_compute.lexicon import kinds as K

    shared = sorted(op.name for op in OPS if op.name in K.BY_KIND)
    assert shared == [], f"an operation shares a kind's name: {shared}"
    # The retired noun spellings resolve to the verbs.
    assert lexicon.resolve("logits/read", warn=False) == "logits/read"
    assert lexicon.resolve("records/tabulate", warn=False) == "records/tabulate"
    assert lexicon.resolve("geometry/compare", warn=False) == "geometry/compare"
    # …and a kind of the same name is untouched.
    assert "logits/decision" in K.BY_KIND and "records/table" in K.BY_KIND


def test_to_dict_is_json_shaped() -> None:
    import json

    for op in OPS:
        json.dumps(op.to_dict())  # REQUIRED never leaks; defaults are JSON values
        for p in op.params:
            d = p.to_dict()
            assert ("default" in d) != d["required"]


def _publishable(where: str, text: str) -> None:
    for pat in INTERNAL:
        m = pat.search(text)
        assert m is None, f"{where}: {m.group(0)!r} is a reference nobody outside this repo can follow"


def test_every_family_is_declared_and_described() -> None:
    """The families are the namespaces of both vocabularies: each one an
    op or a kind belongs to is declared with a summary and a doc, and
    nothing is declared that has no members."""
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
    """A value type is declared with its summary, its doc, and typed,
    described fields; the point vocabulary's page lists exactly the
    points the code accepts."""
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
    """One name, set two ways.

    A surface that shows an operation to a person shows
    `Records :: Cross`; a surface that shows it to a machine — a stored
    graph, `ops.json`, provenance — shows `records/cross`, or its full
    identity `~canonical/ops/records/cross`. The rendering is derived
    from the name by rule, which is what keeps it from BEING a second
    name: a hand-written label per operation drifts out of the lexicon
    with nothing to notice.
    """

    def test_it_reads_as_a_family_and_a_verb(self) -> None:
        assert lexicon.title("records/cross") == "Records :: Cross"
        assert lexicon.title("logits/read-layers") == "Logits :: Read Layers"
        assert (lexicon.title("activations/capture-attention")
                == "Activations :: Capture Attention")

    def test_every_name_in_the_lexicon_reads_back(self) -> None:
        """The line between typography and a new name: a rendering you
        cannot reverse is a name in a different font."""
        for name in (*lexicon.BY_NAME, *(k.name for k in lexicon.KINDS)):
            assert lexicon.name_of_title(lexicon.title(name)) == name, name

    def test_it_takes_a_stored_path_as_readily_as_a_name(self) -> None:
        assert lexicon.title("~canonical/ops/records/cross") == "Records :: Cross"
        assert lexicon.title("~canonical/ops/records/cross/1") == "Records :: Cross"

    def test_the_identity_is_the_path_and_the_name_is_bare(self) -> None:
        # Three forms, and this is all of them: the name a graph stores,
        # the identity provenance records, and the rendering a person
        # reads. Nothing else may name an operation.
        assert lexicon.canonical_path("records/cross") == "~canonical/ops/records/cross"
        assert lexicon.display_name("~canonical/ops/records/cross") == "records/cross"
        assert lexicon.title("records/cross") == "Records :: Cross"

    def test_it_needs_no_table_of_its_own(self) -> None:
        """An operation nobody has declared yet renders like any other,
        which is the property a lookup table cannot have."""
        assert lexicon.title("weights/decompose") == "Weights :: Decompose"
        assert lexicon.title("nothing/here-yet") == "Nothing :: Here Yet"


class TestWhatAnOperationNeeds:
    """`requires` is declared in the lexicon and proved against the
    code.

    It decides which machine may run a node: the composer copies it onto
    every node and `/jobs/next` filters claims by it, so a wrong value
    routes work to a machine that cannot do it. Declared anywhere but
    beside the code it describes, it drifts — a merge marked as needing
    local weights that never loads a model.
    """

    @staticmethod
    def _loads_model(block: str) -> bool:
        """Does running this block put the model in memory? Asked of the
        executor, not of a list: the dispatch arm, and any handler it
        calls, reaching `_model_loaded` (which `_run_model_block` wraps)."""
        import inspect
        import pathlib
        import re

        from mechbench_compute import protocol

        src = pathlib.Path(inspect.getfile(protocol)).read_text()
        bodies = dict(re.findall(
            r'\n    def (_block_[a-z_]+)\(.*?\n(.*?)(?=\n    def |\Z)', src, re.S))
        arms = dict(re.findall(
            r'block == "([a-z0-9-]+/[a-z0-9-]+)":\s*\n(.*?)(?=\n\s*elif block ==|\n\s*else:)',
            src, re.S))
        body = arms.get(block, "")
        if "_run_model_block" in body:
            return True
        return any(
            "_model_loaded" in bodies.get(h, "") or "_run_model_block" in bodies.get(h, "")
            for h in re.findall(r"self\.(_block_[a-z_]+)", body)
        )

    def test_every_operation_declares_one_of_the_four(self) -> None:
        for op in lexicon.OPS:
            assert op.requires in ("pure", "mlx-local", "remote", "by-model"), op.name

    def test_a_pure_operation_never_touches_the_model(self) -> None:
        """The direction that matters for routing: a node declared pure
        may be claimed by a machine with no weights at all."""
        for op in lexicon.OPS:
            if op.requires == "pure":
                assert not self._loads_model(op.name), (
                    f"{op.name} is declared pure and loads the model")

    def test_an_operation_that_loads_the_model_says_so(self) -> None:
        from mechbench_compute.protocol import REMOTE_BLOCKS

        for op in lexicon.OPS:
            if self._loads_model(op.name):
                expect = "by-model" if op.name in REMOTE_BLOCKS else "mlx-local"
                assert op.requires == expect, (
                    f"{op.name} loads the model; declared {op.requires!r}")

    def test_the_operations_that_run_either_side_are_the_remote_blocks(self) -> None:
        """`by-model` and the executor's own idea of what may be remote
        are one fact. Chat and judge are the same operation whichever
        side answers, which is the point of them."""
        from mechbench_compute.protocol import REMOTE_BLOCKS

        assert {op.name for op in lexicon.OPS if op.requires == "by-model"} == set(REMOTE_BLOCKS)

    def test_the_pure_registry_is_pure(self) -> None:
        from mechbench_compute.blocks import PURE_BLOCKS

        for name in PURE_BLOCKS:
            if name in lexicon.BY_NAME:
                assert lexicon.BY_NAME[name].requires == "pure", name

    def test_only_publishing_needs_the_network_without_a_model(self) -> None:
        """One op is `remote` for a reason that is not a model: it pushes
        an adapter to a hub, which needs the network and the owner's
        credentials. If a second one appears, this line is where someone
        decides whether the value still means what it says."""
        assert {op.name for op in lexicon.OPS if op.requires == "remote"} == {"adapter/publish"}
