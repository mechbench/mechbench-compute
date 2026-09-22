"""The kinds are declared once and the declaration is fit to publish
(docs/LEXICON.md §3–§6).

Every op emits a declared kind; every declared kind that is not a
platform kind is emitted by some op; names are two-level and bare; the
one container is `collection`; a collection sorts by its key and the
sort is idempotent; every retired string resolves.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import mechbench_compute

from mechbench_compute import lexicon as L
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon._base import COLLECTION, KIND_ROOT, Kind

INTERNAL = [
    re.compile(r"\b0\d{5}\b"),
    re.compile(r"\b(?:task|tasks|epic|epics)\s+\d", re.I),
    re.compile(r"\bstep\s+\d{2}\b", re.I),
    re.compile(r"\bexperiment\s+0\d{2}\b", re.I),
]

FAMILIES = {"records", "text", "eval", "logits", "activations", "geometry",
            "intervene", "direction", "trajectory", "adapter", "weights",
            "tools",
            # platform families: kinds no op produces
            "sandbox", "provider", "model", "run"}


@pytest.mark.parametrize("kind", K.KINDS, ids=lambda k: k.name)
def test_a_kind_is_named_and_described(kind: Kind) -> None:
    if kind.name != COLLECTION:
        assert re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", kind.name), kind.name
        assert kind.family in FAMILIES, kind.family
        assert kind.path == f"{KIND_ROOT}{kind.name}"
    assert kind.summary.strip() and kind.summary.strip()[-1] in ".?!"
    for pat in INTERNAL:
        for text in (kind.summary, kind.doc, *[f["description"] for f in kind.fields.values()],
                     *kind.header.values()):
            assert not pat.search(text), f"{kind.name}: house idiom {pat.pattern!r}"
    for f, spec in kind.fields.items():
        assert spec.get("type") and spec.get("description"), f"{kind.name}.{f}: type and description"
    inherited = K.all_fields(kind)
    for r in kind.required:
        assert r in inherited, f"{kind.name}: required {r!r} is not a field of it or an ancestor"
    if kind.extends:
        assert kind.extends in K.BY_KIND, f"{kind.name} extends unknown {kind.extends!r}"
    json.dumps(kind.to_dict())


def test_names_are_unique_and_the_container_is_one() -> None:
    names = [k.name for k in K.KINDS]
    assert len(names) == len(set(names))
    assert COLLECTION in K.BY_KIND
    assert sum(1 for k in K.KINDS if k.name == COLLECTION) == 1


def test_every_op_emits_a_declared_kind_or_nothing() -> None:
    for op in L.OPS:
        if op.output is None:
            assert op.family == "tools", f"{op.name} emits nothing and is not a tool"
            continue
        assert op.output.kind in K.BY_KIND, f"{op.name} produces undeclared {op.output.kind!r}"
        kind = K.BY_KIND[op.output.kind]
        if op.output.collection:
            assert kind.collectable, f"{op.name} emits a collection of {kind.name}, which declares no key"


def test_every_op_kind_is_emitted_by_some_op() -> None:
    emitted = {op.output.kind for op in L.OPS if op.output}
    # Ancestors and the container are declared for the lattice, not
    # emitted directly; platform kinds are produced by the platform; an
    # authored input — a corpus, a word list, an intervention's spec —
    # is written, not emitted.
    exempt = {COLLECTION, "records/record", "records/condition", "records/pair",
              "logits/distribution", "activations/grid", "text/word-list",
              "intervene/spec"}
    orphans = sorted(k.name for k in K.KINDS
                     if not k.platform and k.name not in emitted and k.name not in exempt)
    assert orphans == [], orphans


def test_no_source_literal_names_an_undeclared_kind() -> None:
    """Every `"kind": "…"` literal in the package names a declared kind:
    the retired spellings are read through the aliases, never written."""
    root = Path(mechbench_compute.__file__).parent
    pat = re.compile(r'"kind":\s*"([^"]+)"')
    bad = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "_smoke_bench.py" or "lexicon" in path.parts or "ops" in path.parts:
            continue
        for m in pat.finditer(path.read_text()):
            if m.group(1) not in K.BY_KIND:
                bad.append(f"{path.relative_to(root)}: {m.group(1)!r}")
    assert bad == [], bad


def test_the_lattice_is_acyclic_and_rooted() -> None:
    for kind in K.KINDS:
        seen, cur = [], kind
        while cur.extends:
            assert cur.extends not in seen, f"cycle at {kind.name}"
            seen.append(cur.extends)
            cur = K.BY_KIND[cur.extends]


class TestAliases:
    @pytest.mark.parametrize("old,target", sorted(K.KIND_ALIASES.items()))
    def test_a_retired_string_resolves_to_a_declared_kind(self, old, target):
        name, plural = target
        assert name in K.BY_KIND, f"{old!r} -> undeclared {name!r}"
        assert K.resolve_kind(old) == (name, plural)
        assert old not in K.BY_KIND or old == COLLECTION

    def test_bare_and_registered_spellings_resolve(self):
        assert K.resolve_kind("records/table") == ("records/table", False)
        assert K.resolve_kind(f"{KIND_ROOT}records/table") == ("records/table", False)
        assert K.resolve_kind(COLLECTION) == (COLLECTION, True)
        with pytest.raises(KeyError):
            K.resolve_kind("records/tabel")

    def test_a_legacy_document_collection_is_documents(self):
        """A corpus generated before the typology satisfies a records
        port: its items are documents, which are records. Resolving the
        old plural to the bare container refused every such corpus."""
        from mechbench_compute.block_params import check_inputs

        assert K.item_kind_of({"kind": "document_collection", "items": []}) == "text/document"
        assert K.resolve_kind("document_collection", warn=False) == ("text/document", True)
        check_inputs("activations/capture", {"records": {"kind": "document_collection", "items": [{"id": "a", "text": "t"}]}})

    def test_a_legacy_plural_names_its_item_kind(self):
        assert K.item_kind_of({"kind": "residual_vectors"}) == "activations/vector"
        assert K.item_kind_of({"kind": COLLECTION, "item_kind": "logits/decision"}) == "logits/decision"
        assert K.item_kind_of({"kind": "records/table"}) is None
        assert K.item_kind_of({"kind": "not-a-kind"}) is None


class TestTheContainer:
    def test_collection_builds_with_the_kinds_key(self):
        c = K.collection("logits/decision", [{"id": "x"}], model="m", nothing=None)
        assert c == {"kind": COLLECTION, "item_kind": "logits/decision", "key": ["id"],
                     "items": [{"id": "x"}], "model": "m"}

    def test_canonical_collection_sorts_by_key_and_is_idempotent(self):
        def sp(layer, head=None):
            return {"model": "m", "layer": layer, "point": "resid_post", "head": head, "d": 2}
        c = K.collection("activations/vector", [
            {"id": "b", "space": sp(2)}, {"id": "a", "space": sp(3)},
            {"id": "a", "space": sp(1, 1)}, {"id": "a", "space": sp(1)},
        ])
        s = K.canonical_collection(c)
        assert [(i["id"], i["space"]["layer"], i["space"].get("head")) for i in s["items"]] == [
            ("a", 1, None), ("a", 1, 1), ("a", 3, None), ("b", 2, None)]
        assert K.canonical_collection(s) == s
        # The same items in any order hash the same.
        import random
        from mechbench_compute.resume import content_hash
        shuffled = dict(c, items=random.Random(3).sample(c["items"], len(c["items"])))
        assert content_hash(K.canonical_collection(shuffled)) == content_hash(s)
        assert content_hash(shuffled) != content_hash(s) or shuffled["items"] == s["items"]

    def test_canonical_collection_leaves_other_values_alone(self):
        for v in ({"kind": "records/table", "rows": [2, 1]}, [3, 1, 2], "x", None):
            assert K.canonical_collection(v) == v

    def test_items_of_reads_every_spelling(self):
        assert K.items_of([1, 2]) == [1, 2]
        assert K.items_of({"kind": COLLECTION, "items": [1]}) == [1]
        assert K.items_of({"kind": "residual_vectors", "rows": [1]}) == [1]
        assert K.items_of({"kind": "decision_read", "conditions": [1]}) == [1]
        assert K.items_of({"kind": "record_set", "records": [1]}) == [1]
        assert K.items_of({"records": [1]}) == [1]
        # A container written with its list under an older name still reads.
        assert K.items_of({"kind": COLLECTION, "item_kind": "records/record", "records": [1]}) == [1]
        with pytest.raises(ValueError):
            K.items_of({"kind": "records/table", "columns": []})
        with pytest.raises(ValueError):
            K.items_of(3)


class TestTheRegistryFollowsTheLexicon:
    """`platform_kinds.manifests()` is generated from the declarations:
    one manifest per renderable kind at its canonical path, naming the
    registered paths it supersedes."""

    def test_every_renderable_kind_has_a_manifest_at_its_path(self):
        from mechbench_compute import platform_kinds

        by_path = {m.path: m for m in platform_kinds.manifests()}
        for kind in K.KINDS:
            if kind.renderer is None or kind.name == COLLECTION:
                continue
            assert kind.path in by_path, f"{kind.name} declares a renderer but registers nothing"
            m = by_path[kind.path]
            assert m.renderer.primitive == kind.renderer["primitive"]
            assert m.item_schema["type"] == "object"
            assert m.version == "1"
            if kind.name == "sandbox/snapshot":
                continue  # the snapshot codec's own schema, not the declaration's
            assert set(m.item_schema.get("required", [])) == set(kind.required)
            for f in K.all_fields(kind):
                assert f in m.item_schema["properties"], f"{kind.name}.{f} missing from the manifest"

    def test_a_manifest_names_the_paths_it_supersedes(self):
        from mechbench_compute import platform_kinds

        by_path = {m.path: m for m in platform_kinds.manifests()}
        doc = by_path[f"{KIND_ROOT}text/document"]
        assert f"{KIND_ROOT}text" in doc.supersedes
        funnel = by_path[f"{KIND_ROOT}logits/funnel"]
        assert {f"{KIND_ROOT}lens-trajectory", f"{KIND_ROOT}lens-trajectory/2"} <= set(funnel.supersedes)
        for m in by_path.values():
            for old in m.supersedes:
                assert old.startswith(KIND_ROOT) and old != m.path

    def test_no_manifest_registers_a_retired_path(self):
        from mechbench_compute import platform_kinds

        retired = {old for old in K.KIND_ALIASES if old.startswith(KIND_ROOT)}
        assert not retired & {m.path for m in platform_kinds.manifests()}


class TestRetiredSpellingsWarn:
    def test_a_retired_string_warns_once(self):
        K._warned.discard("metric_table")
        with pytest.warns(K.RetiredKindName, match="retired spelling of 'records/table'"):
            assert K.resolve_kind("metric_table") == ("records/table", False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert K.resolve_kind("metric_table") == ("records/table", False)
            assert K.resolve_kind("records/table", warn=True) == ("records/table", False)
