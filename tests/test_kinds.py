from __future__ import annotations

import pytest

from mechbench_compute import lexicon as L
from mechbench_compute.conformance import check_core, format_findings, select_findings
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon._base import COLLECTION, KIND_ROOT, Kind


def assert_clean(codes: tuple[str, ...], subject: str | None = None) -> None:
    allowed = {op.name: frozenset({"NO_OUTPUT"}) for op in L.OPS if op.family == "tools"}
    hits = select_findings(check_core(), codes, subject=subject, allowed=allowed)
    assert not hits, format_findings(hits)


FAMILIES = {"records", "text", "eval", "logits", "activations", "geometry",
            "intervene", "direction", "trajectory", "adapter", "weights",
            "tools",
            "sandbox", "provider", "model", "platform", "run"}


@pytest.mark.parametrize("kind", K.KINDS, ids=lambda k: k.name)
def test_a_kind_is_named_and_described(kind: Kind) -> None:
    if kind.name != COLLECTION:
        assert kind.family in FAMILIES, kind.family
        assert kind.path == f"{KIND_ROOT}{kind.name}"
    assert_clean(("KIND_NAME_INVALID", "KIND_NOT_DESCRIBED", "HOUSE_IDIOM", "KIND_FIELD_UNDESCRIBED",
                  "KIND_REQUIRED_UNKNOWN", "KIND_EXTENDS_UNKNOWN", "NOT_JSON"), kind.name)


def test_names_are_unique_and_the_container_is_one() -> None:
    assert_clean(("KIND_DUPLICATE",))
    assert COLLECTION in K.BY_KIND
    assert sum(1 for k in K.KINDS if k.name == COLLECTION) == 1


def test_every_op_emits_a_declared_kind_or_nothing() -> None:
    assert_clean(("NO_OUTPUT", "OUTPUT_KIND_UNKNOWN", "OUTPUT_NOT_COLLECTABLE"))


def test_every_op_kind_is_emitted_by_some_op() -> None:
    emitted = {op.output.kind for op in L.OPS if op.output}
    emitted |= {o.kind for op in L.OPS if op.output for o in op.output.otherwise}
    exempt = {COLLECTION, "records/record", "records/condition", "records/pair",
              "logits/distribution", "activations/grid", "text/word-list",
              "intervene/spec"}
    orphans = sorted(k.name for k in K.KINDS
                     if not k.platform and k.name not in emitted and k.name not in exempt)
    assert orphans == [], orphans


def test_no_source_literal_names_an_undeclared_kind() -> None:
    assert_clean(("KIND_LITERAL_UNKNOWN",))


def test_the_lattice_is_acyclic_and_rooted() -> None:
    assert_clean(("KIND_CYCLE", "KIND_EXTENDS_UNKNOWN"))


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
        assert K.items_of({"kind": COLLECTION, "item_kind": "records/record", "records": [1]}) == [1]
        with pytest.raises(ValueError):
            K.items_of({"kind": "records/table", "columns": []})
        with pytest.raises(ValueError):
            K.items_of(3)


class TestTheRegistryFollowsTheLexicon:
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
                continue
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


SILENT_KINDS = frozenset({
    "activations/attention", "activations/coordinate", "activations/divergence", "activations/grid",
    "activations/vector", "adapter/checkpoint", "adapter/delta", "adapter/lora", "adapter/push",
    "collection", "direction/vocab", "eval/verdict", "geometry/mst",
    "geometry/similarity", "intervene/ablation", "intervene/heads", "intervene/readout",
    "intervene/spec", "intervene/trace", "logits/attribution", "logits/decision",
    "logits/distribution", "logits/funnel", "logits/lens", "model/pointer", "model/ref",
    "provider/call", "provider/cassette", "provider/completion", "records/chart",
    "records/condition", "records/pair", "records/record", "records/table", "run/ladder",
    "run/replay", "run/result", "sandbox/call", "sandbox/image", "sandbox/snapshot",
    "text/annotation", "text/document", "text/tokenization", "text/transcript", "text/word-list",
    "trajectory/comparison", "trajectory/point", "trajectory/summary", "weights/parameter",
})


class TestEveryKindSpeaks:
    def test_every_kind_speaks(self):
        silent = {k.name for k in K.KINDS if not k.speak}
        assert silent <= SILENT_KINDS, (
            f"{sorted(silent - SILENT_KINDS)} say nothing: give each a `speak` template")

    def test_the_silent_set_only_shrinks(self):
        speaking = {k.name for k in K.KINDS if k.speak} & SILENT_KINDS
        assert not speaking, f"{sorted(speaking)} speak now: take them off SILENT_KINDS"
        assert SILENT_KINDS <= set(K.BY_KIND), sorted(SILENT_KINDS - set(K.BY_KIND))

    def test_speak_draw_and_version_reach_the_published_dict(self):
        from mechbench_compute.lexicon._base import Draw

        k = Kind("x/y", "s", speak="{a} at layer {layer}", version=2,
                 draw=Draw("heat", {"x": "layer", "value": "a"}))
        d = k.to_dict()
        assert d["speak"] == "{a} at layer {layer}" and d["version"] == 2
        assert d["draw"] == {"mark": "heat", "encoding": {"x": "layer", "value": "a"}}
        assert Kind("x/y", "s").to_dict()["draw"] is None


NO_NOTABLE = frozenset({
    "activations/attention", "activations/coordinate", "activations/divergence", "activations/feature",
    "activations/vector", "adapter/delta", "direction/vector", "direction/vocab", "eval/verdict",
    "geometry/alignment", "geometry/mst", "geometry/similarity", "intervene/ablation", "intervene/circuit",
    "intervene/faithfulness", "intervene/heads", "intervene/trace", "logits/lens", "platform/noise",
    "records/chart", "records/table", "text/annotation", "text/branch-point", "text/document",
    "text/tokenization", "text/transcript", "trajectory/comparison", "trajectory/point",
    "trajectory/summary", "weights/parameter",
})


def read_try_kinds() -> set[str]:
    from mechbench_compute.live.run_try import TRAINING
    from mechbench_compute.registry import CORE

    out: set[str] = set()
    for mod in CORE.modules().values():
        op = mod.OP
        if op.name in TRAINING or op.requires == "remote":
            continue
        for o in ([op.output] if op.output else []) + list((op.outputs or {}).values()):
            out.add(o.kind)
            out.update(x.kind for x in o.otherwise)
    return out


class TestEveryKindATryReturnsHasANotable:
    def test_every_kind_a_try_returns_declares_a_notable(self):
        unnoted = {k for k in read_try_kinds() if K.BY_KIND[k].notable is None}
        assert unnoted <= NO_NOTABLE, (
            f"NO_NOTABLE: {sorted(unnoted - NO_NOTABLE)} can be a try's result and declare no "
            "`notable`: give each a `Notable(field, key, metric, threshold)`")

    def test_the_exemptions_only_shrink(self):
        noted = {k for k in NO_NOTABLE if K.BY_KIND[k].notable is not None}
        assert not noted, f"{sorted(noted)} declare a notable now: take them off NO_NOTABLE"
        assert NO_NOTABLE <= read_try_kinds(), sorted(NO_NOTABLE - read_try_kinds())

    def test_a_notable_reaches_the_published_dict_only_when_declared(self):
        from mechbench_compute.lexicon._base import Notable

        k = Kind("x/y", "s", notable=Notable("p", ("id",), "difference", 0.5, control={"factor": 0}))
        assert k.to_dict()["notable"] == {"field": "p", "key": ["id"], "metric": "difference",
                                          "threshold": 0.5, "control": {"factor": 0}}
        assert "notable" not in Kind("x/y", "s").to_dict()

    def test_the_generated_kind_table_carries_the_notable(self):
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parents[1] / "scripts" / "dump_kinds_ts.py"
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True).stdout
        rows = {line.split(":", 1)[0].strip().strip('"'): line for line in out.splitlines()
                if line.startswith('  "')}
        assert ('notable: {"field": "entropy_bits", "key": ["id", "layer"], "metric": "difference", '
                '"threshold": 0.5}') in rows["logits/funnel"]
        assert "notable" not in rows["records/table"]

    def test_a_declared_notable_names_a_metric_its_kind_compares_by(self):
        from mechbench_compute import metrics
        from mechbench_compute.lexicon._base import DIFFERENCE

        for k in K.KINDS:
            if k.notable is not None and k.notable.metric != DIFFERENCE:
                assert k.notable.metric in metrics.declared(k.name), k.name


class TestOneKindPerFile:
    def test_a_kinds_path_is_a_function_of_its_name(self):
        import importlib

        for k in K.KINDS:
            mod = importlib.import_module(K.resolve_kind_module(k.name))
            assert mod.KIND is k, k.name

    def test_a_platform_kind_says_so(self):
        platform_families = {"sandbox", "provider", "model", "platform", "run"}
        for k in K.KINDS:
            assert k.platform == (k.family in platform_families), k.name
