from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from mechbench_compute import bench, lexicon, ops
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon.extension import Extension, Package, declare, encode_canonical, hash_extension
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.protocol.check_graph import check_graph
from mechbench_compute.registry import CORE, GROUP, REGISTRY, InstalledSource, RestartRequired

FIXTURE = Path(__file__).parent / "fixtures" / "ext_pkg"
MODELS = Path(__file__).resolve().parents[2] / "mechbench-models"
EXTENSION = "alice/interp-extras/extensions/interp-extras"
ADDRESS = "alice/interp-extras/ops/geometry/align"
ALIGNMENT = "alice/interp-extras/kinds/geometry/alignment"
CROSS = {"factors": [{"name": "x", "levels": [{"key": "a"}, {"key": "b"}]}]}


def point(value: str = "mb_fixture_ext:MANIFEST", name: str = "interp-extras"):
    return importlib.metadata.EntryPoint(name=name, value=value, group=GROUP)


def install(*points, installed=lambda: []):
    source = InstalledSource(find=lambda: list(points), installed=installed)
    REGISTRY.sources = (CORE, source)
    REGISTRY.refresh()
    return source


@pytest.fixture
def swap_sources(monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURE))
    held = REGISTRY.sources
    try:
        yield install
    finally:
        REGISTRY.sources = held
        REGISTRY.refresh()


@pytest.fixture
def installed(swap_sources):
    return swap_sources(point())


def manifest():
    return sys.modules["mb_fixture_ext"].MANIFEST


def graph(block: str) -> dict:
    return {"dataflow": 2,
            "nodes": [{"id": "x", "block": "records/cross", "params": CROSS},
                      {"id": "al", "block": block, "params": {"method": "jaccard"}}],
            "edges": [{"from": {"node": "x", "port": "records"}, "to": {"node": "al", "port": port}}
                      for port in ("a", "b")]}


class TestResolution:
    def test_an_extension_op_resolves_by_its_address(self, installed):
        r = REGISTRY.resolve(ADDRESS)
        assert (r.tier, r.name, r.version) == ("installed", ADDRESS, 2)
        assert r.module.__name__ == "mb_fixture_ext.ops.geometry.align"
        assert r.source == f"{EXTENSION}@{r.digest}"
        assert r.op.name == ADDRESS and r.op.family == "geometry"
        assert r.op.output.kind == ALIGNMENT
        assert lexicon.resolve(ADDRESS) == ADDRESS
        assert ADDRESS in lexicon.BY_NAME and lexicon.BY_NAME[ADDRESS] is r.op

    def test_a_core_name_still_resolves_first_and_as_before(self, installed):
        r = REGISTRY.resolve("~canonical/ops/records/filter")
        assert (r.tier, r.name, r.source) == ("core", "records/filter", "core")
        assert r.pinned == "~canonical/ops/records/filter"

    def test_a_counter_resolves_only_when_it_is_the_installed_version(self, installed):
        assert REGISTRY.resolve(f"{ADDRESS}@2") is REGISTRY.resolve(ADDRESS)
        with pytest.raises(KeyError, match="is @2, not @3"):
            REGISTRY.resolve(f"{ADDRESS}@3")

    def test_a_pin_resolves_only_when_it_is_the_installed_digest(self, installed):
        r = REGISTRY.resolve(ADDRESS)
        assert REGISTRY.resolve(r.pinned) is r
        wrong = "sha256:" + "0" * 64
        with pytest.raises(KeyError) as e:
            REGISTRY.resolve(f"{ADDRESS}@{wrong}")
        assert wrong in str(e.value) and r.digest in str(e.value)
        assert r.digest in lexicon.explain_unknown(f"{ADDRESS}@{wrong}")

    def test_a_missing_extension_says_so(self, installed):
        assert "no installed extension provides" in lexicon.explain_unknown("bob/other/ops/geometry/align")

    def test_the_runners_recorded_hash_is_the_digest(self, swap_sources):
        pinned = "sha256:" + "c" * 64
        swap_sources(point(), installed=lambda: [{"address": EXTENSION, "version": 2, "hash": pinned}])
        assert REGISTRY.resolve(f"{ADDRESS}@{pinned}").digest == pinned


class TestKinds:
    def test_an_extension_kind_satisfies_the_core_kind_it_extends(self, installed):
        assert K.satisfies(ALIGNMENT, "records/record")
        assert K.ancestry(ALIGNMENT) == (ALIGNMENT, "records/record")
        assert {"id", "coords", "score"} <= set(K.all_fields(K.BY_KIND[ALIGNMENT]))
        assert K.resolve_kind(ALIGNMENT) == (ALIGNMENT, False)
        assert REGISTRY.kind(ALIGNMENT).family == "geometry"
        assert ALIGNMENT in {k.name for k in K.KINDS}

    def test_the_manifest_keeps_the_authors_spelling_and_carries_its_own_path(self, installed):
        d = manifest().to_dict()
        assert d["provides"]["ops"][0]["name"] == "geometry/align"
        assert d["provides"]["ops"][0]["path"] == ADDRESS
        assert d["provides"]["kinds"][0]["path"] == ALIGNMENT
        assert d["provides"]["ops"][0]["output"]["kind"] == "geometry/alignment"
        assert d["provides"]["ops"][0]["entry"] == "mb_fixture_ext.ops.geometry.align"
        assert d["provides"]["kinds"][0]["extends"] == "records/record"
        assert d["provides"]["marks"] == [{"name": "pairs", "version": 1,
                                           "channels": {"x": "required", "y": "required"}}]
        assert (d["owner"], d["project"], d["name"], d["version"]) == ("alice", "interp-extras", "interp-extras", 2)
        assert d["needs"] == [] and d["state"] == "draft" and d["conformance"] is None


class TestRunning:
    def test_check_graph_accepts_a_graph_naming_it(self, installed):
        g = graph(ADDRESS)
        nodes = {n["id"]: n for n in g["nodes"]}
        check_graph(nodes, g["edges"], ["x", "al"])
        g["nodes"][1]["params"]["nope"] = 1
        with pytest.raises(ValueError, match="does not accept 'nope'"):
            check_graph({n["id"]: n for n in g["nodes"]}, g["edges"], ["x", "al"])

    def test_the_executor_runs_it_in_process_with_its_declared_needs(self, installed, monkeypatch):
        lent = []
        real = ops.Context.for_op.__func__

        def spy(cls, op, executor=None, **kw):
            lent.append((op.name, op.needs))
            return real(cls, op, executor, **kw)

        monkeypatch.setattr(ops.Context, "for_op", classmethod(spy))
        out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                                  extra={"graph": graph(ADDRESS)}))
        assert out.payload["nodes_executed"] == ["x", "al"]
        assert (ADDRESS, frozenset()) in lent

    def test_a_bare_spelling_and_its_pin_fingerprint_alike(self, installed):
        seen = []

        def run(block):
            ex = ProtocolExecutor(on_node_start=lambda nid, fp: seen.append((nid, fp)))
            ex.run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={"graph": graph(block)}))
            return dict(seen)["al"]

        bare = run(ADDRESS)
        seen.clear()
        assert run(REGISTRY.resolve(ADDRESS).pinned) == bare

    def test_the_standalone_and_fused_sets_are_core_plus_the_extension(self, installed):
        core = {n for n in ops.find_standalone() if n in ops.load_modules()}
        assert ops.find_standalone() == frozenset(core | {ADDRESS})
        assert not ops.fuses_adapter(ADDRESS) and not ops.fuses_adapter_locally(ADDRESS)
        assert ops.run_standalone(ADDRESS, {"a": [{"id": "1"}], "b": [{"id": "1"}, {"id": "2"}]},
                                  {"method": "jaccard"})["items"][0]["score"] == 0.5


class TestScope:
    def test_the_registered_entries_carry_their_address_as_path(self, installed):
        assert REGISTRY.resolve(ADDRESS).op.to_dict()["path"] == ADDRESS
        assert REGISTRY.kind(ALIGNMENT).to_dict()["path"] == ALIGNMENT
        assert REGISTRY.resolve("records/filter").op.to_dict()["path"] == "~canonical/ops/records/filter"

    def test_a_short_port_kind_is_rewritten_at_registration(self, installed):
        op = REGISTRY.resolve(ADDRESS).op
        assert [p.kind for p in op.inputs] == ["records/record", "records/record"]
        assert op.output.kind == ALIGNMENT

    def test_an_extension_ops_context_carries_its_scope(self, installed):
        assert ops.Context.for_op(REGISTRY.resolve(ADDRESS).op).scope == "alice/interp-extras"
        assert ops.Context.for_op(REGISTRY.resolve("records/filter").op).scope is None

    def test_a_short_kind_resolves_in_the_given_scope_then_core(self, installed):
        scope = "alice/interp-extras"
        assert K.qualify_kind("geometry/alignment", scope) == ALIGNMENT
        assert K.qualify_kind("geometry/alignment") == "geometry/alignment"
        assert K.qualify_kind("records/record", scope) == "records/record"
        assert K.resolve_kind("geometry/alignment", scope=scope) == (ALIGNMENT, False)
        assert K.satisfies("geometry/alignment", "records/record", scope=scope)
        assert lexicon.kinds.collection("geometry/alignment", [], scope=scope)["item_kind"] == ALIGNMENT
        with pytest.raises(KeyError):
            lexicon.kinds.collection("geometry/alignment", [])

    def test_inside_an_extension_op_the_registry_knows_the_scope(self, installed):
        with REGISTRY.within("alice/interp-extras"):
            assert K.resolve_kind("geometry/alignment") == (ALIGNMENT, False)
            with REGISTRY.within(None):
                assert K.qualify_kind("geometry/alignment") == "geometry/alignment"
        out = ops.run_standalone(ADDRESS, {"a": [{"id": "1"}], "b": [{"id": "1"}]}, {})
        assert out["item_kind"] == ALIGNMENT

    def test_a_kind_named_as_a_core_kind_is_refused(self, swap_sources, monkeypatch, tmp_path):
        root = tmp_path / "mb_clash_ext"
        shutil.copytree(FIXTURE / "mb_fixture_ext", root)
        (root / "kinds" / "geometry" / "similarity.py").write_text(
            (root / "kinds" / "geometry" / "alignment.py").read_text().replace(
                '"geometry/alignment"', '"geometry/similarity"'))
        monkeypatch.syspath_prepend(str(tmp_path))
        TestRefusals().refuse(swap_sources, monkeypatch, "bob/clash", "core kind's name", "mb_clash_ext")


class TestProvenance:
    def run(self, monkeypatch, g):
        emitted = {}

        def emit(target, payload, *, inputs=(), operation=None, extension=None, **kw):
            emitted[target] = {"operation": operation, "extension": extension}
            return {"path": target}

        monkeypatch.setattr(bench, "emit", emit)
        out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                                  extra={"graph": g, "resultPath": "a/b/results/j"}))
        return emitted, out.payload["resolved"]

    def test_an_extension_ops_result_records_the_pin(self, installed, monkeypatch):
        r = REGISTRY.resolve(ADDRESS)
        pin = {"address": EXTENSION, "version": 2, "hash": r.digest}
        assert r.pin == pin
        emitted, resolved = self.run(monkeypatch, graph(ADDRESS))
        assert emitted["a/b/results/j/al"] == {"operation": ADDRESS, "extension": pin}
        assert resolved["extensions"] == {EXTENSION: pin}

    def test_a_core_ops_result_records_none(self, installed, monkeypatch):
        emitted, resolved = self.run(monkeypatch, {"dataflow": 2, "edges": [], "nodes": [
            {"id": "x", "block": "records/cross", "params": CROSS}]})
        assert emitted == {"a/b/results/j/x": {"operation": "~canonical/ops/records/cross", "extension": None}}
        assert "extensions" not in resolved

    def test_the_schema_accepts_the_pin_and_leaves_it_out_when_absent(self, installed):
        import mechbench_schema as ms

        base = {"created_at": "2026-09-29T00:00:00Z", "produced_by": {"tool": "t", "version": "1"},
                "schema_version": ms.__version__, "operation": ADDRESS}
        pin = REGISTRY.resolve(ADDRESS).pin
        assert ms.Provenance(**base, extension=pin).model_dump(mode="json")["extension"] == pin
        assert "extension" not in ms.Provenance(**base).model_dump(mode="json")


class TestRefusals:
    def refuse(self, swap_sources, monkeypatch, name, message, module="mb_fixture_ext"):
        bad = types.ModuleType("mb_refused_ext")
        bad.MANIFEST = Extension(name=name, extension="interp-extras", version=1, module=module,
                                 package=Package("mechbench-ext-x"), min_compute="0.165.0")
        monkeypatch.setitem(sys.modules, "mb_refused_ext", bad)
        swap_sources(point("mb_refused_ext:MANIFEST", "refused"))
        with pytest.raises(KeyError, match=message):
            REGISTRY.resolve(f"{name}/ops/geometry/align")
        refused = REGISTRY.table().refused
        assert list(refused) == [f"{name}/extensions/interp-extras"]
        assert REGISTRY.refresh()["refused"] == refused

    def test_an_owner_that_is_a_core_family_is_refused(self, swap_sources, monkeypatch):
        self.refuse(swap_sources, monkeypatch, "records/x", "reserved")

    def test_a_kind_extending_a_kind_nobody_declares_is_refused(self, swap_sources, monkeypatch, tmp_path):
        root = tmp_path / "mb_orphan_ext"
        shutil.copytree(FIXTURE / "mb_fixture_ext", root)
        path = root / "kinds" / "geometry" / "alignment.py"
        path.write_text(path.read_text().replace('extends="records/record"', 'extends="other/thing"'))
        monkeypatch.syspath_prepend(str(tmp_path))
        self.refuse(swap_sources, monkeypatch, "bob/orphan", "neither a core kind", "mb_orphan_ext")

    def test_a_changed_version_after_load_asks_for_a_restart(self, installed, monkeypatch):
        monkeypatch.setattr(sys.modules["mb_fixture_ext"], "MANIFEST", dataclasses.replace(manifest(), version=3))
        with pytest.raises(RestartRequired, match="must restart"):
            REGISTRY.refresh()
        assert REGISTRY.resolve(f"{ADDRESS}@2").version == 2


class TestRefresh:
    def test_a_first_load_reports_the_extension_added(self, swap_sources):
        found = [point()]
        source = swap_sources(*found)
        assert source.added == [EXTENSION]
        assert REGISTRY.refresh() == {"added": [], "dropped": {}, "refused": {}}

    def test_an_entry_point_that_is_gone_is_forgotten(self, swap_sources):
        found = [point()]
        swap_sources(*found)
        source = REGISTRY.sources[1]
        source.find = lambda: []
        report = REGISTRY.refresh()
        assert list(report["dropped"]) == [EXTENSION] and "entry point" in report["dropped"][EXTENSION]
        assert "no installed extension provides" in lexicon.explain_unknown(ADDRESS)
        source.find = lambda: [point()]
        assert REGISTRY.refresh()["added"] == [EXTENSION]
        assert REGISTRY.resolve(ADDRESS).version == 2

    def test_a_pin_that_left_installed_json_is_forgotten(self, swap_sources):
        pinned = "sha256:" + "c" * 64
        records = [{"address": EXTENSION, "version": 2, "hash": pinned}]
        swap_sources(point(), installed=lambda: list(records))
        assert REGISTRY.resolve(ADDRESS).digest == pinned
        records.clear()
        report = REGISTRY.refresh()
        assert "no longer in installed.json" in report["dropped"][EXTENSION]
        with pytest.raises(KeyError):
            REGISTRY.resolve(ADDRESS)
        assert REGISTRY.refresh() == {"added": [], "dropped": {}, "refused": {}}
        records.append({"address": EXTENSION, "version": 2, "hash": pinned})
        assert REGISTRY.refresh()["added"] == [EXTENSION]

    def test_an_extension_loaded_without_a_record_stays_loaded(self, installed):
        assert REGISTRY.refresh()["dropped"] == {}
        assert REGISTRY.resolve(ADDRESS).version == 2

    def test_a_forgotten_extension_back_at_another_version_asks_for_a_restart(self, swap_sources, monkeypatch):
        swap_sources(point())
        source = REGISTRY.sources[1]
        source.find = lambda: []
        REGISTRY.refresh()
        monkeypatch.setattr(sys.modules["mb_fixture_ext"], "MANIFEST", dataclasses.replace(manifest(), version=3))
        source.find = lambda: [point()]
        with pytest.raises(RestartRequired, match="must restart"):
            REGISTRY.refresh()


class TestDigest:
    def test_the_digest_is_stable_and_ignores_what_the_platform_writes(self, installed):
        d = manifest().to_dict()
        first = hash_extension(d)
        assert first == hash_extension(json.loads(json.dumps(d))) == REGISTRY.resolve(ADDRESS).digest
        assert hash_extension({**d, "state": "verified", "visibility": "public", "flags": [1]}) == first
        assert hash_extension({**d, "min_compute": "0.166.0"}) != first

    def test_canonical_json_is_the_javascript_spelling(self):
        value = {"b": 1.0, "a": [1e-7, 0.05, 1e21, 123.456, -0.0, "é\n"], "B": None, "é": True}
        assert encode_canonical(value) == '{"B":null,"a":[1e-7,0.05,1e+21,123.456,0,"é\\n"],"b":1,"é":true}'


NEEDS_MODELS = pytest.mark.skipif(
    not (MODELS / "src" / "extension.ts").exists() or not (MODELS / "node_modules" / ".bin" / "vite-node").exists(),
    reason="mechbench-models with src/extension.ts and its node_modules is not checked out beside compute")

HAND_BUILT = {
    "kind": "extension", "owner": "alice", "project": "interp-extras", "name": "interp-extras", "version": 1,
    "tier": "installed", "min_compute": "0.169.0",
    "package": {"name": "mechbench-ext-x", "python": ">=3.12", "sdist": "~hash/sha256:" + "e" * 64},
    "provenance": {"published_by": "usr_fixture", "source": "élan"},
    "provides": {
        "ops": [{"name": "geometry/align", "summary": "Aligns two sets, café.",
                 "params": [{"name": "scale", "type": "float", "default": 1.0},
                            {"name": "spec", "type": "object", "fields": [{"name": "k", "type": "int"}]}],
                 "inputs": [{"name": "a", "kind": "records/record"}],
                 "output": {"kind": "geometry/alignment",
                            "otherwise": [{"kind": "records/record", "when": {"port": "a"}}]},
                 "resume": {"level": "exchangeable"}}],
        "kinds": [{"name": "geometry/alignment", "summary": "One score.", "metrics": [{"name": "score"}]}],
    },
}


def models_pin(tmp_path, manifest):
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    script = tmp_path / "check.ts"
    script.write_text(textwrap.dedent(f"""
        import {{ createHash }} from "node:crypto";
        import {{ readFileSync }} from "node:fs";
        import {{ ExtensionManifestSchema, declarationOf }} from "{MODELS}/src/extension.ts";
        import {{ canonicalJson }} from "{MODELS}/src/protocol_file.ts";
        const parsed = ExtensionManifestSchema.parse(JSON.parse(readFileSync("{tmp_path / 'm.json'}", "utf8")));
        const hash = "sha256:" + createHash("sha256").update(canonicalJson(declarationOf(parsed)), "utf8").digest("hex");
        console.log(JSON.stringify({{ parsed, hash }}));
    """))
    run = subprocess.run([str(MODELS / "node_modules" / ".bin" / "vite-node"), str(script)],
                         capture_output=True, text=True, cwd=MODELS, timeout=120)
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@NEEDS_MODELS
def test_the_manifest_validates_against_models_and_hashes_as_the_api_does(installed, tmp_path):
    d = manifest().to_dict()
    d["package"]["sdist"] = "~hash/sha256:" + "d" * 64
    d["provenance"]["published_by"] = "usr_fixture"
    got = models_pin(tmp_path, d)
    assert hash_extension(got["parsed"]) == hash_extension(d) == got["hash"]


@NEEDS_MODELS
def test_a_manifest_written_short_hashes_as_models_fills_it(tmp_path):
    got = models_pin(tmp_path, HAND_BUILT)
    assert hash_extension(HAND_BUILT) == got["hash"]
    assert declare(HAND_BUILT) == {k: got["parsed"][k] for k in declare(HAND_BUILT)}


class TestDeclaration:
    def test_defaults_are_filled_and_optional_keys_stay_absent(self):
        d = declare(HAND_BUILT)
        assert set(d) == {"owner", "project", "name", "version", "tier", "provides", "needs",
                          "min_compute", "package", "links"}
        op, kind = d["provides"]["ops"][0], d["provides"]["kinds"][0]
        assert op["inputs"][0] == {"name": "a", "kind": "records/record", "doc": "", "required": True,
                                   "many": False, "variadic": False, "on_missing": "fail"}
        assert op["params"][1]["fields"][0] == {"name": "k", "type": "int", "doc": ""}
        assert "required" not in op["params"][0] and "example" not in op
        assert op["resume"] == {"level": "exchangeable", "items": False}
        assert op["output"]["collection"] is False and op["output"]["otherwise"][0]["collection"] is False
        assert kind["extends"] is None and kind["version"] == 1 and kind["metrics"] == [{"name": "score", "doc": ""}]
        assert d["provides"]["marks"] == [] and d["needs"] == [] and d["links"] == {}
        assert "kind" not in d and "provenance" not in d

    def test_owner_and_project_hash_lowercased(self):
        assert hash_extension({**HAND_BUILT, "owner": "Alice", "project": "Interp-Extras"}) == hash_extension(HAND_BUILT)
