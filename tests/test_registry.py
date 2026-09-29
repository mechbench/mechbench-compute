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

from mechbench_compute import lexicon, ops
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon.extension import Extension, Package, encode_canonical, hash_extension
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

    def test_the_manifest_keeps_the_authors_spelling(self, installed):
        d = manifest().to_dict()
        assert d["provides"]["ops"][0]["name"] == "geometry/align"
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


class TestRefusals:
    def refuse(self, swap_sources, monkeypatch, name, message, module="mb_fixture_ext"):
        bad = types.ModuleType("mb_refused_ext")
        bad.MANIFEST = Extension(name=name, extension="interp-extras", version=1, module=module,
                                 package=Package("mechbench-ext-x"), min_compute="0.165.0")
        monkeypatch.setitem(sys.modules, "mb_refused_ext", bad)
        swap_sources(point("mb_refused_ext:MANIFEST", "refused"))
        with pytest.raises(KeyError, match=message):
            REGISTRY.resolve(f"{name}/ops/geometry/align")

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


@pytest.mark.skipif(not (MODELS / "src" / "extension.ts").exists()
                    or not (MODELS / "node_modules" / ".bin" / "vite-node").exists(),
                    reason="mechbench-models with src/extension.ts and its node_modules is not checked out beside compute")
def test_the_manifest_validates_against_models_and_hashes_as_the_api_does(installed, tmp_path):
    d = manifest().to_dict()
    d["package"]["sdist"] = "~hash/sha256:" + "d" * 64
    d["provenance"]["published_by"] = "usr_fixture"
    (tmp_path / "m.json").write_text(json.dumps(d))
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
    got = json.loads(run.stdout.strip().splitlines()[-1])
    assert hash_extension(got["parsed"]) == got["hash"]
