from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mechbench_compute.conformance import (
    WARNING,
    Conformance,
    check_core,
    check_extension,
    check_manifest,
    check_package,
    format_findings,
    read_inputs_from,
    run_examples,
    select_findings,
)
from mechbench_compute.registry import REGISTRY
from tests.test_lexicon import ALLOWED

FIXTURE = Path(__file__).parent / "fixtures" / "ext_pkg"
INPUTS = FIXTURE / "inputs"
ADDRESS = "alice/interp-extras/ops/geometry/align"
ALIGNMENT = "alice/interp-extras/kinds/geometry/alignment"


@pytest.fixture
def fixture_ext(monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURE))
    return importlib.import_module("mb_fixture_ext").MANIFEST


def copy_fixture(tmp_path, monkeypatch, module: str, edits: dict[str, list[tuple[str, str]]]):
    root = tmp_path / module
    shutil.copytree(FIXTURE / "mb_fixture_ext", root, ignore=shutil.ignore_patterns("__pycache__"))
    for rel, pairs in edits.items():
        path = root / rel
        text = path.read_text()
        for old, new in pairs:
            assert old in text, (rel, old)
            text = text.replace(old, new)
        path.write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module(module).MANIFEST


def codes(findings) -> list[str]:
    return sorted(f.code for f in findings)


class TestTheDeclarations:
    def test_the_fixture_extensions_one_finding_is_that_its_kind_shadows_cores(self, fixture_ext):
        assert [(f.code, f.at, f.severity) for f in check_manifest(fixture_ext)] == [
            ("KIND_SHADOWS_CORE", "geometry/alignment", WARNING)]
        assert check_package(fixture_ext) == []

    def test_its_wire_form_reads_the_same(self, fixture_ext):
        assert check_manifest(json.loads(json.dumps(fixture_ext.to_dict()))) == check_manifest(fixture_ext)

    def test_a_broken_copy_fails_by_name(self, tmp_path, monkeypatch):
        broken = copy_fixture(tmp_path, monkeypatch, "mb_broken_ext", {
            "__init__.py": [('name="alice/interp-extras"', 'name="bob/broken"')],
            "ops/geometry/align.py": [
                ('summary="How far two sets of records share their ids, as one score."', 'summary=""'),
                ('In("b", "records/record"', 'In("b", "geometry/nothing"'),
                ("def run(ctx, inputs, params):\n",
                 "def overlap_of(a, b):\n    return len(a & b)\n\n\n"
                 "def run(ctx, inputs, params):\n    if params.get(\"probe\"):\n        ctx.model(\"m\")\n"),
            ],
            "kinds/geometry/alignment.py": [('    speak="{header.method} of {a} and {b} records: {score}",\n', "")],
        })
        found = [*check_manifest(broken), *check_package(broken)]
        assert codes(found) == ["KIND_SHADOWS_CORE", "NAME_NOT_VERB", "NEED_UNDECLARED", "NO_SPEAK", "NO_SUMMARY",
                                "PORT_KIND_UNKNOWN"], format_findings(found)
        by = {f.code: f.at for f in found}
        assert by["NAME_NOT_VERB"] == "geometry/align#overlap_of"
        assert by["PORT_KIND_UNKNOWN"] == "geometry/align:b"
        assert by["NO_SPEAK"] == "geometry/alignment"

    def test_a_need_outside_the_vocabulary_fails_by_name(self, fixture_ext):
        d = json.loads(json.dumps(fixture_ext.to_dict()))
        d["provides"]["ops"][0]["needs"] = ["gpu"]
        assert codes(check_manifest(d)) == ["KIND_SHADOWS_CORE", "NEED_UNKNOWN"]

    def test_an_extension_kind_is_extend_only(self, fixture_ext):
        d = json.loads(json.dumps(fixture_ext.to_dict()))
        kind = d["provides"]["kinds"][0]
        d["provides"]["kinds"].append({**kind, "name": "run/thing", "path": None, "extends": None})
        kind["fields"]["id"] = {"type": "integer", "description": "An id that is a number."}
        found = check_manifest(d)
        assert codes(found) == ["KIND_CHANGES_ANCESTOR", "KIND_NO_EXTENDS", "KIND_SEALED_FAMILY",
                                "KIND_SHADOWS_CORE"], (
            format_findings(found))

    def test_core_has_no_findings_but_its_grandfathers(self):
        stray = [f for f in check_core() if f.code not in ALLOWED.get(f.at, frozenset())]
        assert not stray, format_findings(stray)
        assert select_findings(check_core(), ("OP_NOT_VERB",), subject="records/filter") == []


class TestTheExamples:
    def test_the_fixture_runs_twice_identically(self, fixture_ext):
        [r] = run_examples(fixture_ext, resolve_inputs=read_inputs_from(INPUTS))
        assert (r.op, r.status, r.kind, r.satisfies, r.deterministic) == (ADDRESS, "identical", ALIGNMENT,
                                                                          True, True)
        assert len(r.seconds) == 2 and r.findings == ()
        assert REGISTRY.find(ADDRESS) is None

    def test_a_reference_without_a_resolver_is_a_finding(self, fixture_ext):
        [r] = run_examples(fixture_ext)
        assert r.status == "failed" and codes(r.findings) == ["EXAMPLE_INPUTS_UNRESOLVED"]

    def test_a_model_op_is_skipped_without_a_model(self, tmp_path, monkeypatch):
        ext = copy_fixture(tmp_path, monkeypatch, "mb_model_ext", {
            "__init__.py": [('name="alice/interp-extras"', 'name="carol/model"')],
            "ops/geometry/align.py": [('    example={"method": "overlap"},\n',
                                       '    example={"method": "overlap"},\n    needs=frozenset({"model.forward"}),\n')],
        })
        result = check_extension(ext, resolve_inputs=read_inputs_from(INPUTS))
        [r] = result.examples
        assert r.status == "skipped" and codes(r.findings) == ["EXAMPLE_NEEDS_MODEL"]
        assert result.double_run == "skipped"

    def test_an_output_of_the_wrong_kind_is_a_finding(self, tmp_path, monkeypatch):
        ext = copy_fixture(tmp_path, monkeypatch, "mb_wrong_ext", {
            "__init__.py": [('name="alice/interp-extras"', 'name="dave/wrong"')],
            "ops/geometry/align.py": [('return collection("geometry/alignment", ', 'return collection("records/record", ')],
        })
        [r] = run_examples(ext, resolve_inputs=read_inputs_from(INPUTS))
        assert r.status == "identical" and r.satisfies is False
        assert codes(r.findings) == ["EXAMPLE_WRONG_KIND"]


class TestTheReport:
    def test_to_dict_is_the_extension_objects_conformance(self, fixture_ext):
        result = check_extension(fixture_ext, resolve_inputs=read_inputs_from(INPUTS))
        assert isinstance(result, Conformance) and result.passed
        d = result.to_dict()
        assert set(d) == {"compute", "declarations", "double_run", "installs_beside"}
        assert (d["declarations"], d["double_run"]) == ("passed", "identical")
        assert d["compute"] == d["installs_beside"] and "+" not in d["compute"]

    def run_cli(self, *args: str, path: Path = FIXTURE) -> subprocess.CompletedProcess:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(path), str(Path(__file__).parents[1])])}
        return subprocess.run([sys.executable, "-m", "mechbench_compute.conformance", *args],
                              capture_output=True, text=True, env=env, timeout=300)

    def test_the_cli_prints_the_report_as_json(self):
        run = self.run_cli("mb_fixture_ext", "--inputs", str(INPUTS))
        assert run.returncode == 0, run.stderr[-2000:]
        report = json.loads(run.stdout)
        assert report["declarations"] == "passed" and report["double_run"] == "identical"
        assert report["examples"][0]["op"] == ADDRESS
        assert [(f["code"], f["severity"]) for f in report["findings"]] == [("KIND_SHADOWS_CORE", WARNING)]

    def test_the_cli_fails_on_a_refused_example(self):
        run = self.run_cli("mb_fixture_ext:MANIFEST")
        assert run.returncode == 1
        assert json.loads(run.stdout)["examples"][0]["findings"][0]["code"] == "EXAMPLE_INPUTS_UNRESOLVED"
