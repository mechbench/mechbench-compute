from __future__ import annotations

import ast
import pathlib

import pytest

OPS = pathlib.Path(__file__).resolve().parent.parent / "mechbench_compute" / "ops"

FREE = frozenset({"mechbench_compute.api", "mechbench_compute._mlx"})

IMPORTS_INTERNALS: frozenset[str] = frozenset({
    "activations/capture.py",
    "activations/capture_attention.py",
    "activations/capture_tokens.py",
    "activations/contrast.py",
    "activations/examples.py",
    "adapter/measure.py",
    "adapter/merge.py",
    "adapter/publish.py",
    "adapter/train.py",
    "direction/add.py",
    "direction/average.py",
    "direction/classify.py",
    "direction/decompose.py",
    "direction/fit.py",
    "direction/normalize.py",
    "direction/orthogonalize.py",
    "direction/project.py",
    "direction/regress.py",
    "direction/unembed.py",
    "eval/benchmark.py",
    "eval/expect.py",
    "eval/judge.py",
    "geometry/compare.py",
    "intervene/ablate_heads.py",
    "intervene/ablate_layers.py",
    "intervene/apply.py",
    "intervene/patch.py",
    "intervene/path.py",
    "intervene/steer.py",
    "logits/attribute.py",
    "logits/read.py",
    "logits/read_layers.py",
    "logits/scan.py",
    "records/cross.py",
    "records/derive.py",
    "records/diff.py",
    "records/filter.py",
    "records/group.py",
    "records/join.py",
    "records/jq.py",
    "records/plot.py",
    "records/python.py",
    "records/sort.py",
    "records/tabulate.py",
    "records/union.py",
    "records/unnest.py",
    "records/zip.py",
    "text/chat.py",
    "text/extend.py",
    "text/generate.py",
    "text/measure.py",
    "text/render.py",
    "text/tokenize.py",
    "tools/lookup.py",
    "trajectory/aggregate.py",
    "trajectory/capture.py",
    "trajectory/compare.py",
    "trajectory/project.py",
    "weights/capture.py",
    "weights/circuit.py",
    "weights/decompose.py",})


def read_internal_imports(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                found.append("." * node.level + module)
            elif module == "mechbench_compute":
                found += [f"mechbench_compute.{a.name}" for a in node.names
                          if f"mechbench_compute.{a.name}" not in FREE]
            elif module.split(".")[0] == "mechbench_compute" and module not in FREE:
                found.append(module)
        elif isinstance(node, ast.Import):
            found += [a.name for a in node.names
                      if a.name.split(".")[0] == "mechbench_compute" and a.name not in FREE]
    return found


def op_files() -> dict[str, str]:
    return {str(p.relative_to(OPS)): p.read_text() for p in sorted(OPS.rglob("*.py"))
            if p.name != "__init__.py"}


class TestOperationsUseTheApi:
    def test_an_unlisted_operation_imports_only_the_api(self):
        reaching = {rel: read_internal_imports(src) for rel, src in op_files().items()
                    if rel not in IMPORTS_INTERNALS}
        reaching = {rel: mods for rel, mods in reaching.items() if mods}
        assert not reaching, (
            f"{reaching}: an operation imports mechbench_compute through "
            "mechbench_compute.api (docs/PLUGIN_API.md). Add what it needs to "
            "the api, or import it from there; IMPORTS_INTERNALS only shrinks.")

    @pytest.mark.parametrize("rel", sorted(IMPORTS_INTERNALS))
    def test_a_listed_operation_still_imports_internals(self, rel):
        files = op_files()
        assert rel in files, f"{rel} is listed and is not under ops/; take it off the list"
        assert read_internal_imports(files[rel]), (
            f"{rel} imports only the api now; take it off IMPORTS_INTERNALS")

    def test_the_api_offers_every_name_it_lists(self):
        from mechbench_compute import api

        for name in api.__all__:
            assert getattr(api, name) is not None, name
        with pytest.raises(AttributeError, match="has no 'nothing'"):
            api.nothing  # noqa: B018
