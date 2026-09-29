from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any

from mechbench_compute.conformance.check_manifest import check_manifest
from mechbench_compute.conformance.check_package import check_package
from mechbench_compute.conformance.finding import Finding, has_errors
from mechbench_compute.conformance.run_examples import ExampleResult, Resolver, read_inline, run_examples
from mechbench_compute.lexicon.extension import Extension


@dataclass(frozen=True)
class Conformance:
    compute: str
    declarations: str
    findings: tuple[Finding, ...]
    examples: tuple[ExampleResult, ...]
    double_run: str
    installs_beside: str

    @property
    def passed(self) -> bool:
        return self.declarations == "passed" and not has_errors(f for r in self.examples for f in r.findings)

    def to_dict(self) -> dict[str, Any]:
        return {"compute": self.compute, "declarations": self.declarations,
                "double_run": self.double_run, "installs_beside": self.installs_beside}

    def report(self) -> dict[str, Any]:
        return {**self.to_dict(), "passed": self.passed,
                "findings": [f.to_dict() for f in self.findings],
                "examples": [r.to_dict() for r in self.examples]}


def read_compute_version() -> str:
    from mechbench_compute import __version__

    return __version__.split("+", 1)[0]


def read_double_run(examples: list[ExampleResult]) -> str:
    ran = [r.status for r in examples if r.status in ("identical", "differs")]
    if "differs" in ran:
        return "differs"
    return "identical" if ran and all(r.status != "failed" for r in examples) else "skipped"


def check_extension(extension: Extension, *, resolve_inputs: Resolver = read_inline,
                    model: bool = False) -> Conformance:
    findings = [*check_manifest(extension), *check_package(extension)]
    examples = run_examples(extension, resolve_inputs=resolve_inputs, model=model)
    compute = read_compute_version()
    return Conformance(compute=compute, declarations="failed" if has_errors(findings) else "passed",
                       findings=tuple(findings), examples=tuple(examples),
                       double_run=read_double_run(examples), installs_beside=compute)


def load_extension(spec: str) -> Extension:
    from mechbench_compute.registry import GROUP

    if ":" in spec:
        module, _, attr = spec.partition(":")
        found = getattr(importlib.import_module(module), attr)
    else:
        points = [ep for ep in importlib.metadata.entry_points(group=GROUP) if ep.name == spec]
        found = points[0].load() if points else getattr(importlib.import_module(spec), "MANIFEST", None)
    if not isinstance(found, Extension):
        raise ValueError(f"{spec!r} names no Extension: give a package with a MANIFEST, "
                         "`module:ATTR`, or an entry point in `mechbench.extensions`")
    return found
