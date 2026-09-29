from __future__ import annotations

from functools import cache
from typing import Any

from mechbench_compute.conformance.check_kinds import check_kinds
from mechbench_compute.conformance.check_ops import INTERNAL, check_ops
from mechbench_compute.conformance.check_package import check_package
from mechbench_compute.conformance.check_types import check_param, check_types, walk_params
from mechbench_compute.conformance.finding import Finding
from mechbench_compute.conformance.manifest import Manifest, read_core, read_manifest
from mechbench_compute.lexicon.extension import Extension


def read_common_params() -> list[dict[str, Any]]:
    from mechbench_compute.lexicon.common import COMMON

    return [p.to_dict() for p in COMMON]


def check_manifest(extension: Extension | dict[str, Any] | Manifest) -> list[Finding]:
    m = read_manifest(extension)
    return [*check_ops(m), *check_types(m, read_common_params()), *check_kinds(m)]


@cache
def check_core() -> tuple[Finding, ...]:
    from mechbench_compute.lexicon.values import VALUES
    from mechbench_compute.registry import CORE

    findings = check_manifest(read_core())
    common = read_common_params()
    for at, p in walk_params(common, "common"):
        findings += check_param(at, p)
        if not str(p.get("type") or "").strip() or not str(p.get("doc") or "").strip():
            findings.append(Finding("PARAM_UNDOCUMENTED", at, "a common param has a type and a description"))
        for pat in INTERNAL:
            if pat.search(str(p.get("doc") or "")):
                findings.append(Finding("HOUSE_IDIOM", at, f"{pat.pattern!r} in its doc"))
    declared = [(at, p["choices"]) for at, p in walk_params(common, "common") if p.get("choices")]
    declared += [(f"value {v.name}.{name}", spec["choices"]) for v in VALUES
                 for name, spec in v.fields.items() if spec.get("choices")]
    findings += check_package(CORE, skip_literals=("lexicon", "ops", "_smoke_bench.py"), skip_sets=("lexicon",),
                              closed_sets=declared)
    return tuple(findings)
