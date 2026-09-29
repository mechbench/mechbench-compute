from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mechbench_compute.conformance.finding import ERROR, WARNING, Finding
from mechbench_compute.lexicon._base import COLLECTION, Op, Output
from mechbench_compute.lexicon.address import address_op
from mechbench_compute.lexicon.extension import Extension

Resolver = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]

LOCAL_NEEDS = ("model.", "runtime.mlx")


@dataclass(frozen=True)
class ExampleResult:
    op: str
    status: str
    kind: str | None = None
    satisfies: bool | None = None
    deterministic: bool | None = None
    seconds: tuple[float, ...] = ()
    findings: tuple[Finding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "status": self.status, "kind": self.kind, "satisfies": self.satisfies,
                "deterministic": self.deterministic, "seconds": list(self.seconds),
                "findings": [f.to_dict() for f in self.findings]}


def is_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {"$ref"}


def read_inline(op: str, example_inputs: Mapping[str, Any]) -> dict[str, Any]:
    refs = sorted(port for port, v in example_inputs.items() if is_ref(v))
    if refs:
        raise ValueError(f"{op}'s example reads {refs} by reference, and no resolver was given")
    return dict(example_inputs)


def read_inputs_from(root: Path | str) -> Resolver:
    base = Path(root)

    def resolve(op: str, example_inputs: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for port, v in example_inputs.items():
            if not is_ref(v):
                out[port] = v
                continue
            ref = v["$ref"]
            where = ref.get("file") or ref.get("bench") if isinstance(ref, Mapping) else None
            if not where:
                raise ValueError(f"{op} port {port!r}: {ref!r} names no file and no bench path")
            path = base / str(where)
            if not path.is_file() and path.with_suffix(".json").is_file():
                path = path.with_suffix(".json")
            if not path.is_file():
                raise FileNotFoundError(f"{op} port {port!r}: {path} does not exist")
            out[port] = json.loads(path.read_text())
        return out

    return resolve


@contextmanager
def install_extension(extension: Extension) -> Iterator[str | None]:
    from mechbench_compute.registry import CORE, REGISTRY, InstalledSource

    point = SimpleNamespace(name=extension.extension, value=f"{extension.module}:MANIFEST",
                            dist=None, load=lambda: extension)
    held = REGISTRY.sources
    REGISTRY.sources = (CORE, InstalledSource(find=lambda: [point], installed=lambda: []))
    try:
        REGISTRY.refresh()
        yield REGISTRY.table().refused.get(extension.address)
    finally:
        REGISTRY.sources = held
        REGISTRY.refresh()


def read_key(kind_name: str) -> tuple[str, ...]:
    from mechbench_compute.lexicon import kinds as K
    from mechbench_compute.registry import REGISTRY

    for name in K.ancestry(kind_name):
        key = REGISTRY.kind(name).key
        if key:
            return tuple(key)
    return ()


def check_output(value: Any, out: Output, at: str) -> tuple[str | None, list[Finding]]:
    from mechbench_compute.lexicon import kinds as K

    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        return None, [Finding("EXAMPLE_WRONG_KIND", at, f"the output has no kind; it declares `{out.kind}`")]
    allowed = [out.kind, *(o.kind for o in out.otherwise)]
    if value["kind"] == COLLECTION:
        actual = str(value.get("item_kind") or "")
        items = list(value.get("items") or ())
    else:
        try:
            actual = K.resolve_kind(value["kind"], warn=False)[0]
        except KeyError:
            actual = value["kind"]
        items = []
    problems: list[Finding] = []
    if not any(K.satisfies(actual, k) for k in allowed):
        problems.append(Finding("EXAMPLE_WRONG_KIND", at, f"the output is `{actual}`; it declares `{out.kind}`"))
    key = read_key(actual) if actual in {k.name for k in K.KINDS} else ()
    for i, item in enumerate(items):
        missing = [f for f in key if not isinstance(item, Mapping) or f not in item]
        if missing:
            problems.append(Finding("EXAMPLE_WRONG_KIND", at, f"item {i} lacks the key fields {missing}"))
            break
    return actual, problems


def check_outputs(result: Any, op: Op, at: str) -> tuple[str | None, list[Finding]]:
    if op.output is not None:
        return check_output(result, op.output, at)
    kinds, problems = [], []
    for port, out in (op.outputs or {}).items():
        got = result.get(port) if isinstance(result, Mapping) else None
        kind, found = check_output(got, out, f"{at}:{port}")
        kinds.append(f"{port}={kind}")
        problems += found
    return (", ".join(kinds) or None), problems


def dump_result(result: Any) -> bytes:
    from mechbench_schema import dump_canonical

    from mechbench_compute.lexicon.kinds import canonical_collection

    return dump_canonical(canonical_collection(result))


def run_once(resolved: Any, inputs: Mapping[str, Any], params: Mapping[str, Any], executor: Any) -> Any:
    from mechbench_compute.ops import run_standalone

    if executor is None:
        return run_standalone(resolved, copy.deepcopy(dict(inputs)), dict(params))
    return executor.run_sub(resolved.name, copy.deepcopy(dict(inputs)), dict(params))


def run_example(extension: Extension, short: str, resolve_inputs: Resolver, model: bool) -> ExampleResult:
    from mechbench_compute.registry import REGISTRY

    address = address_op(extension.name, short)
    resolved = REGISTRY.resolve(address)
    declared = resolved.op
    if declared.example is None:
        return ExampleResult(address, "skipped", findings=(Finding("NO_EXAMPLE", address, "declares no example; "
                                                                   "verification runs each operation's example", WARNING),))
    local = any(n.startswith(LOCAL_NEEDS) for n in declared.needs)
    if local and not model:
        return ExampleResult(address, "skipped", findings=(Finding(
            "EXAMPLE_NEEDS_MODEL", address, "needs a model, and none is available here", WARNING),))
    try:
        inputs = dict(resolve_inputs(address, dict(declared.example_inputs or {})))
    except Exception as e:  # noqa: BLE001
        return ExampleResult(address, "failed", findings=(Finding("EXAMPLE_INPUTS_UNRESOLVED", address, str(e)),))
    executor = None
    if declared.needs:
        from mechbench_compute.protocol import ProtocolExecutor

        executor = ProtocolExecutor()
    results, seconds = [], []
    for _ in range(2):
        start = time.perf_counter()
        try:
            with REGISTRY.within(resolved.scope):
                results.append(run_once(resolved, inputs, declared.example, executor))
        except Exception as e:  # noqa: BLE001
            return ExampleResult(address, "failed", seconds=tuple(seconds), findings=(Finding(
                "EXAMPLE_FAILED", address, f"{type(e).__name__}: {e}"),))
        seconds.append(time.perf_counter() - start)
    with REGISTRY.within(resolved.scope):
        kind, findings = check_outputs(results[0], declared, address)
    try:
        same = dump_result(results[0]) == dump_result(results[1])
    except Exception as e:  # noqa: BLE001
        findings.append(Finding("EXAMPLE_NOT_CANONICAL", address, f"the output has no canonical bytes: {e}"))
        same = False
    if not same:
        findings.append(Finding("EXAMPLE_DIFFERS", address, "two runs of the example differ",
                                ERROR if declared.deterministic else WARNING))
    return ExampleResult(address, "identical" if same else "differs", kind=kind,
                         satisfies=not any(f.code == "EXAMPLE_WRONG_KIND" for f in findings),
                         deterministic=same, seconds=tuple(seconds), findings=tuple(findings))


def run_examples(extension: Extension, *, resolve_inputs: Resolver = read_inline,
                 model: bool = False) -> list[ExampleResult]:
    with install_extension(extension) as refused:
        if refused is not None:
            return [ExampleResult(extension.address, "failed",
                                  findings=(Finding("EXTENSION_REFUSED", extension.address, refused),))]
        return [run_example(extension, short, resolve_inputs, model) for short in sorted(extension.modules())]
