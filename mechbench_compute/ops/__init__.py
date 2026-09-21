"""The operations, one file each. See docs/OPS_LAYOUT.md.

An operation named `family/name` is the module
`mechbench_compute.ops.<family>.<name>` (a hyphen in the name is an
underscore in the path). That module holds `OP`, the declaration, and
`run(ctx, inputs, params)`, the one entry point. Nothing lists the
operations: `modules()` finds them by walking this package.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from functools import cache
from types import ModuleType
from typing import Any, Callable, Mapping


@dataclass
class Context:
    """What the executor lends an operation for one node.

    Every field has a default, so a test that needs only a model builds
    one in a line: `run(Context(model=fake), inputs, params)`.
    """

    #: A model already loaded, for a test. Otherwise `executor` loads it.
    loaded: Any = None
    executor: Any = None
    on_item: Callable[..., None] | None = None
    on_start: Callable[..., None] | None = None
    on_checkpoint: Callable[..., None] | None = None
    resume_items: Mapping[str, Any] | None = None
    resume_state: Any = None
    secrets: Mapping[str, Any] | None = None
    input_paths: Mapping[str, str] = field(default_factory=dict)
    bindings: Mapping[str, Any] | None = None
    result_base: str | None = None

    def model(self, ref: Any) -> Any:
        """The loaded model a `model` param names."""
        if self.loaded is not None:
            return self.loaded
        if self.executor is None:
            raise RuntimeError(
                "this operation needs a model and the context has neither "
                "one loaded nor an executor to load it")
        return self.executor._model_loaded(ref)


def module_name(op: str) -> str:
    """`logits/read-layers` -> `mechbench_compute.ops.logits.read_layers`."""
    family, name = op.split("/", 1)
    return f"{__name__}.{family}.{name.replace('-', '_')}"


@cache
def modules() -> dict[str, ModuleType]:
    """Every operation's module, by the operation's name."""
    found: dict[str, ModuleType] = {}
    for info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
        if info.ispkg or info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        mod = importlib.import_module(info.name)
        op = getattr(mod, "OP", None)
        if op is None:
            raise ImportError(f"{info.name} is under ops/ and declares no OP")
        if module_name(op.name) != info.name:
            raise ImportError(
                f"{info.name} declares {op.name!r}, which belongs at "
                f"{module_name(op.name)}: an operation's path is a function of its name")
        found[op.name] = mod
    return found


def find(op: str) -> ModuleType | None:
    return modules().get(op)


def fuses_adapter(op: str) -> bool:
    """Whether the executor loads the model and fuses an adapter around
    this operation: it needs local weights, and an adapter may arrive."""
    declared = modules()[op].OP
    return declared.requires == "mlx-local" and declared.port("adapter") is not None


@cache
def standalone() -> frozenset[str]:
    """The operations that run with no executor: pure, and their `run`
    never reads `ctx`. What a tool handler or a chunked reduce may call."""
    names = set()
    for name, mod in modules().items():
        if mod.OP.requires != "pure":
            continue
        tree = ast.parse(inspect.getsource(mod.run))
        if not any(isinstance(n, ast.Name) and n.id == "ctx" and isinstance(n.ctx, ast.Load)
                   for n in ast.walk(tree)):
            names.add(name)
    return frozenset(names)


def run_standalone(op: str, inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return modules()[op].run(Context(), inputs, params)


# Last, and deliberately: an operation's file imports its declaration's
# vocabulary from `mechbench_compute.lexicon`, and the lexicon, as it
# initialises, walks this package to find the operations. If an
# operation's file were the first thing imported, the lexicon would walk
# into that same half-imported file and find no OP in it. A parent
# package initialises before any module inside it starts, so importing
# the lexicon here means it is always whole, or at least past the point
# of walking, before any operation's file begins to execute.
import mechbench_compute.lexicon  # noqa: E402,F401
