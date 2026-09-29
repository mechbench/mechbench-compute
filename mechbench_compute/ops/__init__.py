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
    loaded: Any = None
    executor: Any = None
    on_item: Callable[..., None] | None = None
    on_start: Callable[..., None] | None = None
    on_checkpoint: Callable[..., None] | None = None
    resume_items: Mapping[str, Any] | None = None
    resume_state: Any = None
    secrets: Mapping[str, Any] | None = None
    input_paths: Mapping[str, str] = field(default_factory=dict)
    run_params: Mapping[str, Any] | None = None
    result_base: str | None = None
    on_token: Callable[..., None] | None = None

    def model(self, ref: Any) -> Any:
        if self.loaded is not None:
            return self.loaded
        if self.executor is None:
            raise RuntimeError(
                "this operation needs a model and the context has neither "
                "one loaded nor an executor to load it")
        return self.executor._model_loaded(ref)


def resolve_module_name(op: str) -> str:
    family, name = op.split("/", 1)
    return f"{__name__}.{family}.{name.replace('-', '_')}"


@cache
def load_modules() -> dict[str, ModuleType]:
    found: dict[str, ModuleType] = {}
    for info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
        if info.ispkg or info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        mod = importlib.import_module(info.name)
        op = getattr(mod, "OP", None)
        if op is None:
            raise ImportError(f"{info.name} is under ops/ and declares no OP")
        if resolve_module_name(op.name) != info.name:
            raise ImportError(
                f"{info.name} declares {op.name!r}, which belongs at "
                f"{resolve_module_name(op.name)}: an operation's path is a function of its name")
        found[op.name] = mod
    return found


def find(op: str) -> ModuleType | None:
    return load_modules().get(op)


def fuses_adapter(op: str) -> bool:
    declared = load_modules()[op].OP
    return declared.requires == "mlx-local" and declared.port("adapter") is not None


def read_context_uses(mod: ModuleType) -> frozenset[str]:
    tree = ast.parse(inspect.getsource(mod))
    funcs = {n.name: n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    uses: set[str] = set()
    seen: set[str] = set()
    todo = ["run"]
    while todo:
        name = todo.pop()
        if name in seen or name not in funcs:
            continue
        seen.add(name)
        fn = funcs[name]
        attributed = {id(n.value) for n in ast.walk(fn)
                      if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                      and n.value.id == "ctx"}
        followed: set[int] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute) and id(n.value) in attributed:
                uses.add(n.attr)
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "getattr" and len(n.args) >= 2
                  and isinstance(n.args[0], ast.Name) and n.args[0].id == "ctx"
                  and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)):
                uses.add(n.args[1].value)
                followed.add(id(n.args[0]))
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id in funcs):
                for a in (*n.args, *(k.value for k in n.keywords)):
                    if isinstance(a, ast.Name) and a.id == "ctx":
                        todo.append(n.func.id)
                        followed.add(id(a))
        for n in ast.walk(fn):
            if (isinstance(n, ast.Name) and n.id == "ctx" and isinstance(n.ctx, ast.Load)
                    and id(n) not in attributed and id(n) not in followed):
                uses.add("ctx")
    return frozenset(uses)


@cache
def find_standalone() -> frozenset[str]:
    return frozenset(name for name, mod in load_modules().items()
                     if mod.OP.requires == "pure" and not read_context_uses(mod))


def run_standalone(op: str, inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return load_modules()[op].run(Context(), inputs, params)


import mechbench_compute.lexicon  # noqa: E402,F401
