from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping


MODEL_NEEDS = frozenset({"model.forward", "model.sample", "model.backward"})

MEMBER_NEEDS: dict[str, frozenset[str]] = {
    "model": MODEL_NEEDS,
    "loaded": MODEL_NEEDS,
    "evict_model": MODEL_NEEDS,
    "executor": frozenset({"executor.sub"}),
    "sub": frozenset({"executor.sub"}),
    "provider": frozenset({"provider.chat", "provider.embed"}),
    "memo": frozenset({"memo"}),
    "materialize": frozenset({"objects.read"}),
    "secrets": frozenset({"secrets"}),
}

REACHING_PAST_NEEDS: dict[str, frozenset[str]] = {}


class NeedNotDeclared(RuntimeError):
    pass


class Refused:
    def __init__(self, message: str) -> None:
        object.__setattr__(self, "_message", message)

    def refuse(self, *_: Any, **__: Any) -> Any:
        raise NeedNotDeclared(object.__getattribute__(self, "_message"))

    def __getattr__(self, _name: str) -> Any:
        self.refuse()

    __setattr__ = __getitem__ = __call__ = __iter__ = __bool__ = __len__ = __contains__ = refuse


def check_member(op: Any, member: str) -> str | None:
    wanted = MEMBER_NEEDS.get(member)
    if wanted is None or wanted & op.needs or member in REACHING_PAST_NEEDS.get(op.name, ()):
        return None
    return f"{op.name} did not declare {' or '.join(sorted(wanted))}"


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

    refused: Mapping[str, str] = field(default_factory=dict)
    host: Any = None
    run_secrets: Mapping[str, Any] | None = None

    @classmethod
    def for_op(cls, op: Any, executor: Any = None, **lent: Any) -> Context:
        refused = {m: msg for m in MEMBER_NEEDS if (msg := check_member(op, m)) is not None}
        run_secrets = lent.get("secrets")
        for member in ("executor", "secrets"):
            if member in refused:
                lent[member] = Refused(refused[member])
        if "executor" not in refused:
            lent["executor"] = executor
        return cls(refused=refused, host=executor, run_secrets=run_secrets, **lent)

    def check(self, member: str) -> None:
        if member in self.refused:
            raise NeedNotDeclared(self.refused[member])

    def find_host(self, member: str, doing: str) -> Any:
        self.check(member)
        host = self.executor if self.host is None else self.host
        if host is None:
            raise RuntimeError(
                f"this operation {doing}, and the context has no executor to do it")
        return host

    def model(self, ref: Any) -> Any:
        self.check("model")
        if self.loaded is not None:
            return self.loaded
        return self.find_host("model", "needs a model and none is loaded")._model_loaded(ref)

    def sub(self, target: str | Mapping[str, Any], inputs: Mapping[str, Any],
            params: Mapping[str, Any], *, budget: float | None = None,
            on_item: Callable[..., None] | None = None,
            on_start: Callable[..., None] | None = None) -> Any:
        host = self.find_host("sub", "runs a sub-run")
        secrets = self.run_secrets
        if secrets is None and not isinstance(self.secrets, Refused):
            secrets = self.secrets
        return host.run_sub(target, inputs, params, secrets=secrets, budget=budget,
                            on_item=on_item, on_start=on_start)

    def provider(self, model_ref: Any) -> Any:
        self.check("provider")
        host = self.executor if self.host is None else self.host
        if host is None:
            from mechbench_compute.providers.provider_client import ProviderClient

            return ProviderClient(model_ref, secrets=self.secrets)
        return host.open_provider(model_ref, self.secrets)

    def memo(self, key: Any) -> Any:
        if not key:
            return None
        return self.find_host("memo", "keeps a memo")._open_memo({"cache": key})

    def materialize(self, ref: str) -> Path:
        return self.find_host("materialize", "reads a stored checkpoint")._materialize_checkpoint(ref)

    def evict_model(self) -> None:
        self.check("evict_model")
        self.loaded = None
        host = self.executor if self.host is None else self.host
        if host is not None:
            host.evict_model()


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
    return "model.forward" in declared.needs and declared.port("adapter") is not None


def fuses_adapter_locally(op: str) -> bool:
    declared = load_modules()[op].OP
    return declared.requires == "by-model" and declared.port("adapter") is not None


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
                     if not mod.OP.needs and not read_context_uses(mod))


def run_standalone(op: str, inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return load_modules()[op].run(Context(), inputs, params)


import mechbench_compute.lexicon  # noqa: E402,F401
