from __future__ import annotations

import ast
import inspect
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
    "open_budget": frozenset({"executor.sub"}),
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
    scope: str | None = None
    node: str | None = None
    name: str | None = None
    fused: list[Any] = field(default_factory=list)

    @classmethod
    def for_op(cls, op: Any, executor: Any = None, **lent: Any) -> Context:
        from mechbench_compute.registry import scope_of

        op = getattr(op, "op", op)
        lent.setdefault("scope", scope_of(op.name))
        lent.setdefault("name", op.name)
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
        host = self.find_host("model", "needs a model and none is loaded")
        model = host._model_loaded(ref)
        host._reference_fused(model, ref, self.fused, self.name)
        return model

    def sub(self, target: str | Mapping[str, Any], inputs: Mapping[str, Any],
            params: Mapping[str, Any], *, budget: Any = None,
            on_item: Callable[..., None] | None = None,
            on_start: Callable[..., None] | None = None) -> Any:
        host = self.find_host("sub", "runs a sub-run")
        secrets = self.run_secrets
        if secrets is None and not isinstance(self.secrets, Refused):
            secrets = self.secrets
        return host.run_sub(target, inputs, params, secrets=secrets, budget=budget,
                            on_item=on_item, on_start=on_start)

    def open_budget(self, cap_usd: float) -> Any:
        return self.find_host("open_budget", "caps a sub-run's spend").open_child_budget(cap_usd)

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
    from mechbench_compute.lexicon.walk import name_module

    return name_module(__name__, op)


def load_modules() -> dict[str, ModuleType]:
    from mechbench_compute.registry import CORE

    return CORE.modules()


def resolve_op(op: Any) -> Any:
    if hasattr(op, "module"):
        return op
    from mechbench_compute.registry import REGISTRY

    return REGISTRY.resolve(op)


def find(op: str) -> ModuleType | None:
    from mechbench_compute.registry import REGISTRY

    hit = REGISTRY.find(op)
    return hit.module if hit is not None else None


def fuses_adapter(op: Any) -> bool:
    declared = resolve_op(op).op
    return "model.forward" in declared.needs and declared.port("adapter") is not None


def fuses_adapter_locally(op: Any) -> bool:
    declared = resolve_op(op).op
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


def find_standalone() -> frozenset[str]:
    from mechbench_compute.registry import REGISTRY

    return read_standalone(REGISTRY, REGISTRY.generation)


@cache
def read_standalone(registry: Any, generation: int) -> frozenset[str]:
    return frozenset(r.name for r in registry.resolved()
                     if not r.op.needs and not read_context_uses(r.module))


def run_standalone(op: Any, inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    from mechbench_compute.registry import REGISTRY

    resolved = resolve_op(op)
    with REGISTRY.within(resolved.scope):
        return resolved.module.run(Context(scope=resolved.scope), inputs, params)
