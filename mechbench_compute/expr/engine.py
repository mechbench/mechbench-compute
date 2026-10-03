from __future__ import annotations

import atexit
import functools
import hashlib
import json
import pathlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ENGINE_VERSION = "v0.2.0"
ENGINE_SHA256 = "5a0b2996be6c922bf482c4c4712fd139f82281219704ea188222f0fe7c5ffcb9"
ENGINE_URL = (
    "https://github.com/mechbench/mechbench-expr/releases/download/"
    f"{ENGINE_VERSION}/mbexpr-{ENGINE_VERSION}.wasm"
)
WASM_PATH = pathlib.Path(__file__).with_name("mbexpr.wasm")
ROOTS = ("params", "header")
FUEL = 1_000_000
FUEL_MARGIN = 2
FUEL_REMEDY = (
    "Read less per record: a comprehension spends steps on every item of the list it walks, and a "
    "built-in over the list (`sum(xs)`, `max(xs)`, `len(xs)`) two in all; a captured vector already "
    "carries its `norm`, and `top: k` on `activations/capture` names its largest coordinates.")


class FuelRefused(ValueError):
    code = "EXPRESSION_FUEL"

    def __init__(self, expr: str, rows: int, share: int) -> None:
        self.expr = expr
        self.rows = rows
        self.estimate = share * rows
        self.limit = FUEL
        self.remedy = FUEL_REMEDY
        super().__init__(
            f"{self.code}: `{expr}` spends more than {share:,} steps on the first of {rows:,} records, "
            f"so at that rate the records need more than {self.estimate:,}, and one call has "
            f"{self.limit:,} for all of them. {self.remedy}")

    @property
    def issue(self) -> dict[str, Any]:
        return {"code": self.code, "expr": self.expr, "rows": self.rows, "estimate": self.estimate,
                "limit": self.limit, "remedy": self.remedy,
                "message": str(self).removeprefix(f"{self.code}: ")}


class ExprError(ValueError):
    def __init__(self, expr: str, error: Mapping[str, Any]) -> None:
        self.expr = expr
        self.kind = str(error.get("kind", "error"))
        self.detail = str(error.get("message", ""))
        self.start = error.get("start")
        self.end = error.get("end")
        self.record = error.get("record")
        at = f" in record {self.record}" if self.record is not None else ""
        super().__init__(f"{self.detail}{at}: {expr}")

    @classmethod
    def of(cls, expr: str, kind: str, message: str) -> ExprError:
        return cls(expr, dict(zip(("kind", "message"), (kind, message), strict=True)))


@dataclass(frozen=True)
class Evaluated:
    values: list[Any]
    undefined: dict[str, int]


@dataclass(frozen=True)
class Filtered:
    kept: list[int]
    unknown: int
    undefined: dict[str, int]


class Engine:
    def __init__(self, wasm: bytes) -> None:
        import wasmtime

        self._wasmtime = wasmtime
        self._engine = wasmtime.Engine()
        self._module = wasmtime.Module(self._engine, wasm)
        self._local = threading.local()

    def _instantiate(self) -> tuple[Any, Any]:
        store = self._wasmtime.Store(self._engine)
        instance = self._wasmtime.Instance(store, self._module, [])
        return store, instance.exports(store)

    def call(self, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            data = json.dumps(request, allow_nan=False, default=_make_jsonable).encode()
        except ValueError as e:
            raise ExprError.of(str(request.get("expr", "")), "type", f"a value is not JSON: {e}") from None
        store, x = self._instantiate()
        memory = x["memory"]
        ptr = x["mbexpr_alloc"](store, len(data))
        memory.write(store, data, ptr)
        packed = x["mbexpr_call"](store, ptr, len(data))
        x["mbexpr_free"](store, ptr, len(data))
        out_ptr, out_len = (packed >> 32) & 0xFFFFFFFF, packed & 0xFFFFFFFF
        answer = json.loads(memory.read(store, out_ptr, out_ptr + out_len))
        x["mbexpr_free"](store, out_ptr, out_len)
        return answer

    def _ask(self, expr: str, request: Mapping[str, Any]) -> dict[str, Any]:
        request = {**request, "fuel": FUEL}
        records = request.get("records")
        if records is not None and len(records) > FUEL_MARGIN:
            self._check_fuel(expr, request, len(records))
        answer = self.call(request)
        if not answer.get("ok"):
            raise ExprError(expr, answer.get("error") or {})
        return answer

    def _check_fuel(self, expr: str, request: Mapping[str, Any], rows: int) -> None:
        share = -(-FUEL * FUEL_MARGIN // rows)
        first = self.call({**request, "records": request["records"][:1], "fuel": share})
        error = first.get("error") or {}
        if error.get("kind") == "limit" and error.get("record") is not None:
            raise FuelRefused(expr, rows, share)

    def check(self, expr: str) -> dict[str, Any]:
        return self._ask(expr, {"op": "check", "expr": expr})

    def evaluate(
        self,
        exprs: str | Mapping[str, str],
        records: Sequence[Mapping[str, Any]] | None = None,
        params: Mapping[str, Any] | None = None,
        header: Mapping[str, Any] | None = None,
    ) -> Evaluated:
        sources = [exprs] if isinstance(exprs, str) else list(exprs.values())
        request: dict[str, Any] = {"op": "eval", "params": params or {}, "header": header or {}}
        request["expr" if isinstance(exprs, str) else "exprs"] = exprs
        if records is not None:
            request["records"] = self._project(sources, records)
        answer = self._ask(sources[0] if len(sources) == 1 else "; ".join(sources), request)
        return Evaluated(answer["values"], answer.get("undefined") or {})

    def filter(
        self,
        expr: str,
        records: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any] | None = None,
        header: Mapping[str, Any] | None = None,
    ) -> Filtered:
        answer = self._ask(expr, {
            "op": "filter", "expr": expr, "records": self._project([expr], records),
            "params": params or {}, "header": header or {},
        })
        return Filtered(answer["kept"], int(answer.get("unknown") or 0), answer.get("undefined") or {})

    def render(
        self,
        template: str,
        records: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any] | None = None,
        header: Mapping[str, Any] | None = None,
    ) -> Evaluated:
        answer = self._ask(template, {
            "op": "template", "template": template, "records": list(records),
            "params": params or {}, "header": header or {},
        })
        return Evaluated(answer["values"], answer.get("undefined") or {})

    def split(self, expr: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._ask(expr, {"op": "split", "expr": expr, "params": params or {}})

    def _project(self, sources: Sequence[str], records: Sequence[Mapping[str, Any]]) -> list[Any]:
        fields = self._read_roots(tuple(sources))
        if fields is None:
            return list(records)
        return [{k: r[k] for k in fields if k in r} for r in records]

    def _read_roots(self, sources: tuple[str, ...]) -> frozenset[str] | None:
        cache = getattr(self._local, "roots", None)
        if cache is None:
            cache = self._local.roots = {}
        if sources not in cache:
            roots: set[str] = set()
            whole = False
            for src in sources:
                for path in self.check(src)["reads"]:
                    root = path.split(".", 1)[0].split("[", 1)[0]
                    if root == "record":
                        whole = True
                    elif root not in ROOTS:
                        roots.add(root)
            cache[sources] = None if whole or "record" in " ".join(sources) else frozenset(roots)
        return cache[sources]


def _make_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def verify_wasm(data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != ENGINE_SHA256:
        raise RuntimeError(f"the expression engine's module is {digest}, not the pinned {ENGINE_SHA256}")


@functools.cache
def load_engine() -> Engine:
    data = WASM_PATH.read_bytes()
    verify_wasm(data)
    return Engine(data)


atexit.register(load_engine.cache_clear)
