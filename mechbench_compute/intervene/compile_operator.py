from __future__ import annotations

import ast
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mechbench_compute.intervene.array_ops import read_array_ops
from mechbench_compute.intervene.operator_refused import OperatorRefused

NUMBER, CONDITION = "number", "condition"

ARITHMETIC = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide",
              ast.FloorDiv: "floor_divide", ast.Mod: "remainder", ast.Pow: "power"}

COMPARISONS = {ast.Eq: "equal", ast.NotEq: "not_equal", ast.Lt: "less", ast.LtE: "less_equal",
               ast.Gt: "greater", ast.GtE: "greater_equal"}

ONE_ARGUMENT = {"abs": "abs", "ceil": "ceil", "exp": "exp", "floor": "floor", "log2": "log2",
                "log10": "log10", "round": "round", "sqrt": "sqrt"}

ARITY = {"log": (1, 2, "one or two arguments"), "pow": (2, 2, "two arguments"),
         "min": (2, None, "two or more arguments"), "max": (2, None, "two or more arguments")}

FUNCTIONS = tuple(sorted({*ONE_ARGUMENT, *ARITY}))

UNDEFINED_SOMEWHERE = frozenset({"exp", "log", "log2", "log10", "sqrt"})

HINTS = {"where": "; the language writes an elementwise choice as `a if c else b`",
         "clip": "; write `min(max(x, lo), hi)`"}

CONSTRUCTS = {ast.Attribute: "field access", ast.Subscript: "indexing", ast.List: "a list",
              ast.Tuple: "a tuple", ast.Dict: "an object", ast.Set: "a set",
              ast.ListComp: "a comprehension", ast.GeneratorExp: "a comprehension"}

Fn = Callable[[Mapping[str, Any], Any], Any]


@dataclass(frozen=True)
class Operator:
    written: str
    canonical: str
    names: tuple[str, ...]
    undefinable: bool
    run: Fn

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        return self.run(env, read_array_ops(env["x"]).verbs)


@functools.lru_cache(maxsize=256)
def compile_operator(f: str) -> Operator:
    from mechbench_compute.expr.engine import ExprError, load_engine

    try:
        canonical = str(load_engine().check(f)["canonical"])
    except ExprError as e:
        hint = "; power is `**`, as in Python" if "^" in f else ""
        spans = isinstance(e.start, int) and isinstance(e.end, int)
        raise OperatorRefused(
            "OPERATOR_SYNTAX",
            f"`f` is not an expression of the platform's expression language: {e.detail}{hint}: `{f}`",
            construct=f[e.start:e.end] if spans else f) from None
    compiler = _Compiler(canonical)
    fn, kind = compiler.compile(ast.parse(canonical, mode="eval").body)
    if kind != NUMBER:
        raise OperatorRefused(
            "OPERATOR_TYPE", f"`{canonical}` gives a condition, and `f` gives the new coordinate, a "
            "number: write `1 if … else 0` for an indicator", construct=canonical)
    return Operator(f, canonical, tuple(sorted(compiler.names)), compiler.undefinable, fn)


def _is_whole(node: ast.AST) -> bool:
    v = getattr(node, "value", None) if isinstance(node, ast.Constant) else None
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and float(v).is_integer() and 0 <= v <= 64)


def _raise_to(base: Fn, exponent: Fn, node: ast.AST) -> Fn:
    if not _is_whole(node):
        return lambda env, xp: xp.power(base(env, xp), exponent(env, xp))
    n = int(node.value)

    def multiply(env, xp):
        b, out, k = base(env, xp), None, n
        while k:
            if k & 1:
                out = b if out is None else xp.multiply(out, b)
            k >>= 1
            if k:
                b = xp.multiply(b, b)
        return xp.ones_like(xp.array(b)) if out is None else out

    return multiply


class _Compiler:
    def __init__(self, canonical: str) -> None:
        self.canonical = canonical
        self.names: set[str] = set()
        self.undefinable = False

    def refuse(self, code: str, node: ast.AST, message: str, construct: str | None = None) -> None:
        snippet = ast.unparse(node)
        raise OperatorRefused(code, f"{message}: `{snippet}` in `{self.canonical}`",
                              construct=construct or snippet)

    def compile(self, node: ast.AST) -> tuple[Fn, str]:
        method = getattr(self, f"compile_{type(node).__name__.lower()}", None)
        if method is None:
            what = CONSTRUCTS.get(type(node), "this construct")
            self.refuse("OPERATOR_UNSUPPORTED", node, f"{what} has no elementwise form; `f` reads "
                        "numbers, `x` and its constants")
        return method(node)

    def number(self, node: ast.AST) -> Fn:
        fn, kind = self.compile(node)
        if kind != NUMBER:
            self.refuse("OPERATOR_TYPE", node, "a condition where a number is needed")
        return fn

    def condition(self, node: ast.AST) -> Fn:
        fn, kind = self.compile(node)
        if kind != CONDITION:
            self.refuse("OPERATOR_TYPE", node, "a number where a condition is needed: compare it, "
                        "as in `x > 0`")
        return fn

    def compile_constant(self, node: ast.Constant) -> tuple[Fn, str]:
        v = node.value
        if isinstance(v, bool):
            return (lambda env, xp: xp.array(v)), CONDITION
        if isinstance(v, (int, float)):
            value = float(v)
            return (lambda env, xp: value), NUMBER
        self.refuse("OPERATOR_UNSUPPORTED", node, "a string or None has no elementwise value")

    def compile_name(self, node: ast.Name) -> tuple[Fn, str]:
        name = node.id
        if name in ("params", "header", "record"):
            self.refuse("OPERATOR_UNSUPPORTED", node, f"`f` reads `x` and its constants, not `{name}`: "
                        "bind a value as a constant (a `$param` there is bound with the run)")
        if name != "x":
            self.names.add(name)
        return (lambda env, xp: env[name]), NUMBER

    def compile_binop(self, node: ast.BinOp) -> tuple[Fn, str]:
        verb = ARITHMETIC.get(type(node.op))
        if verb is None:
            self.refuse("OPERATOR_UNSUPPORTED", node, "this operator has no elementwise form")
        left, right = self.number(node.left), self.number(node.right)
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)) or (
                isinstance(node.op, ast.Pow) and not _is_whole(node.right)):
            self.undefinable = True
        if isinstance(node.op, ast.Pow):
            return _raise_to(left, right, node.right), NUMBER
        return (lambda env, xp: getattr(xp, verb)(left(env, xp), right(env, xp))), NUMBER

    def compile_unaryop(self, node: ast.UnaryOp) -> tuple[Fn, str]:
        if isinstance(node.op, ast.Not):
            inner = self.condition(node.operand)
            return (lambda env, xp: xp.logical_not(inner(env, xp))), CONDITION
        operand = self.number(node.operand)
        if isinstance(node.op, ast.USub):
            return (lambda env, xp: xp.negative(operand(env, xp))), NUMBER
        if isinstance(node.op, ast.UAdd):
            return operand, NUMBER
        self.refuse("OPERATOR_UNSUPPORTED", node, "this operator has no elementwise form")

    def compile_compare(self, node: ast.Compare) -> tuple[Fn, str]:
        verbs = []
        for op in node.ops:
            verb = COMPARISONS.get(type(op))
            if verb is None:
                self.refuse("OPERATOR_UNSUPPORTED", node, "`in`, `not in`, `is` and `is not` test "
                            "membership and identity, which have no elementwise form")
            verbs.append(verb)
        operands = [self.number(n) for n in (node.left, *node.comparators)]

        def compare(env, xp):
            values = [o(env, xp) for o in operands]
            out = None
            for verb, a, b in zip(verbs, values, values[1:]):
                c = getattr(xp, verb)(a, b)
                out = c if out is None else xp.logical_and(out, c)
            return out

        return compare, CONDITION

    def compile_boolop(self, node: ast.BoolOp) -> tuple[Fn, str]:
        parts = [self.condition(v) for v in node.values]
        verb = "logical_and" if isinstance(node.op, ast.And) else "logical_or"

        def combine(env, xp):
            out = parts[0](env, xp)
            for p in parts[1:]:
                out = getattr(xp, verb)(out, p(env, xp))
            return out

        return combine, CONDITION

    def compile_ifexp(self, node: ast.IfExp) -> tuple[Fn, str]:
        test = self.condition(node.test)
        (body, kind), (orelse, other) = self.compile(node.body), self.compile(node.orelse)
        if kind != other:
            self.refuse("OPERATOR_TYPE", node, "one branch gives a number and the other a condition")
        return (lambda env, xp: xp.where(test(env, xp), body(env, xp), orelse(env, xp))), kind

    def compile_call(self, node: ast.Call) -> tuple[Fn, str]:
        if not isinstance(node.func, ast.Name):
            self.refuse("OPERATOR_UNSUPPORTED", node, "a method call has no elementwise form")
        name = node.func.id
        if name not in FUNCTIONS:
            self.refuse("OPERATOR_FUNCTION_UNKNOWN", node,
                        f"`{name}` is not a function the operator evaluates; `f` takes "
                        f"{', '.join(FUNCTIONS)}{HINTS.get(name, '')}", construct=name)
        base = [k for k in node.keywords if name == "log" and k.arg == "base"]
        for k in node.keywords:
            if k not in base:
                self.refuse("OPERATOR_UNSUPPORTED", node, f"`{name}` takes no `{k.arg}` here",
                            construct=f"{k.arg}=")
        nodes = [*node.args, *(k.value for k in base)]
        args = [self.number(a) for a in nodes]
        n = len(args)
        if name in ("min", "max") and n == 1:
            self.refuse("OPERATOR_UNSUPPORTED", node, f"`{name}` of one argument is the "
                        f"{'least' if name == 'min' else 'greatest'} of a list; elementwise it takes "
                        "two or more")
        if name == "round" and n == 2:
            self.refuse("OPERATOR_UNSUPPORTED", node, "rounding to decimal places has no elementwise "
                        "form here; `round(x)` rounds to a whole number, half to even")
        least, most, said = ARITY.get(name, (1, 1, "one argument"))
        if n < least or (most is not None and n > most):
            self.refuse("OPERATOR_TYPE", node, f"`{name}` takes {said}, not {n}")
        if name in UNDEFINED_SOMEWHERE or (name == "pow" and not _is_whole(nodes[1])):
            self.undefinable = True
        if name in ONE_ARGUMENT:
            verb, (only,) = ONE_ARGUMENT[name], args
            return (lambda env, xp: getattr(xp, verb)(only(env, xp))), NUMBER
        if name == "log":
            if n == 1:
                return (lambda env, xp: xp.log(args[0](env, xp))), NUMBER
            return (lambda env, xp: xp.divide(xp.log(args[0](env, xp)), xp.log(args[1](env, xp)))), NUMBER
        if name == "pow":
            return _raise_to(args[0], args[1], nodes[1]), NUMBER
        verb = "minimum" if name == "min" else "maximum"

        def fold(env, xp):
            out = args[0](env, xp)
            for a in args[1:]:
                out = getattr(xp, verb)(out, a(env, xp))
            return out

        return fold, NUMBER
