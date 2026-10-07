from __future__ import annotations

import ast
import math
import operator
from collections.abc import Mapping
from typing import Any

from mechbench_compute.api import Op, P, Resume

OP = Op(
    name="tools/calc",
    resume=Resume("restart"),
    summary=(
        "Evaluate an arithmetic expression — numbers and operators only — as "
        "a tool a model may call."
    ),
    description="""\
The expression is parsed and refused if it contains anything but numeric
literals and `+ - * / // % **` (and parentheses): a tool a model can steer
must not be an evaluator. An integer result past 4300 digits is refused
before it is computed. Offered to a `chat` node by naming `"calc"`
in its `tools`; the model's call supplies `arguments:
{expression}`. A tool has no input ports — its arguments come from the
call.
""",
    inputs=(),
    output=None,
    params=(
        P("expression", "string",
          "The expression, when the block is run directly rather than as a "
          "tool call.",
          None),
    ),
    example={"expression": "(3 + 4) * 12 / 7"},
)


def run(ctx, inputs, params):
    return calculate(inputs, params)


class CalcRefused(ValueError):
    pass


MAX_LENGTH = 10_000
# external: CPython — 4300 digits is the default limit on converting an int to text
MAX_DIGITS = 4300
MAX_BITS = int(MAX_DIGITS * math.log2(10)) + 1

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    args = dict(inputs.get("arguments") or {})
    expression = str(args.get("expression") or params.get("expression") or "")
    if not expression:
        raise ValueError("calc needs an `expression`")
    if len(expression) > MAX_LENGTH:
        raise CalcRefused(f"calc refuses an expression over {MAX_LENGTH} characters")
    tree = ast.parse(expression, mode="eval")
    return {"expression": expression, "result": evaluate_node(tree.body)}


def evaluate_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, complex)):
            raise CalcRefused(
                f"calc refuses a {type(node.value).__name__} literal: it evaluates numbers")
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(evaluate_node(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](check_number(evaluate_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = check_number(evaluate_node(node.left))
        right = check_number(evaluate_node(node.right))
        check_size(node.op, left, right)
        return _BINARY[type(node.op)](left, right)
    raise CalcRefused(
        f"calc refuses {type(node).__name__}: it evaluates arithmetic, not code")


def check_number(value: Any) -> Any:
    if isinstance(value, tuple):
        raise CalcRefused("calc refuses arithmetic on a tuple: it evaluates numbers")
    return value


def check_size(op: ast.operator, left: Any, right: Any) -> None:
    if isinstance(op, ast.Pow) and isinstance(left, int) and isinstance(right, int):
        if right > 0 and abs(left) > 1 and right * math.log2(abs(left)) > MAX_BITS:
            raise CalcRefused(
                f"calc refuses {left} ** {right}: the result would pass {MAX_DIGITS} digits")
    elif (isinstance(op, ast.Mult) and isinstance(left, int) and isinstance(right, int)
          and left.bit_length() + right.bit_length() > MAX_BITS + 1):
        raise CalcRefused(f"calc refuses a product past {MAX_DIGITS} digits")
