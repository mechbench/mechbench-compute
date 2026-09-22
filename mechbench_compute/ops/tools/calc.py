from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import Op, P

OP = Op(
    name="tools/calc",
    summary=(
        "Evaluate an arithmetic expression — numbers and operators only — as "
        "a tool a model may call."
    ),
    description="""\
The expression is parsed and refused if it contains anything but numeric
literals and `+ - * / // % **` (and parentheses): a tool a model can steer
must not be an evaluator. Offered to a `chat` node by naming `"calc"`
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
    return calc(inputs, params)


class CalcRefused(ValueError):
    """`calc` met something that is not arithmetic. A refusal, not a
    type error: the expression parsed fine, it just is not allowed."""


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Tuple, ast.Load,
)


def calc(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`tools/calc` — arithmetic, and ONLY arithmetic.

    Parsed, walked, and refused if it contains anything but numbers and
    operators: a tool a model can steer must not be an eval.
    """
    args = dict(inputs.get("arguments") or {})
    expression = str(args.get("expression") or params.get("expression") or "")
    if not expression:
        raise ValueError("calc needs an `expression`")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise CalcRefused(
                f"calc refuses {type(node).__name__}: it evaluates arithmetic, "
                "not code")
    value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    return {"expression": expression, "result": value}
