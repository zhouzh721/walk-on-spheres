"""Restricted mathematical expressions used by the geometry generators."""

from __future__ import annotations

import ast
import math
from collections.abc import Iterable


FUNCTIONS = {
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "cos": math.cos,
    "cosh": math.cosh,
    "exp": math.exp,
    "log": math.log,
    "max": max,
    "min": min,
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "tanh": math.tanh,
}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
)


class MathExpression:
    """Compile an arithmetic expression without allowing arbitrary Python code."""

    def __init__(self, source: str, variables: Iterable[str]):
        self.source = source
        self.variables = frozenset(variables)
        tree = ast.parse(source, mode="eval")
        allowed_names = self.variables | FUNCTIONS.keys() | CONSTANTS.keys()

        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise ValueError(
                    f"unsupported syntax {type(node).__name__!s} in expression {source!r}"
                )
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError(f"only numeric constants are allowed in {source!r}")
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                raise ValueError(f"unknown name {node.id!r} in expression {source!r}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                    raise ValueError(f"unsupported function call in expression {source!r}")
                if node.keywords:
                    raise ValueError(f"keyword arguments are not allowed in {source!r}")

        self._code = compile(tree, "<geometry-expression>", "eval")

    def __call__(self, **variables: float) -> float:
        missing = self.variables - variables.keys()
        if missing:
            raise ValueError(f"missing expression variables: {sorted(missing)}")
        namespace = dict(FUNCTIONS)
        namespace.update(CONSTANTS)
        namespace.update(variables)
        value = float(eval(self._code, {"__builtins__": {}}, namespace))
        if not math.isfinite(value):
            raise ValueError(f"expression {self.source!r} produced a non-finite value")
        return value
