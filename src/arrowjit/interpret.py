"""Run lowered IR one row at a time, as the reference for what each instruction means.

Tests check lowering against Python through it, and later the compiled kernel
against it, so it favors being obviously right over being fast.
"""

import math
from collections.abc import Sequence

from arrowjit.ir import Constant, Function, Op


def wrap(value: int) -> int:
    """Reduce an exact integer to int64, the way two's complement arithmetic does."""
    return (value + 2**63) % 2**64 - 2**63


def interpret(function: Function, row: Sequence[Constant]) -> Constant | None:
    """Run a lowered function on one row. None means Python raises for that row."""
    values: list[Constant] = []
    for instr in function.instrs:
        if instr.op is Op.PARAM:
            assert isinstance(instr.attr, int)
            values.append(row[instr.attr])
        elif instr.op is Op.CONST:
            assert instr.attr is not None
            values.append(instr.attr)
        else:
            values.append(_apply(instr.op, [values[arg] for arg in instr.args]))
    return None if values[function.error] else values[function.result]


def _apply(op: Op, operands: list[Constant]) -> Constant:
    match op, operands:
        case Op.NOT, [bool() as a]:
            return not a
        case Op.AND, [bool() as a, bool() as b]:
            return a and b
        case Op.OR, [bool() as a, bool() as b]:
            return a or b
        case Op.SELECT, [bool() as condition, a, b]:
            return a if condition else b
        case Op.EQ, [a, b]:
            return a == b
        case Op.NE, [a, b]:
            return a != b
        case Op.LT, [a, b]:
            return a < b
        case Op.LE, [a, b]:
            return a <= b
        case Op.GT, [a, b]:
            return a > b
        case Op.GE, [a, b]:
            return a >= b
        case Op.TO_FLOAT, [int() as a]:
            return float(a)
        case Op.NEG, [int() as a]:
            return wrap(-a)
        case Op.ABS, [int() as a]:
            return wrap(abs(a))
        case Op.ADD, [int() as a, int() as b]:
            return wrap(a + b)
        case Op.SUB, [int() as a, int() as b]:
            return wrap(a - b)
        case Op.MUL, [int() as a, int() as b]:
            return wrap(a * b)
        # Where Python would raise, lowering already flags the row, so any value will do.
        case Op.FLOORDIV, [int() as a, int() as b]:
            return wrap(a // b) if b else 0
        case Op.MOD, [int() as a, int() as b]:
            return wrap(a % b) if b else 0
        case Op.NEG, [float() as x]:
            return -x
        case Op.ABS, [float() as x]:
            return abs(x)
        case Op.ADD, [float() as x, float() as y]:
            return x + y
        case Op.SUB, [float() as x, float() as y]:
            return x - y
        case Op.MUL, [float() as x, float() as y]:
            return x * y
        case Op.DIV, [float() as x, float() as y]:
            return x / y if y else math.nan
        case Op.FLOORDIV, [float() as x, float() as y]:
            return x // y if y else math.nan
        case Op.MOD, [float() as x, float() as y]:
            return x % y if y else math.nan
        case Op.SQRT, [float() as x]:
            return math.nan if x < 0.0 else math.sqrt(x)
        case Op.LOG, [float() as x]:
            return math.nan if x <= 0.0 else math.log(x)
        case Op.EXP, [float() as x]:
            try:
                return math.exp(x)
            except OverflowError:
                # Lowering detects overflow by finding inf here, as it will in machine code.
                return math.inf
    raise AssertionError(f"no {op.value} instruction takes {operands!r}")
