"""The typed tree the checker produces and lowering consumes.

Every expression knows its type, and Python's silent int-to-float conversions
show up as explicit Cast nodes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import assert_never

from arrowjit.diagnostics import Span
from arrowjit.scalar import ScalarType

type Expr = Const | Var | Unary | Binary | Compare | Logical | Conditional | Call | Cast
type Stmt = Assign | If | Return


class UnaryOp(Enum):
    NEG = "-"
    NOT = "not"


class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    TRUEDIV = "/"
    FLOORDIV = "//"
    MOD = "%"


class CompareOp(Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class LogicalOp(Enum):
    AND = "and"
    OR = "or"


class Builtin(Enum):
    ABS = "abs"
    MIN = "min"
    MAX = "max"
    SQRT = "math.sqrt"
    EXP = "math.exp"
    LOG = "math.log"


@dataclass(frozen=True, slots=True)
class Const:
    value: int | float | bool
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Var:
    name: str
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Unary:
    op: UnaryOp
    operand: Expr
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Binary:
    op: BinaryOp
    left: Expr
    right: Expr
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Compare:
    op: CompareOp
    left: Expr
    right: Expr
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Logical:
    op: LogicalOp
    left: Expr
    right: Expr
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Conditional:
    test: Expr
    body: Expr
    orelse: Expr
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Call:
    func: Builtin
    args: tuple[Expr, ...]
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Cast:
    operand: Expr
    type: ScalarType
    span: Span


@dataclass(frozen=True, slots=True)
class Assign:
    name: str
    value: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class If:
    test: Expr
    body: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...]
    span: Span


@dataclass(frozen=True, slots=True)
class Return:
    value: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    type: ScalarType


@dataclass(frozen=True, slots=True)
class Function:
    name: str
    params: tuple[Param, ...]
    return_type: ScalarType
    body: tuple[Stmt, ...]


def render(expr: Expr) -> str:
    """Render an expression with its casts visible, for tests and debugging."""
    match expr:
        case Const(value=value):
            return repr(value)
        case Var(name=name):
            return name
        case Unary(operand=operand):
            prefix = "-" if expr.op is UnaryOp.NEG else "not "
            return f"({prefix}{render(operand)})"
        case (
            Binary(op=op, left=left, right=right)
            | Compare(op=op, left=left, right=right)
            | Logical(op=op, left=left, right=right)
        ):
            return f"({render(left)} {op.value} {render(right)})"
        case Conditional(test=test, body=body, orelse=orelse):
            return f"({render(body)} if {render(test)} else {render(orelse)})"
        case Call(func=func, args=args):
            return f"{func.value}({', '.join(render(arg) for arg in args)})"
        case Cast(operand=operand, type=target):
            return f"{target.python_name}({render(operand)})"
        case _:
            assert_never(expr)
