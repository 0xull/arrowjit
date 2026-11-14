"""Lower the typed tree into predicated IR, where branches become selects."""

import math
from typing import assert_never

from arrowjit import ir, tree
from arrowjit.ir import Builder, Op
from arrowjit.scalar import ScalarType
from arrowjit.tree import (
    Assign,
    Binary,
    BinaryOp,
    Builtin,
    Call,
    Cast,
    Compare,
    CompareOp,
    Conditional,
    Const,
    Expr,
    If,
    Logical,
    LogicalOp,
    Return,
    Stmt,
    Unary,
    UnaryOp,
    Var,
)

FLOAT64, BOOL = ScalarType.FLOAT64, ScalarType.BOOL

BINARY_OPS = {
    BinaryOp.ADD: Op.ADD,
    BinaryOp.SUB: Op.SUB,
    BinaryOp.MUL: Op.MUL,
    BinaryOp.TRUEDIV: Op.DIV,
    BinaryOp.FLOORDIV: Op.FLOORDIV,
    BinaryOp.MOD: Op.MOD,
}

COMPARE_OPS = {
    CompareOp.EQ: Op.EQ,
    CompareOp.NE: Op.NE,
    CompareOp.LT: Op.LT,
    CompareOp.LE: Op.LE,
    CompareOp.GT: Op.GT,
    CompareOp.GE: Op.GE,
}

# Python raises ZeroDivisionError for each of these when the divisor is zero.
DIVISIONS = frozenset({BinaryOp.TRUEDIV, BinaryOp.FLOORDIV, BinaryOp.MOD})


def lower(function: tree.Function) -> ir.Function:
    """Lower a type-checked function into predicated IR."""
    return _Lowering(function).run()


class _Lowering:
    def __init__(self, function: tree.Function) -> None:
        self.function = function
        self.b = Builder()
        self.env = {
            param.name: self.b.param(position, param.type)
            for position, param in enumerate(function.params)
        }
        # Masks over rows. reached is the rows whose control flow gets to the current
        # statement, and returned is the rows that already hit a return.
        self.reached = self.b.const(True, BOOL)
        self.returned = self.b.const(False, BOOL)
        self.error = self.b.const(False, BOOL)
        self.result: int | None = None

    def run(self) -> ir.Function:
        self.block(self.function.body)
        # The checker rejects functions with a path that doesn't return.
        assert self.result is not None
        params = tuple(param.type for param in self.function.params)
        return self.b.finish(self.function.name, params, self.result, self.error)

    def block(self, statements: tuple[Stmt, ...]) -> None:
        for statement in statements:
            self.statement(statement)

    def statement(self, statement: Stmt) -> None:
        match statement:
            case Assign(name=name, value=value):
                self.env[name] = self.expr(value, self.running())
            case Return(value=value):
                running = self.running()
                lowered = self.expr(value, running)
                # A row keeps the first return it reaches, because later returns only
                # overwrite rows that are still running.
                earlier = lowered if self.result is None else self.result
                self.result = self.select(running, lowered, earlier)
                self.returned = self.or_(self.returned, running)
            case If(test=test, body=body, orelse=orelse):
                condition = self.expr(test, self.running())
                reached, returned, env = self.reached, self.returned, self.env
                then_reached = self.and_(reached, condition)
                else_reached = self.and_(reached, self.not_(condition))
                then_returned, then_env = self.branch(body, then_reached, returned, env)
                else_returned, else_env = self.branch(orelse, else_reached, returned, env)
                self.reached = reached
                self.returned = self.or_(then_returned, else_returned)
                self.env = self.merge(condition, then_env, else_env)
            case _:
                assert_never(statement)

    def branch(
        self, body: tuple[Stmt, ...], reached: int, returned: int, env: dict[str, int]
    ) -> tuple[int, dict[str, int]]:
        """Lower one side of an if, starting from the state before the if."""
        self.reached, self.returned, self.env = reached, returned, dict(env)
        self.block(body)
        return self.returned, self.env

    def merge(
        self, condition: int, then_env: dict[str, int], else_env: dict[str, int]
    ) -> dict[str, int]:
        """Join the variables after an if, giving each row the value from its branch."""
        # The checker only lets code read a name one branch assigned if the other
        # branch returned, so that branch's value is right for every row that gets here.
        merged = then_env | else_env
        for name, value in then_env.items():
            if name in else_env:
                merged[name] = self.select(condition, value, else_env[name])
        return merged

    def expr(self, expr: Expr, mask: int) -> int:
        """Lower an expression whose checks apply to the rows in `mask`."""
        match expr:
            case Const(value=value):
                return self.b.const(value, expr.type)
            case Var(name=name):
                return self.env[name]
            case Unary(operand=operand):
                inner = self.expr(operand, mask)
                if expr.op is UnaryOp.NOT:
                    return self.not_(inner)
                return self.b.emit(Op.NEG, expr.type, inner)
            case Binary(left=left, right=right):
                lhs, rhs = self.expr(left, mask), self.expr(right, mask)
                if expr.op in DIVISIONS:
                    self.check(mask, self.b.emit(Op.EQ, BOOL, rhs, self.zero(right.type)))
                return self.b.emit(BINARY_OPS[expr.op], expr.type, lhs, rhs)
            case Compare(left=left, right=right):
                lhs, rhs = self.expr(left, mask), self.expr(right, mask)
                return self.b.emit(COMPARE_OPS[expr.op], BOOL, lhs, rhs)
            case Logical(left=left, right=right):
                lhs = self.expr(left, mask)
                # Python only evaluates the right side for rows the left side doesn't decide.
                if expr.op is LogicalOp.AND:
                    return self.and_(lhs, self.expr(right, self.and_(mask, lhs)))
                return self.or_(lhs, self.expr(right, self.and_(mask, self.not_(lhs))))
            case Conditional(test=test, body=body, orelse=orelse):
                condition = self.expr(test, mask)
                if_true = self.expr(body, self.and_(mask, condition))
                if_false = self.expr(orelse, self.and_(mask, self.not_(condition)))
                return self.select(condition, if_true, if_false)
            case Call(args=args):
                return self.call(expr.func, [self.expr(arg, mask) for arg in args], mask)
            case Cast(operand=operand):
                return self.b.emit(Op.TO_FLOAT, FLOAT64, self.expr(operand, mask))
            case _:
                assert_never(expr)

    def call(self, func: Builtin, args: list[int], mask: int) -> int:
        match func, args:
            case Builtin.ABS, [x]:
                return self.b.emit(Op.ABS, self.b.type_of(x), x)
            case Builtin.MIN, [x, y]:
                # Python's min keeps the first argument unless the second is smaller, and
                # that rule decides the result for NaN and for 0.0 against -0.0.
                return self.select(self.b.emit(Op.LT, BOOL, y, x), y, x)
            case Builtin.MAX, [x, y]:
                return self.select(self.b.emit(Op.GT, BOOL, y, x), y, x)
            case Builtin.SQRT, [x]:
                # ValueError for x < 0. -0.0 isn't less than 0, so sqrt(-0.0) is fine.
                self.check(mask, self.b.emit(Op.LT, BOOL, x, self.zero(FLOAT64)))
                return self.b.emit(Op.SQRT, FLOAT64, x)
            case Builtin.LOG, [x]:
                # ValueError for x <= 0, which includes -0.0.
                self.check(mask, self.b.emit(Op.LE, BOOL, x, self.zero(FLOAT64)))
                return self.b.emit(Op.LOG, FLOAT64, x)
            case Builtin.EXP, [x]:
                result = self.b.emit(Op.EXP, FLOAT64, x)
                # OverflowError when the result is inf but x isn't.
                inf = self.b.const(math.inf, FLOAT64)
                result_is_inf = self.b.emit(Op.EQ, BOOL, result, inf)
                x_is_not_inf = self.b.emit(Op.NE, BOOL, x, inf)
                self.check(mask, self.and_(result_is_inf, x_is_not_inf))
                return result
        raise AssertionError(f"the checker let {func.value} through with {len(args)} arguments")

    def check(self, mask: int, fails: int) -> None:
        """Flag the rows in `mask` where `fails` is true, because Python raises there."""
        self.error = self.or_(self.error, self.and_(mask, fails))

    def running(self) -> int:
        """The rows that reach the current statement and haven't returned yet."""
        return self.and_(self.reached, self.not_(self.returned))

    def zero(self, type: ScalarType) -> int:
        return self.b.const(0.0 if type is FLOAT64 else 0, type)

    def select(self, condition: int, if_true: int, if_false: int) -> int:
        return self.b.emit(Op.SELECT, self.b.type_of(if_true), condition, if_true, if_false)

    def and_(self, left: int, right: int) -> int:
        return self.b.emit(Op.AND, BOOL, left, right)

    def or_(self, left: int, right: int) -> int:
        return self.b.emit(Op.OR, BOOL, left, right)

    def not_(self, value: int) -> int:
        return self.b.emit(Op.NOT, BOOL, value)
