"""Type-check a parsed function and translate it into the typed tree."""

import ast
from collections.abc import Sequence

from arrowjit.diagnostics import CompileError, Source, Span, span_of
from arrowjit.frontend import ParsedFunction
from arrowjit.scalar import NUMERIC, ScalarType
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
    Function,
    If,
    Logical,
    LogicalOp,
    Param,
    Return,
    Stmt,
    Unary,
    UnaryOp,
    Var,
)

INT64, FLOAT64, BOOL = ScalarType.INT64, ScalarType.FLOAT64, ScalarType.BOOL
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1

COLUMN_TYPES = {"int": INT64, "float": FLOAT64}
BUILTINS = {"abs": Builtin.ABS, "min": Builtin.MIN, "max": Builtin.MAX}
MATH_FUNCTIONS = {"sqrt": Builtin.SQRT, "exp": Builtin.EXP, "log": Builtin.LOG}

BINARY_OPS: dict[type[ast.operator], BinaryOp] = {
    ast.Add: BinaryOp.ADD,
    ast.Sub: BinaryOp.SUB,
    ast.Mult: BinaryOp.MUL,
    ast.Div: BinaryOp.TRUEDIV,
    ast.FloorDiv: BinaryOp.FLOORDIV,
    ast.Mod: BinaryOp.MOD,
}

COMPARE_OPS: dict[type[ast.cmpop], CompareOp] = {
    ast.Eq: CompareOp.EQ,
    ast.NotEq: CompareOp.NE,
    ast.Lt: CompareOp.LT,
    ast.LtE: CompareOp.LE,
    ast.Gt: CompareOp.GT,
    ast.GtE: CompareOp.GE,
}

type Node = ast.expr | ast.stmt | ast.arg


def check(parsed: ParsedFunction) -> Function:
    """Type-check a parsed function, or raise CompileError pointing at the problem."""
    return _Checker(parsed.source).function(parsed.tree)


def widen(value: Expr, target: ScalarType) -> Expr:
    """Make Python's silent int-to-float conversion an explicit Cast."""
    return value if value.type is target else Cast(value, target, value.span)


def fits(source: ScalarType, target: ScalarType) -> bool:
    """Whether a value of type `source` can go where `target` is expected."""
    return source is target or (source is INT64 and target is FLOAT64)


def without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    match body:
        case [ast.Expr(value=ast.Constant(value=str())), *rest]:
            return rest
    return body


class _Checker:
    def __init__(self, source: Source) -> None:
        self.source = source
        self.types: dict[str, ScalarType] = {}
        self.known: frozenset[str] = frozenset()
        self.return_type = FLOAT64

    def error(self, message: str, at: Node | Span) -> CompileError:
        span = at if isinstance(at, Span) else span_of(at)
        return CompileError(message, span, self.source)

    def function(self, node: ast.FunctionDef) -> Function:
        args = node.args
        if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults:
            raise self.error("only plain positional parameters are supported", node)

        params: list[Param] = []
        for arg in args.args:
            param_type = self.column_type(arg.annotation, arg, f"parameter {arg.arg!r}")
            params.append(Param(arg.arg, param_type))
            self.types[arg.arg] = param_type
        self.return_type = self.column_type(node.returns, node, "the return value")

        # Every name the body assigns somewhere, so "used before it's assigned"
        # can be told apart from a name that doesn't exist at all.
        stored = {
            name.id
            for name in ast.walk(node)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
        }
        self.known = frozenset(self.types) | stored

        body, _, returns = self.block(without_docstring(node.body), frozenset(self.types))
        if not returns:
            raise self.error("some paths through the function don't return a value", node)
        return Function(node.name, tuple(params), self.return_type, body)

    def column_type(self, annotation: ast.expr | None, owner: Node, what: str) -> ScalarType:
        if isinstance(annotation, ast.Name) and annotation.id in COLUMN_TYPES:
            return COLUMN_TYPES[annotation.id]
        raise self.error(f"{what} must be annotated int or float", annotation or owner)

    def block(
        self, statements: Sequence[ast.stmt], assigned: frozenset[str]
    ) -> tuple[tuple[Stmt, ...], frozenset[str], bool]:
        """Check statements in order, tracking assigned names and whether they all return."""
        checked: list[Stmt] = []
        returns = False
        for node in statements:
            if returns:
                raise self.error("this can never run because of the return above it", node)
            statement, assigned, returns = self.statement(node, assigned)
            checked.append(statement)
        return tuple(checked), assigned, returns

    def statement(
        self, node: ast.stmt, assigned: frozenset[str]
    ) -> tuple[Stmt, frozenset[str], bool]:
        span = span_of(node)
        match node:
            case ast.Return(value=ast.expr() as value):
                result = self.return_value(self.expr(value, assigned), value)
                return Return(result, span), assigned, True
            case ast.Assign(targets=[ast.Name(id=name)], value=value):
                result = self.assign(name, self.expr(value, assigned), node)
                return result, assigned | {name}, False
            case ast.AugAssign(target=ast.Name(id=name) as target, op=op, value=value):
                current = self.variable(name, target, assigned)
                updated = self.binary(op, current, self.expr(value, assigned), node)
                return self.assign(name, updated, node), assigned, False
            case ast.If(test=test, body=body, orelse=orelse):
                condition = self.condition(test, assigned)
                then_body, then_assigned, then_returns = self.block(body, assigned)
                else_body, else_assigned, else_returns = self.block(orelse, assigned)
                # A branch that returns never reaches the code after the if,
                # so only the other branch's assignments count there.
                if then_returns:
                    after = else_assigned
                elif else_returns:
                    after = then_assigned
                else:
                    after = then_assigned & else_assigned
                returns = then_returns and else_returns
                return If(condition, then_body, else_body, span), after, returns
            case ast.Return():
                raise self.error("return needs a value", node)
            case ast.Assign() | ast.AugAssign():
                raise self.error("only assignments to a single plain name are supported", node)
            case ast.Expr():
                raise self.error("a bare expression has no effect here", node)
            case _:
                raise self.error(f"{type(node).__name__} statements aren't supported", node)

    def return_value(self, value: Expr, node: ast.expr) -> Expr:
        if not fits(value.type, self.return_type):
            raise self.error(
                f"returns {value.type.python_name}, "
                f"but the function is annotated -> {self.return_type.python_name}",
                node,
            )
        return widen(value, self.return_type)

    def assign(self, name: str, value: Expr, node: ast.stmt) -> Assign:
        # A variable's type is fixed by its first assignment in the source.
        declared = self.types.setdefault(name, value.type)
        if not fits(value.type, declared):
            raise self.error(
                f"{name!r} holds {declared.python_name} values, "
                f"so it can't be assigned a {value.type.python_name}",
                node,
            )
        return Assign(name, widen(value, declared), span_of(node))

    def expr(self, node: ast.expr, assigned: frozenset[str]) -> Expr:
        span = span_of(node)
        match node:
            case ast.Constant(value=bool() as value):
                return Const(value, BOOL, span)
            case ast.Constant(value=int() as value):
                if not INT64_MIN <= value <= INT64_MAX:
                    raise self.error("this integer doesn't fit in int64", node)
                return Const(value, INT64, span)
            case ast.Constant(value=float() as value):
                return Const(value, FLOAT64, span)
            case ast.Constant(value=value):
                raise self.error(f"{type(value).__name__} constants aren't supported", node)
            case ast.Name(id=name):
                return self.variable(name, node, assigned)
            case ast.UnaryOp(op=op, operand=operand):
                return self.unary(op, self.expr(operand, assigned), node)
            case ast.BinOp(left=left, op=op, right=right):
                lhs, rhs = self.expr(left, assigned), self.expr(right, assigned)
                return self.binary(op, lhs, rhs, node)
            case ast.Compare(left=left, ops=[op], comparators=[right]):
                lhs, rhs = self.expr(left, assigned), self.expr(right, assigned)
                return self.compare(op, lhs, rhs, node)
            case ast.BoolOp(op=op, values=values):
                operands = [self.condition(value, assigned) for value in values]
                return self.logical(op, operands, node)
            case ast.IfExp(test=test, body=body, orelse=orelse):
                condition = self.condition(test, assigned)
                return self.conditional(
                    condition, self.expr(body, assigned), self.expr(orelse, assigned), node
                )
            case ast.Call(func=func, args=args, keywords=[]):
                return self.call(func, [self.expr(arg, assigned) for arg in args], node)
            case ast.Compare():
                raise self.error(
                    "chained comparisons aren't supported, write a < b and b < c", node
                )
            case ast.Call():
                raise self.error("keyword arguments aren't supported", node)
            case _:
                raise self.error(f"{type(node).__name__} expressions aren't supported", node)

    def variable(self, name: str, node: ast.expr, assigned: frozenset[str]) -> Var:
        if name in assigned:
            return Var(name, self.types[name], span_of(node))
        if name in self.known:
            raise self.error(f"{name!r} might be used before it's assigned", node)
        raise self.error(
            f"unknown name {name!r}, only parameters and local variables can be used", node
        )

    def unary(self, op: ast.unaryop, operand: Expr, node: ast.expr) -> Expr:
        span = span_of(node)
        if isinstance(op, ast.USub) and operand.type in NUMERIC:
            return Unary(UnaryOp.NEG, operand, operand.type, span)
        if isinstance(op, ast.Not) and operand.type is BOOL:
            return Unary(UnaryOp.NOT, operand, BOOL, span)
        if isinstance(op, ast.Not):
            raise self.error("not needs a bool, for example not (x > 0)", node)
        raise self.error("only - on numbers and not on bools are supported", node)

    def binary(self, op: ast.operator, left: Expr, right: Expr, node: Node) -> Expr:
        binary_op = BINARY_OPS.get(type(op))
        if binary_op is None:
            raise self.error("this operator isn't supported, use + - * / // or %", node)
        if left.type not in NUMERIC or right.type not in NUMERIC:
            raise self.error(f"{binary_op.value} needs int or float operands", node)
        # As in Python, / always produces a float, and so does anything with a float operand.
        floating = binary_op is BinaryOp.TRUEDIV or FLOAT64 in (left.type, right.type)
        result = FLOAT64 if floating else INT64
        return Binary(binary_op, widen(left, result), widen(right, result), result, span_of(node))

    def compare(self, op: ast.cmpop, left: Expr, right: Expr, node: ast.expr) -> Expr:
        compare_op = COMPARE_OPS.get(type(op))
        if compare_op is None:
            raise self.error("this comparison isn't supported, use == != < <= > or >=", node)
        span = span_of(node)
        both_bool = left.type is BOOL and right.type is BOOL
        if both_bool and compare_op in (CompareOp.EQ, CompareOp.NE):
            return Compare(compare_op, left, right, BOOL, span)
        if left.type not in NUMERIC or right.type not in NUMERIC:
            raise self.error("bools can only be compared with each other, using == or !=", node)
        operand = FLOAT64 if FLOAT64 in (left.type, right.type) else INT64
        return Compare(compare_op, widen(left, operand), widen(right, operand), BOOL, span)

    def logical(self, op: ast.boolop, operands: list[Expr], node: ast.expr) -> Expr:
        logical_op = LogicalOp.AND if isinstance(op, ast.And) else LogicalOp.OR
        # a and b and c arrives as one node with three values, so fold it left to right.
        result = operands[0]
        for operand in operands[1:]:
            result = Logical(logical_op, result, operand, BOOL, span_of(node))
        return result

    def conditional(self, test: Expr, body: Expr, orelse: Expr, node: ast.expr) -> Expr:
        if body.type is orelse.type:
            result = body.type
        elif body.type in NUMERIC and orelse.type in NUMERIC:
            result = FLOAT64
        else:
            raise self.error("both branches of the conditional must have the same type", node)
        return Conditional(test, widen(body, result), widen(orelse, result), result, span_of(node))

    def call(self, func: ast.expr, args: list[Expr], node: ast.expr) -> Expr:
        match func:
            case ast.Name(id="float"):
                (value,) = self.numeric_arguments(args, 1, node)
                return widen(value, FLOAT64)
            case ast.Name(id=name) if name in BUILTINS:
                builtin = BUILTINS[name]
            case ast.Attribute(value=ast.Name(id="math"), attr=attr) if attr in MATH_FUNCTIONS:
                builtin = MATH_FUNCTIONS[attr]
            case _:
                raise self.error(
                    "only abs, min, max, float, math.sqrt, math.exp and math.log can be called",
                    node,
                )

        span = span_of(node)
        if builtin is Builtin.ABS:
            (value,) = self.numeric_arguments(args, 1, node)
            return Call(builtin, (value,), value.type, span)
        if builtin in (Builtin.MIN, Builtin.MAX):
            left, right = self.numeric_arguments(args, 2, node)
            # Python returns the winning argument unchanged, so with mixed types
            # the result's type would depend on the data.
            if left.type is not right.type:
                raise self.error(f"{builtin.value}() needs two arguments of the same type", node)
            return Call(builtin, (left, right), left.type, span)
        # math.sqrt, math.exp and math.log take and return floats.
        (value,) = self.numeric_arguments(args, 1, node)
        return Call(builtin, (widen(value, FLOAT64),), FLOAT64, span)

    def numeric_arguments(self, args: list[Expr], count: int, node: ast.expr) -> list[Expr]:
        if len(args) != count:
            raise self.error(f"expected {count} argument(s), got {len(args)}", node)
        for arg in args:
            if arg.type not in NUMERIC:
                raise self.error(f"expected int or float, got {arg.type.python_name}", arg.span)
        return args

    def condition(self, node: ast.expr, assigned: frozenset[str]) -> Expr:
        result = self.expr(node, assigned)
        if result.type is not BOOL:
            raise self.error(
                f"a condition must be a bool, not {result.type.python_name}. "
                "Compare explicitly, for example x != 0",
                node,
            )
        return result
