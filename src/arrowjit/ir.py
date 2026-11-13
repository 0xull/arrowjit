"""The predicated IR that lowering produces and code generation consumes.

A function is one block of SSA instructions that runs once per row. Every
instruction is pure and total, so it can safely run on rows where Python would
never have reached it. Operations follow Python's semantics, except that int64
arithmetic wraps and an operation that would raise produces an unspecified
value instead. Lowering reports those cases through the row's error flag.
"""

from dataclasses import dataclass, replace
from enum import Enum

from arrowjit.scalar import ScalarType

BOOL = ScalarType.BOOL

type Constant = int | float | bool


class Op(Enum):
    PARAM = "param"
    CONST = "const"
    NEG = "neg"
    NOT = "not"
    ABS = "abs"
    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"
    TO_FLOAT = "to_float"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    FLOORDIV = "floordiv"
    MOD = "mod"
    AND = "and"
    OR = "or"
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    SELECT = "select"


@dataclass(frozen=True, slots=True)
class Instr:
    """An instruction, referred to by its index in the function's instruction list."""

    op: Op
    type: ScalarType
    args: tuple[int, ...] = ()
    # The parameter's position for PARAM, the value for CONST, and None otherwise.
    attr: Constant | None = None


@dataclass(frozen=True, slots=True)
class Function:
    """A lowered function. Each row produces `result`, or raises if `error` is true."""

    name: str
    params: tuple[ScalarType, ...]
    instrs: tuple[Instr, ...]
    result: int
    error: int

    @property
    def return_type(self) -> ScalarType:
        """The result's type, which the checker made equal to the declared return type."""
        return self.instrs[self.result].type

    def __str__(self) -> str:
        params = ", ".join(param.value for param in self.params)
        lines = [f"{self.name}({params}) -> {self.return_type.value}"]
        for index, instr in enumerate(self.instrs):
            lines.append(f"  %{index}: {instr.type.value} = {_format(instr)}")
        lines += [f"  return %{self.result}", f"  raise if %{self.error}"]
        return "\n".join(lines)


def _format(instr: Instr) -> str:
    if instr.op in (Op.PARAM, Op.CONST):
        return f"{instr.op.value} {instr.attr!r}"
    operands = ", ".join(f"%{arg}" for arg in instr.args)
    return f"{instr.op.value} {operands}"


class Builder:
    """Appends instructions in order, so every operand comes before its users."""

    def __init__(self) -> None:
        self.instrs: list[Instr] = []
        # No instruction has side effects, so an identical one can always be reused.
        # The key includes repr because 0.0 == -0.0 in Python.
        self._existing: dict[tuple[Instr, str], int] = {}

    def type_of(self, value: int) -> ScalarType:
        return self.instrs[value].type

    def param(self, position: int, type: ScalarType) -> int:
        return self._add(Instr(Op.PARAM, type, attr=position))

    def const(self, value: Constant, type: ScalarType) -> int:
        return self._add(Instr(Op.CONST, type, attr=value))

    def emit(self, op: Op, type: ScalarType, *args: int) -> int:
        folded = self._fold(op, args)
        return folded if folded is not None else self._add(Instr(op, type, args))

    def finish(
        self, name: str, params: tuple[ScalarType, ...], result: int, error: int
    ) -> Function:
        """Build the function from the instructions its result and error depend on."""
        live = [False] * len(self.instrs)
        live[result] = live[error] = True
        # Operands come before their users, so one backward pass finds everything live.
        for index in reversed(range(len(self.instrs))):
            if live[index]:
                for arg in self.instrs[index].args:
                    live[arg] = True

        renumbered: dict[int, int] = {}
        instrs: list[Instr] = []
        for index, instr in enumerate(self.instrs):
            if live[index]:
                renumbered[index] = len(instrs)
                instrs.append(replace(instr, args=tuple(renumbered[arg] for arg in instr.args)))
        return Function(name, params, tuple(instrs), renumbered[result], renumbered[error])

    def _add(self, instr: Instr) -> int:
        key = (instr, repr(instr.attr))
        if key not in self._existing:
            self._existing[key] = len(self.instrs)
            self.instrs.append(instr)
        return self._existing[key]

    def _known(self, value: int) -> bool | None:
        """The value of a bool constant, or None if this isn't one."""
        attr = self.instrs[value].attr
        return attr if isinstance(attr, bool) else None

    def _fold(self, op: Op, args: tuple[int, ...]) -> int | None:
        """Fold logic on constant bools, since predicates start as True and errors as False."""
        known = [self._known(arg) for arg in args]
        if op is Op.NOT and known[0] is not None:
            return self.const(not known[0], BOOL)
        if op in (Op.AND, Op.OR):
            # True is the identity of and, and absorbs or. False is the other way around.
            identity = op is Op.AND
            left, right = args
            if known[0] is not None:
                return right if known[0] is identity else left
            if known[1] is not None:
                return left if known[1] is identity else right
        if op is Op.SELECT:
            _, if_true, if_false = args
            if known[0] is not None:
                return if_true if known[0] else if_false
            if if_true == if_false:
                return if_true
        return None
