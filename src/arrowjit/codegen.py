"""Generate an LLVM kernel that runs a lowered function over whole columns.

The kernel walks the rows in blocks of 64, so a block's error flags fit in one
i64 and line up with one word of the Arrow validity bitmap. It returns the
index of the first valid row whose error flag is set, or -1 if there is none.
"""

import struct

from arrowjit.ir import Constant, Function, Instr, Op
from arrowjit.scalar import ScalarType

INT64, FLOAT64, BOOL = ScalarType.INT64, ScalarType.FLOAT64, ScalarType.BOOL
INT64_MIN = -(2**63)

LLVM_TYPES = {INT64: "i64", FLOAT64: "double", BOOL: "i1"}

# Without nsw or fast-math flags, LLVM's integer arithmetic wraps and its float
# arithmetic is exact IEEE 754, which is what Python's semantics need.
ARITHMETIC = {
    (Op.ADD, INT64): "add",
    (Op.SUB, INT64): "sub",
    (Op.MUL, INT64): "mul",
    (Op.ADD, FLOAT64): "fadd",
    (Op.SUB, FLOAT64): "fsub",
    (Op.MUL, FLOAT64): "fmul",
    (Op.DIV, FLOAT64): "fdiv",
    (Op.AND, BOOL): "and",
    (Op.OR, BOOL): "or",
}

INT_COMPARISONS = {Op.EQ: "eq", Op.NE: "ne", Op.LT: "slt", Op.LE: "sle", Op.GT: "sgt", Op.GE: "sge"}
# Ordered comparisons are false when either side is NaN, as in Python. != is the
# exception, because Python makes nan != nan true.
FLOAT_COMPARISONS = {
    Op.EQ: "oeq",
    Op.NE: "une",
    Op.LT: "olt",
    Op.LE: "ole",
    Op.GT: "ogt",
    Op.GE: "oge",
}

KERNEL = """\
define i64 @kernel({params}) {{
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %block

block:
  %start = phi i64 [ 0, %entry ], [ %next.start, %next ]
  %limit = add nuw i64 %start, 64
  %short = icmp ult i64 %n, %limit
  %end = select i1 %short, i64 %n, i64 %limit
  br label %row

row:
  %i = phi i64 [ %start, %block ], [ %i.next, %row ]
  %errors = phi i64 [ 0, %block ], [ %errors.next, %row ]
{body}
  %lane = sub nuw i64 %i, %start
  %flag = zext i1 {error} to i64
  %bit = shl i64 %flag, %lane
  %errors.next = or i64 %errors, %bit
  %i.next = add nuw i64 %i, 1
  %row.done = icmp eq i64 %i.next, %end
  br i1 %row.done, label %check, label %row

check:
  %has.valid = icmp ne ptr %valid, null
  br i1 %has.valid, label %masked, label %test

masked:
  %word = lshr i64 %start, 6
  %valid.ptr = getelementptr inbounds i64, ptr %valid, i64 %word
  %valid.bits = load i64, ptr %valid.ptr, align 8
  %errors.valid = and i64 %errors.next, %valid.bits
  br label %test

test:
  %failed = phi i64 [ %errors.next, %check ], [ %errors.valid, %masked ]
  %any = icmp ne i64 %failed, 0
  br i1 %any, label %fail, label %next

fail:
  %offset = call i64 @llvm.cttz.i64(i64 %failed, i1 true)
  %first = add nuw i64 %start, %offset
  ret i64 %first

next:
  %next.start = add nuw i64 %start, 64
  %more = icmp ult i64 %next.start, %n
  br i1 %more, label %block, label %done

done:
  ret i64 -1
}}

declare i64 @llvm.abs.i64(i64, i1 immarg)
declare double @llvm.fabs.f64(double)
declare double @llvm.sqrt.f64(double)
declare double @llvm.exp.f64(double)
declare double @llvm.log.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.copysign.f64(double, double)
declare i64 @llvm.cttz.i64(i64, i1 immarg)
"""


def generate(function: Function) -> str:
    """Emit LLVM IR for a kernel named kernel that runs `function` over whole columns."""
    return _Generator(function).module()


def _literal(value: Constant, type: ScalarType) -> str:
    if type is BOOL:
        return "true" if value else "false"
    if type is FLOAT64:
        # LLVM rejects decimal floats that aren't exact in binary, like 0.1, so write the bits.
        (bits,) = struct.unpack("<Q", struct.pack("<d", value))
        return f"0x{bits:016X}"
    return str(value)


class _Generator:
    def __init__(self, function: Function) -> None:
        self.function = function

    def module(self) -> str:
        columns = (f"ptr noalias %in{position}" for position in range(len(self.function.params)))
        params = ", ".join(["ptr noalias %out", "ptr noalias %valid", "i64 %n", *columns])
        result = LLVM_TYPES[self.function.return_type]
        body = [
            line
            for index, instr in enumerate(self.function.instrs)
            for line in self.instruction(index, instr)
        ]
        body += [
            f"%out.ptr = getelementptr inbounds {result}, ptr %out, i64 %i",
            f"store {result} {self.value(self.function.result)}, ptr %out.ptr, align 8",
        ]
        return KERNEL.format(
            params=params,
            body="\n".join(f"  {line}" for line in body),
            error=self.value(self.function.error),
        )

    def value(self, index: int) -> str:
        """How an instruction's value is written as an operand."""
        instr = self.function.instrs[index]
        if instr.op is not Op.CONST:
            return f"%v{index}"
        assert instr.attr is not None
        return _literal(instr.attr, instr.type)

    def instruction(self, index: int, instr: Instr) -> list[str]:
        """The LLVM lines that compute one IR instruction into %v<index>."""
        v, t = f"%v{index}", LLVM_TYPES[instr.type]
        args = [self.value(arg) for arg in instr.args]
        match instr.op, instr.type:
            case Op.CONST, _:
                # Constants are written inline wherever they're used.
                return []
            case Op.PARAM, _:
                return [
                    f"{v}.ptr = getelementptr inbounds {t}, ptr %in{instr.attr}, i64 %i",
                    f"{v} = load {t}, ptr {v}.ptr, align 8",
                ]
            case Op.NEG, ScalarType.INT64:
                return [f"{v} = sub i64 0, {args[0]}"]
            case Op.NEG, ScalarType.FLOAT64:
                return [f"{v} = fneg double {args[0]}"]
            case Op.NOT, _:
                return [f"{v} = xor i1 {args[0]}, true"]
            case Op.ABS, ScalarType.INT64:
                return [f"{v} = call i64 @llvm.abs.i64(i64 {args[0]}, i1 false)"]
            case Op.ABS, ScalarType.FLOAT64:
                return [f"{v} = call double @llvm.fabs.f64(double {args[0]})"]
            case (Op.SQRT | Op.EXP | Op.LOG) as op, _:
                return [f"{v} = call double @llvm.{op.value}.f64(double {args[0]})"]
            case Op.TO_FLOAT, _:
                return [f"{v} = sitofp i64 {args[0]} to double"]
            case (Op.FLOORDIV | Op.MOD) as op, ScalarType.INT64:
                return self.int_division(v, op, args[0], args[1])
            case (Op.FLOORDIV | Op.MOD) as op, ScalarType.FLOAT64:
                return self.float_division(v, op, args[0], args[1])
            case Op.SELECT, _:
                condition, if_true, if_false = args
                return [f"{v} = select i1 {condition}, {t} {if_true}, {t} {if_false}"]
            case (Op.EQ | Op.NE | Op.LT | Op.LE | Op.GT | Op.GE) as op, _:
                operand = self.function.instrs[instr.args[0]].type
                if operand is FLOAT64:
                    predicate = f"fcmp {FLOAT_COMPARISONS[op]}"
                else:
                    predicate = f"icmp {INT_COMPARISONS[op]}"
                return [f"{v} = {predicate} {LLVM_TYPES[operand]} {args[0]}, {args[1]}"]
            # Everything left is a single arithmetic or logic instruction.
            case key if key in ARITHMETIC:
                return [f"{v} = {ARITHMETIC[key]} {t} {args[0]}, {args[1]}"]
        raise AssertionError(f"no code for {instr.op.value} on {instr.type.value}")

    def int_division(self, v: str, op: Op, a: str, b: str) -> list[str]:
        # sdiv and srem are undefined for a zero divisor and for INT64_MIN / -1, and
        # x86's idiv traps on both, so those rows divide by 1 instead. Lowering flags
        # division by zero, and INT64_MIN // 1 is the wrapped INT64_MIN // -1.
        lines = [
            f"{v}.zero = icmp eq i64 {b}, 0",
            f"{v}.min = icmp eq i64 {a}, {INT64_MIN}",
            f"{v}.minus_one = icmp eq i64 {b}, -1",
            f"{v}.overflow = and i1 {v}.min, {v}.minus_one",
            f"{v}.unsafe = or i1 {v}.zero, {v}.overflow",
            f"{v}.d = select i1 {v}.unsafe, i64 1, i64 {b}",
            f"{v}.q = sdiv i64 {a}, {v}.d",
            f"{v}.r = srem i64 {a}, {v}.d",
            # sdiv rounds toward zero and Python rounds down. They differ exactly when
            # the remainder is nonzero and has the opposite sign to the divisor.
            f"{v}.inexact = icmp ne i64 {v}.r, 0",
            f"{v}.signs = xor i64 {v}.r, {v}.d",
            f"{v}.opposite = icmp slt i64 {v}.signs, 0",
            f"{v}.adjust = and i1 {v}.inexact, {v}.opposite",
        ]
        if op is Op.FLOORDIV:
            return [
                *lines,
                f"{v}.step = zext i1 {v}.adjust to i64",
                f"{v} = sub i64 {v}.q, {v}.step",
            ]
        return [
            *lines,
            f"{v}.shift = select i1 {v}.adjust, i64 {v}.d, i64 0",
            f"{v} = add i64 {v}.r, {v}.shift",
        ]

    def float_division(self, v: str, op: Op, x: str, y: str) -> list[str]:
        # These are the steps of CPython's float_rem and _float_div_mod, so the results
        # match Python's exactly, signed zeros included.
        lines = [
            f"{v}.fmod = frem double {x}, {y}",
            f"{v}.nonzero = fcmp une double {v}.fmod, 0.0",
            f"{v}.y.neg = fcmp olt double {y}, 0.0",
            f"{v}.fmod.neg = fcmp olt double {v}.fmod, 0.0",
            f"{v}.differ = xor i1 {v}.y.neg, {v}.fmod.neg",
            f"{v}.fix = and i1 {v}.nonzero, {v}.differ",
        ]
        if op is Op.MOD:
            return [
                *lines,
                f"{v}.fixed = fadd double {v}.fmod, {y}",
                f"{v}.kept = select i1 {v}.fix, double {v}.fixed, double {v}.fmod",
                f"{v}.zero = call double @llvm.copysign.f64(double 0.0, double {y})",
                f"{v} = select i1 {v}.nonzero, double {v}.kept, double {v}.zero",
            ]
        return [
            *lines,
            f"{v}.exact = fsub double {x}, {v}.fmod",
            f"{v}.div = fdiv double {v}.exact, {y}",
            f"{v}.lower = fsub double {v}.div, 1.0",
            f"{v}.q = select i1 {v}.fix, double {v}.lower, double {v}.div",
            f"{v}.floor = call double @llvm.floor.f64(double {v}.q)",
            f"{v}.frac = fsub double {v}.q, {v}.floor",
            f"{v}.round.up = fcmp ogt double {v}.frac, 0.5",
            f"{v}.up = fadd double {v}.floor, 1.0",
            f"{v}.snapped = select i1 {v}.round.up, double {v}.up, double {v}.floor",
            f"{v}.ratio = fdiv double {x}, {y}",
            f"{v}.zero = call double @llvm.copysign.f64(double 0.0, double {v}.ratio)",
            f"{v}.q.nonzero = fcmp une double {v}.q, 0.0",
            f"{v} = select i1 {v}.q.nonzero, double {v}.snapped, double {v}.zero",
        ]
