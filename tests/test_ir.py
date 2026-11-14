"""The IR builder's reuse, folding and pruning, and how a function prints."""

from arrowjit.ir import Builder, Op
from arrowjit.scalar import ScalarType

INT64, FLOAT64, BOOL = ScalarType.INT64, ScalarType.FLOAT64, ScalarType.BOOL


def test_identical_instructions_are_reused() -> None:
    b = Builder()
    x = b.param(0, INT64)

    first = b.emit(Op.ADD, INT64, x, b.const(1, INT64))
    second = b.emit(Op.ADD, INT64, x, b.const(1, INT64))

    assert first == second
    assert len(b.instrs) == 3


def test_positive_and_negative_zero_are_different_constants() -> None:
    b = Builder()

    assert b.const(0.0, FLOAT64) != b.const(-0.0, FLOAT64)


def test_logic_on_constant_bools_folds_away() -> None:
    b = Builder()
    negative = b.emit(Op.LT, BOOL, b.param(0, FLOAT64), b.const(0.0, FLOAT64))
    true, false = b.const(True, BOOL), b.const(False, BOOL)

    assert b.emit(Op.AND, BOOL, true, negative) == negative
    assert b.emit(Op.AND, BOOL, negative, false) == false
    assert b.emit(Op.OR, BOOL, false, negative) == negative
    assert b.emit(Op.OR, BOOL, negative, true) == true
    assert b.emit(Op.NOT, BOOL, true) == false


def test_select_folds_when_the_choice_is_known() -> None:
    b = Builder()
    x, y = b.param(0, FLOAT64), b.param(1, FLOAT64)
    condition = b.emit(Op.LT, BOOL, x, y)

    assert b.emit(Op.SELECT, FLOAT64, b.const(True, BOOL), x, y) == x
    assert b.emit(Op.SELECT, FLOAT64, b.const(False, BOOL), x, y) == y
    assert b.emit(Op.SELECT, FLOAT64, condition, x, x) == x


def test_finish_keeps_only_what_the_result_and_error_use() -> None:
    b = Builder()
    x = b.param(0, FLOAT64)
    b.param(1, FLOAT64)  # never used
    doubled = b.emit(Op.MUL, FLOAT64, x, b.const(2.0, FLOAT64))

    function = b.finish("double", (FLOAT64, FLOAT64), doubled, b.const(False, BOOL))

    assert str(function).splitlines() == [
        "double(float64, float64) -> float64",
        "  %0: float64 = param 0",
        "  %1: float64 = const 2.0",
        "  %2: float64 = mul %0, %1",
        "  %3: bool = const False",
        "  return %2",
        "  raise if %3",
    ]


def test_double_negation_and_repeated_operands_fold_away() -> None:
    b = Builder()
    negative = b.emit(Op.LT, BOOL, b.param(0, FLOAT64), b.const(0.0, FLOAT64))

    assert b.emit(Op.NOT, BOOL, b.emit(Op.NOT, BOOL, negative)) == negative
    assert b.emit(Op.AND, BOOL, negative, negative) == negative
    assert b.emit(Op.OR, BOOL, negative, negative) == negative
