"""What the checker accepts, how it types it, and what it rejects."""

import math
from collections.abc import Callable
from typing import Any

import pytest

from arrowjit.checker import check
from arrowjit.diagnostics import CompileError
from arrowjit.frontend import parse_function
from arrowjit.scalar import ScalarType
from arrowjit.tree import Assign, Function, If, Param, Return, render

INT64, FLOAT64 = ScalarType.INT64, ScalarType.FLOAT64
SCALE = 2.0


def typed(fn: Callable[..., Any]) -> Function:
    return check(parse_function(fn))


def returned(function: Function) -> str:
    """Render the expression a single-statement function returns."""
    (statement,) = function.body
    assert isinstance(statement, Return)
    return render(statement.value)


def test_mixed_arithmetic_casts_the_int_side() -> None:
    def f(a: int, b: float) -> float:
        return a + b

    function = typed(f)

    assert function.params == (Param("a", INT64), Param("b", FLOAT64))
    assert function.return_type is FLOAT64
    assert returned(function) == "(float(a) + b)"


def test_true_division_of_ints_is_float() -> None:
    def f(a: int, b: int) -> float:
        return a / b

    assert returned(typed(f)) == "(float(a) / float(b))"


def test_floor_division_and_modulo_of_ints_stay_int() -> None:
    def f(a: int, b: int) -> int:
        return a // b + a % b

    assert returned(typed(f)) == "((a // b) + (a % b))"


def test_returning_an_int_from_a_float_function_casts_it() -> None:
    def f(a: int) -> float:
        return a

    assert returned(typed(f)) == "float(a)"


def test_conditional_branches_are_unified_to_float() -> None:
    def f(a: int, b: float) -> float:
        return a if a > 0 else b

    assert returned(typed(f)) == "(float(a) if (a > 0) else b)"


def test_math_functions_take_and_return_float() -> None:
    def f(a: int) -> float:
        return math.sqrt(a) + abs(a)

    assert returned(typed(f)) == "(math.sqrt(float(a)) + float(abs(a)))"


def test_a_local_keeps_the_type_of_its_first_assignment() -> None:
    def f(x: float) -> float:
        y = x * 2.0
        if x < 0.0:
            y = 0
        return y

    _, branch, _ = typed(f).body
    assert isinstance(branch, If)
    (reassignment,) = branch.body
    assert isinstance(reassignment, Assign)
    assert render(reassignment.value) == "float(0)"


def test_an_early_return_counts_as_returning() -> None:
    def f(x: float) -> float:
        if x < 0.0:
            return 0.0
        return x

    assert len(typed(f).body) == 2


def test_a_docstring_is_skipped() -> None:
    def f(x: float) -> float:
        """Double x."""
        return x * 2.0

    assert returned(typed(f)) == "(x * 2.0)"


def bad_pow(x: float) -> float:
    return x**2


def bad_loop(x: float) -> float:
    total = 0.0
    for _ in range(3):
        total += x
    return total


def bad_missing_return(x: float) -> float:  # type: ignore[return]
    if x > 0.0:
        return x


def bad_maybe_unassigned(x: float) -> float:
    if x > 0.0:
        y = x
    return y


def bad_global(x: float) -> float:
    return x * SCALE


def bad_truthiness(x: float) -> float:
    if x:
        return 1.0
    return 0.0


def bad_type_change(x: float) -> float:
    y = 1.0
    y = x > 0.0
    return y


def bad_min(a: int, b: float) -> float:
    return min(a, b)


def bad_chained(x: float) -> float:
    return 1.0 if 0.0 < x < 1.0 else 0.0


def bad_keyword(x: float) -> float:
    return round(x, ndigits=2)


def bad_bool_param(flag: bool) -> float:
    return 1.0 if flag else 0.0


def bad_not(x: int) -> int:
    return 1 if not x else 0


@pytest.mark.parametrize(
    ("fn", "message"),
    [
        (bad_pow, "operator isn't supported"),
        (bad_loop, "For statements aren't supported"),
        (bad_missing_return, "don't return a value"),
        (bad_maybe_unassigned, "might be used before it's assigned"),
        (bad_global, "unknown name 'SCALE'"),
        (bad_truthiness, "must be a bool"),
        (bad_type_change, "can't be assigned"),
        (bad_min, "same type"),
        (bad_chained, "chained comparisons"),
        (bad_keyword, "keyword arguments"),
        (bad_bool_param, "must be annotated int or float"),
        (bad_not, "not needs a bool"),
    ],
)
def test_rejects_code_outside_the_subset(fn: Callable[..., Any], message: str) -> None:
    with pytest.raises(CompileError, match=message):
        typed(fn)


def test_errors_point_at_the_offending_code() -> None:
    with pytest.raises(CompileError) as caught:
        typed(bad_pow)

    message = str(caught.value)
    assert "return x**2" in message
    assert message.splitlines()[-1].strip() == "^" * 4
