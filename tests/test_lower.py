"""The predicated IR that lowering produces for each kind of control flow."""

from collections.abc import Callable
from typing import Any

from arrowjit.checker import check
from arrowjit.frontend import parse_function
from arrowjit.lower import lower


def lowered(fn: Callable[..., Any]) -> list[str]:
    return str(lower(check(parse_function(fn)))).splitlines()


def test_a_variable_reassigned_in_a_branch_becomes_a_select() -> None:
    def f(x: float) -> float:
        y = x
        if x > 1.0:
            y = 1.0
        return y

    assert lowered(f) == [
        "f(float64) -> float64",
        "  %0: float64 = param 0",
        "  %1: bool = const False",
        "  %2: float64 = const 1.0",
        "  %3: bool = gt %0, %2",
        "  %4: float64 = select %3, %2, %0",
        "  return %4",
        "  raise if %1",
    ]


def test_division_flags_rows_where_the_divisor_is_zero() -> None:
    def f(a: int, b: int) -> int:
        return a // b

    assert lowered(f) == [
        "f(int64, int64) -> int64",
        "  %0: int64 = param 0",
        "  %1: int64 = param 1",
        "  %2: int64 = const 0",
        "  %3: bool = eq %1, %2",
        "  %4: int64 = floordiv %0, %1",
        "  return %4",
        "  raise if %3",
    ]


def test_a_check_in_a_branch_only_covers_rows_that_take_it() -> None:
    def f(a: int, b: int) -> int:
        if b != 0:
            return a // b
        return 0

    assert lowered(f) == [
        "f(int64, int64) -> int64",
        "  %0: int64 = param 0",
        "  %1: int64 = param 1",
        "  %2: int64 = const 0",
        "  %3: bool = ne %1, %2",
        "  %4: bool = not %3",
        "  %5: bool = eq %1, %2",
        "  %6: bool = and %3, %5",
        "  %7: int64 = floordiv %0, %1",
        "  %8: int64 = select %4, %2, %7",
        "  return %8",
        "  raise if %6",
    ]


def test_a_check_after_an_early_return_skips_rows_that_returned() -> None:
    def f(a: int, b: int) -> int:
        if b == 0:
            return 0
        return a // b

    assert lowered(f) == [
        "f(int64, int64) -> int64",
        "  %0: int64 = param 0",
        "  %1: int64 = param 1",
        "  %2: int64 = const 0",
        "  %3: bool = eq %1, %2",
        "  %4: bool = not %3",
        "  %5: bool = and %4, %3",
        "  %6: int64 = floordiv %0, %1",
        "  %7: int64 = select %4, %6, %2",
        "  return %7",
        "  raise if %5",
    ]


def test_the_right_side_of_and_is_only_checked_where_python_evaluates_it() -> None:
    def f(a: int, b: int) -> int:
        return 1 if b != 0 and a // b > 1 else 0

    assert lowered(f) == [
        "f(int64, int64) -> int64",
        "  %0: int64 = param 0",
        "  %1: int64 = param 1",
        "  %2: int64 = const 0",
        "  %3: bool = ne %1, %2",
        "  %4: bool = eq %1, %2",
        "  %5: bool = and %3, %4",
        "  %6: int64 = floordiv %0, %1",
        "  %7: int64 = const 1",
        "  %8: bool = gt %6, %7",
        "  %9: bool = and %3, %8",
        "  %10: int64 = select %9, %7, %2",
        "  return %10",
        "  raise if %5",
    ]
