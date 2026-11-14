"""Lowered functions agree with the Python they came from, row by row."""

import math
from collections.abc import Callable
from functools import cache
from typing import Any

import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st

from arrowjit import ir
from arrowjit.checker import check
from arrowjit.frontend import parse_function
from arrowjit.interpret import interpret, wrap
from arrowjit.ir import Constant
from arrowjit.kernel import Kernel
from arrowjit.lower import lower
from arrowjit.scalar import ScalarType

ANY_INTS = st.integers(-(2**63), 2**63 - 1)
# Ints up to 2^53 in magnitude convert to float exactly, so mixed int and float
# arithmetic on them matches Python's exactly.
EXACT_INTS = st.integers(-(2**53), 2**53)
# Every float64, including nan, inf, -0.0 and subnormals.
ANY_FLOATS = st.floats()
ARROW_TYPES = {ScalarType.INT64: pa.int64(), ScalarType.FLOAT64: pa.float64()}


def mixed(a: int, x: float) -> float:
    return a * x - a / 3 + x // 2.0


def guarded(a: int, b: int) -> int:
    if b == 0:
        return 0
    return a // b + a % b


def piecewise(x: float, y: float) -> float:
    if x < 0.0:
        return math.sqrt(y)
    z = x / y
    if z > 1.0:
        z = 1.0
    elif z < -1.0:
        return math.log(-z)
    return z


def short_circuit(a: int, b: int) -> int:
    return 1 if (b != 0 and a // b > 2) or a < 0 else 0


def ternary(x: float) -> float:
    return math.log(x) if x > 0.0 else -1.0


def maths(x: float) -> float:
    return math.sqrt(x) + math.exp(x) - math.log(abs(x) + 1.0)


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def ratio(a: int, b: int) -> float:
    return a / b


def float_division(x: float, y: float) -> float:
    return x % y - x // y


def wrapping(a: int, b: int) -> int:
    return abs(a) * b - 3 * a + -b


def comparisons(x: float, y: float) -> int:
    score = 0
    if x != y:
        score += 1
    if x <= y:
        score += 2
    if x >= y:
        score += 4
    if x == y:
        score += 8
    return score


type Case = tuple[Callable[..., Any], tuple[st.SearchStrategy[Any], ...]]

CASES: list[Case] = [
    (mixed, (EXACT_INTS, ANY_FLOATS)),
    (guarded, (ANY_INTS, ANY_INTS)),
    (piecewise, (ANY_FLOATS, ANY_FLOATS)),
    (short_circuit, (EXACT_INTS, EXACT_INTS)),
    (ternary, (ANY_FLOATS,)),
    (maths, (ANY_FLOATS,)),
    (clamp, (ANY_FLOATS, ANY_FLOATS, ANY_FLOATS)),
    (ratio, (EXACT_INTS, EXACT_INTS)),
    (float_division, (ANY_FLOATS, ANY_FLOATS)),
    (wrapping, (ANY_INTS, ANY_INTS)),
    (comparisons, (ANY_FLOATS, ANY_FLOATS)),
]


@cache
def lowered(fn: Callable[..., Any]) -> ir.Function:
    return lower(check(parse_function(fn)))


def python_outcome(
    fn: Callable[..., Any], row: tuple[Any, ...], column_type: ScalarType
) -> Constant | None:
    """What Python gives for a row, in the column's type, or None if it raises."""
    try:
        value = fn(*row)
    except ArithmeticError, ValueError:
        return None
    # Wrapping once at the end equals wrapping after every +, - and *, and in these
    # functions overflow is only ever followed by those operations.
    return float(value) if column_type is ScalarType.FLOAT64 else wrap(value)


def same(a: Constant | None, b: Constant | None) -> bool:
    """Compare outcomes, telling -0.0 from 0.0 and treating every NaN as equal."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return type(a) is type(b) and a == b


@pytest.mark.parametrize(("fn", "columns"), CASES, ids=[fn.__name__ for fn, _ in CASES])
@given(data=st.data())
def test_lowering_matches_python_row_by_row(
    fn: Callable[..., Any], columns: tuple[st.SearchStrategy[Any], ...], data: st.DataObject
) -> None:
    function = lowered(fn)
    row = data.draw(st.tuples(*columns), label="row")
    assert same(interpret(function, row), python_outcome(fn, row, function.return_type))


def test_wrap_reduces_to_int64_like_twos_complement() -> None:
    assert wrap(2**63) == -(2**63)
    assert wrap(-(2**63) - 1) == 2**63 - 1
    assert wrap(2**64 + 5) == 5


@cache
def compiled(fn: Callable[..., Any]) -> Kernel:
    return Kernel(lowered(fn))


def run_kernel(
    kernel: Kernel, rows: list[tuple[Any, ...]], valid: list[bool] | None
) -> tuple[list[Any], int]:
    """Run a kernel over rows given as Python values, returning its output and first error."""
    n = len(rows)
    columns = [
        pa.array([row[position] for row in rows], type=ARROW_TYPES[param])
        for position, param in enumerate(kernel.function.params)
    ]
    out = pa.allocate_buffer(n * 8)
    # The kernel reads the bitmap a 64-bit word at a time, so pad it to whole words.
    bitmap = None if valid is None else pa.array(valid + [False] * (-n % 64), type=pa.bool_())
    first = kernel(
        out.address,
        None if bitmap is None else bitmap.buffers()[1].address,
        n,
        [column.buffers()[1].address for column in columns],
    )
    values = pa.Array.from_buffers(ARROW_TYPES[kernel.function.return_type], n, [None, out])
    return values.to_pylist(), first


@pytest.mark.parametrize(("fn", "columns"), CASES, ids=[fn.__name__ for fn, _ in CASES])
@given(data=st.data())
def test_compiled_kernel_matches_python_on_whole_columns(
    fn: Callable[..., Any], columns: tuple[st.SearchStrategy[Any], ...], data: st.DataObject
) -> None:
    kernel = compiled(fn)
    n = data.draw(st.integers(0, 150), label="n")
    rows = data.draw(st.lists(st.tuples(*columns), min_size=n, max_size=n), label="rows")
    valid = data.draw(st.none() | st.lists(st.booleans(), min_size=n, max_size=n), label="valid")
    values, first = run_kernel(kernel, rows, valid)

    expected = [python_outcome(fn, row, kernel.function.return_type) for row in rows]
    checked = [i for i in range(n) if valid is None or valid[i]]
    raising = [i for i in checked if expected[i] is None]
    assert first == (raising[0] if raising else -1)
    # The kernel returns as soon as a block has an error, so later rows aren't written.
    stop = n if first < 0 else first
    for i in checked:
        if i < stop:
            assert same(values[i], expected[i]), f"row {i} is {rows[i]}"


def test_int64_min_floor_divided_by_minus_one_wraps_instead_of_trapping() -> None:
    assert run_kernel(compiled(guarded), [(-(2**63), -1)], None) == ([-(2**63)], -1)


def test_each_block_reads_its_own_validity_word() -> None:
    # Rows 100 and 130 divide by zero, and rows 2 and 100 are null. Lanes repeat every
    # 64 rows, so a kernel that read another block's validity word would stop at row 100
    # or skip row 130.
    rows = [(1, 0) if i in (100, 130) else (1, 1) for i in range(150)]
    valid = [i not in (2, 100) for i in range(150)]
    _, first = run_kernel(compiled(ratio), rows, valid)
    assert first == 130
