"""The jit decorator over Arrow arrays, slices and chunked columns."""

import pyarrow as pa
import pytest

from arrowjit import CompileError, jit


@jit
def scaled(x: float, y: float) -> float:
    return x * y + 1.0


@jit
def ratio(a: int, b: int) -> float:
    return a / b


def test_runs_over_arrays_and_returns_a_float64_array() -> None:
    result = scaled(pa.array([1.0, 2.0, 3.0]), pa.array([2.0, 0.5, -1.0]))
    assert result.type == pa.float64()
    assert result.to_pylist() == [3.0, 2.0, -2.0]


def test_a_null_in_any_input_makes_the_row_null() -> None:
    result = scaled(pa.array([1.0, None, 3.0]), pa.array([2.0, 2.0, None]))
    assert result.to_pylist() == [3.0, None, None]


def test_slices_are_read_from_their_offset() -> None:
    x = pa.array([9.0, 1.0, None, 3.0])[1:]
    y = pa.array([9.0, 9.0, 2.0, 2.0, 2.0])[2:]
    assert scaled(x, y).to_pylist() == [3.0, None, 7.0]


def test_chunked_columns_can_have_different_chunk_boundaries() -> None:
    x = pa.chunked_array([[1.0, 2.0], [3.0, 4.0, 5.0]])
    y = pa.chunked_array([[1.0], [1.0, 1.0, 1.0, 2.0]])
    result = scaled(x, y)
    assert isinstance(result, pa.ChunkedArray)
    assert result.to_pylist() == [2.0, 3.0, 4.0, 5.0, 11.0]


def test_the_first_failing_row_raises_pythons_own_exception() -> None:
    with pytest.raises(ZeroDivisionError, match="division by zero") as caught:
        ratio(pa.array([1, 2, 3, 4]), pa.array([1, 2, 0, 0]))
    assert caught.value.__notes__ == ["arrowjit: raised for row 2, with arguments (3, 0)"]


def test_a_failing_row_in_a_later_chunk_reports_its_row_in_the_whole_column() -> None:
    with pytest.raises(ZeroDivisionError) as caught:
        ratio(pa.chunked_array([[1, 2], [3, 4]]), pa.chunked_array([[1, 1], [1, 0]]))
    assert caught.value.__notes__ == ["arrowjit: raised for row 3, with arguments (4, 0)"]


def test_a_null_row_never_raises() -> None:
    # pyarrow stores 0 under the null, so only the validity bitmap keeps this row quiet.
    assert ratio(pa.array([1, 2, 3]), pa.array([1, None, 2])).to_pylist() == [1.0, None, 1.5]


def test_empty_columns_give_an_empty_column() -> None:
    assert scaled(pa.array([], pa.float64()), pa.array([], pa.float64())).to_pylist() == []


def test_rejects_the_wrong_number_of_columns() -> None:
    with pytest.raises(TypeError, match="takes 2 columns, not 1"):
        scaled(pa.array([1.0]))


def test_rejects_a_column_of_the_wrong_type() -> None:
    with pytest.raises(TypeError, match="'y' must have Arrow type double, not int64"):
        scaled(pa.array([1.0]), pa.array([1]))


def test_rejects_columns_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        scaled(pa.array([1.0]), pa.array([1.0, 2.0]))


def test_unsupported_code_fails_when_decorated() -> None:
    def squared(x: float) -> float:
        return x**2

    with pytest.raises(CompileError, match="operator isn't supported"):
        jit(squared)
