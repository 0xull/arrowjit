"""The jit decorator, which runs a compiled function over Arrow columns.

Calling a jit function behaves like calling the original once per row. A null in
any input makes that row's output null, and the first valid row that raises
makes the whole call raise the same exception Python would.
"""

import functools
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

import pyarrow as pa
import pyarrow.compute as pc

from arrowjit.checker import check
from arrowjit.frontend import parse_function
from arrowjit.kernel import Kernel
from arrowjit.lower import lower
from arrowjit.scalar import ScalarType

ARROW_TYPES = {ScalarType.INT64: pa.int64(), ScalarType.FLOAT64: pa.float64()}


def jit(fn: Callable[..., Any]) -> JitFunction:
    """Compile a typed function to run over whole Arrow columns."""
    return JitFunction(fn)


class JitFunction:
    """A function compiled by @jit. Call it with one Arrow column per parameter."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        checked = check(parse_function(fn))
        if not checked.params:
            raise TypeError(f"{fn.__name__} needs at least one parameter to take a column")
        self.fn = fn
        self.params = checked.params
        self.lowered = lower(checked)
        self.kernel = Kernel(self.lowered)
        self.result_type = ARROW_TYPES[self.lowered.return_type]
        functools.update_wrapper(self, fn)

    def __call__(self, *columns: Any) -> Any:
        """Return a column of results, chunked if any input column is chunked."""
        self._check_columns(columns)
        if all(isinstance(column, pa.Array) for column in columns):
            return self._run(columns, 0)
        # Splitting a table into record batches cuts every column at the same rows, so
        # each batch is a set of equal-length arrays, without copying anything.
        table = pa.Table.from_arrays(list(columns), names=[param.name for param in self.params])
        chunks: list[Any] = []
        start = 0
        for batch in table.to_batches():
            chunks.append(self._run(batch.columns, start))
            start += batch.num_rows
        return pa.chunked_array(chunks, type=self.result_type)

    def _check_columns(self, columns: tuple[Any, ...]) -> None:
        name = self.fn.__name__
        if len(columns) != len(self.params):
            raise TypeError(f"{name}() takes {len(self.params)} columns, not {len(columns)}")
        for param, column in zip(self.params, columns, strict=True):
            expected = ARROW_TYPES[param.type]
            if not isinstance(column, pa.Array | pa.ChunkedArray):
                raise TypeError(
                    f"{param.name!r} must be an Arrow array, not {type(column).__name__}"
                )
            if column.type != expected:
                raise TypeError(
                    f"{param.name!r} must have Arrow type {expected}, not {column.type}"
                )
        lengths = {len(column) for column in columns}
        if len(lengths) > 1:
            raise ValueError(f"all columns must have the same length, got {sorted(lengths)}")

    def _run(self, columns: Sequence[Any], start: int) -> Any:
        """Run the kernel over equal-length arrays that begin at row `start` of the input."""
        n = len(columns[0])
        if n == 0:
            return pa.array([], type=self.result_type)
        valid = _validity(columns)
        out = pa.allocate_buffer(n * 8)
        first = self.kernel(
            out.address,
            None if valid is None else valid.address,
            n,
            [_values_address(column) for column in columns],
        )
        if first >= 0:
            self._raise_for_row(columns, first, start)
        return pa.Array.from_buffers(self.result_type, n, [valid, out])

    def _raise_for_row(self, columns: Sequence[Any], row: int, start: int) -> NoReturn:
        args = tuple(column[row].as_py() for column in columns)
        try:
            self.fn(*args)
        except Exception as error:
            error.add_note(f"arrowjit: raised for row {start + row}, with arguments {args}")
            raise
        raise AssertionError(f"the kernel flagged row {start + row}, but Python didn't raise")


def _validity(columns: Sequence[Any]) -> Any:
    """A bitmap of the rows where every column is valid, or None if all of them are."""
    masks = [column.is_valid() for column in columns if column.null_count > 0]
    if not masks:
        return None
    combined = masks[0]
    for mask in masks[1:]:
        combined = pc.call_function("and", [combined, mask])
    padding = pa.array([False] * (-len(combined) % 64), type=pa.bool_())
    return pa.concat_arrays([combined, padding]).buffers()[1]


def _values_address(column: Any) -> int:
    """The address of a column's first value, after its slice offset."""
    address: int = column.buffers()[1].address + column.offset * 8
    if address % 8:
        raise ValueError("arrowjit needs columns whose values are 8-byte aligned")
    return address
