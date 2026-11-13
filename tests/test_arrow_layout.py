"""Pin down the parts of Arrow's memory layout that arrowjit relies on.

Each test states one fact about how pyarrow lays out a column. If a pyarrow
upgrade ever changes one of them, the failing test names the broken assumption.
"""

import struct

import pyarrow as pa

ALIGNMENT = 64


def validity_bit(bitmap: bytes, row: int) -> int:
    """Return the validity bit for `row`, numbered least-significant bit first."""
    return (bitmap[row // 8] >> (row % 8)) & 1


def test_validity_bitmap_is_lsb_first_and_one_means_valid() -> None:
    validity, _ = pa.array([3, None, 7], type=pa.int64()).buffers()
    bitmap = validity.to_pybytes()

    assert bitmap[0] & 0b111 == 0b101
    assert [validity_bit(bitmap, row) for row in range(3)] == [1, 0, 1]


def test_arrays_without_nulls_have_no_validity_bitmap() -> None:
    arr = pa.array([1, 2, 3], type=pa.int64())

    assert arr.null_count == 0
    assert arr.buffers()[0] is None


def test_int64_values_are_8_byte_little_endian() -> None:
    _, values = pa.array([3, None, 7], type=pa.int64()).buffers()
    data = values.to_pybytes()

    assert struct.unpack_from("<q", data, 0 * 8) == (3,)
    assert struct.unpack_from("<q", data, 2 * 8) == (7,)


def test_float64_values_are_ieee754_doubles() -> None:
    _, values = pa.array([1.5, -0.25], type=pa.float64()).buffers()

    assert struct.unpack_from("<2d", values.to_pybytes(), 0) == (1.5, -0.25)


def test_pyarrow_writes_zero_into_null_slots() -> None:
    _, values = pa.array([3, None, 7], type=pa.int64()).buffers()

    assert struct.unpack_from("<q", values.to_pybytes(), 1 * 8) == (0,)


def test_new_buffers_are_64_byte_aligned() -> None:
    validity, values = pa.array([3, None, 7], type=pa.int64()).buffers()

    assert validity.address % ALIGNMENT == 0
    assert values.address % ALIGNMENT == 0


def test_slices_share_buffers_and_shift_rows_by_the_offset() -> None:
    parent = pa.array([10, None, 30, 40], type=pa.int64())
    child = parent.slice(1, 2)
    validity, values = child.buffers()
    bitmap = validity.to_pybytes()

    assert child.offset == 1
    assert child.to_pylist() == [None, 30]
    assert values.address == parent.buffers()[1].address
    assert validity_bit(bitmap, child.offset + 0) == 0
    assert validity_bit(bitmap, child.offset + 1) == 1
    assert struct.unpack_from("<q", values.to_pybytes(), (child.offset + 1) * 8) == (30,)


def test_booleans_are_bit_packed() -> None:
    _, values = pa.array([True, False, True, True]).buffers()

    assert values.to_pybytes()[0] & 0b1111 == 0b1101


def test_table_columns_are_chunked_arrays() -> None:
    table = pa.concat_tables([pa.table({"x": [1, 2]}), pa.table({"x": [3]})])
    column = table.column("x")

    assert isinstance(column, pa.ChunkedArray)
    assert [chunk.to_pylist() for chunk in column.chunks] == [[1, 2], [3]]
