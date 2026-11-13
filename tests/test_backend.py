"""Check that LLVM IR becomes callable, vectorized machine code."""

import ctypes
import gc
from pathlib import Path

import pyarrow as pa
import pytest

from arrowjit.backend import CompiledModule

ADD_IR = """
define i64 @add(i64 %a, i64 %b) {
entry:
  %sum = add i64 %a, %b
  ret i64 %sum
}
"""

SCALE_IR = Path(__file__).parent / "ir" / "scale.ll"
ADD = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
SCALE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_double)


def test_compiled_function_is_callable_from_python() -> None:
    compiled = CompiledModule(ADD_IR)
    add = ADD(compiled.function_address("add"))

    assert add(40, 2) == 42
    # int64 addition wraps around instead of growing like a Python int.
    assert add(-(2**63), -1) == 2**63 - 1


def test_compiling_after_an_engine_is_freed_does_not_crash() -> None:
    # Regression test for a segfault. Engines used to share one target machine,
    # and freeing the first engine also freed the machine the second one needed.
    first = CompiledModule(ADD_IR)
    del first
    gc.collect()

    second = CompiledModule(ADD_IR)
    assert ADD(second.function_address("add"))(1, 2) == 3


def test_unknown_function_name_raises() -> None:
    compiled = CompiledModule(ADD_IR)

    with pytest.raises(KeyError, match="mul"):
        compiled.function_address("mul")


def test_scale_loop_runs_over_an_arrow_buffer() -> None:
    n = 1003  # not a multiple of 16, so the scalar remainder loop runs too
    _, values = pa.array([float(i) for i in range(n)], type=pa.float64()).buffers()
    out = pa.allocate_buffer(n * 8)

    compiled = CompiledModule(SCALE_IR.read_text())
    scale = SCALE(compiled.function_address("scale"))
    scale(out.address, values.address, n, 2.5)

    result = pa.Array.from_buffers(pa.float64(), n, [None, out])
    assert result.to_pylist() == [2.5 * i for i in range(n)]


def test_scale_loop_is_vectorized() -> None:
    assembly = CompiledModule(SCALE_IR.read_text()).assembly()
    # On any x86-64 CPU, including CI's, a vectorized loop multiplies with mulpd
    # or vmulpd and a scalar loop with mulsd or vmulsd.
    assert "mulpd" in assembly
