"""Compile a lowered function into a kernel that Python calls through ctypes."""

import ctypes
from collections.abc import Sequence

from arrowjit.backend import CompiledModule
from arrowjit.codegen import generate
from arrowjit.ir import Function


class Kernel:
    """A lowered function compiled for the host CPU, called on raw column buffers."""

    def __init__(self, function: Function) -> None:
        self.function = function
        self.module = CompiledModule(generate(function))
        columns = [ctypes.c_void_p] * len(function.params)
        signature = ctypes.CFUNCTYPE(
            ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, *columns
        )
        self._run = signature(self.module.function_address("kernel"))

    def __call__(self, out: int, valid: int | None, n: int, columns: Sequence[int]) -> int:
        """Run n rows and return the first valid row that raises, or -1.

        Every address must be 8-byte aligned. out and each column hold n values, and
        valid is None when every row is valid or else holds ceil(n / 64) words.
        """
        first: int = self._run(out, valid, n, *columns)
        return first
