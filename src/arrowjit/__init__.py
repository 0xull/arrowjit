"""Compile typed Python functions into native kernels over Apache Arrow columns."""

from importlib.metadata import version

from arrowjit.diagnostics import CompileError
from arrowjit.runtime import jit

__version__ = version("arrowjit")

__all__ = ["CompileError", "__version__", "jit"]
