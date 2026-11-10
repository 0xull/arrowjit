"""Compile typed Python functions into native kernels over Apache Arrow columns."""

from importlib.metadata import version

__version__ = version("arrowjit")

__all__ = ["__version__"]
