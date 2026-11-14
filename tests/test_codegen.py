"""The machine code that generated kernels compile to."""

from collections.abc import Callable
from typing import Any

from arrowjit.backend import CompiledModule
from arrowjit.checker import check
from arrowjit.codegen import generate
from arrowjit.frontend import parse_function
from arrowjit.lower import lower


def assembly(fn: Callable[..., Any]) -> str:
    return CompiledModule(generate(lower(check(parse_function(fn))))).assembly()


# On any x86-64 CPU, including CI's, LLVM's loop vectorizer turns these into packed
# instructions like mulpd or vmulpd, while a scalar loop would use mulsd or vmulsd.


def test_arithmetic_is_vectorized() -> None:
    def f(x: float, y: float) -> float:
        return x * y + 1.0

    assert "mulpd" in assembly(f)


def test_a_guarded_division_is_still_vectorized() -> None:
    def f(x: float, y: float) -> float:
        if y == 0.0:
            return 0.0
        return x / y

    assert "divpd" in assembly(f)
