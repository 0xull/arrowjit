"""Read a Python function's source and parse it."""

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from arrowjit.diagnostics import CompileError, Source


@dataclass(frozen=True, slots=True)
class ParsedFunction:
    """A function's syntax tree together with the source it came from."""

    tree: ast.FunctionDef
    source: Source


def parse_function(fn: Callable[..., Any]) -> ParsedFunction:
    """Parse the source of `fn`, which must be a function defined with def."""
    try:
        lines, first_line = inspect.getsourcelines(fn)
    except (OSError, TypeError) as error:
        raise CompileError(
            f"can't read the source code of {fn!r}, which arrowjit needs to compile it"
        ) from error

    text = textwrap.dedent("".join(lines))
    try:
        module = ast.parse(text)
    except SyntaxError as error:
        raise CompileError(f"can't parse the source code of {fn!r}") from error

    source = Source(inspect.getsourcefile(fn) or "<unknown>", first_line, tuple(text.splitlines()))
    node = module.body[0]
    if not isinstance(node, ast.FunctionDef):
        raise CompileError("arrowjit compiles functions defined with def, not lambdas")
    return ParsedFunction(node, source)
