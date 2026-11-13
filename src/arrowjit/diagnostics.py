"""Source locations and compile errors that point at the offending code."""

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    """A range in the parsed source, with Python's 1-based lines and 0-based columns."""

    line: int
    col: int
    end_line: int
    end_col: int


def span_of(node: ast.expr | ast.stmt | ast.arg) -> Span:
    """Return the source range Python's parser recorded for a node."""
    end_line = node.end_lineno if node.end_lineno is not None else node.lineno
    end_col = node.end_col_offset if node.end_col_offset is not None else node.col_offset + 1
    return Span(node.lineno, node.col_offset, end_line, end_col)


@dataclass(frozen=True, slots=True)
class Source:
    """The text a function was parsed from, and where it starts in its file."""

    filename: str
    first_line: int
    lines: tuple[str, ...]

    def excerpt(self, span: Span) -> str:
        """Show the line a span starts on, with carets under the span, like a traceback."""
        text = self.lines[span.line - 1]
        stripped = text.lstrip()
        start = span.col - (len(text) - len(stripped))
        end = span.end_col if span.end_line == span.line else len(text)
        carets = "^" * max(1, end - span.col)
        return (
            f'  File "{self.filename}", line {self.first_line + span.line - 1}\n'
            f"    {stripped}\n"
            f"    {' ' * start}{carets}"
        )


class CompileError(Exception):
    """Raised when a function uses something arrowjit can't compile."""

    def __init__(
        self, message: str, span: Span | None = None, source: Source | None = None
    ) -> None:
        self.message = message
        self.span = span
        self.source = source
        details = f"\n{source.excerpt(span)}" if span is not None and source is not None else ""
        super().__init__(f"{message}{details}")
