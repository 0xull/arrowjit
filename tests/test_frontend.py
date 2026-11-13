"""Tests for reading a function's source and pointing errors back at it."""

import pytest

from arrowjit.diagnostics import CompileError, Span
from arrowjit.frontend import parse_function


def top_level(x: float) -> float:
    return x * 2.0


def test_parses_a_top_level_function() -> None:
    parsed = parse_function(top_level)

    assert parsed.tree.name == "top_level"
    assert parsed.source.filename.endswith("test_frontend.py")
    assert parsed.source.lines[0] == "def top_level(x: float) -> float:"


def test_dedents_a_nested_function() -> None:
    def nested(x: float) -> float:
        return x + 1.0

    parsed = parse_function(nested)

    assert parsed.tree.name == "nested"
    assert parsed.source.lines[0] == "def nested(x: float) -> float:"


def test_rejects_a_lambda() -> None:
    with pytest.raises(CompileError, match="defined with def"):
        parse_function(lambda x: x * 2)


def test_rejects_a_function_without_python_source() -> None:
    with pytest.raises(CompileError, match="can't read the source"):
        parse_function(len)


def test_excerpt_points_at_the_span_with_the_real_line_number() -> None:
    def nested(x: float) -> float:
        return x * 2.0

    source = parse_function(nested).source
    excerpt = source.excerpt(Span(line=2, col=11, end_line=2, end_col=18))

    file_line, code, carets = excerpt.splitlines()
    assert file_line.endswith(f"line {source.first_line + 1}")
    assert code == "    return x * 2.0"
    assert carets == " " * 11 + "^" * 7
