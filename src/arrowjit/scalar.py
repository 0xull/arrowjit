"""The scalar types arrowjit compiles, which Python spells int, float and bool."""

from enum import Enum


class ScalarType(Enum):
    INT64 = "int64"
    FLOAT64 = "float64"
    BOOL = "bool"

    @property
    def python_name(self) -> str:
        """How Python spells this type, for error messages."""
        return _PYTHON_NAMES[self]


_PYTHON_NAMES = {ScalarType.INT64: "int", ScalarType.FLOAT64: "float", ScalarType.BOOL: "bool"}

# Types arithmetic accepts. bool is left out on purpose, even though Python allows True + 1.
NUMERIC = frozenset({ScalarType.INT64, ScalarType.FLOAT64})
