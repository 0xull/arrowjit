from importlib.metadata import version

import arrowjit


def test_package_exposes_installed_version() -> None:
    assert arrowjit.__version__ == version("arrowjit")
