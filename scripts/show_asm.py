"""Print the optimized LLVM IR and the assembly for an LLVM IR file."""

import argparse
from pathlib import Path

from arrowjit.backend import CompiledModule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir_file", type=Path, help="path to a .ll file")
    args = parser.parse_args()

    compiled = CompiledModule(args.ir_file.read_text())
    print("; ---- optimized LLVM IR ----")
    print(compiled.optimized_ir)
    print("; ---- assembly ----")
    print(compiled.assembly())


if __name__ == "__main__":
    main()
