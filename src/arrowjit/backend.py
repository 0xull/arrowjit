"""JIT-compile LLVM IR into machine code for the host CPU.

The host CPU is the processor running this Python process. LLVM detects which
extensions it supports, such as AVX2 and AVX-512, and the generated code may
use any of them, so it only runs on CPUs with the same features.

llvmlite has no type hints, so its objects are typed Any in this module and
everything the module returns is a plain int or str.
"""

from functools import cache
from typing import Any

import llvmlite.binding as llvm


@cache
def _initialize_native_target() -> None:
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()


def _new_target_machine() -> Any:
    """Create a target machine for the host CPU.

    An MCJIT engine owns the target machine it's built with and frees it along
    with itself, so every engine needs its own.
    """
    _initialize_native_target()
    target = llvm.Target.from_triple(llvm.get_process_triple())
    return target.create_target_machine(
        cpu=llvm.get_host_cpu_name(),
        features=llvm.get_host_cpu_features().flatten(),
        opt=3,
    )


class CompiledModule:
    """LLVM IR optimized at O3 and compiled to machine code for the host CPU.

    Function addresses are only valid while this object is alive, because the
    machine code lives in the engine it owns.
    """

    def __init__(self, ir: str) -> None:
        target_machine = _new_target_machine()
        module = llvm.parse_assembly(ir)
        # Tell the optimizer which target this is and how it lays out types in memory.
        module.triple = llvm.get_process_triple()
        module.data_layout = str(target_machine.target_data)
        module.verify()

        tuning = llvm.create_pipeline_tuning_options(speed_level=3)
        tuning.loop_vectorization = True
        tuning.slp_vectorization = True
        passes = llvm.create_pass_builder(target_machine, tuning)
        passes.getModulePassManager().run(module, passes)

        self.optimized_ir: str = str(module)
        # MCJIT treats an unknown function name as a fatal error, so names are
        # checked against this set before the engine ever sees them.
        self._defined: frozenset[str] = frozenset(
            function.name for function in module.functions if not function.is_declaration
        )
        self._engine: Any = llvm.create_mcjit_compiler(module, target_machine)
        self._engine.finalize_object()

    def function_address(self, name: str) -> int:
        """Return a compiled function's address, to call it through ctypes."""
        if name not in self._defined:
            raise KeyError(f"no function named {name!r} in this module")
        address: int = self._engine.get_function_address(name)
        return address

    def assembly(self) -> str:
        """Return the module's assembly for the host CPU, for inspecting codegen."""
        module = llvm.parse_assembly(self.optimized_ir)
        text: str = _new_target_machine().emit_assembly(module)
        return text
