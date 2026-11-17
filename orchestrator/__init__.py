"""
Fed-LCCF Local Multi-Process Simulation Orchestrator.

Exports the core orchestrator class and utilities for managing the lifecycle
of the central cloud server and federated client agents on a single machine.
"""

from typing import Any

__all__ = [
    "LocalSimulationOrchestrator",
    "check_port_available",
    "_kill_process_tree",
    "parse_args",
    "main",
]


def __getattr__(name: str) -> Any:
    """Lazy export to prevent runpy RuntimeWarning on python -m execution."""
    if name in __all__:
        from fedlccf_prototype.orchestrator import run_simulation
        return getattr(run_simulation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
