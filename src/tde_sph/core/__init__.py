"""
Core module: interfaces, simulation orchestrator, and utilities.
"""

from tde_sph.core.interfaces import (
    Metric,
    GravitySolver,
    EOS,
    RadiationModel,
    TimeIntegrator,
    ICGenerator,
    Visualizer,
)
from tde_sph.core.simulation import (
    Simulation,
    SimulationConfig,
    SimulationState,
)

__all__ = [
    "Metric",
    "GravitySolver",
    "EOS",
    "RadiationModel",
    "TimeIntegrator",
    "ICGenerator",
    "Visualizer",
    "Simulation",
    "SimulationConfig",
    "SimulationState",
]
