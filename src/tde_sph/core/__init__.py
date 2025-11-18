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
from tde_sph.core.energy_diagnostics import EnergyDiagnostics

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
    "EnergyDiagnostics",
]
