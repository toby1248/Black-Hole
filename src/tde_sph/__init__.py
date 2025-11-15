"""
TDE-SPH: Relativistic SPH framework for stellar tidal disruption events.

A modular Python/CUDA framework for simulating tidal disruption events around
supermassive black holes with support for both general relativistic and
Newtonian dynamics.
"""

__version__ = "1.0.0"
__author__ = "TDE-SPH Dev Team"

# Core imports for convenience
from tde_sph.core.interfaces import (
    Metric,
    GravitySolver,
    EOS,
    RadiationModel,
    TimeIntegrator,
    ICGenerator,
    Visualizer,
)

__all__ = [
    "Metric",
    "GravitySolver",
    "EOS",
    "RadiationModel",
    "TimeIntegrator",
    "ICGenerator",
    "Visualizer",
]
