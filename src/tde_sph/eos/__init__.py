"""
EOS module: equations of state (ideal gas, gas + radiation).

Available EOS implementations:
- IdealGas: Simple ideal gas with adiabatic index γ (Phase 1)
- RadiationGasEOS: Combined gas + radiation pressure for optically thick gas (Phase 3, TASK-021)
"""

from .ideal_gas import IdealGas
from .radiation_gas import RadiationGasEOS

__all__ = ["IdealGas", "RadiationGasEOS"]
