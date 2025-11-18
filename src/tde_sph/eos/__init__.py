"""
EOS module: equations of state (ideal gas, gas + radiation).
"""

from .ideal_gas import IdealGas
from .radiation_gas import RadiationGas

__all__ = ["IdealGas", "RadiationGas"]
