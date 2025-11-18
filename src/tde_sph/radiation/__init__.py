"""
Radiation module: cooling models and radiative transfer.
"""

from .simple_cooling import SimpleCoolingModel, CoolingRates

# Backward compatibility alias
SimpleCooling = SimpleCoolingModel

__all__ = ["SimpleCoolingModel", "SimpleCooling", "CoolingRates"]
