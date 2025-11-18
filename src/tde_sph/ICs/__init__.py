"""
Initial conditions module: stellar models and orbit generators.
"""

from tde_sph.ICs.polytrope import Polytrope
from tde_sph.ICs.disc import DiscGenerator

__all__ = ["Polytrope", "DiscGenerator"]
