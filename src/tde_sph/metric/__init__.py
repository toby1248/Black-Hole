"""
Metric module: spacetime metrics (Minkowski, Schwarzschild, Kerr).

This module provides spacetime metric implementations for general relativistic
simulations of tidal disruption events. All metrics inherit from the Metric ABC
defined in tde_sph.core.interfaces.

Available metrics:
- MinkowskiMetric: Flat spacetime (testing/validation)
- SchwarzschildMetric: Non-rotating black hole
- KerrMetric: Rotating black hole with spin parameter

Coordinate utilities:
- cartesian_to_bl_spherical: Convert Cartesian to Boyer-Lindquist coordinates
- bl_spherical_to_cartesian: Convert Boyer-Lindquist to Cartesian coordinates
- velocity transformations and singularity regularization

References
----------
- Misner, Thorne & Wheeler (1973) - Gravitation
- Bardeen, Press & Teukolsky (1972), ApJ 178, 347
- Tejeda et al. (2017), MNRAS 469, 4483 [arXiv:1701.00303]
- Liptai & Price (2019), MNRAS 485, 819 [arXiv:1901.08064]
"""

from tde_sph.metric.minkowski import MinkowskiMetric
from tde_sph.metric.schwarzschild import SchwarzschildMetric
from tde_sph.metric.kerr import KerrMetric
from tde_sph.metric.coordinates import (
    cartesian_to_bl_spherical,
    bl_spherical_to_cartesian,
    velocity_cartesian_to_bl,
    velocity_bl_to_cartesian,
    check_coordinate_validity,
    regularize_near_singularity,
)

__all__ = [
    "MinkowskiMetric",
    "SchwarzschildMetric",
    "KerrMetric",
    "cartesian_to_bl_spherical",
    "bl_spherical_to_cartesian",
    "velocity_cartesian_to_bl",
    "velocity_bl_to_cartesian",
    "check_coordinate_validity",
    "regularize_near_singularity",
]
