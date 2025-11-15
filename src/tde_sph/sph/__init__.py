"""
SPH module: particles, kernels, neighbour search, and hydrodynamic forces.
"""

from .particles import ParticleSystem
from .kernels import CubicSplineKernel, default_kernel
from .neighbours_cpu import (
    find_neighbours_bruteforce,
    compute_density_summation,
    update_smoothing_lengths
)
from .hydro_forces import (
    compute_hydro_acceleration,
    compute_viscosity_timestep,
    compute_thermal_conductivity
)

__all__ = [
    # Particle management
    "ParticleSystem",

    # Kernels
    "CubicSplineKernel",
    "default_kernel",

    # Neighbour search
    "find_neighbours_bruteforce",
    "compute_density_summation",
    "update_smoothing_lengths",

    # Hydrodynamic forces
    "compute_hydro_acceleration",
    "compute_viscosity_timestep",
    "compute_thermal_conductivity",
]
