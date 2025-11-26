"""
SPH module: particles, kernels, neighbour search, and hydrodynamic forces.
"""

from .particles import ParticleSystem
from .kernels import CubicSplineKernel, default_kernel
from .neighbours_cpu import (
    find_neighbours_bruteforce,
    find_neighbours_octree,
    compute_density_summation,
    update_smoothing_lengths
)
from .hydro_forces import (
    compute_hydro_acceleration,
    compute_viscosity_timestep,
    compute_thermal_conductivity
)

# GPU neighbour search
try:
    from .neighbours_gpu import (
        find_neighbours_octree_gpu_integrated,
        find_neighbours_gpu,
        GPUNeighbourSearchCache,
        GPUNeighborSearchCache,
    )
except ImportError:
    find_neighbours_octree_gpu_integrated = None
    find_neighbours_gpu = None
    GPUNeighbourSearchCache = None
    GPUNeighborSearchCache = None

__all__ = [
    # Particle management
    "ParticleSystem",

    # Kernels
    "CubicSplineKernel",
    "default_kernel",

    # Neighbour search (CPU)
    "find_neighbours_bruteforce",
    "find_neighbours_octree",
    "compute_density_summation",
    "update_smoothing_lengths",

    # Neighbour search (GPU)
    "find_neighbours_octree_gpu_integrated",
    "find_neighbours_gpu",
    "GPUNeighbourSearchCache",
    "GPUNeighborSearchCache",

    # Hydrodynamic forces
    "compute_hydro_acceleration",
    "compute_viscosity_timestep",
    "compute_thermal_conductivity",
]
