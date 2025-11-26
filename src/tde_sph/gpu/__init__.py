"""
GPU acceleration module using CUDA (via Numba/CuPy).

This module provides GPU-accelerated implementations of:
- Gravity (N-body with Barnes-Hut octree)
- SPH Neighbour Search (octree-based)
- Hydrodynamics
- Time integration

It requires an NVIDIA GPU and the `cupy` and `numba` packages.
"""

import warnings

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    warnings.warn("CuPy not available. GPU acceleration disabled.")

from .manager import GPUManager
from .kernels import (
    compute_gravity_bruteforce_gpu,
    compute_density_gpu,
    compute_hydro_gpu,
    update_smoothing_lengths_gpu
)

# Import octree functionality
try:
    from .octree_gpu import (
        GPUOctree,
        find_neighbours_octree_gpu,
        compute_gravity_gpu
    )
except ImportError:
    # Octree not available
    GPUOctree = None
    find_neighbours_octree_gpu = None
    compute_gravity_gpu = None

# Import TreeSPH optimized kernels
try:
    from .treesph_kernels import (
        compute_density_treesph,
        compute_hydro_treesph
    )
except ImportError:
    compute_density_treesph = None
    compute_hydro_treesph = None

# Import optimized smoothing length kernels
try:
    from .smoothing_length_kernels import (
        update_smoothing_lengths_gpu as update_smoothing_lengths_gpu_octree
    )
except ImportError:
    update_smoothing_lengths_gpu_octree = None

# Import energy computation kernels
try:
    from .energy_kernels import (
        compute_energies_gpu,
        compute_energies_gpu_simple
    )
except ImportError:
    compute_energies_gpu = None
    compute_energies_gpu_simple = None

__all__ = [
    'HAS_CUDA',
    'GPUManager',
    'compute_gravity_bruteforce_gpu',
    'compute_density_gpu',
    'compute_hydro_gpu',
    'update_smoothing_lengths_gpu',
    'update_smoothing_lengths_gpu_octree',
    'GPUOctree',
    'find_neighbours_octree_gpu',
    'compute_gravity_gpu',
    'compute_density_treesph',
    'compute_hydro_treesph',
    'compute_energies_gpu',
    'compute_energies_gpu_simple',
]
