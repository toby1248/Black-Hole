"""
GPU acceleration module using CUDA (via Numba/CuPy).

This module provides GPU-accelerated implementations of:
- Gravity (N-body)
- SPH Neighbour Search
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
