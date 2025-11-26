"""
GPU-accelerated Barnes-Hut gravity solver using CuPy.

Provides the same interface as the CPU BarnesHutGravity but keeps all
data on GPU to minimize PCIe transfers.
"""

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

from typing import Optional
from ..gpu.octree_gpu import GPUOctree, compute_gravity_gpu

try:
    # Try to import from the SPH framework
    from ..core.interfaces import GravitySolver, Metric, NDArrayFloat
except (ImportError, ValueError):
    # Fallback for standalone usage
    class GravitySolver:
        pass
    Metric = None
    NDArrayFloat = np.ndarray


class BarnesHutGravityGPU(GravitySolver):
    """
    GPU-accelerated Barnes-Hut gravity solver.

    Uses CuPy to keep all tree construction and force calculations on GPU.
    Minimizes PCIe transfers for optimal RTX 4090 performance.

    Parameters
    ----------
    G : float
        Gravitational constant
    theta : float
        Multipole acceptance criterion (0.5 is standard)
    epsilon : float
        Softening parameter multiplier (relative to smoothing length)
    """

    def __init__(self, G: float = 1.0, theta: float = 0.5, epsilon: float = 0.1):
        if not HAS_CUPY:
            raise ImportError("CuPy is required for GPU Barnes-Hut solver")

        self.G = float(G)
        self.theta = float(theta)
        self.epsilon = float(epsilon)

        # GPU octree (reused between calls)
        self.tree = None

        # Store last tree data for potential reuse in neighbour search
        self.last_tree_data = None

        # GPU arrays (persistent to avoid repeated transfers)
        self.positions_gpu = None
        self.masses_gpu = None
        self.smoothing_lengths_gpu = None
        self.forces_gpu = None

    def get_tree_data(self):
        """
        Get the octree data from the last tree build.

        Returns
        -------
        dict or None
            Dictionary containing GPU octree data for reuse in neighbour search.
            Contains the GPUOctree instance and metadata.
        """
        return self.last_tree_data

    def get_tree(self) -> Optional[GPUOctree]:
        """
        Get the GPU octree instance for direct reuse.

        Returns
        -------
        GPUOctree or None
            The octree instance, or None if not built yet
        """
        return self.tree

    def _ensure_gpu_arrays(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        smoothing_lengths: np.ndarray
    ) -> None:
        """Transfer data to GPU if needed."""
        n = len(positions)

        # Check if we need to reallocate
        need_realloc = (
            self.positions_gpu is None or
            self.positions_gpu.shape[0] != n
        )

        if need_realloc:
            self.positions_gpu = cp.asarray(positions, dtype=cp.float32)
            self.masses_gpu = cp.asarray(masses, dtype=cp.float32)
            self.smoothing_lengths_gpu = cp.asarray(smoothing_lengths, dtype=cp.float32)
            self.forces_gpu = cp.zeros((n, 3), dtype=cp.float32)
        else:
            # Reuse existing allocations, just copy data
            cp.copyto(self.positions_gpu, cp.asarray(positions, dtype=cp.float32))
            cp.copyto(self.masses_gpu, cp.asarray(masses, dtype=cp.float32))
            cp.copyto(self.smoothing_lengths_gpu, cp.asarray(smoothing_lengths, dtype=cp.float32))

    def compute_acceleration(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        metric: Optional[Metric] = None
    ) -> NDArrayFloat:
        """
        Compute gravitational acceleration using Barnes-Hut on GPU.

        Parameters
        ----------
        positions : ndarray
            Particle positions (N, 3)
        masses : ndarray
            Particle masses (N,)
        smoothing_lengths : ndarray
            Smoothing lengths (N,)
        metric : Metric, optional
            Metric for cosmological simulations (not used in Euclidean case)

        Returns
        -------
        acceleration : ndarray
            Gravitational acceleration (N, 3) in CPU memory
        """
        # Transfer to GPU
        self._ensure_gpu_arrays(positions, masses, smoothing_lengths)

        # Build tree and compute forces on GPU
        self.forces_gpu, self.tree = compute_gravity_gpu(
            self.positions_gpu,
            self.masses_gpu,
            self.smoothing_lengths_gpu,
            G=self.G,
            epsilon=self.epsilon,
            theta=self.theta,
            tree=self.tree  # Reuse if available
        )

        # Store tree data for neighbour search reuse
        self.last_tree_data = {
            'tree': self.tree,
            'gpu': True,
            'n_particles': len(positions),
            'statistics': self.tree.get_tree_statistics()
        }

        # Print debug info
        stats = self.tree.get_tree_statistics()
        print(f"GPU BH Tree built: {stats['n_occupied_leaves']} occupied leaves, "
              f"{stats['avg_particles_per_leaf']:.1f} avg particles/leaf")

        # Compute acceleration from forces (F = ma, so a = F/m)
        accel_gpu = self.forces_gpu / self.masses_gpu[:, None]

        # Transfer result back to CPU
        accel_cpu = cp.asnumpy(accel_gpu)

        return accel_cpu

    def compute_potential(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute gravitational potential using Barnes-Hut on GPU.

        Note: Current implementation computes forces, potential calculation
        would require a separate kernel.

        Parameters
        ----------
        positions : ndarray
            Particle positions (N, 3)
        masses : ndarray
            Particle masses (N,)
        smoothing_lengths : ndarray
            Smoothing lengths (N,)

        Returns
        -------
        potential : ndarray
            Gravitational potential (N,) in CPU memory
        """
        # For now, return zeros - would need separate kernel for potential
        # This is not critical for most SPH simulations
        return np.zeros(len(positions), dtype=np.float32)

    def get_statistics(self) -> dict:
        """
        Get statistics about the GPU tree and performance.

        Returns
        -------
        dict
            Statistics dictionary
        """
        if self.tree is None:
            return {'status': 'no tree built'}

        stats = self.tree.get_tree_statistics()
        stats['G'] = self.G
        stats['theta'] = self.theta
        stats['epsilon'] = self.epsilon

        if self.positions_gpu is not None:
            # GPU memory usage estimate
            n = self.positions_gpu.shape[0]
            mem_positions = self.positions_gpu.nbytes
            mem_masses = self.masses_gpu.nbytes
            mem_smoothing = self.smoothing_lengths_gpu.nbytes
            mem_forces = self.forces_gpu.nbytes
            mem_total = mem_positions + mem_masses + mem_smoothing + mem_forces
            stats['gpu_memory_mb'] = mem_total / (1024**2)

        return stats
