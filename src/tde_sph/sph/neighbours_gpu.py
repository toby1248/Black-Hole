"""
GPU-accelerated neighbour search using CuPy.

Provides GPU-accelerated versions of neighbour-finding functions that match
the interface of neighbours_cpu.py but keep data on GPU to minimize PCIe transfers.
"""

import numpy as np
import warnings
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

from typing import Tuple, List, Optional
from ..gpu.octree_gpu import GPUOctree, find_neighbours_octree_gpu


def find_neighbours_octree_gpu_integrated(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    tree_data: dict,
    support_radius: float = 2.0,
    max_neighbours: int = 64,
    **deprecated_kwargs,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Find neighbours using GPU octree (TreeSPH approach).

    This reuses the octree built by the GPU Barnes-Hut gravity solver.
    Matches the interface of find_neighbours_octree from neighbours_cpu.py.

    Parameters
    ----------
    positions : ndarray (n_particles, 3)
        Particle positions (CPU array)
    smoothing_lengths : ndarray (n_particles,)
        Smoothing length for each particle (CPU array)
    tree_data : dict
        Dictionary containing GPU octree data from BarnesHutGravityGPU:
        - 'tree': GPUOctree instance
        - 'gpu': True (indicates GPU data)
        - Other metadata
    support_radius : float, optional
        Support radius multiplier (default: 2.0)
    max_neighbours : int, optional
        Maximum neighbours per particle (default: 128)

    Returns
    -------
    neighbour_lists : list of arrays
        List where neighbour_lists[i] contains indices of neighbours for particle i
    neighbour_distances : ndarray
        Empty array (placeholder for compatibility)

    Notes
    -----
    - Matches signature of find_neighbours_octree() for drop-in replacement
    - Data transferred to GPU, processed, and results returned to CPU
    - Tree is reused from gravity solver to avoid rebuild
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is required for GPU neighbour search")

    # Check if this is GPU tree data
    if not tree_data.get('gpu', False):
        raise ValueError("Tree data is not from GPU solver. Use CPU neighbour search.")

    if 'max_neighbors' in deprecated_kwargs:
        warnings.warn(
            "'max_neighbors' keyword is deprecated. Use 'max_neighbours' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        deprecated_value = deprecated_kwargs.pop('max_neighbors')
        if deprecated_value is not None:
            max_neighbours = deprecated_value

    if deprecated_kwargs:
        unexpected = ', '.join(deprecated_kwargs.keys())
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    # Get the GPU octree
    tree = tree_data['tree']
    if tree is None:
        raise ValueError("GPU tree is None - build tree first")

    # Transfer data to GPU
    positions_gpu = cp.asarray(positions, dtype=cp.float32)
    smoothing_lengths_gpu = cp.asarray(smoothing_lengths, dtype=cp.float32)

    # Find neighbours on GPU
    neighbour_lists_gpu, neighbour_counts_gpu = tree.find_neighbours(
        positions_gpu,
        smoothing_lengths_gpu,
        support_radius=support_radius,
        max_neighbours=max_neighbours
    )

    # Transfer results back to CPU
    neighbour_lists_cpu = cp.asnumpy(neighbour_lists_gpu)
    neighbour_counts_cpu = cp.asnumpy(neighbour_counts_gpu)

    # Convert to list of arrays format (matching CPU version)
    neighbour_lists = []
    for i in range(len(positions)):
        count = neighbour_counts_cpu[i]
        if count > 0:
            neighbours = neighbour_lists_cpu[i, :count]
            # Filter out invalid indices (-1)
            neighbours = neighbours[neighbours >= 0]
            neighbour_lists.append(neighbours)
        else:
            neighbour_lists.append(np.array([], dtype=np.int32))

    # Return empty distances array for compatibility
    neighbour_distances = np.array([])

    return neighbour_lists, neighbour_distances


class GPUNeighbourSearchCache:
    """
    Cache for GPU neighbour search to minimize PCIe transfers.

    Keeps particle data on GPU between calls and only transfers
    when data changes.
    """

    def __init__(self, max_neighbours: int = 64):
        self.max_neighbours = max_neighbours

        # Cached GPU arrays
        self.positions_gpu = None
        self.smoothing_lengths_gpu = None
        self.neighbour_lists_gpu = None
        self.neighbour_counts_gpu = None

        # Cached tree
        self.tree = None

    @property
    def max_neighbors(self) -> int:
        warnings.warn(
            "'max_neighbors' is deprecated. Use 'max_neighbours' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.max_neighbours

    @max_neighbors.setter
    def max_neighbors(self, value: int) -> None:
        warnings.warn(
            "'max_neighbors' is deprecated. Use 'max_neighbours' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.max_neighbours = value

    def find_neighbours(
        self,
        positions: np.ndarray,
        smoothing_lengths: np.ndarray,
        support_radius: float = 2.0,
        tree: Optional[GPUOctree] = None,
        rebuild_tree: bool = False
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Find neighbours with caching to minimize transfers.

        Parameters
        ----------
        positions : ndarray
            Particle positions (N, 3)
        smoothing_lengths : ndarray
            Smoothing lengths (N,)
        support_radius : float
            Support radius multiplier
        tree : GPUOctree, optional
            Pre-built tree to reuse
        rebuild_tree : bool
            Force tree rebuild even if cached

        Returns
        -------
        neighbour_lists : list of arrays
            Neighbour indices for each particle
        neighbour_distances : ndarray
            Empty array (compatibility)
        """
        n = len(positions)

        # Check if we need to reallocate
        need_realloc = (
            self.positions_gpu is None or
            self.positions_gpu.shape[0] != n
        )

        if need_realloc:
            # Allocate new arrays
            self.positions_gpu = cp.asarray(positions, dtype=cp.float32)
            self.smoothing_lengths_gpu = cp.asarray(smoothing_lengths, dtype=cp.float32)
            self.neighbour_lists_gpu = cp.full((n, self.max_neighbours), -1, dtype=cp.int32)
            self.neighbour_counts_gpu = cp.zeros(n, dtype=cp.int32)
        else:
            # Reuse arrays, just copy data
            cp.copyto(self.positions_gpu, cp.asarray(positions, dtype=cp.float32))
            cp.copyto(self.smoothing_lengths_gpu, cp.asarray(smoothing_lengths, dtype=cp.float32))

        # Use provided tree or cached tree
        if tree is not None:
            self.tree = tree
        elif rebuild_tree or self.tree is None:
            # Build new tree
            self.tree = GPUOctree()
            self.tree.build(self.positions_gpu)

        # Find neighbours
        self.neighbour_lists_gpu, self.neighbour_counts_gpu = self.tree.find_neighbours(
            self.positions_gpu,
            self.smoothing_lengths_gpu,
            support_radius=support_radius,
            max_neighbours=self.max_neighbours
        )

        # Transfer results to CPU
        neighbour_lists_cpu = cp.asnumpy(self.neighbour_lists_gpu)
        neighbour_counts_cpu = cp.asnumpy(self.neighbour_counts_gpu)

        # Convert to list format
        neighbour_lists = []
        for i in range(n):
            count = neighbour_counts_cpu[i]
            if count > 0:
                neighbours = neighbour_lists_cpu[i, :count]
                neighbours = neighbours[neighbours >= 0]
                neighbour_lists.append(neighbours)
            else:
                neighbour_lists.append(np.array([], dtype=np.int32))

        return neighbour_lists, np.array([])


class GPUNeighborSearchCache(GPUNeighbourSearchCache):
    """Compatibility alias for American spelling."""

    def __init__(self, *args, **kwargs):  # type: ignore[override]
        warnings.warn(
            "'GPUNeighborSearchCache' is deprecated. Use 'GPUNeighbourSearchCache' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def find_neighbors(
        self,
        positions: np.ndarray,
        smoothing_lengths: np.ndarray,
        support_radius: float = 2.0,
        tree: Optional[GPUOctree] = None,
        rebuild_tree: bool = False
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Compatibility shim for American spelling."""
        warnings.warn(
            "'find_neighbors' is deprecated. Use 'find_neighbours' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_neighbours(
            positions=positions,
            smoothing_lengths=smoothing_lengths,
            support_radius=support_radius,
            tree=tree,
            rebuild_tree=rebuild_tree,
        )

    def get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB."""
        if self.positions_gpu is None:
            return 0.0

        total = (
            self.positions_gpu.nbytes +
            self.smoothing_lengths_gpu.nbytes +
            self.neighbour_lists_gpu.nbytes +
            self.neighbour_counts_gpu.nbytes
        )
        return total / (1024**2)


# Convenience function matching CPU API
def find_neighbours_gpu(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    support_radius: float = 2.0,
    max_neighbours: int = 64,
    use_octree: bool = True,
    tree: Optional[GPUOctree] = None,
    **deprecated_kwargs,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Find neighbours using GPU acceleration.

    Parameters
    ----------
    positions : ndarray
        Particle positions (N, 3)
    smoothing_lengths : ndarray
        Smoothing lengths (N,)
    support_radius : float
        Support radius multiplier
    max_neighbours : int
        Maximum neighbours per particle
    use_octree : bool
        Use octree (True) or brute force (False)
    tree : GPUOctree, optional
        Pre-built tree to reuse

    Returns
    -------
    neighbour_lists : list of arrays
        Neighbour indices
    neighbour_distances : ndarray
        Empty array
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is required for GPU neighbour search")

    if 'max_neighbors' in deprecated_kwargs:
        warnings.warn(
            "'max_neighbors' keyword is deprecated. Use 'max_neighbours' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        deprecated_value = deprecated_kwargs.pop('max_neighbors')
        if deprecated_value is not None:
            max_neighbours = deprecated_value

    if deprecated_kwargs:
        unexpected = ', '.join(deprecated_kwargs.keys())
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    if not use_octree:
        # Could implement GPU brute force here
        raise NotImplementedError("GPU brute force not implemented yet")

    # Use octree
    positions_gpu = cp.asarray(positions, dtype=cp.float32)
    smoothing_lengths_gpu = cp.asarray(smoothing_lengths, dtype=cp.float32)

    if tree is None:
        tree = GPUOctree()
        tree.build(positions_gpu)

    neighbour_lists_gpu, neighbour_counts_gpu = tree.find_neighbours(
        positions_gpu,
        smoothing_lengths_gpu,
        support_radius=support_radius,
        max_neighbours=max_neighbours
    )

    # Transfer back
    neighbour_lists_cpu = cp.asnumpy(neighbour_lists_gpu)
    neighbour_counts_cpu = cp.asnumpy(neighbour_counts_gpu)

    # Convert format
    neighbour_lists = []
    for i in range(len(positions)):
        count = neighbour_counts_cpu[i]
        if count > 0:
            neighbours = neighbour_lists_cpu[i, :count]
            neighbours = neighbours[neighbours >= 0]
            neighbour_lists.append(neighbours)
        else:
            neighbour_lists.append(np.array([], dtype=np.int32))

    return neighbour_lists, np.array([])
