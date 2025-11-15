"""
CPU-based neighbour search for SPH particles.

This module implements a simple O(N²) pairwise distance neighbour search
as a baseline implementation. This is suitable for small particle counts
(N < 10⁵) and serves as a reference for testing more advanced methods.

Future optimizations (TASK-029):
- GPU-accelerated uniform grid/hash
- Tree-based search (octree/kd-tree)
- Cell lists with spatial hashing

Design follows REQ-001 (SPH core) with vectorized numpy operations.
"""

import numpy as np
import numpy.typing as npt
from typing import List, Tuple

# Type alias for clarity
NDArrayFloat = npt.NDArray[np.float32]


def find_neighbours_bruteforce(
    positions: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    support_radius: float = 2.0
) -> Tuple[List[npt.NDArray[np.int32]], NDArrayFloat]:
    """
    Find neighbours within kernel support using brute-force pairwise search.

    For each particle i, finds all particles j where |r_i - r_j| < support_radius × h_i.
    This is the standard SPH neighbour criterion (Price 2012, Section 3.1).

    Parameters
    ----------
    positions : NDArrayFloat, shape (N, 3)
        Particle positions.
    smoothing_lengths : NDArrayFloat, shape (N,)
        Smoothing lengths h for each particle.
    support_radius : float, optional
        Kernel support radius in units of h (default 2.0 for cubic spline).

    Returns
    -------
    neighbour_lists : List[NDArray[int32]]
        List of neighbour indices for each particle.
        neighbour_lists[i] contains indices j of all neighbours of particle i.
    neighbour_distances : NDArrayFloat, shape matches flattened neighbours
        Distances |r_i - r_j| for all neighbour pairs (for efficient reuse).

    Notes
    -----
    This is an O(N²) algorithm suitable for small systems or testing.
    For large N (> 10⁵), use tree-based or grid-based methods (TASK-029).

    References
    ----------
    Price, D. J. (2012), "Smoothed particle hydrodynamics and
    magnetohydrodynamics", Journal of Computational Physics, 231, 759.

    Examples
    --------
    >>> positions = np.random.randn(100, 3).astype(np.float32)
    >>> h = np.full(100, 0.1, dtype=np.float32)
    >>> neighbours, distances = find_neighbours_bruteforce(positions, h)
    >>> print(f"Particle 0 has {len(neighbours[0])} neighbours")
    """
    n_particles = positions.shape[0]
    neighbour_lists = []

    # Store all distances for efficient reuse (optional optimization)
    # For now, we'll recompute in hydro_forces to keep this function simple

    for i in range(n_particles):
        # Compute distance vectors to all other particles
        r_ij_vec = positions[i] - positions  # Shape (N, 3)
        r_ij = np.linalg.norm(r_ij_vec, axis=1).astype(np.float32)  # Shape (N,)

        # Find neighbours within support radius
        # Use the maximum of h_i and h_j for symmetry (standard SPH practice)
        h_max = np.maximum(smoothing_lengths[i], smoothing_lengths)
        support_dist = support_radius * h_max

        # Neighbour criterion: r_ij < support_radius × h
        # Exclude self (r_ij > 0)
        is_neighbour = (r_ij > 0) & (r_ij < support_dist)
        neighbour_indices = np.where(is_neighbour)[0].astype(np.int32)

        neighbour_lists.append(neighbour_indices)

    # Placeholder for distances (can be enhanced later)
    neighbour_distances = np.array([], dtype=np.float32)

    return neighbour_lists, neighbour_distances


def compute_density_summation(
    positions: NDArrayFloat,
    masses: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    neighbour_lists: List[npt.NDArray[np.int32]],
    kernel_func: callable
) -> NDArrayFloat:
    """
    Compute SPH density using neighbour lists and kernel summation.

    Implements the standard SPH density estimator:
        ρ_i = ∑_j m_j W(|r_i - r_j|, h_i)

    Parameters
    ----------
    positions : NDArrayFloat, shape (N, 3)
        Particle positions.
    masses : NDArrayFloat, shape (N,)
        Particle masses.
    smoothing_lengths : NDArrayFloat, shape (N,)
        Smoothing lengths.
    neighbour_lists : List[NDArray[int32]]
        Neighbour indices from find_neighbours_bruteforce.
    kernel_func : callable
        Kernel function W(r, h). Should accept (r, h) and return scalar/array.

    Returns
    -------
    density : NDArrayFloat, shape (N,)
        Computed densities ρ_i.

    Notes
    -----
    Following Price (2012), the density includes the particle's self-contribution
    (W(0, h_i) term) for better behaved density estimates.
    """
    n_particles = positions.shape[0]
    density = np.zeros(n_particles, dtype=np.float32)

    for i in range(n_particles):
        # Self-contribution
        W_self = kernel_func(0.0, smoothing_lengths[i])
        rho_i = masses[i] * W_self

        # Neighbour contributions
        neighbours = neighbour_lists[i]
        if len(neighbours) > 0:
            r_ij_vec = positions[i] - positions[neighbours]
            r_ij = np.linalg.norm(r_ij_vec, axis=1).astype(np.float32)

            W_ij = kernel_func(r_ij, smoothing_lengths[i])
            rho_i += np.sum(masses[neighbours] * W_ij)

        density[i] = rho_i

    return density


def update_smoothing_lengths(
    positions: NDArrayFloat,
    masses: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    target_neighbours: int = 50,
    max_iterations: int = 10,
    tolerance: float = 0.05
) -> NDArrayFloat:
    """
    Update smoothing lengths to maintain target neighbour count.

    Iteratively adjusts h_i so that each particle has approximately
    target_neighbours neighbours within the kernel support (adaptive h).
    This is essential for modern SPH (Price 2012, Section 3.2).

    Parameters
    ----------
    positions : NDArrayFloat, shape (N, 3)
        Particle positions.
    masses : NDArrayFloat, shape (N,)
        Particle masses (used for convergence criterion).
    smoothing_lengths : NDArrayFloat, shape (N,)
        Initial smoothing lengths.
    target_neighbours : int, optional
        Target number of neighbours (default 50, typical for 3D SPH).
    max_iterations : int, optional
        Maximum number of iterations (default 10).
    tolerance : float, optional
        Fractional tolerance for convergence (default 0.05 = 5%).

    Returns
    -------
    h_new : NDArrayFloat, shape (N,)
        Updated smoothing lengths.

    Notes
    -----
    Uses a simple Newton-Raphson-like iteration:
        h_new = h_old × (n_target / n_actual)^(1/3)
    This assumes roughly uniform density locally (3D scaling).

    For production code, consider more sophisticated schemes from
    Price & Monaghan (2007) or PHANTOM implementation.

    References
    ----------
    Price, D. J., & Monaghan, J. J. (2007), "An energy-conserving
    formalism for adaptive gravitational force softening in smoothed
    particle hydrodynamics and N-body codes", MNRAS, 374, 1347.
    """
    h_new = smoothing_lengths.copy()
    n_particles = positions.shape[0]

    for iteration in range(max_iterations):
        # Find current neighbours
        neighbour_lists, _ = find_neighbours_bruteforce(positions, h_new)

        # Count neighbours for each particle
        n_neighbours = np.array([len(nb) for nb in neighbour_lists], dtype=np.float32)

        # Check convergence
        frac_error = np.abs(n_neighbours - target_neighbours) / target_neighbours
        if np.max(frac_error) < tolerance:
            break

        # Update h using 3D scaling: h ∝ n^(-1/3)
        # h_new = h_old × (n_target / n_actual)^(1/3)
        ratio = np.maximum(n_neighbours, 1.0) / target_neighbours
        h_new *= ratio**(-1.0 / 3.0)

        # Clamp to reasonable range to avoid instabilities
        h_new = np.clip(h_new, 1e-6, 1e2)

    return h_new


# TODO (TASK-029): Implement GPU-accelerated neighbour search
# - Uniform grid with spatial hashing (O(N) average case)
# - Octree or kd-tree for non-uniform distributions
# - CUDA kernels for parallel distance computation
# - Benchmark against CPU version for crossover point
#
# Example API:
# def find_neighbours_gpu(positions, smoothing_lengths, support_radius=2.0):
#     """GPU-accelerated neighbour search using uniform grid."""
#     # Implementation using CuPy/Numba CUDA
#     pass
