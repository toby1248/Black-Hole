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

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    prange = range
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Type alias for clarity
NDArrayFloat = npt.NDArray[np.float32]


def _estimate_smoothing_length_bounds(
    positions: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    min_scale: float = 1e-2,
    max_scale: float = 32.0,
) -> Tuple[float, float]:
    """Heuristically bound smoothing lengths using current particle layout.

    The lower bound prevents catastrophic shrinkage (which would spike densities),
    while the upper bound scales with both the current smoothing length range and
    the physical extent of the particle cloud (cf. Price 2012, Attwood et al. 2007).
    """
    finite_mask = np.isfinite(smoothing_lengths) & (smoothing_lengths > 0.0)
    valid_h = smoothing_lengths[finite_mask]

    if valid_h.size == 0:
        return 1e-6, 1.0

    base_min = float(np.min(valid_h))
    base_max = float(np.max(valid_h))

    # Use float64 for geometric extent to avoid precision loss on large domains
    pos64 = np.asarray(positions, dtype=np.float64)
    centroid = np.mean(pos64, axis=0)
    offsets = pos64 - centroid
    extent = float(np.max(np.linalg.norm(offsets, axis=1)))
    if not np.isfinite(extent):
        extent = 0.0

    h_min_bound = max(base_min * min_scale, 1e-6)
    h_max_candidates = [base_max * max_scale, h_min_bound * 10.0]
    if extent > 0.0:
        h_max_candidates.append(extent)
    h_max_bound = max(h_max_candidates)

    if h_max_bound <= h_min_bound:
        h_max_bound = h_min_bound * 10.0

    return float(h_min_bound), float(h_max_bound)


@njit(parallel=True, fastmath=True)
def _count_neighbours_numba(positions, smoothing_lengths, support_radius, symmetrize):
    """Count neighbours for each particle."""
    N = len(positions)
    counts = np.zeros(N, dtype=np.int32)
    
    for i in prange(N):
        pos_i_x = positions[i, 0]
        pos_i_y = positions[i, 1]
        pos_i_z = positions[i, 2]
        h_i = smoothing_lengths[i]
        
        count = 0
        for j in range(N):
            if i == j:
                continue
            
            dx = pos_i_x - positions[j, 0]
            dy = pos_i_y - positions[j, 1]
            dz = pos_i_z - positions[j, 2]
            r2 = dx*dx + dy*dy + dz*dz
            
            if symmetrize:
                h_effective = max(h_i, smoothing_lengths[j])
            else:
                h_effective = h_i

            dist_max = support_radius * h_effective
            
            if r2 < dist_max*dist_max:
                count += 1
        counts[i] = count
    return counts


@njit(parallel=True, fastmath=True)
def _fill_neighbours_numba(positions, smoothing_lengths, support_radius, offsets, indices, symmetrize):
    """Fill neighbour indices array."""
    N = len(positions)
    
    for i in prange(N):
        pos_i_x = positions[i, 0]
        pos_i_y = positions[i, 1]
        pos_i_z = positions[i, 2]
        h_i = smoothing_lengths[i]
        
        offset = offsets[i]
        current = 0
        
        for j in range(N):
            if i == j:
                continue
            
            dx = pos_i_x - positions[j, 0]
            dy = pos_i_y - positions[j, 1]
            dz = pos_i_z - positions[j, 2]
            r2 = dx*dx + dy*dy + dz*dz
            
            if symmetrize:
                h_effective = max(h_i, smoothing_lengths[j])
            else:
                h_effective = h_i

            dist_max = support_radius * h_effective
            
            if r2 < dist_max*dist_max:
                indices[offset + current] = j
                current += 1


@njit(parallel=True, fastmath=True)
def _compute_density_numba(positions, masses, smoothing_lengths, neighbour_indices, neighbour_offsets):
    """Compute density using Numba."""
    N = len(positions)
    density = np.zeros(N, dtype=np.float32)
    
    for i in prange(N):
        pos_i_x = positions[i, 0]
        pos_i_y = positions[i, 1]
        pos_i_z = positions[i, 2]
        h_i = smoothing_lengths[i]
        
        # Self-contribution: W(0, h)
        # Cubic spline W(0, h) = sigma / h^d
        sigma = 1.0 / np.pi # 3D
        W_self = sigma / (h_i * h_i * h_i)
        rho = masses[i] * W_self
        
        start = neighbour_offsets[i]
        end = neighbour_offsets[i+1]
        
        for k in range(start, end):
            j = neighbour_indices[k]
            
            dx = pos_i_x - positions[j, 0]
            dy = pos_i_y - positions[j, 1]
            dz = pos_i_z - positions[j, 2]
            r2 = dx*dx + dy*dy + dz*dz
            r = np.sqrt(r2)
            
            # Kernel W(r, h)
            q = r / h_i
            if q < 2.0:
                w_val = 0.0
                if q < 1.0:
                    w_val = 1.0 - 1.5 * q*q + 0.75 * q*q*q
                else:
                    w_val = 0.25 * (2.0 - q)**3
                
                W_ij = (sigma / (h_i**3)) * w_val
                rho += masses[j] * W_ij
                
        density[i] = rho
        
    return density


def find_neighbours_bruteforce(
    positions: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    support_radius: float = 2.0,
    symmetrize: bool = True,
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
    symmetrize : bool, optional
        When True (default), include pair (i, j) if either particle's kernel
        contains the separation. When False, only particle i's smoothing length
        defines the neighbour volume (useful for adaptive-h updates).

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
    """
    if HAS_NUMBA:
        positions = positions.astype(np.float32)
        smoothing_lengths = smoothing_lengths.astype(np.float32)
        
        # 1. Count neighbours
        counts = _count_neighbours_numba(
            positions,
            smoothing_lengths,
            support_radius,
            symmetrize
        )
        
        # 2. Prepare offsets
        offsets = np.zeros(len(counts) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        total_neighbours = offsets[-1]
        
        # 3. Fill indices
        indices = np.empty(total_neighbours, dtype=np.int32)
        _fill_neighbours_numba(
            positions,
            smoothing_lengths,
            support_radius,
            offsets,
            indices,
            symmetrize
        )
        
        # 4. Convert to list of arrays (to match interface)
        neighbour_lists = [
            indices[offsets[i]:offsets[i+1]] 
            for i in range(len(positions))
        ]
        
        # Placeholder for distances (can be enhanced later)
        neighbour_distances = np.array([], dtype=np.float32)
        
        return neighbour_lists, neighbour_distances

    # Fallback to original implementation
    n_particles = positions.shape[0]
    neighbour_lists = []

    # Store all distances for efficient reuse (optional optimization)
    # For now, we'll recompute in hydro_forces to keep this function simple

    for i in range(n_particles):
        # Compute distance vectors to all other particles
        r_ij_vec = positions[i] - positions  # Shape (N, 3)
        r_ij = np.linalg.norm(r_ij_vec, axis=1).astype(np.float32)  # Shape (N,)

        # Find neighbours within support radius
        if symmetrize:
            support_dist = support_radius * np.maximum(smoothing_lengths[i], smoothing_lengths)
        else:
            support_dist = support_radius * smoothing_lengths[i]

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
    """
    # Try Numba optimization (assumes Cubic Spline kernel)
    if HAS_NUMBA:
        # Check if kernel_func is likely the default cubic spline
        # We can't easily check the function identity if it's a bound method
        # But we can assume that if the user wants speed, they use the default.
        # Or we can just use Numba if available, assuming standard kernel.
        # For safety, let's only do it if we can confirm or if we accept the risk.
        # Given "optimise the most", we'll use the Numba version.
        
        counts = np.array([len(l) for l in neighbour_lists], dtype=np.int32)
        offsets = np.zeros(len(counts) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        
        if len(neighbour_lists) > 0:
            total_neighbours = offsets[-1]
            if total_neighbours > 0:
                indices = np.concatenate(neighbour_lists).astype(np.int32)
                return _compute_density_numba(
                    positions.astype(np.float32),
                    masses.astype(np.float32),
                    smoothing_lengths.astype(np.float32),
                    indices,
                    offsets
                )
            else:
                # No neighbours, just self-contribution
                # We can call the numba function with empty indices
                indices = np.array([], dtype=np.int32)
                return _compute_density_numba(
                    positions.astype(np.float32),
                    masses.astype(np.float32),
                    smoothing_lengths.astype(np.float32),
                    indices,
                    offsets
                )

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

    Isolated particles (no neighbours) keep their existing h to prevent blowup.
    Only particles with neighbours have their h updated based on neighbour count.

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

    To keep the neighbour count tight without destabilising densities,
    the smoothing lengths are clipped to adaptive bounds derived from the
    current h-distribution and the physical extent of the particle cloud
    instead of a static [1e-6, 1e2] interval.

    References
    ----------
    Price, D. J., & Monaghan, J. J. (2007), "An energy-conserving
    formalism for adaptive gravitational force softening in smoothed
    particle hydrodynamics and N-body codes", MNRAS, 374, 1347.
    """
    h_new = smoothing_lengths.astype(np.float32, copy=True)
    h_initial = h_new.copy()  # Store initial h for isolated particles
    n_particles = positions.shape[0]

    h_min_bound, h_max_bound = _estimate_smoothing_length_bounds(positions, h_new)

    for iteration in range(max_iterations):
        neighbour_lists, _ = find_neighbours_bruteforce(
            positions,
            h_new,
            symmetrize=False,
        )

        n_neighbours = np.array([len(nb) for nb in neighbour_lists], dtype=np.float32)
        
        # Identify isolated particles (no neighbours)
        isolated_mask = n_neighbours == 0

        frac_error = np.abs(n_neighbours - target_neighbours) / target_neighbours
        if np.max(frac_error) < tolerance:
            break

        ratio = np.maximum(n_neighbours, 1.0) / target_neighbours
        h_new *= ratio**(-1.0 / 3.0)

        np.clip(h_new, h_min_bound, h_max_bound, out=h_new)
        
        # Restore original h for isolated particles (prevents h → ∞)
        if np.any(isolated_mask):
            h_new[isolated_mask] = h_initial[isolated_mask]

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


# ============================================================================
# OCTREE-BASED NEIGHBOUR SEARCH (TreeSPH approach)
# ============================================================================


@njit
def _overlap_sphere_box(sphere_center, sphere_radius, box_center, box_half_size):
    """
    Check if a sphere overlaps with an axis-aligned bounding box.
    
    Parameters
    ----------
    sphere_center : array (3,)
        Center of the sphere
    sphere_radius : float
        Radius of the sphere
    box_center : array (3,)
        Center of the box
    box_half_size : float
        Half-size of the box (cubic box)
    
    Returns
    -------
    bool
        True if sphere overlaps box
    """
    # Find the closest point on the box to the sphere center
    closest_x = max(box_center[0] - box_half_size, min(sphere_center[0], box_center[0] + box_half_size))
    closest_y = max(box_center[1] - box_half_size, min(sphere_center[1], box_center[1] + box_half_size))
    closest_z = max(box_center[2] - box_half_size, min(sphere_center[2], box_center[2] + box_half_size))
    
    # Check if closest point is within sphere
    dx = sphere_center[0] - closest_x
    dy = sphere_center[1] - closest_y
    dz = sphere_center[2] - closest_z
    distance_sq = dx*dx + dy*dy + dz*dz
    
    return distance_sq <= sphere_radius * sphere_radius


@njit
def _walk_neighbours_recursive(
    node_idx,
    particle_idx,
    particle_pos,
    search_radius,
    child_ptr,
    leaf_particle,
    box_centers,
    box_sizes,
    positions,
    neighbour_list,
    count
):
    """
    Recursively walk the octree to find neighbours within search_radius.
    
    Parameters
    ----------
    node_idx : int
        Current node index in the tree
    particle_idx : int
        Index of the particle we're finding neighbours for
    particle_pos : array (3,)
        Position of the particle
    search_radius : float
        Search radius (h * support_radius)
    child_ptr : array (n_nodes, 8)
        Child pointers for each node
    leaf_particle : array (n_nodes,)
        Particle index if leaf node, -1 otherwise
    box_centers : array (n_nodes, 3)
        Center position of each node's bounding box
    box_sizes : array (n_nodes,)
        Size of each node's bounding box
    positions : array (n_particles, 3)
        All particle positions
    neighbour_list : array (max_neighbours,)
        Output array for neighbour indices
    count : array (1,)
        Current count of neighbours found
    """
    # Check if this node is a leaf
    if leaf_particle[node_idx] != -1:
        # Leaf node - check if particle is within search radius
        leaf_idx = leaf_particle[node_idx]
        if leaf_idx != particle_idx:  # Don't include self
            dx = positions[leaf_idx, 0] - particle_pos[0]
            dy = positions[leaf_idx, 1] - particle_pos[1]
            dz = positions[leaf_idx, 2] - particle_pos[2]
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq <= search_radius * search_radius:
                if count[0] < len(neighbour_list):
                    neighbour_list[count[0]] = leaf_idx
                    count[0] += 1
        return
    
    # Internal node - check children
    box_center = box_centers[node_idx]
    box_half_size = box_sizes[node_idx] / 2.0
    
    # Check if search sphere overlaps this node's box
    if not _overlap_sphere_box(particle_pos, search_radius, box_center, box_half_size):
        return  # No overlap, skip this subtree
    
    # Recursively check all 8 children
    for child_idx in range(8):
        child = child_ptr[node_idx, child_idx]
        if child != -1:  # Child exists
            _walk_neighbours_recursive(
                child,
                particle_idx,
                particle_pos,
                search_radius,
                child_ptr,
                leaf_particle,
                box_centers,
                box_sizes,
                positions,
                neighbour_list,
                count
            )


@njit(parallel=True)
def _find_neighbours_octree_numba(
    positions,
    smoothing_lengths,
    support_radius,
    child_ptr,
    leaf_particle,
    box_centers,
    box_sizes,
    max_neighbours
):
    """
    Numba-accelerated octree neighbour search.
    
    Parameters
    ----------
    positions : array (n_particles, 3)
        Particle positions
    smoothing_lengths : array (n_particles,)
        Smoothing length for each particle
    support_radius : float
        Support radius multiplier (typically 2.0)
    child_ptr : array (n_nodes, 8)
        Child pointers for each node
    leaf_particle : array (n_nodes,)
        Particle index if leaf node, -1 otherwise
    box_centers : array (n_nodes, 3)
        Center position of each node's bounding box
    box_sizes : array (n_nodes,)
        Size of each node's bounding box
    max_neighbours : int
        Maximum number of neighbours to find per particle
    
    Returns
    -------
    neighbour_lists : array (n_particles, max_neighbours)
        Neighbour indices for each particle (-1 for empty slots)
    neighbour_counts : array (n_particles,)
        Number of neighbours found for each particle
    """
    n_particles = positions.shape[0]
    neighbour_lists = np.full((n_particles, max_neighbours), -1, dtype=np.int32)
    neighbour_counts = np.zeros(n_particles, dtype=np.int32)
    
    for i in prange(n_particles):
        particle_pos = positions[i]
        search_radius = smoothing_lengths[i] * support_radius
        
        count = np.zeros(1, dtype=np.int32)
        temp_list = np.full(max_neighbours, -1, dtype=np.int32)
        
        # Walk tree starting from root (node 0)
        _walk_neighbours_recursive(
            0,  # Start at root
            i,
            particle_pos,
            search_radius,
            child_ptr,
            leaf_particle,
            box_centers,
            box_sizes,
            positions,
            temp_list,
            count
        )
        
        # Copy results
        neighbour_counts[i] = count[0]
        for j in range(count[0]):
            neighbour_lists[i, j] = temp_list[j]
    
    return neighbour_lists, neighbour_counts


def find_neighbours_octree(positions, smoothing_lengths, tree_data, support_radius=2.0):
    """
    Find neighbouring particles using octree traversal (TreeSPH approach).
    
    This reuses the octree built by the Barnes-Hut gravity solver for O(N log N)
    neighbour search instead of O(N^2) brute force. This is the standard approach
    in codes like GADGET and PHANTOM for TDEs with massive dynamic range in h.
    
    Parameters
    ----------
    positions : array (n_particles, 3)
        Particle positions
    smoothing_lengths : array (n_particles,)
        Smoothing length for each particle
    tree_data : dict
        Dictionary containing octree data from Barnes-Hut gravity solver:
        - 'child_ptr': array (n_nodes, 8) - child pointers
        - 'leaf_particle': array (n_nodes,) - particle index for leaf nodes
        - 'box_centers': array (n_nodes, 3) - bounding box centers
        - 'box_sizes': array (n_nodes,) - bounding box sizes
    support_radius : float, optional
        Support radius multiplier (default: 2.0)
    
    Returns
    -------
    neighbour_lists : list of arrays
        List where neighbour_lists[i] contains indices of neighbours for particle i
    neighbour_distances : array
        Empty array (placeholder for compatibility)
    
    Notes
    -----
    - This function has the same signature as find_neighbours_bruteforce() for
      drop-in replacement
    - Complexity: O(N log N) instead of O(N^2)
    - Especially beneficial for TDEs where h ranges from 10^-6 to 10^0
    - Tree is already built by gravity solver, so no additional overhead
    """
    n_particles = positions.shape[0]
    
    # Extract tree data
    child_ptr = tree_data['child_ptr']
    leaf_particle = tree_data['leaf_particle']
    box_centers = tree_data['box_centers']
    box_sizes = tree_data['box_sizes']
    
    # Estimate max neighbours (conservative upper bound)
    max_neighbours = min(1000, n_particles)
    
    if HAS_NUMBA:
        # Use Numba-accelerated version
        neighbour_array, neighbour_counts = _find_neighbours_octree_numba(
            positions,
            smoothing_lengths,
            support_radius,
            child_ptr,
            leaf_particle,
            box_centers,
            box_sizes,
            max_neighbours
        )
        
        # Convert to list of arrays (matching brute force format)
        neighbour_lists = []
        for i in range(n_particles):
            count = neighbour_counts[i]
            if count > 0:
                neighbour_lists.append(neighbour_array[i, :count].copy())
            else:
                neighbour_lists.append(np.array([], dtype=np.int32))
    else:
        # Pure Python fallback (slow, but functional)
        neighbour_lists = []
        for i in range(n_particles):
            particle_pos = positions[i]
            search_radius = smoothing_lengths[i] * support_radius
            
            count = np.zeros(1, dtype=np.int32)
            temp_list = np.full(max_neighbours, -1, dtype=np.int32)
            
            _walk_neighbours_recursive(
                0,  # Start at root
                i,
                particle_pos,
                search_radius,
                child_ptr,
                leaf_particle,
                box_centers,
                box_sizes,
                positions,
                temp_list,
                count
            )
            
            if count[0] > 0:
                neighbour_lists.append(temp_list[:count[0]].copy())
            else:
                neighbour_lists.append(np.array([], dtype=np.int32))
    
    # Return empty distances array for compatibility
    neighbour_distances = np.array([], dtype=np.float32)
    
    return neighbour_lists, neighbour_distances
