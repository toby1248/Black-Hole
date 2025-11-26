"""
GPU-accelerated smoothing length updates using octree-based neighbor counting.

Provides O(N log N) smoothing length adaptation by counting neighbors via
octree traversal, avoiding O(N²) brute force searches.
"""
import numpy as np
import cupy as cp
from typing import Tuple

SMOOTHING_LENGTH_CUDA_SOURCE = r'''
extern "C" __global__
void count_neighbors_octree(
    const float* positions,
    const float* smoothing_lengths,
    const int* sorted_indices,
    const int* internal_children_left,
    const int* internal_children_right,
    const float* leaf_box_min,
    const float* leaf_box_max,
    const float* internal_box_min,
    const float* internal_box_max,
    int* neighbor_counts,
    int n_particles,
    int n_internal,
    float support_radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];
    float h = smoothing_lengths[idx];
    float search_r = support_radius * h;
    float search_r2 = search_r * search_r;

    int count = 0;
    
    // Stack for octree traversal
    const int MAX_STACK = 128;
    int stack[MAX_STACK];
    int stack_ptr = 0;
    
    // Start at root
    if (n_internal > 0) {
        stack[stack_ptr++] = 0;
    }
    
    // Safety counter
    int iterations = 0;
    const int MAX_ITERATIONS = n_particles * 10;
    
    while (stack_ptr > 0 && iterations < MAX_ITERATIONS) {
        iterations++;
        
        if (stack_ptr >= MAX_STACK) {
            break;
        }
        
        int node = stack[--stack_ptr];
        
        if (node < 0) continue;
        
        // Check if internal or leaf node
        if (node < n_internal) {
            // Internal node - check bounding box overlap
            float box_min_x = internal_box_min[node * 3 + 0];
            float box_min_y = internal_box_min[node * 3 + 1];
            float box_min_z = internal_box_min[node * 3 + 2];
            float box_max_x = internal_box_max[node * 3 + 0];
            float box_max_y = internal_box_max[node * 3 + 1];
            float box_max_z = internal_box_max[node * 3 + 2];
            
            // Sphere-box intersection test
            float dx = fmaxf(box_min_x - px, fmaxf(0.0f, px - box_max_x));
            float dy = fmaxf(box_min_y - py, fmaxf(0.0f, py - box_max_y));
            float dz = fmaxf(box_min_z - pz, fmaxf(0.0f, pz - box_max_z));
            float dist2 = dx*dx + dy*dy + dz*dz;
            
            if (dist2 <= search_r2) {
                // Add children to stack
                int left = internal_children_left[node];
                int right = internal_children_right[node];
                
                if (left >= 0 && left < n_internal + n_particles && stack_ptr < MAX_STACK) {
                    stack[stack_ptr++] = left;
                }
                if (right >= 0 && right < n_internal + n_particles && stack_ptr < MAX_STACK) {
                    stack[stack_ptr++] = right;
                }
            }
        } else {
            // Leaf node - count particles within search radius
            int leaf_id = node - n_internal;
            if (leaf_id < 0 || leaf_id >= n_particles) continue;
            
            int other_idx = sorted_indices[leaf_id];
            
            if (other_idx != idx && other_idx >= 0 && other_idx < n_particles) {
                float ox = positions[other_idx * 3 + 0];
                float oy = positions[other_idx * 3 + 1];
                float oz = positions[other_idx * 3 + 2];
                
                float dx = ox - px;
                float dy = oy - py;
                float dz = oz - pz;
                float r2 = dx*dx + dy*dy + dz*dz;
                
                if (r2 <= search_r2) {
                    count++;
                }
            }
        }
    }

    neighbor_counts[idx] = count;
}

extern "C" __global__
void update_h_from_counts(
    float* smoothing_lengths,
    const int* neighbor_counts,
    int target_neighbors,
    int n_particles,
    float h_min,
    float h_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;
    
    int n_actual = neighbor_counts[idx];
    float h_old = smoothing_lengths[idx];
    
    // 3D scaling: h_new = h_old * (n_target / n_actual)^(-1/3)
    // NO damping - apply full correction like working CPU/GPU algorithms
    float ratio = fmaxf((float)n_actual, 1.0f) / (float)target_neighbors;
    float scale = powf(ratio, -1.0f / 3.0f);
    
    float h_new = h_old * scale;
    
    // Clamp to reasonable range
    h_new = fminf(fmaxf(h_new, h_min), h_max);
    
    smoothing_lengths[idx] = h_new;
}

extern "C" __global__
void check_convergence(
    const int* neighbor_counts,
    int target_neighbors,
    float tolerance,
    int* converged_flags,
    int n_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;
    
    int n_actual = neighbor_counts[idx];
    float frac_error = fabsf((float)n_actual - (float)target_neighbors) / (float)target_neighbors;
    
    converged_flags[idx] = (frac_error < tolerance) ? 1 : 0;
}
'''

# Compile CUDA kernels
_smoothing_module = cp.RawModule(code=SMOOTHING_LENGTH_CUDA_SOURCE)
_count_neighbors_kernel = _smoothing_module.get_function('count_neighbors_octree')
_update_h_kernel = _smoothing_module.get_function('update_h_from_counts')
_check_convergence_kernel = _smoothing_module.get_function('check_convergence')


def _estimate_dynamic_h_bounds(
    positions: cp.ndarray,
    smoothing_lengths: cp.ndarray,
    min_scale: float = 1e-2,
    max_scale: float = 32.0
) -> Tuple[float, float]:
    """
    Compute adaptive smoothing-length bounds from current particle distribution.
    
    This is the KEY to stability - bounds scale with actual particle layout,
    not static values. Matches working CPU/GPU algorithm.
    """
    valid_mask = cp.logical_and(cp.isfinite(smoothing_lengths), smoothing_lengths > 0.0)
    if not cp.any(valid_mask):
        return 1e-6, 1.0
    
    h_valid = smoothing_lengths[valid_mask]
    base_min = float(cp.min(h_valid).get())
    base_max = float(cp.max(h_valid).get())
    
    # Physical extent of particle cloud
    centroid = cp.mean(positions, axis=0)
    offsets = positions - centroid
    extent = float(cp.max(cp.linalg.norm(offsets, axis=1)).get())
    if not np.isfinite(extent):
        extent = 0.0
    
    # Dynamic bounds that scale with distribution
    h_min_bound = max(base_min * min_scale, 1e-6)
    h_max_candidates = [base_max * max_scale, h_min_bound * 10.0]
    if extent > 0.0:
        h_max_candidates.append(extent)
    h_max_bound = max(h_max_candidates)
    
    if h_max_bound <= h_min_bound:
        h_max_bound = h_min_bound * 10.0
    
    return float(h_min_bound), float(h_max_bound)


def update_smoothing_lengths_gpu(
    positions: cp.ndarray,
    smoothing_lengths: cp.ndarray,
    target_neighbours: int = 50,
    tolerance: float = 0.05,
    max_iter: int = 10,
    h_min: float = None,
    h_max: float = None,
    support_radius: float = 2.0,
    octree = None
) -> cp.ndarray:
    """
    Update smoothing lengths on GPU using octree-based neighbor counting.
    
    Iteratively adjusts smoothing lengths until each particle has approximately
    target_neighbours within its support radius. Uses O(N log N) octree traversal
    instead of O(N²) brute force.
    
    Parameters
    ----------
    positions : cp.ndarray, shape (N, 3)
        Particle positions on GPU
    smoothing_lengths : cp.ndarray, shape (N,)
        Current smoothing lengths on GPU
    target_neighbours : int
        Target neighbor count (default 50)
    tolerance : float
        Convergence tolerance as fraction of target (default 0.05 = 5%)
    max_iter : int
        Maximum iterations (default 10)
    h_min : float
        Minimum allowed smoothing length (default 1e-6)
    h_max : float
        Maximum allowed smoothing length (default 1e2)
    support_radius : float
        Support radius multiplier (default 2.0 for cubic spline)
    octree : GPUOctree, optional
        Pre-built octree. If None, will build one internally.
        
    Returns
    -------
    cp.ndarray, shape (N,)
        Updated smoothing lengths
    """
    n = positions.shape[0]
    
    if n == 0:
        return smoothing_lengths
    
    # Build octree if not provided
    build_octree_here = octree is None
    if build_octree_here:
        from .octree_gpu import GPUOctree
        octree = GPUOctree()
        octree.build(positions, smoothing_lengths)
    
    # Get octree data
    n_internal = octree.n_internal
    sorted_indices = octree.sorted_indices
    internal_children_left = octree.internal_children_left
    internal_children_right = octree.internal_children_right
    leaf_box_min = octree.leaf_box_min
    leaf_box_max = octree.leaf_box_max
    internal_box_min = octree.internal_box_min
    internal_box_max = octree.internal_box_max
    
    # Working arrays
    h_new = smoothing_lengths.copy()
    neighbor_counts = cp.zeros(n, dtype=cp.int32)
    converged_flags = cp.zeros(n, dtype=cp.int32)
    
    # CUDA launch parameters
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    # Iterative refinement with dynamic bounds
    for iteration in range(max_iter):
        # Compute adaptive bounds each iteration based on current distribution
        # This is KEY to stability - bounds scale with particle layout
        if h_min is None or h_max is None:
            h_min_iter, h_max_iter = _estimate_dynamic_h_bounds(positions, h_new)
        else:
            h_min_iter, h_max_iter = h_min, h_max
        
        # 1. Count neighbors using octree traversal
        _count_neighbors_kernel(
            (grid_size,), (block_size,),
            (positions.ravel(), h_new, sorted_indices,
             internal_children_left, internal_children_right,
             leaf_box_min.ravel(), leaf_box_max.ravel(),
             internal_box_min.ravel(), internal_box_max.ravel(),
             neighbor_counts, n, n_internal, cp.float32(support_radius))
        )
        
        # 2. Check convergence
        _check_convergence_kernel(
            (grid_size,), (block_size,),
            (neighbor_counts, target_neighbours, cp.float32(tolerance),
             converged_flags, n)
        )
        
        # Check if all particles converged
        n_converged = int(cp.sum(converged_flags))
        if n_converged == n:
            break
        
        # 3. Update smoothing lengths based on neighbor counts
        _update_h_kernel(
            (grid_size,), (block_size,),
            (h_new, neighbor_counts, target_neighbours, n,
             cp.float32(h_min_iter), cp.float32(h_max_iter))
        )
    
    return h_new


def update_smoothing_lengths_gpu_fast(
    positions: cp.ndarray,
    smoothing_lengths: cp.ndarray,
    octree,
    target_neighbours: int = 50,
    max_iter: int = 10,
    h_min: float = 1e-3,
    h_max: float = 1e2
) -> cp.ndarray:
    """
    Fast version with reduced iterations for use during simulation.
    
    Uses a pre-built octree and fewer iterations for better performance
    during time-stepping where perfect convergence is not critical.
    
    Parameters
    ----------
    positions : cp.ndarray, shape (N, 3)
        Particle positions on GPU
    smoothing_lengths : cp.ndarray, shape (N,)
        Current smoothing lengths on GPU
    octree : GPUOctree
        Pre-built octree
    target_neighbours : int
        Target neighbor count (default 50)
    max_iter : int
        Maximum iterations (default 3, reduced for speed)
    h_min : float
        Minimum allowed smoothing length (default 1e-6)
    h_max : float
        Maximum allowed smoothing length (default 1e2)
        
    Returns
    -------
    cp.ndarray, shape (N,)
        Updated smoothing lengths
    """
    return update_smoothing_lengths_gpu(
        positions,
        smoothing_lengths,
        target_neighbours=target_neighbours,
        tolerance=0.1,  # Relaxed tolerance for speed
        max_iter=max_iter,
        h_min=h_min,
        h_max=h_max,
        support_radius=2.0,
        octree=octree
    )
