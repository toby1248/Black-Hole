"""
GPU-accelerated octree construction and traversal using CuPy.

Implements unified octree for both:
- TreeSPH neighbour search
- Barnes-Hut gravitational force calculations

Based on algorithms from:
- Karras 2012: Parallel radix tree construction
- Bédorf et al 2012: GPU octree for N-body simulations
- Keller et al 2023: Cornerstone method for scalable octree construction

All operations designed to minimize PCIe transfers by keeping data on GPU.
"""

import cupy as cp
import numpy as np
from typing import Tuple, Optional
import math
import warnings


# CUDA kernels as raw strings for custom operations
MORTON_CODE_KERNEL = r'''
extern "C" __global__
void compute_morton_codes(
    const float* positions,
    const float* bbox_min,
    const float* bbox_max,
    unsigned long long* morton_codes,
    int* sorted_indices,
    int n_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    // Normalize position to [0, 1]^3
    float x = (positions[idx * 3 + 0] - bbox_min[0]) / (bbox_max[0] - bbox_min[0]);
    float y = (positions[idx * 3 + 1] - bbox_min[1]) / (bbox_max[1] - bbox_min[1]);
    float z = (positions[idx * 3 + 2] - bbox_min[2]) / (bbox_max[2] - bbox_min[2]);

    // Clamp to [0, 1]
    x = fmaxf(0.0f, fminf(1.0f, x));
    y = fmaxf(0.0f, fminf(1.0f, y));
    z = fmaxf(0.0f, fminf(1.0f, z));

    // Convert to fixed-point coordinates (21 bits per dimension for 63-bit codes)
    unsigned long long xi = (unsigned long long)(x * 2097151.0f);  // 2^21 - 1
    unsigned long long yi = (unsigned long long)(y * 2097151.0f);
    unsigned long long zi = (unsigned long long)(z * 2097151.0f);

    // Interleave bits to create Morton code (Z-order curve)
    unsigned long long morton = 0;
    for (int i = 0; i < 21; i++) {
        morton |= ((xi & (1ULL << i)) << (2 * i)) |
                  ((yi & (1ULL << i)) << (2 * i + 1)) |
                  ((zi & (1ULL << i)) << (2 * i + 2));
    }

    morton_codes[idx] = morton;
    sorted_indices[idx] = idx;
}
'''

KARRAS_TREE_KERNEL = r'''
/* Karras (2012) - Parallel radix tree construction kernel */
__device__ int delta(const unsigned long long* morton_codes, int i, int j, int n) {
    // Compute common prefix length between morton_codes[i] and morton_codes[j]
    if (j < 0 || j >= n) return -1;
    
    if (morton_codes[i] == morton_codes[j]) {
        // Tie-break using indices
        return __clzll(morton_codes[i] ^ morton_codes[j]) + __clz(i ^ j);
    }
    return __clzll(morton_codes[i] ^ morton_codes[j]);
}

__device__ int2 determine_range(const unsigned long long* morton_codes, int i, int n) {
    // Determine the range of keys covered by internal node i
    if (i == 0) {
        return make_int2(0, n - 1);
    }
    
    // Determine direction of the range
    int d = (delta(morton_codes, i, i + 1, n) - delta(morton_codes, i, i - 1, n)) >= 0 ? 1 : -1;
    
    // Compute upper bound for range length
    int delta_min = delta(morton_codes, i, i - d, n);
    int l_max = 2;
    int safety_counter = 0;
    const int MAX_ITERATIONS = 30;  // Enough for 2^30 > 1 billion particles
    
    while (delta(morton_codes, i, i + l_max * d, n) > delta_min && safety_counter < MAX_ITERATIONS) {
        l_max *= 2;
        safety_counter++;
    }
    
    // Binary search to find the other end
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(morton_codes, i, i + (l + t) * d, n) > delta_min) {
            l += t;
        }
    }
    
    int j = i + l * d;
    
    // Clamp to valid range
    if (j < 0) j = 0;
    if (j >= n) j = n - 1;
    
    return make_int2(min(i, j), max(i, j));
}

__device__ int find_split(const unsigned long long* morton_codes, int first, int last, int n) {
    // Find the split position between first and last
    if (first == last) return first;
    
    int delta_node = delta(morton_codes, first, last, n);
    
    // Binary search for split
    int s = 0;
    int safety_counter = 0;
    const int MAX_ITERATIONS = 64;  // Log2 of max range
    
    for (int t = (last - first + 1) / 2; t >= 1; t /= 2) {
        if (safety_counter++ >= MAX_ITERATIONS) break;
        
        int test_pos = first + s + t;
        if (test_pos <= last && delta(morton_codes, first, test_pos, n) > delta_node) {
            s += t;
        }
    }
    
    // Clamp result to valid range
    int split = first + s;
    if (split < first) split = first;
    if (split >= last) split = last - 1;
    
    return split;
}

extern "C" __global__
void build_karras_tree(
    const unsigned long long* sorted_morton,
    int* internal_children_left,
    int* internal_children_right,
    int* internal_parent,
    int* leaf_parent,
    int n_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_internal = n_particles - 1;
    
    if (idx >= n_internal) return;
    if (n_particles < 2) return;  // Need at least 2 particles
    
    // Determine range covered by this node
    int2 range = determine_range(sorted_morton, idx, n_particles);
    int first = range.x;
    int last = range.y;
    
    // Validate range
    if (first < 0 || first >= n_particles || last < 0 || last >= n_particles || first > last) {
        return;  // Invalid range, skip
    }
    
    // Find split position
    int split = find_split(sorted_morton, first, last, n_particles);
    
    // Validate split
    if (split < first || split >= last) {
        split = (first + last) / 2;  // Fallback to midpoint
    }
    
    // Determine children (internal nodes or leaves)
    int left_child = (split == first) ? (split + n_internal) : split;
    int right_child = (split + 1 == last) ? (last + n_internal) : (split + 1);
    
    // Validate children indices
    if (left_child < 0 || left_child >= n_internal + n_particles) return;
    if (right_child < 0 || right_child >= n_internal + n_particles) return;
    
    internal_children_left[idx] = left_child;
    internal_children_right[idx] = right_child;
    
    // Set parent pointers for children
    if (split == first) {
        // Left child is a leaf
        if (split >= 0 && split < n_particles) {
            leaf_parent[split] = idx;
        }
    } else {
        // Left child is internal
        if (split >= 0 && split < n_internal) {
            internal_parent[split] = idx;
        }
    }
    
    if (split + 1 == last) {
        // Right child is a leaf
        if (last >= 0 && last < n_particles) {
            leaf_parent[last] = idx;
        }
    } else {
        // Right child is internal
        if (split + 1 >= 0 && split + 1 < n_internal) {
            internal_parent[split + 1] = idx;
        }
    }
}

/* Bottom-up bounding box and COM computation - GPU parallel version */
extern "C" __global__
void compute_internal_properties_bottomup(
    const int* internal_children_left,
    const int* internal_children_right,
    const int* internal_parent,
    const int* leaf_parent,
    const float* leaf_box_min,
    const float* leaf_box_max,
    const float* leaf_com,
    const float* leaf_mass,
    float* internal_box_min,
    float* internal_box_max,
    float* internal_com,
    float* internal_mass,
    int* node_flags,
    int n_internal,
    int n_particles
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= n_particles) return;
    
    // Start from this leaf and walk up
    int current = leaf_parent[leaf_idx];
    
    // Safety limit to prevent infinite loops
    int max_iterations = 64;
    int iteration = 0;
    
    while (current >= 0 && current < n_internal && iteration < max_iterations) {
        // Atomically increment flag
        int old = atomicAdd(&node_flags[current], 1);
        
        if (old == 0) {
            // First child - exit
            break;
        }
        
        // Second child - compute properties
        int left = internal_children_left[current];
        int right = internal_children_right[current];
        
        float left_min[3], left_max[3], left_com_arr[3], left_m;
        float right_min[3], right_max[3], right_com_arr[3], right_m;
        
        // Get left child data
        if (left >= n_internal) {
            int lid = left - n_internal;
            for (int d = 0; d < 3; d++) {
                left_min[d] = leaf_box_min[lid * 3 + d];
                left_max[d] = leaf_box_max[lid * 3 + d];
                left_com_arr[d] = leaf_com[lid * 3 + d];
            }
            left_m = leaf_mass[lid];
        } else {
            for (int d = 0; d < 3; d++) {
                left_min[d] = internal_box_min[left * 3 + d];
                left_max[d] = internal_box_max[left * 3 + d];
                left_com_arr[d] = internal_com[left * 3 + d];
            }
            left_m = internal_mass[left];
        }
        
        // Get right child data
        if (right >= n_internal) {
            int rid = right - n_internal;
            for (int d = 0; d < 3; d++) {
                right_min[d] = leaf_box_min[rid * 3 + d];
                right_max[d] = leaf_box_max[rid * 3 + d];
                right_com_arr[d] = leaf_com[rid * 3 + d];
            }
            right_m = leaf_mass[rid];
        } else {
            for (int d = 0; d < 3; d++) {
                right_min[d] = internal_box_min[right * 3 + d];
                right_max[d] = internal_box_max[right * 3 + d];
                right_com_arr[d] = internal_com[right * 3 + d];
            }
            right_m = internal_mass[right];
        }
        
        // Compute union bbox
        for (int d = 0; d < 3; d++) {
            internal_box_min[current * 3 + d] = fminf(left_min[d], right_min[d]);
            internal_box_max[current * 3 + d] = fmaxf(left_max[d], right_max[d]);
        }
        
        // Compute COM and total mass
        float total_m = left_m + right_m;
        internal_mass[current] = total_m;
        
        if (total_m > 0.0f) {
            for (int d = 0; d < 3; d++) {
                internal_com[current * 3 + d] = 
                    (left_com_arr[d] * left_m + right_com_arr[d] * right_m) / total_m;
            }
        } else {
            for (int d = 0; d < 3; d++) {
                internal_com[current * 3 + d] = (left_com_arr[d] + right_com_arr[d]) * 0.5f;
            }
        }
        
        // Move to parent
        if (current == 0) break;
        
        // Find parent
        int parent = internal_parent[current];
        if (parent < 0 || parent >= n_internal) break;
        
        current = parent;
        iteration++;
    }
}
'''

NEIGHBOUR_SEARCH_KERNEL = r'''
extern "C" __global__
void find_neighbours(
    const float* positions,
    const float* smoothing_lengths,
    const int* sorted_indices,
    const int* internal_children_left,
    const int* internal_children_right,
    const float* leaf_box_min,
    const float* leaf_box_max,
    const float* internal_box_min,
    const float* internal_box_max,
    int* neighbour_lists,
    int* neighbour_counts,
    int n_particles,
    int n_internal,
    int max_neighbours,
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
    
    // Dynamic stack allocation - increased size for safety
    const int MAX_STACK = 128;
    int stack[MAX_STACK];
    int stack_ptr = 0;
    
    // Start at root (node 0)
    if (n_internal > 0) {
        stack[stack_ptr++] = 0;
    }
    
    // Safety counter to prevent infinite loops
    int iterations = 0;
    const int MAX_ITERATIONS = n_particles * 10;
    
    while (stack_ptr > 0 && count < max_neighbours && iterations < MAX_ITERATIONS) {
        iterations++;
        
        if (stack_ptr >= MAX_STACK) {
            // Stack overflow - truncate search
            break;
        }
        
        int node = stack[--stack_ptr];
        
        // Bounds check
        if (node < 0) continue;
        
        // Check if this is internal or leaf
        if (node < n_internal) {
            // Internal node - check bounding box
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
                
                // Bounds check before adding to stack
                if (left >= 0 && left < n_internal + n_particles && stack_ptr < MAX_STACK) {
                    stack[stack_ptr++] = left;
                }
                if (right >= 0 && right < n_internal + n_particles && stack_ptr < MAX_STACK) {
                    stack[stack_ptr++] = right;
                }
            }
        } else {
            // Leaf node (particle)
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
                
                if (r2 <= search_r2 && count < max_neighbours) {
                    neighbour_lists[idx * max_neighbours + count] = other_idx;
                    count++;
                }
            }
        }
    }

    neighbour_counts[idx] = count;
}
'''

BARNES_HUT_KERNEL = r'''
extern "C" __global__
void compute_gravity_forces(
    const float* positions,
    const float* masses,
    const float* smoothing_lengths,
    const int* sorted_indices,
    const int* internal_children_left,
    const int* internal_children_right,
    const float* leaf_box_min,
    const float* leaf_box_max,
    const float* leaf_com,
    const float* leaf_mass,
    const float* internal_box_min,
    const float* internal_box_max,
    const float* internal_com,
    const float* internal_mass,
    float* forces,
    int n_particles,
    int n_internal,
    float G,
    float theta,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];
    float h = smoothing_lengths[idx];
    float eps = epsilon * h;
    float eps2 = eps * eps;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    // Stack for tree traversal
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
        
        if (stack_ptr >= MAX_STACK) break;
        
        int node = stack[--stack_ptr];
        
        if (node < 0) continue;
        
        if (node < n_internal) {
            // Internal node - check MAC
            float cx = internal_com[node * 3 + 0];
            float cy = internal_com[node * 3 + 1];
            float cz = internal_com[node * 3 + 2];
            float M = internal_mass[node];
            
            float dx = cx - px;
            float dy = cy - py;
            float dz = cz - pz;
            float r2 = dx*dx + dy*dy + dz*dz;
            float r = sqrtf(r2);
            
            // Compute cell size
            float sx = internal_box_max[node * 3 + 0] - internal_box_min[node * 3 + 0];
            float sy = internal_box_max[node * 3 + 1] - internal_box_min[node * 3 + 1];
            float sz = internal_box_max[node * 3 + 2] - internal_box_min[node * 3 + 2];
            float cell_size = fmaxf(sx, fmaxf(sy, sz));
            
            // Multipole Acceptance Criterion (MAC)
            if (r > 0.0f && cell_size / r < theta) {
                // Use multipole approximation
                float r_soft = sqrtf(r2 + eps2);
                float f_mag = G * M / (r_soft * r_soft * r_soft);
                fx += f_mag * dx;
                fy += f_mag * dy;
                fz += f_mag * dz;
            } else {
                // Open cell - add children to stack
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
            // Leaf node (single particle)
            int leaf_id = node - n_internal;
            if (leaf_id < 0 || leaf_id >= n_particles) continue;
            
            int other_idx = sorted_indices[leaf_id];
            
            if (other_idx != idx && other_idx >= 0 && other_idx < n_particles) {
                float ox = positions[other_idx * 3 + 0];
                float oy = positions[other_idx * 3 + 1];
                float oz = positions[other_idx * 3 + 2];
                float m = masses[other_idx];
                
                float dx = ox - px;
                float dy = oy - py;
                float dz = oz - pz;
                float r2 = dx*dx + dy*dy + dz*dz;
                float r_soft = sqrtf(r2 + eps2);
                
                float f_mag = G * m / (r_soft * r_soft * r_soft);
                fx += f_mag * dx;
                fy += f_mag * dy;
                fz += f_mag * dz;
            }
        }
    }

    forces[idx * 3 + 0] = fx;
    forces[idx * 3 + 1] = fy;
    forces[idx * 3 + 2] = fz;
}
'''

ADAPTIVE_NEIGHBOUR_COUNT_KERNEL = r'''
extern "C" __global__
void count_adaptive_neighbours(
    const float* positions,
    const float* smoothing_lengths,
    const int* sorted_indices,
    const int* internal_children_left,
    const int* internal_children_right,
    const float* leaf_box_min,
    const float* leaf_box_max,
    const float* internal_box_min,
    const float* internal_box_max,
    int* neighbour_counts,
    int n_particles,
    int n_internal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];
    float h_i = smoothing_lengths[idx];

    int count = 0;
    
    // Dynamic stack
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
        
        if (stack_ptr >= MAX_STACK) break;
        
        int node = stack[--stack_ptr];
        
        if (node < 0) continue;
        
        if (node < n_internal) {
            // Internal node - check bounding box with conservative radius
            float box_min_x = internal_box_min[node * 3 + 0];
            float box_min_y = internal_box_min[node * 3 + 1];
            float box_min_z = internal_box_min[node * 3 + 2];
            float box_max_x = internal_box_max[node * 3 + 0];
            float box_max_y = internal_box_max[node * 3 + 1];
            float box_max_z = internal_box_max[node * 3 + 2];
            
            // Conservative radius: use a large enough radius to catch all possible neighbours
            // Since h varies, we use a large value, but in practice, limit to avoid too many traversals
            float conservative_r = 4.0f * h_i;  // Should be sufficient for most cases
            
            // Sphere-box intersection
            float dx = fmaxf(box_min_x - px, fmaxf(0.0f, px - box_max_x));
            float dy = fmaxf(box_min_y - py, fmaxf(0.0f, py - box_max_y));
            float dz = fmaxf(box_min_z - pz, fmaxf(0.0f, pz - box_max_z));
            float dist2 = dx*dx + dy*dy + dz*dz;
            
            if (dist2 <= conservative_r * conservative_r) {
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
            // Leaf node
            int leaf_id = node - n_internal;
            if (leaf_id < 0 || leaf_id >= n_particles) continue;
            
            int other_idx = sorted_indices[leaf_id];
            
            if (other_idx != idx && other_idx >= 0 && other_idx < n_particles) {
                float ox = positions[other_idx * 3 + 0];
                float oy = positions[other_idx * 3 + 1];
                float oz = positions[other_idx * 3 + 2];
                float h_j = smoothing_lengths[other_idx];
                
                float dx = ox - px;
                float dy = oy - py;
                float dz = oz - pz;
                float r2 = dx*dx + dy*dy + dz*dz;
                float r = sqrtf(r2);
                
                float h_max = fmaxf(h_i, h_j);
                float r_max = 2.0f * h_max;
                
                if (r <= r_max) {
                    count++;
                }
            }
        }
    }
    
    neighbour_counts[idx] = count;
}
'''


class GPUOctree:
    """
    GPU-accelerated octree for unified neighbour search and Barnes-Hut gravity.

    All data kept on GPU to minimize PCIe transfers.
    """

    def __init__(self, max_depth: int = 21, theta: float = 0.5):
        """
        Initialize GPU octree.

        Parameters
        ----------
        max_depth : int
            Maximum tree depth (21 for 63-bit Morton codes)
        theta : float
            Barnes-Hut opening angle criterion
        """
        self.max_depth = max_depth
        self.theta = theta

        # Compile CUDA kernels
        self.morton_kernel = cp.RawKernel(MORTON_CODE_KERNEL, 'compute_morton_codes')
        self.karras_kernel = cp.RawKernel(KARRAS_TREE_KERNEL, 'build_karras_tree')
        self.bbox_kernel = cp.RawKernel(KARRAS_TREE_KERNEL, 'compute_internal_properties_bottomup')
        self.neighbour_kernel = cp.RawKernel(NEIGHBOUR_SEARCH_KERNEL, 'find_neighbours')
        self.gravity_kernel = cp.RawKernel(BARNES_HUT_KERNEL, 'compute_gravity_forces')
        self.adaptive_count_kernel = cp.RawKernel(ADAPTIVE_NEIGHBOUR_COUNT_KERNEL, 'count_adaptive_neighbours')

        # Tree data (all on GPU)
        self.morton_codes = None
        self.sorted_indices = None
        
        # Karras tree structure
        self.internal_children_left = None
        self.internal_children_right = None
        self.internal_parent = None
        self.leaf_parent = None
        
        # Bounding boxes
        self.leaf_box_min = None
        self.leaf_box_max = None
        self.internal_box_min = None
        self.internal_box_max = None
        
        # COM and mass for Barnes-Hut
        self.leaf_com = None
        self.leaf_mass = None
        self.internal_com = None
        self.internal_mass = None
        
        self.n_particles = None
        self.n_internal = None

        # Bounding box
        self.bbox_min = None
        self.bbox_max = None

    def build(self, positions: cp.ndarray, masses: Optional[cp.ndarray] = None) -> None:
        """
        Build octree from particle positions using Karras (2012) algorithm.

        Parameters
        ----------
        positions : cp.ndarray
            Particle positions (N, 3) on GPU
        masses : cp.ndarray, optional
            Particle masses (N,) on GPU
        """
        self.n_particles = positions.shape[0]
        self.n_internal = max(1, self.n_particles - 1)

        if self.n_particles < 2:
            # Degenerate case - single particle
            print("Warning: Only 1 particle, tree build skipped")
            return

        # Compute bounding box
        self.bbox_min = cp.min(positions, axis=0).astype(cp.float32)
        self.bbox_max = cp.max(positions, axis=0).astype(cp.float32)

        # Add padding to avoid division by zero
        padding = cp.maximum((self.bbox_max - self.bbox_min) * 0.01, 1e-6)
        self.bbox_min -= padding
        self.bbox_max += padding

        # Allocate arrays for Morton codes
        self.morton_codes = cp.zeros(self.n_particles, dtype=cp.uint64)
        self.sorted_indices = cp.arange(self.n_particles, dtype=cp.int32)

        # Compute Morton codes
        block_size = 256
        grid_size = (self.n_particles + block_size - 1) // block_size

        self.morton_kernel(
            (grid_size,), (block_size,),
            (positions.ravel(), self.bbox_min, self.bbox_max,
             self.morton_codes, self.sorted_indices, self.n_particles)
        )

        # Sort by Morton code (radix sort on GPU)
        sort_idx = cp.argsort(self.morton_codes)
        self.morton_codes = self.morton_codes[sort_idx]
        self.sorted_indices = self.sorted_indices[sort_idx]

        # Allocate tree structure arrays (Karras tree)
        self.internal_children_left = cp.zeros(self.n_internal, dtype=cp.int32)
        self.internal_children_right = cp.zeros(self.n_internal, dtype=cp.int32)
        self.internal_parent = cp.full(self.n_internal, -1, dtype=cp.int32)
        self.leaf_parent = cp.full(self.n_particles, -1, dtype=cp.int32)

        # Build tree structure using Karras algorithm
        grid_size_internal = (self.n_internal + block_size - 1) // block_size
        self.karras_kernel(
            (grid_size_internal,), (block_size,),
            (self.morton_codes, self.internal_children_left, self.internal_children_right,
             self.internal_parent, self.leaf_parent, self.n_particles)
        )

        # Compute leaf properties (bounding boxes, COM, mass)
        self._compute_leaf_properties(positions, masses)
        
        # Allocate internal node arrays
        self.internal_box_min = cp.zeros((self.n_internal, 3), dtype=cp.float32)
        self.internal_box_max = cp.zeros((self.n_internal, 3), dtype=cp.float32)
        self.internal_com = cp.zeros((self.n_internal, 3), dtype=cp.float32)
        self.internal_mass = cp.zeros(self.n_internal, dtype=cp.float32)
        
        # Compute internal properties bottom-up (GPU parallel)
        node_flags = cp.zeros(self.n_internal, dtype=cp.int32)
        self.bbox_kernel(
            (grid_size,), (block_size,),
            (self.internal_children_left, self.internal_children_right,
             self.internal_parent,
             self.leaf_parent,
             self.leaf_box_min.ravel(), self.leaf_box_max.ravel(),
             self.leaf_com.ravel(), self.leaf_mass,
             self.internal_box_min.ravel(), self.internal_box_max.ravel(),
             self.internal_com.ravel(), self.internal_mass,
             node_flags, self.n_internal, self.n_particles)
        )
        
        # Synchronize to ensure all properties computed
        cp.cuda.Stream.null.synchronize()

    def _compute_leaf_properties(self, positions: cp.ndarray, masses: Optional[cp.ndarray]) -> None:
        """Compute bounding boxes, center of mass, and total mass for each leaf (particle)."""
        # In Karras tree, leaves are individual particles
        self.leaf_box_min = cp.zeros((self.n_particles, 3), dtype=cp.float32)
        self.leaf_box_max = cp.zeros((self.n_particles, 3), dtype=cp.float32)
        self.leaf_com = cp.zeros((self.n_particles, 3), dtype=cp.float32)
        self.leaf_mass = cp.zeros(self.n_particles, dtype=cp.float32)

        # Reorder by sorted indices
        sorted_positions = positions[self.sorted_indices]
        if masses is not None:
            sorted_masses = masses[self.sorted_indices]
        else:
            sorted_masses = cp.ones(self.n_particles, dtype=cp.float32)

        # For single particle leaves, box is just the point (with small epsilon)
        epsilon = 1e-6
        self.leaf_box_min = sorted_positions - epsilon
        self.leaf_box_max = sorted_positions + epsilon
        self.leaf_com = sorted_positions.copy()
        self.leaf_mass = sorted_masses.copy()

    def find_neighbours(
        self,
        positions: cp.ndarray,
        smoothing_lengths: cp.ndarray,
        support_radius: float = 2.0,
        max_neighbours: int = 64
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Find neighbours using octree traversal.

        Parameters
        ----------
        positions : cp.ndarray
            Particle positions (N, 3) on GPU
        smoothing_lengths : cp.ndarray
            Smoothing lengths (N,) on GPU
        support_radius : float
            Support radius multiplier
        max_neighbours : int
            Maximum neighbours per particle

        Returns
        -------
        neighbour_lists : cp.ndarray
            Neighbour indices (N, max_neighbours) on GPU
        neighbour_counts : cp.ndarray
            Number of neighbours for each particle (N,) on GPU
        """
        n = positions.shape[0]
        neighbour_lists = cp.full((n, max_neighbours), -1, dtype=cp.int32)
        neighbour_counts = cp.zeros(n, dtype=cp.int32)

        if self.n_particles < 2:
            return neighbour_lists, neighbour_counts

        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        self.neighbour_kernel(
            (grid_size,), (block_size,),
            (positions.ravel(), smoothing_lengths, self.sorted_indices,
             self.internal_children_left, self.internal_children_right,
             self.leaf_box_min.ravel(), self.leaf_box_max.ravel(),
             self.internal_box_min.ravel(), self.internal_box_max.ravel(),
             neighbour_lists, neighbour_counts, n, self.n_internal, max_neighbours,
             cp.float32(support_radius))
        )

        return neighbour_lists, neighbour_counts

    def find_neighbors(
        self,
        positions: cp.ndarray,
        smoothing_lengths: cp.ndarray,
        support_radius: float = 2.0,
        max_neighbors: int = 64
    ) -> Tuple[cp.ndarray, cp.ndarray]:
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
            max_neighbours=max_neighbors,
        )

    def compute_gravity(
        self,
        positions: cp.ndarray,
        masses: cp.ndarray,
        smoothing_lengths: cp.ndarray,
        G: float = 1.0,
        epsilon: float = 0.1
    ) -> cp.ndarray:
        """
        Compute gravitational forces using Barnes-Hut algorithm.

        Parameters
        ----------
        positions : cp.ndarray
            Particle positions (N, 3) on GPU
        masses : cp.ndarray
            Particle masses (N,) on GPU
        smoothing_lengths : cp.ndarray
            Smoothing lengths (N,) on GPU
        G : float
            Gravitational constant
        epsilon : float
            Softening parameter (multiplied by smoothing length)

        Returns
        -------
        forces : cp.ndarray
            Gravitational forces (N, 3) on GPU
        """
        n = positions.shape[0]
        forces = cp.zeros((n, 3), dtype=cp.float32)

        if self.n_particles < 2:
            return forces

        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        self.gravity_kernel(
            (grid_size,), (block_size,),
            (positions.ravel(), masses, smoothing_lengths, self.sorted_indices,
             self.internal_children_left, self.internal_children_right,
             self.leaf_box_min.ravel(), self.leaf_box_max.ravel(),
             self.leaf_com.ravel(), self.leaf_mass,
             self.internal_box_min.ravel(), self.internal_box_max.ravel(),
             self.internal_com.ravel(), self.internal_mass,
             forces.ravel(), n, self.n_internal,
             cp.float32(G), cp.float32(self.theta), cp.float32(epsilon))
        )

        return forces

    def count_adaptive_neighbours(
        self,
        positions: cp.ndarray,
        smoothing_lengths: cp.ndarray
    ) -> cp.ndarray:
        """
        Count adaptive neighbours using octree traversal.

        Uses r < 2 * max(h_i, h_j) criterion for adaptive smoothing lengths.

        Parameters
        ----------
        positions : cp.ndarray
            Particle positions (N, 3) on GPU
        smoothing_lengths : cp.ndarray
            Smoothing lengths (N,) on GPU

        Returns
        -------
        neighbour_counts : cp.ndarray
            Number of neighbours for each particle (N,) on GPU
        """
        n = positions.shape[0]
        neighbour_counts = cp.zeros(n, dtype=cp.int32)

        if self.n_particles < 2:
            return neighbour_counts

        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        self.adaptive_count_kernel(
            (grid_size,), (block_size,),
            (positions.ravel(), smoothing_lengths, self.sorted_indices,
             self.internal_children_left, self.internal_children_right,
             self.leaf_box_min.ravel(), self.leaf_box_max.ravel(),
             self.internal_box_min.ravel(), self.internal_box_max.ravel(),
             neighbour_counts, n, self.n_internal)
        )

        return neighbour_counts

    def count_adaptive_neighbors(
        self,
        positions: cp.ndarray,
        smoothing_lengths: cp.ndarray
    ) -> cp.ndarray:
        """Compatibility shim for American spelling."""
        warnings.warn(
            "'count_adaptive_neighbors' is deprecated. Use 'count_adaptive_neighbours' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.count_adaptive_neighbours(positions, smoothing_lengths)


def find_neighbours_octree_gpu(
    positions: cp.ndarray,
    smoothing_lengths: cp.ndarray,
    masses: Optional[cp.ndarray] = None,
    support_radius: float = 2.0,
    max_neighbours: int = 64,
    tree: Optional[GPUOctree] = None,
    **deprecated_kwargs,
) -> Tuple[cp.ndarray, cp.ndarray, GPUOctree]:
    """
    Find neighbours using GPU octree (TreeSPH approach).

    Parameters
    ----------
    positions : cp.ndarray
        Particle positions (N, 3) on GPU
    smoothing_lengths : cp.ndarray
        Smoothing lengths (N,) on GPU
    masses : cp.ndarray, optional
        Particle masses (N,) on GPU (for tree building)
    support_radius : float
        Support radius multiplier
    max_neighbours : int
        Maximum neighbours per particle
    tree : GPUOctree, optional
        Existing tree to reuse

    Returns
    -------
    neighbour_lists : cp.ndarray
        Neighbour indices (N, max_neighbours) on GPU
    neighbour_counts : cp.ndarray
        Number of neighbours (N,) on GPU
    tree : GPUOctree
        The octree (for reuse)
    """
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

    if tree is None:
        tree = GPUOctree()
        tree.build(positions, masses)

    neighbour_lists, neighbour_counts = tree.find_neighbours(
        positions, smoothing_lengths, support_radius, max_neighbours
    )

    return neighbour_lists, neighbour_counts, tree


def compute_gravity_gpu(
    positions: cp.ndarray,
    masses: cp.ndarray,
    smoothing_lengths: cp.ndarray,
    G: float = 1.0,
    epsilon: float = 0.1,
    theta: float = 0.5,
    tree: Optional[GPUOctree] = None
) -> Tuple[cp.ndarray, GPUOctree]:
    """
    Compute gravitational forces using GPU Barnes-Hut.

    Parameters
    ----------
    positions : cp.ndarray
        Particle positions (N, 3) on GPU
    masses : cp.ndarray
        Particle masses (N,) on GPU
    smoothing_lengths : cp.ndarray
        Smoothing lengths (N,) on GPU
    G : float
        Gravitational constant
    epsilon : float
        Softening parameter
    theta : float
        Barnes-Hut opening angle
    tree : GPUOctree, optional
        Existing tree to reuse

    Returns
    -------
    forces : cp.ndarray
        Gravitational forces (N, 3) on GPU
    tree : GPUOctree
        The octree (for reuse)
    """
    if tree is None:
        tree = GPUOctree(theta=theta)
        tree.build(positions, masses)

    forces = tree.compute_gravity(positions, masses, smoothing_lengths, G, epsilon)

    return forces, tree
