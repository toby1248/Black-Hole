
"""
Barnes-Hut gravity solver for self-gravitating SPH particles.

Implements O(N log N) gravitational force calculation using an Octree.
This is a critical optimization for large N simulations (N > 10^4).

Reference:
    Barnes & Hut (1986) - A Hierarchical O(N log N) Force-Calculation Algorithm
"""

from typing import Optional
import numpy as np
try:
    from numba import njit, prange, int32, float32
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback mocks
    prange = range
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    int32 = int
    float32 = float

from ..core.interfaces import GravitySolver, Metric, NDArrayFloat

# Constants
MAX_NODES_FACTOR = 20  # Maximum nodes relative to particles (safety buffer)
THETA = 0.5            # Multipole acceptance criterion (0.5-0.7 is standard)
G_CONST = 1.0          # Default G

@njit(fastmath=True)
def _get_octant(x, y, z, cx, cy, cz):
    """Determine which octant (0-7) a point lies in relative to center."""
    idx = 0
    if x >= cx: idx |= 1
    if y >= cy: idx |= 2
    if z >= cz: idx |= 4
    return idx

@njit(fastmath=True)
def _insert_particle_recursive(p_idx, positions, masses, node_idx, 
                               child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, 
                               n_nodes_ptr, max_nodes):
    """Recursive insertion of a particle into the tree."""
    
    px = positions[p_idx, 0]
    py = positions[p_idx, 1]
    pz = positions[p_idx, 2]
    pm = masses[p_idx]
    
    # Update mass/COM of current node
    node_mass[node_idx] += pm
    node_com[node_idx, 0] += px * pm
    node_com[node_idx, 1] += py * pm
    node_com[node_idx, 2] += pz * pm
    
    # Check if leaf (no children)
    is_leaf = True
    for k in range(8):
        if child_ptr[node_idx, k] != -1:
            is_leaf = False
            break
            
    if is_leaf:
        if leaf_particle[node_idx] == -1:
            # Empty leaf -> Place particle here
            leaf_particle[node_idx] = p_idx
            return True
        else:
            # Occupied leaf -> Split
            p_old = leaf_particle[node_idx]
            leaf_particle[node_idx] = -1 # No longer a leaf holding a particle directly
            
            # We must push p_old down
            if not _push_to_child(p_old, positions, masses, node_idx, child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, n_nodes_ptr, max_nodes):
                return False
                
            # Then push p_new down
            if not _push_to_child(p_idx, positions, masses, node_idx, child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, n_nodes_ptr, max_nodes):
                return False
                
            return True
    else:
        # Internal node -> Push p_new down
        return _push_to_child(p_idx, positions, masses, node_idx, child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, n_nodes_ptr, max_nodes)

@njit(fastmath=True)
def _push_to_child(p_idx, positions, masses, node_idx, 
                   child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, 
                   n_nodes_ptr, max_nodes):
    """Helper to determine octant and recurse."""
    px = positions[p_idx, 0]
    py = positions[p_idx, 1]
    pz = positions[p_idx, 2]
    
    bc = box_center[node_idx]
    octant = _get_octant(px, py, pz, bc[0], bc[1], bc[2])
    
    child = child_ptr[node_idx, octant]
    
    if child == -1:
        # Create new child
        if n_nodes_ptr[0] >= max_nodes:
            return False
        child = n_nodes_ptr[0]
        n_nodes_ptr[0] += 1
        child_ptr[node_idx, octant] = child
        
        # Setup child box
        bs = box_size[node_idx] * 0.5
        off_x = bs * 0.5 * (1 if (octant & 1) else -1)
        off_y = bs * 0.5 * (1 if (octant & 2) else -1)
        off_z = bs * 0.5 * (1 if (octant & 4) else -1)
        
        box_center[child, 0] = bc[0] + off_x
        box_center[child, 1] = bc[1] + off_y
        box_center[child, 2] = bc[2] + off_z
        box_size[child] = bs
        
        # Initialize child
        leaf_particle[child] = -1
        node_mass[child] = 0.0
        node_com[child] = 0.0
        
    # Recurse
    return _insert_particle_recursive(p_idx, positions, masses, child, 
                                      child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, 
                                      n_nodes_ptr, max_nodes)

@njit(fastmath=True)
def _build_tree(positions, masses, n_particles, max_nodes):
    """
    Build the Octree from particle data.
    """
    # Allocate arrays
    child_ptr = np.full((max_nodes, 8), -1, dtype=np.int32)
    node_mass = np.zeros(max_nodes, dtype=np.float32)
    node_com = np.zeros((max_nodes, 3), dtype=np.float32)
    leaf_particle = np.full(max_nodes, -1, dtype=np.int32)
    box_center = np.zeros((max_nodes, 3), dtype=np.float32)
    box_size = np.zeros(max_nodes, dtype=np.float32)
    
    # 1. Determine root bounding box
    min_x = np.min(positions[:, 0])
    max_x = np.max(positions[:, 0])
    min_y = np.min(positions[:, 1])
    max_y = np.max(positions[:, 1])
    min_z = np.min(positions[:, 2])
    max_z = np.max(positions[:, 2])
    
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    size = max(size_x, max(size_y, size_z)) * 1.01
    
    cx = min_x + size_x * 0.5
    cy = min_y + size_y * 0.5
    cz = min_z + size_z * 0.5
    
    # Root node (index 0)
    n_nodes_ptr = np.array([1], dtype=np.int32)
    box_center[0] = [cx, cy, cz]
    box_size[0] = size
    
    # 2. Insert particles
    for i in range(n_particles):
        if not _insert_particle_recursive(i, positions, masses, 0, 
                                          child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, 
                                          n_nodes_ptr, max_nodes):
            return child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, -1

    # 3. Finalize COM
    n_nodes = n_nodes_ptr[0]
    for k in range(n_nodes):
        if node_mass[k] > 0:
            inv_m = 1.0 / node_mass[k]
            node_com[k, 0] *= inv_m
            node_com[k, 1] *= inv_m
            node_com[k, 2] *= inv_m
            
    return child_ptr, node_mass, node_com, leaf_particle, box_center, box_size, n_nodes


@njit(parallel=True, fastmath=True)
def _compute_accel_tree(positions, masses, smoothing_lengths, G, theta, 
                       child_ptr, node_mass, node_com, leaf_particle, box_size, n_nodes):
    """
    Compute acceleration using Barnes-Hut tree walk.
    """
    N = len(positions)
    accel = np.zeros((N, 3), dtype=np.float32)
    
    # Stack for tree traversal (per thread)
    # Since we can't allocate dynamic memory easily in parallel loop, 
    # we use a fixed size stack or recursion. 
    # Numba supports recursion in some cases, but iterative with explicit stack is safer for GPU/parallel.
    # However, allocating a stack per thread in parallel loop is tricky.
    # Let's try recursion first, Numba handles it reasonably well now.
    
    for i in prange(N):
        ax, ay, az = _walk_tree_recursive(
            positions[i, 0], positions[i, 1], positions[i, 2],
            smoothing_lengths[i],
            0, # Start at root
            G, theta,
            child_ptr, node_mass, node_com, leaf_particle, box_size,
            i # Exclude self
        )
        accel[i, 0] = ax
        accel[i, 1] = ay
        accel[i, 2] = az
        
    return accel

@njit(fastmath=True)
def _walk_tree_recursive(px, py, pz, h_i, node_idx, G, theta,
                        child_ptr, node_mass, node_com, leaf_particle, box_size, p_idx):
    """Recursive tree walk for force calculation."""
    # return 1.0, 1.0, 1.0 # DEBUG
    ax = 0.0
    ay = 0.0
    az = 0.0
    
    # Distance to node COM
    dx = px - node_com[node_idx, 0]
    dy = py - node_com[node_idx, 1]
    dz = pz - node_com[node_idx, 2]
    r2 = dx*dx + dy*dy + dz*dz
    r = np.sqrt(r2)
    
    # Check if leaf
    if leaf_particle[node_idx] != -1:
        # It's a leaf
        idx = leaf_particle[node_idx]
        if idx != p_idx:
            # Direct force
            # Softening
            # We don't have h_j here easily unless we pass the whole array or store h in nodes.
            # For BH, usually use fixed softening or h_i.
            # Let's assume h_j ~ h_i or just use h_i for now (Symmetrized is better but requires lookup)
            # To be accurate, we should store h in the leaf nodes.
            # But for now, let's use h_i (Gather formulation)
            epsilon = h_i 
            r2_soft = r2 + epsilon*epsilon
            inv_r3 = r2_soft**(-1.5)
            f = -G * node_mass[node_idx] * inv_r3
            
            ax += f * dx
            ay += f * dy
            az += f * dz
        return ax, ay, az


    # Internal node
    # MAC check
    s = box_size[node_idx]
    if s / r < theta:
        # Accept as multipole (point mass)
        epsilon = h_i
        r2_soft = r2 + epsilon*epsilon
        inv_r3 = r2_soft**(-1.5)
        f = -G * node_mass[node_idx] * inv_r3
        
        ax += f * dx
        ay += f * dy
        az += f * dz
    else:
        # Recurse
        for k in range(8):
            child = child_ptr[node_idx, k]
            if child != -1:
                dax, day, daz = _walk_tree_recursive(
                    px, py, pz, h_i, child, G, theta,
                    child_ptr, node_mass, node_com, leaf_particle, box_size, p_idx
                )
                ax += dax
                ay += day
                az += daz
                
    return ax, ay, az


class BarnesHutGravity(GravitySolver):
    """
    Barnes-Hut O(N log N) gravity solver.
    
    Uses an Octree to approximate long-range forces.
    """
    
    def __init__(self, G: float = 1.0, theta: float = 0.5):
        self.G = np.float32(G)
        self.theta = np.float32(theta)
        
    def compute_acceleration(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        metric: Optional[Metric] = None
    ) -> NDArrayFloat:
        
        positions = positions.astype(np.float32)
        masses = masses.astype(np.float32)
        smoothing_lengths = smoothing_lengths.astype(np.float32)
        
        N = len(positions)
        max_nodes = N * MAX_NODES_FACTOR
        
        # Build Tree
        child_ptr, node_mass, node_com, leaf_particle, box_size, _, n_nodes = _build_tree(
            positions, masses, N, max_nodes
        )
        
        if n_nodes == -1:
            raise RuntimeError(f"Barnes-Hut tree overflowed. Max nodes: {max_nodes}")
            
        # Debug prints
        print(f"BH Tree built: {n_nodes} nodes. Root mass: {node_mass[0]}")

            
        # Compute Forces
        accel = _compute_accel_tree(
            positions, masses, smoothing_lengths, self.G, self.theta,
            child_ptr, node_mass, node_com, leaf_particle, box_size, n_nodes
        )
        
        return accel

    def compute_potential(self, positions, masses, smoothing_lengths):
        # TODO: Implement potential calculation using tree
        return np.zeros(len(positions), dtype=np.float32)

