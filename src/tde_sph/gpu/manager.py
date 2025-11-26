import warnings
import numpy as np
try:
    import cupy as cp
    from .octree_gpu import GPUOctree
except ImportError:
    cp = None
    GPUOctree = None

class GPUManager:
    """
    Manages data transfer and storage on the GPU.
    
    Keeps all particle data on GPU between steps and caches the octree
    to minimize PCIe transfers.
    """
    def __init__(self, particles):
        if cp is None:
            raise RuntimeError("CuPy is not installed.")
        
        self.n_particles = particles.n_particles
        
        # Allocate GPU arrays (persistent throughout simulation)
        self.pos = cp.asarray(particles.positions, dtype=cp.float32)
        self.vel = cp.asarray(particles.velocities, dtype=cp.float32)
        self.mass = cp.asarray(particles.masses, dtype=cp.float32)
        self.h = cp.asarray(particles.smoothing_lengths, dtype=cp.float32)
        self.u = cp.asarray(particles.internal_energy, dtype=cp.float32)
        self.rho = cp.asarray(particles.density, dtype=cp.float32)
        self.pressure = cp.asarray(particles.pressure, dtype=cp.float32)
        self.cs = cp.asarray(particles.sound_speed, dtype=cp.float32)
        
        # Force arrays
        self.acc_grav = cp.zeros_like(self.pos)
        self.acc_hydro = cp.zeros_like(self.pos)
        self.du_dt = cp.zeros(self.n_particles, dtype=cp.float32)
        
        # Cached octree for gravity and neighbour search
        self.octree = None
        self.octree_theta = 0.5  # Barnes-Hut opening angle
        
        # Track octree validity - only rebuild when particles move significantly
        self.octree_built_positions = None
        self.octree_rebuild_threshold = 0.1  # Rebuild when particles move > 10% of mean h
        
        # Cached neighbour lists (computed once per step, reused for density + hydro)
        self._neighbour_lists = None
        self._neighbour_counts = None
        self._max_neighbours = 64  # Default max neighbours per particle
        
        # Track if data needs re-transfer
        self.data_on_gpu = True
        
    @property
    def neighbour_lists(self):
        return self._neighbour_lists

    @neighbour_lists.setter
    def neighbour_lists(self, value):
        self._neighbour_lists = value

    @property
    def neighbour_counts(self):
        return self._neighbour_counts

    @neighbour_counts.setter
    def neighbour_counts(self, value):
        self._neighbour_counts = value

    @property
    def max_neighbours(self):
        return self._max_neighbours

    @max_neighbours.setter
    def max_neighbours(self, value):
        self._max_neighbours = value

    # Backwards compatibility with previous American spelling attributes
    @property
    def neighbor_lists(self):
        warnings.warn("'neighbor_lists' is deprecated. Use 'neighbour_lists' instead.",
                      DeprecationWarning, stacklevel=2)
        return self._neighbour_lists

    @neighbor_lists.setter
    def neighbor_lists(self, value):
        warnings.warn("'neighbor_lists' is deprecated. Use 'neighbour_lists' instead.",
                      DeprecationWarning, stacklevel=2)
        self._neighbour_lists = value

    @property
    def neighbor_counts(self):
        warnings.warn("'neighbor_counts' is deprecated. Use 'neighbour_counts' instead.",
                      DeprecationWarning, stacklevel=2)
        return self._neighbour_counts

    @neighbor_counts.setter
    def neighbor_counts(self, value):
        warnings.warn("'neighbor_counts' is deprecated. Use 'neighbour_counts' instead.",
                      DeprecationWarning, stacklevel=2)
        self._neighbour_counts = value

    @property
    def max_neighbors(self):
        warnings.warn("'max_neighbors' is deprecated. Use 'max_neighbours' instead.",
                      DeprecationWarning, stacklevel=2)
        return self._max_neighbours

    @max_neighbors.setter
    def max_neighbors(self, value):
        warnings.warn("'max_neighbors' is deprecated. Use 'max_neighbours' instead.",
                      DeprecationWarning, stacklevel=2)
        self._max_neighbours = value

    def update_from_cpu(self, particles, fields=None):
        """
        Update specific fields from CPU particles object.
        
        Parameters
        ----------
        particles : ParticleSystem
            Particle data on CPU
        fields : list of str, optional
            Specific fields to update. If None, updates positions, velocities, h.
        """
        if fields is None:
            # Update only what changes between steps
            fields = ['positions', 'velocities', 'smoothing_lengths']
        
        if 'positions' in fields:
            cp.copyto(self.pos, cp.asarray(particles.positions, dtype=cp.float32))
            # Invalidate caches when positions change
            self.invalidate_neighbours()
        if 'velocities' in fields:
            cp.copyto(self.vel, cp.asarray(particles.velocities, dtype=cp.float32))
        if 'masses' in fields:
            cp.copyto(self.mass, cp.asarray(particles.masses, dtype=cp.float32))
        if 'smoothing_lengths' in fields:
            cp.copyto(self.h, cp.asarray(particles.smoothing_lengths, dtype=cp.float32))
            # Invalidate neighbour cache when smoothing lengths change
            self.invalidate_neighbours()
        if 'internal_energy' in fields:
            cp.copyto(self.u, cp.asarray(particles.internal_energy, dtype=cp.float32))
        if 'density' in fields:
            cp.copyto(self.rho, cp.asarray(particles.density, dtype=cp.float32))
        if 'pressure' in fields:
            cp.copyto(self.pressure, cp.asarray(particles.pressure, dtype=cp.float32))
        if 'sound_speed' in fields:
            cp.copyto(self.cs, cp.asarray(particles.sound_speed, dtype=cp.float32))
            
        self.data_on_gpu = True
        
    def sync_to_host(self, particles):
        """Copy data back to CPU particles object (only when needed)."""
        particles.positions = cp.asnumpy(self.pos)
        particles.velocities = cp.asnumpy(self.vel)
        particles.masses = cp.asnumpy(self.mass)
        particles.smoothing_lengths = cp.asnumpy(self.h)
        particles.internal_energy = cp.asnumpy(self.u)
        particles.density = cp.asnumpy(self.rho)
        particles.pressure = cp.asnumpy(self.pressure)
        particles.sound_speed = cp.asnumpy(self.cs)
        
    def sync_to_device(self, particles):
        """Copy data from CPU particles object to GPU (full refresh)."""
        cp.copyto(self.pos, cp.asarray(particles.positions, dtype=cp.float32))
        cp.copyto(self.vel, cp.asarray(particles.velocities, dtype=cp.float32))
        cp.copyto(self.mass, cp.asarray(particles.masses, dtype=cp.float32))
        cp.copyto(self.h, cp.asarray(particles.smoothing_lengths, dtype=cp.float32))
        cp.copyto(self.u, cp.asarray(particles.internal_energy, dtype=cp.float32))
        self.data_on_gpu = True
        self.invalidate_octree()  # Full refresh invalidates octree
        
    def build_octree(self, theta: float = 0.5, rebuild: bool = False):
        """
        Build or reuse cached octree.
        
        Only rebuilds when particles have moved significantly or when forced.
        
        Parameters
        ----------
        theta : float
            Barnes-Hut opening angle
        rebuild : bool
            Force rebuild even if cached
        """
        # Check if we need to rebuild
        needs_rebuild = rebuild or self.octree is None or self.octree_theta != theta
        
        if not needs_rebuild and self.octree_built_positions is not None:
            # Check if particles have moved significantly
            displacement = cp.linalg.norm(self.pos - self.octree_built_positions, axis=1)
            mean_h = cp.mean(self.h)
            max_displacement = cp.max(displacement)
            relative_movement = max_displacement / mean_h
            
            needs_rebuild = relative_movement > self.octree_rebuild_threshold
        
        if needs_rebuild:
            self.octree = GPUOctree(theta=theta)
            self.octree.build(self.pos, self.mass)
            self.octree_theta = theta
            # Cache positions for next check
            self.octree_built_positions = self.pos.copy()
            # Invalidate neighbour cache when tree rebuilt
            self.invalidate_neighbours()
        # Otherwise reuse existing tree
    
    def compute_neighbours(self, support_radius: float = 2.0, max_neighbours: int = 64):
        """
        Compute neighbour lists using octree (cached for reuse).
        
        Parameters
        ----------
        support_radius : float
            Support radius multiplier
        max_neighbours : int
            Maximum neighbours per particle
            
        Returns
        -------
        neighbour_lists : cp.ndarray
            Neighbour indices (N, max_neighbours)
        neighbour_counts : cp.ndarray
            Number of neighbours per particle (N,)
        """
        if self.neighbour_lists is None or self.max_neighbours != max_neighbours:
            if self.octree is None:
                self.build_octree()

            self._neighbour_lists, self._neighbour_counts = self.octree.find_neighbours(
                self.pos,
                self.h,
                support_radius,
                max_neighbours,
            )
            self._max_neighbours = max_neighbours
        
        return self.neighbour_lists, self.neighbour_counts

    def compute_neighbors(self, support_radius: float = 2.0, max_neighbors: int = 64):
        warnings.warn("'compute_neighbors' is deprecated. Use 'compute_neighbours' instead.",
                      DeprecationWarning, stacklevel=2)
        return self.compute_neighbours(support_radius=support_radius, max_neighbours=max_neighbors)

    def get_neighbour_lists_host(self):
        """Return cached neighbour data as CPU-native structures."""
        if self.neighbour_lists is None or self.neighbour_counts is None:
            return None, None

        lists_gpu = self.neighbour_lists
        counts_gpu = self.neighbour_counts

        counts_cpu = cp.asnumpy(counts_gpu).astype(np.int32, copy=False)
        lists_cpu = cp.asnumpy(lists_gpu).astype(np.int32, copy=False)

        host_lists = [lists_cpu[i, :count].copy() for i, count in enumerate(counts_cpu)]
        return host_lists, counts_cpu
        
    def get_octree(self) -> GPUOctree:
        """Get the cached octree instance."""
        return self.octree
    
    def invalidate_octree(self):
        """Invalidate cached octree (call when particle positions change significantly)."""
        self.octree = None
        self.octree_built_positions = None
        self.invalidate_neighbours()
    
    def invalidate_neighbours(self):
        """Invalidate cached neighbour lists."""
        self._neighbour_lists = None
        self._neighbour_counts = None

    def invalidate_neighbors(self):
        warnings.warn("'invalidate_neighbors' is deprecated. Use 'invalidate_neighbours' instead.",
                      DeprecationWarning, stacklevel=2)
        self.invalidate_neighbours()
    
    def update_smoothing_lengths_octree(self, target_neighbours=50, tolerance=0.05, max_iter=10):
        """
        Update smoothing lengths using octree-based neighbour counting (O(N log N)).
        
        Replaces the O(N^2) brute force approach for massive performance gains at large N.
        """
        if self.octree is None:
            self.build_octree()
        
        h_min_bound, h_max_bound = _estimate_gpu_smoothing_bounds(self.pos, self.h)
        
        for i in range(max_iter):
            # Count neighbours using octree (O(N log N))
            counts = self.octree.count_adaptive_neighbours(self.pos, self.h)
            
            n_actual = cp.maximum(counts, 1.0)
            ratio = n_actual / target_neighbours
            factor = ratio ** (-1.0/3.0)
            
            self.h *= factor
            cp.clip(self.h, h_min_bound, h_max_bound, out=self.h)
            
            error = cp.abs(n_actual - target_neighbours) / target_neighbours
            max_error = cp.max(error)
            
            if max_error < tolerance:
                break
                
        # Invalidate neighbour cache since h changed
        self.invalidate_neighbours()
            
        return self.h


def _estimate_gpu_smoothing_bounds(pos, h, min_scale: float = 1e-2, max_scale: float = 32.0):
    """Compute adaptive smoothing-length bounds directly on the GPU arrays."""
    valid_mask = cp.logical_and(cp.isfinite(h), h > 0.0)
    if not cp.any(valid_mask):
        return 1e-2, 32.0

    h_valid = h[valid_mask]
    base_min = float(cp.min(h_valid).get())
    base_max = float(cp.max(h_valid).get())

    centroid = cp.mean(pos, axis=0)
    offsets = pos - centroid
    extent = float(cp.max(cp.linalg.norm(offsets, axis=1)).get())
    if not np.isfinite(extent):
        extent = 0.0

    h_min_bound = max(base_min * min_scale, 1e-2)
    h_max_candidates = [base_max * max_scale, h_min_bound * 10.0]
    if extent > 0.0:
        h_max_candidates.append(extent)
    h_max_bound = max(h_max_candidates)
    if h_max_bound <= h_min_bound:
        h_max_bound = h_min_bound * 10.0

    return float(h_min_bound), float(h_max_bound)
