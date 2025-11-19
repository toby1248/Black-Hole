"""
Newtonian gravity solver for self-gravitating SPH particles.

Implements REQ-004 (dynamic timesteps), REQ-006 (self-gravity).

This module provides direct O(N²) gravitational force calculation with
softening set to the SPH smoothing length. Tree-based acceleration
(Barnes-Hut) is a planned optimization.

Reference:
    Price & Monaghan (2007) - SPH self-gravity implementations
    Tejeda et al. (2017) - Newtonian self-gravity in hybrid GR framework
"""

from typing import Optional
import numpy as np
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback for type hinting or if numba is missing (though performance will suffer)
    prange = range
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..core.interfaces import GravitySolver, Metric, NDArrayFloat


@njit(parallel=True, fastmath=True)
def _compute_accel_numba(positions, masses, smoothing_lengths, G):
    """JIT-compiled N-body acceleration."""
    N = len(positions)
    accel = np.zeros((N, 3), dtype=np.float32)
    
    for i in prange(N):
        # Pre-load particle i data
        pos_i_x = positions[i, 0]
        pos_i_y = positions[i, 1]
        pos_i_z = positions[i, 2]
        h_i = smoothing_lengths[i]
        
        ax = 0.0
        ay = 0.0
        az = 0.0
        
        for j in range(N):
            if i == j:
                continue
            
            # Vector r_ij = r_i - r_j
            dx = pos_i_x - positions[j, 0]
            dy = pos_i_y - positions[j, 1]
            dz = pos_i_z - positions[j, 2]
            
            r2 = dx*dx + dy*dy + dz*dz
            
            # Softening: epsilon^2 = ((h_i + h_j)/2)^2
            h_j = smoothing_lengths[j]
            epsilon = (h_i + h_j) * 0.5
            epsilon2 = epsilon * epsilon
            
            r2_soft = r2 + epsilon2
            inv_r3 = r2_soft**(-1.5)
            
            # Force: -G * m_j * (r_i - r_j) / r_soft^3
            # We want to add to acceleration of i
            factor = -G * masses[j] * inv_r3
            
            ax += factor * dx
            ay += factor * dy
            az += factor * dz
            
        accel[i, 0] = ax
        accel[i, 1] = ay
        accel[i, 2] = az
        
    return accel


@njit(parallel=True, fastmath=True)
def _compute_potential_numba(positions, masses, smoothing_lengths, G):
    """JIT-compiled N-body potential."""
    N = len(positions)
    potential = np.zeros(N, dtype=np.float32)
    
    for i in prange(N):
        pos_i_x = positions[i, 0]
        pos_i_y = positions[i, 1]
        pos_i_z = positions[i, 2]
        h_i = smoothing_lengths[i]
        
        phi = 0.0
        
        for j in range(N):
            if i == j:
                continue
                
            dx = pos_i_x - positions[j, 0]
            dy = pos_i_y - positions[j, 1]
            dz = pos_i_z - positions[j, 2]
            
            r2 = dx*dx + dy*dy + dz*dz
            
            h_j = smoothing_lengths[j]
            epsilon = (h_i + h_j) * 0.5
            epsilon2 = epsilon * epsilon
            
            r_soft = np.sqrt(r2 + epsilon2)
            
            # Potential: -G * m_j / r_soft
            phi += -G * masses[j] / r_soft
            
        potential[i] = phi
        
    return potential


class NewtonianGravity(GravitySolver):
    """
    Newtonian self-gravity solver using direct particle-particle summation.

    Uses gravitational softening equal to the smoothing length to avoid
    singularities and maintain consistency with SPH density estimation.

    Parameters
    ----------
    G : float, optional
        Gravitational constant (default 1.0 for dimensionless units, GUD-003).

    Attributes
    ----------
    G : float
        Gravitational constant.

    Notes
    -----
    Softened gravitational acceleration:
        a_i = -∑_j G m_j (r_i - r_j) / (|r_i - r_j|² + ε²)^(3/2)

    where ε = smoothing_length is the softening scale.

    Softened gravitational potential:
        φ_i = -∑_j G m_j / sqrt(|r_i - r_j|² + ε²)

    The current implementation is O(N²) and suitable for N ≲ 10⁵ particles.
    Future optimization: tree-based solver (Barnes-Hut or FMM) for larger N.
    """

    def __init__(self, G: float = 1.0):
        """
        Initialize Newtonian gravity solver.

        Parameters
        ----------
        G : float, optional
            Gravitational constant (default 1.0 for dimensionless units).
        """
        self.G = np.float32(G)

    def compute_acceleration(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        metric: Optional[Metric] = None
    ) -> NDArrayFloat:
        """
        Compute Newtonian gravitational acceleration on all particles.

        Uses direct O(N²) pairwise summation with softening equal to
        the smoothing length. Optimized with Numba if available.

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions [x, y, z].
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        smoothing_lengths : NDArrayFloat, shape (N,)
            SPH smoothing lengths (used for softening ε).
        metric : Optional[Metric], optional
            Spacetime metric (not used in Newtonian solver).

        Returns
        -------
        accel : NDArrayFloat, shape (N, 3)
            Gravitational acceleration on each particle [ax, ay, az].
        """
        positions = positions.astype(np.float32)
        masses = masses.astype(np.float32)
        smoothing_lengths = smoothing_lengths.astype(np.float32)

        if HAS_NUMBA:
            return _compute_accel_numba(positions, masses, smoothing_lengths, self.G)
        
        # Fallback to vectorized NumPy implementation
        N = len(positions)
        # ... (rest of original numpy implementation if needed, but we'll just use the Numba one or the old one)
        # For brevity in this edit, I'll keep the old implementation as fallback logic
        
        # Initialize acceleration array
        accel = np.zeros((N, 3), dtype=np.float32)

        # Compute pairwise separations: r_ij = r_j - r_i
        # Shape: (N, N, 3)
        r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

        # Compute squared distances: |r_ij|²
        # Shape: (N, N)
        r2 = np.sum(r_ij**2, axis=2)

        # Softening: use average of smoothing lengths for each pair
        # ε² = ((h_i + h_j) / 2)²
        # Shape: (N, N)
        h_i = smoothing_lengths[:, np.newaxis]
        h_j = smoothing_lengths[np.newaxis, :]
        epsilon2 = ((h_i + h_j) / 2.0)**2

        # Softened distance: r_soft² = r² + ε²
        r2_soft = r2 + epsilon2

        # Compute 1 / r_soft³
        # Shape: (N, N)
        # Use np.where to handle r=0 (self-interaction) gracefully
        inv_r3 = np.where(r2 > 0,
                         r2_soft**(-1.5),
                         0.0).astype(np.float32)

        # Gravitational force per unit mass: F_ij / m_i = G m_j r_ij / r_soft³
        # Acceleration: a_i = ∑_j F_ij / m_i
        # Shape: (N, 3)
        # Broadcasting: (N, N, 1) * (N, N, 3) * (N, N, 1) → (N, N, 3) → sum over j
        masses_inv_r3 = (masses[np.newaxis, :] * inv_r3)[:, :, np.newaxis]
        accel = self.G * np.sum(masses_inv_r3 * r_ij, axis=1)

        return accel.astype(np.float32)

    def compute_potential(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute Newtonian gravitational potential at each particle.

        Uses softened potential with ε = smoothing_length. Optimized with Numba.
        """
        positions = positions.astype(np.float32)
        masses = masses.astype(np.float32)
        smoothing_lengths = smoothing_lengths.astype(np.float32)

        if HAS_NUMBA:
            return _compute_potential_numba(positions, masses, smoothing_lengths, self.G)

        # Fallback
        N = len(positions)

        # Compute pairwise distances
        # r_ij = positions[j] - positions[i]
        r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        r2 = np.sum(r_ij**2, axis=2)

        # Softening
        h_i = smoothing_lengths[:, np.newaxis]
        h_j = smoothing_lengths[np.newaxis, :]
        epsilon2 = ((h_i + h_j) / 2.0)**2

        # Softened distance
        r_soft = np.sqrt(r2 + epsilon2)

        # Avoid self-interaction (i=j)
        # Set diagonal to large value to avoid division issues
        np.fill_diagonal(r_soft, np.inf)

        # Potential: φ_i = -G ∑_j m_j / r_soft
        # Shape: (N,)
        potential = -self.G * np.sum(masses[np.newaxis, :] / r_soft, axis=1)

        return potential.astype(np.float32)

    def __repr__(self) -> str:
        """String representation of solver."""
        return f"NewtonianGravity(G={self.G})"
