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

    def __init__(
        self,
        G: float = 1.0,
        bh_mass: float = 0.0,
        bh_position: Optional[np.ndarray] = None,
    ):
        """
        Initialize Newtonian gravity solver.

        Parameters
        ----------
        G : float, optional
            Gravitational constant (default 1.0 for dimensionless units).
        bh_mass : float, optional
            Optional central point mass (e.g., black hole). Default 0 disables it.
        bh_position : array-like, optional
            Cartesian position of central mass (default origin).
        """
        self.G = np.float32(G)
        self.bh_mass = np.float32(bh_mass)
        if bh_position is None:
            self.bh_position = np.zeros(3, dtype=np.float32)
        else:
            pos = np.asarray(bh_position, dtype=np.float32)
            if pos.shape != (3,):
                raise ValueError("bh_position must be a 3-vector")
            self.bh_position = pos

    def _compute_bh_acceleration(self, positions: NDArrayFloat) -> NDArrayFloat:
        """Acceleration from optional central mass."""
        if self.bh_mass <= 0.0:
            return np.zeros_like(positions, dtype=np.float32)

        r_vec = positions - self.bh_position[np.newaxis, :]
        r = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_safe = np.maximum(r, 1e-6).astype(np.float32)
        accel = -self.G * self.bh_mass * r_vec / (r_safe**3)
        return accel.astype(np.float32)

    def _compute_bh_potential(self, positions: NDArrayFloat) -> NDArrayFloat:
        """Potential from optional central mass."""
        if self.bh_mass <= 0.0:
            return np.zeros(positions.shape[0], dtype=np.float32)

        r_vec = positions - self.bh_position[np.newaxis, :]
        r = np.linalg.norm(r_vec, axis=1)
        r_safe = np.maximum(r, 1e-6).astype(np.float32)
        phi = -self.G * self.bh_mass / r_safe
        return phi.astype(np.float32)

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
            accel = _compute_accel_numba(positions, masses, smoothing_lengths, self.G)
        else:
            # Fallback to vectorized NumPy implementation
            N = len(positions)

            # Initialize acceleration array
            accel = np.zeros((N, 3), dtype=np.float32)

            # Compute pairwise separations: r_ij = r_j - r_i
            r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

            # Compute squared distances: |r_ij|²
            r2 = np.sum(r_ij**2, axis=2)

            # Softening: use average of smoothing lengths for each pair
            h_i = smoothing_lengths[:, np.newaxis]
            h_j = smoothing_lengths[np.newaxis, :]
            epsilon2 = ((h_i + h_j) / 2.0)**2

            # Softened distance: r_soft² = r² + ε²
            r2_soft = r2 + epsilon2

            # Compute 1 / r_soft³ with safe self-interaction handling
            inv_r3 = np.where(r2 > 0,
                             r2_soft**(-1.5),
                             0.0).astype(np.float32)

            masses_inv_r3 = (masses[np.newaxis, :] * inv_r3)[:, :, np.newaxis]
            accel = self.G * np.sum(masses_inv_r3 * r_ij, axis=1)

        if self.bh_mass > 0.0:
            accel += self._compute_bh_acceleration(positions)

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
            potential = _compute_potential_numba(positions, masses, smoothing_lengths, self.G)
        else:
            N = len(positions)

            r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
            r2 = np.sum(r_ij**2, axis=2)

            h_i = smoothing_lengths[:, np.newaxis]
            h_j = smoothing_lengths[np.newaxis, :]
            epsilon2 = ((h_i + h_j) / 2.0)**2

            r_soft = np.sqrt(r2 + epsilon2)
            np.fill_diagonal(r_soft, np.inf)

            potential = -self.G * np.sum(masses[np.newaxis, :] / r_soft, axis=1)

        if self.bh_mass > 0.0:
            potential += self._compute_bh_potential(positions)

        return potential.astype(np.float32)

    def __repr__(self) -> str:
        """String representation of solver."""
        return f"NewtonianGravity(G={self.G})"
