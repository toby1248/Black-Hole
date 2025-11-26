"""
Hybrid relativistic gravity solver for TDE simulations.

Implements REQ-002, REQ-006, TASK-014, TASK-018:
Combines exact GR motion in fixed black hole spacetime with Newtonian self-gravity.

Hybrid approach following Tejeda et al. (2017):
    a_total = a_BH(GR) + a_self(Newtonian)

where:
- a_BH(GR): Geodesic acceleration from Metric.geodesic_acceleration()
- a_self(Newtonian): Self-gravity from NewtonianGravity solver

This approximation is valid when:
1. Stellar self-binding energy << BH gravitational binding energy
2. Internal stellar velocities << c
3. Particle separations >> Schwarzschild radius

References:
    Tejeda et al. (2017), MNRAS 469, 4483 [arXiv:1701.00303]
    Liptai & Price (2019), MNRAS 485, 819 [arXiv:1901.08064]
"""

from typing import Optional
import numpy as np
from ..core.interfaces import GravitySolver, Metric, NDArrayFloat
from .newtonian import NewtonianGravity


class RelativisticGravitySolver(GravitySolver):
    """
    Hybrid gravity solver combining GR black hole and Newtonian self-gravity.

    Implements the hybrid relativistic formulation from Tejeda et al. (2017):
    particles move on geodesics in the fixed BH spacetime, perturbed by
    Newtonian self-gravity from other SPH particles.

    Parameters
    ----------
    G : float, optional
        Gravitational constant for self-gravity (default 1.0 in geometric units).
    bh_mass : float, optional
        Black hole mass M_BH (default 1.0 in geometric units where G=c=M_BH=1).

    Attributes
    ----------
    self_gravity_solver : NewtonianGravity
        Solver for Newtonian self-gravity component.
    bh_mass : float
        Black hole mass.
    G : float
        Gravitational constant.

    Notes
    -----
    **Mode Toggle (TASK-018):**
    If `metric=None` is passed to compute_acceleration, the solver falls back
    to pure Newtonian gravity (BH treated as point mass, no GR corrections).

    **Hybrid Approximation:**
    The total acceleration is:
        a_i = a_i^BH(GR) + a_i^self(Newtonian)

    where:
        a_i^BH(GR) = metric.geodesic_acceleration(x_i, v_i)
        a_i^self(Newtonian) = -∑_{j≠i} G m_j (r_i - r_j) / |r_i - r_j|^3

    This is NOT derived from a single Hamiltonian, so energy is not
    exactly conserved. However, the approximation is well-justified for TDEs
    where self-gravity corrections are small compared to BH tidal forces.

    **Coordinate Systems:**
    - SPH particles use Cartesian coordinates (x, y, z)
    - Metric may use spherical (r, θ, φ) internally
    - geodesic_acceleration() handles conversions

    **Precision:**
    - Particle arrays: FP32 (memory efficiency on GPU)
    - Metric calls: FP64 (precision near horizon)
    - Mixed precision handled internally

    **Validation (TASK-019, TASK-020):**
    - Test-particle limit (set all masses to zero except one)
    - Newtonian limit at r >> 6M (compare to pure Newtonian)
    - ISCO behavior at r ≈ 6M (Schwarzschild)
    """

    def __init__(self, G: float = 1.0, bh_mass: float = 1.0):
        """
        Initialize hybrid relativistic gravity solver.

        Parameters
        ----------
        G : float, optional
            Gravitational constant (default 1.0 for geometric units).
        bh_mass : float, optional
            Black hole mass M_BH (default 1.0).
        """
        self.G = np.float32(G)
        self.bh_mass = np.float64(bh_mass)  # FP64 for metric computations

        # Initialize Newtonian solver for self-gravity component
        self.self_gravity_solver = NewtonianGravity(G=G)

    def compute_acceleration(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        metric: Optional[Metric] = None,
        velocities: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat:
        """
        Compute hybrid GR+Newtonian gravitational acceleration.

        Total acceleration: a = a_BH(GR) + a_self(Newtonian)

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions in Cartesian coordinates [x, y, z].
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        smoothing_lengths : NDArrayFloat, shape (N,)
            SPH smoothing lengths (used for softening in self-gravity).
        metric : Optional[Metric], optional
            Spacetime metric for BH gravity. If None, falls back to
            pure Newtonian gravity (mode toggle, TASK-018).
        velocities : Optional[NDArrayFloat], shape (N, 3), optional
            Particle velocities [vx, vy, vz]. Required for GR mode
            (metric != None), ignored in Newtonian mode.

        Returns
        -------
        accel : NDArrayFloat, shape (N, 3)
            Total gravitational acceleration [ax, ay, az].

        Raises
        ------
        ValueError
            If metric is provided but velocities are None.

        Notes
        -----
        **Newtonian Mode (metric=None):**
        Treats BH as a point mass at origin with Newtonian potential:
            φ_BH = -G M_BH / r
            a_BH = -G M_BH r / r³
        Plus Newtonian self-gravity from all particles.

        **GR Mode (metric != None):**
        Uses metric.geodesic_acceleration() for BH component,
        plus Newtonian self-gravity.

        **Performance:**
        - Self-gravity: O(N²) direct summation (tree-based in future)
        - BH gravity: O(N) independent evaluations

        Examples
        --------
        >>> # Newtonian mode
        >>> solver = RelativisticGravitySolver(G=1.0, bh_mass=1e6)
        >>> accel = solver.compute_acceleration(pos, masses, h, metric=None)

        >>> # GR mode with Schwarzschild metric
        >>> from tde_sph.metric import SchwarzschildMetric
        >>> metric = SchwarzschildMetric(M=1e6)
        >>> accel = solver.compute_acceleration(pos, masses, h, metric, velocities=vel)
        """
        N = len(positions)
        positions = positions.astype(np.float32)
        masses = masses.astype(np.float32)
        smoothing_lengths = smoothing_lengths.astype(np.float32)

        # Initialize total acceleration
        accel_total = np.zeros((N, 3), dtype=np.float32)

        # ===================================================================
        # Component 1: Black hole gravity
        # ===================================================================

        if metric is None:
            # Newtonian mode: BH as point mass at origin
            accel_bh = self._compute_bh_newtonian(positions)
        else:
            # GR mode: Use metric's geodesic acceleration
            if velocities is None:
                raise ValueError(
                    "velocities are required for GR mode (metric != None). "
                    "Provide Cartesian 3-velocity components."
                )

            velocities = velocities.astype(np.float32)
            accel_bh = self._compute_bh_relativistic(positions, velocities, metric)

        accel_total += accel_bh

        # ===================================================================
        # Component 2: Self-gravity (always Newtonian)
        # ===================================================================

        accel_self = self.self_gravity_solver.compute_acceleration(
            positions, masses, smoothing_lengths, metric=None
        )

        accel_total += accel_self

        return accel_total.astype(np.float32)

    def _compute_bh_newtonian(self, positions: NDArrayFloat) -> NDArrayFloat:
        """
        Compute Newtonian BH acceleration: a = -G M_BH r / r³.

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.

        Returns
        -------
        accel : NDArrayFloat, shape (N, 3)
            BH acceleration (Newtonian).
        """
        N = len(positions)

        # Distance from BH (at origin)
        r_vec = positions  # r = x (BH at origin)
        r = np.linalg.norm(r_vec, axis=1, keepdims=True)  # Shape: (N, 1)

        # Avoid division by zero (particles at origin)
        r_safe = np.where(r > 0, r, np.inf)

        # Newtonian acceleration: a = -G M_BH r / r³
        accel = -self.G * self.bh_mass * r_vec / (r_safe**3)

        return accel.astype(np.float32)

    def _compute_bh_relativistic(
        self,
        positions: NDArrayFloat,
        velocities: NDArrayFloat,
        metric: Metric
    ) -> NDArrayFloat:
        """
        Compute GR BH acceleration using metric's geodesic equation.

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.
        velocities : NDArrayFloat, shape (N, 3)
            Particle velocities.
        metric : Metric
            Spacetime metric.

        Returns
        -------
        accel : NDArrayFloat, shape (N, 3)
            BH acceleration (GR).

        Notes
        -----
        Metric.geodesic_acceleration() expects Cartesian positions and
        velocities and handles coordinate transformations internally.
        """
        positions_fp64 = positions.astype(np.float64)
        velocities_fp64 = velocities.astype(np.float64)

        accel = metric.geodesic_acceleration(positions_fp64, velocities_fp64)

        return accel.astype(np.float32)

    def compute_potential(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        metric: Optional[Metric] = None
    ) -> NDArrayFloat:
        """
        Compute gravitational potential at each particle.

        WARNING: In GR mode (metric != None), there is no unique scalar
        potential because spacetime is curved. This method returns only
        the Newtonian self-gravity potential in that case.

        For full energy accounting in GR, use conserved Hamiltonian
        (not implemented in this solver).

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        smoothing_lengths : NDArrayFloat, shape (N,)
            Smoothing lengths.
        metric : Optional[Metric], optional
            Spacetime metric (ignored for potential calculation).

        Returns
        -------
        potential : NDArrayFloat, shape (N,)
            Gravitational potential φ.

            - Newtonian mode: φ = φ_BH + φ_self
            - GR mode: φ = φ_self only (BH potential not well-defined)

        Notes
        -----
        In Newtonian mode:
            φ_BH = -G M_BH / r
            φ_self = -∑_j G m_j / |r_i - r_j|

        In GR mode:
            Only φ_self is returned (Newtonian approximation).
            For proper GR energetics, use Killing vectors and conserved
            quantities (e.g., E = -u_t for Schwarzschild).
        """
        N = len(positions)
        positions = positions.astype(np.float32)
        masses = masses.astype(np.float32)
        smoothing_lengths = smoothing_lengths.astype(np.float32)

        # Self-gravity potential (always Newtonian)
        potential_self = self.self_gravity_solver.compute_potential(
            positions, masses, smoothing_lengths
        )

        if metric is None:
            # Newtonian mode: add BH potential
            r = np.linalg.norm(positions, axis=1)
            r_safe = np.where(r > 0, r, np.inf)
            potential_bh = -self.G * self.bh_mass / r_safe

            return (potential_bh + potential_self).astype(np.float32)
        else:
            # GR mode: only self-gravity potential is meaningful
            # (BH contribution requires GR energy formalism)
            return potential_self.astype(np.float32)

    def __repr__(self) -> str:
        """String representation of solver."""
        return (
            f"RelativisticGravitySolver("
            f"G={self.G}, M_BH={self.bh_mass})"
        )
