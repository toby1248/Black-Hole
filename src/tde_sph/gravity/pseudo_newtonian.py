"""
Pseudo-Newtonian gravity solver using Paczyński-Wiita potential.

Implements TASK-014b (optional): simplified GR approximation for comparison
and algorithm development.

The Paczyński-Wiita (1980) potential mimics GR effects near the ISCO
without full spacetime machinery:
    φ = -G M_BH / (r - r_S)

where r_S = 2 M_BH is the Schwarzschild radius.

Advantages:
- Simple to implement and compute
- Reproduces ISCO at r = 6M (Schwarzschild)
- No coordinate transformations needed

Limitations:
- No frame-dragging (Kerr effects)
- No light bending or other GR effects beyond orbital mechanics
- Singular at r = r_S (not r = 2M as in true GR)
- Approximation breaks down near horizon

Use cases:
- Quick comparison with full GR
- Algorithm development without Metric complexity
- Educational demonstrations

References:
    Paczyński & Wiita (1980), A&A 88, 23
    Tejeda & Rosswog (2013), MNRAS 433, 1930 - Generalized pseudo-Newtonian potentials
"""

from typing import Optional
import numpy as np
from ..core.interfaces import GravitySolver, Metric, NDArrayFloat
from .newtonian import NewtonianGravity


class PseudoNewtonianGravity(GravitySolver):
    """
    Pseudo-Newtonian gravity using Paczyński-Wiita potential.

    Combines Paczyński-Wiita BH potential with Newtonian self-gravity:
        φ_total = φ_PW + φ_self

    where:
        φ_PW = -G M_BH / (r - r_S),  r_S = 2 M_BH
        φ_self = standard Newtonian self-gravity

    Parameters
    ----------
    G : float, optional
        Gravitational constant (default 1.0 in geometric units).
    bh_mass : float, optional
        Black hole mass M_BH (default 1.0).

    Attributes
    ----------
    G : float
        Gravitational constant.
    bh_mass : float
        Black hole mass.
    r_schwarzschild : float
        Schwarzschild radius r_S = 2 M_BH.
    self_gravity_solver : NewtonianGravity
        Solver for self-gravity component.

    Notes
    -----
    **ISCO Radius:**
    The Paczyński-Wiita potential gives ISCO at r = 6M, matching
    Schwarzschild GR. For circular orbits:
        L² = G M_BH r² / (r - 3r_S)

    ISCO occurs when dL²/dr = 0, giving r_ISCO = 6M = 3r_S.

    **Singularity:**
    Potential diverges at r = r_S = 2M. Particles should not
    cross this radius. The solver clamps acceleration to avoid
    numerical instabilities near r_S.

    **Comparison with Full GR:**
    - Agrees qualitatively with Schwarzschild for orbital dynamics
    - Misses frame-dragging (no Kerr analog)
    - Simpler and faster than full metric calculations
    - Good for testing integration schemes and SPH coupling

    Examples
    --------
    >>> solver = PseudoNewtonianGravity(G=1.0, bh_mass=1e6)
    >>> accel = solver.compute_acceleration(pos, masses, h)
    """

    def __init__(self, G: float = 1.0, bh_mass: float = 1.0):
        """
        Initialize pseudo-Newtonian gravity solver.

        Parameters
        ----------
        G : float, optional
            Gravitational constant (default 1.0).
        bh_mass : float, optional
            Black hole mass M_BH (default 1.0).
        """
        self.G = np.float32(G)
        self.bh_mass = np.float32(bh_mass)
        self.r_schwarzschild = 2.0 * self.bh_mass

        # Initialize Newtonian solver for self-gravity
        self.self_gravity_solver = NewtonianGravity(G=G)

    def compute_acceleration(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        metric: Optional[Metric] = None
    ) -> NDArrayFloat:
        """
        Compute pseudo-Newtonian gravitational acceleration.

        Total: a = a_PW + a_self

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions [x, y, z].
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        smoothing_lengths : NDArrayFloat, shape (N,)
            SPH smoothing lengths.
        metric : Optional[Metric], optional
            Ignored (included for interface compatibility).

        Returns
        -------
        accel : NDArrayFloat, shape (N, 3)
            Total gravitational acceleration.

        Notes
        -----
        Paczyński-Wiita acceleration:
            a = -∇φ_PW = -G M_BH r / [r (r - r_S)²]

        For r → r_S, acceleration is clamped to avoid divergence.
        """
        N = len(positions)
        positions = positions.astype(np.float32)
        masses = masses.astype(np.float32)
        smoothing_lengths = smoothing_lengths.astype(np.float32)

        # ===================================================================
        # Component 1: Paczyński-Wiita BH gravity
        # ===================================================================

        # Distance from BH (at origin)
        r_vec = positions  # BH at origin
        r = np.linalg.norm(r_vec, axis=1, keepdims=True)  # Shape: (N, 1)

        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)

        # Modified distance: r - r_S
        r_modified = r_safe - self.r_schwarzschild

        # Clamp to avoid singularity: if r < 1.1 * r_S, clamp acceleration
        # (In practice, particles shouldn't be this close to horizon)
        r_min = 1.1 * self.r_schwarzschild
        r_clamped = np.maximum(r_modified, r_min - self.r_schwarzschild)

        # Paczyński-Wiita acceleration: a = -∇φ
        # φ = -G M_BH / (r - r_S)
        # a = -d/dr[-GM/(r - r_S)] r̂ = -GM/(r - r_S)² r̂
        # In vector form: a = -GM r_vec / [r (r - r_S)²]
        accel_pw = -self.G * self.bh_mass * r_vec / (r_safe * r_clamped**2)

        # ===================================================================
        # Component 2: Self-gravity (Newtonian)
        # ===================================================================

        accel_self = self.self_gravity_solver.compute_acceleration(
            positions, masses, smoothing_lengths, metric=None
        )

        # Total acceleration
        accel_total = accel_pw + accel_self

        return accel_total.astype(np.float32)

    def compute_potential(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute pseudo-Newtonian gravitational potential.

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        smoothing_lengths : NDArrayFloat, shape (N,)
            Smoothing lengths.

        Returns
        -------
        potential : NDArrayFloat, shape (N,)
            Gravitational potential φ = φ_PW + φ_self.

        Notes
        -----
        Paczyński-Wiita potential:
            φ_PW = -G M_BH / (r - r_S)

        Diverges at r = r_S = 2M.
        """
        N = len(positions)
        positions = positions.astype(np.float32)
        masses = masses.astype(np.float32)
        smoothing_lengths = smoothing_lengths.astype(np.float32)

        # Self-gravity potential
        potential_self = self.self_gravity_solver.compute_potential(
            positions, masses, smoothing_lengths
        )

        # Paczyński-Wiita BH potential
        r = np.linalg.norm(positions, axis=1)
        r_safe = np.maximum(r, 1e-10)

        # Modified distance
        r_modified = r_safe - self.r_schwarzschild

        # Clamp to avoid singularity
        r_min = 1.1 * self.r_schwarzschild
        r_clamped = np.maximum(r_modified, r_min - self.r_schwarzschild)

        potential_pw = -self.G * self.bh_mass / r_clamped

        return (potential_pw + potential_self).astype(np.float32)

    def __repr__(self) -> str:
        """String representation of solver."""
        return (
            f"PseudoNewtonianGravity("
            f"G={self.G}, M_BH={self.bh_mass}, r_S={self.r_schwarzschild})"
        )
