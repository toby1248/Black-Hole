"""
Kerr metric implementation for rotating black holes.

This module implements the Kerr solution describing a rotating black hole
in Boyer-Lindquist coordinates, including frame-dragging and ergosphere effects.

References
----------
- Kerr, R. P. (1963), Phys. Rev. Lett. 11, 237
- Boyer & Lindquist (1967), J. Math. Phys. 8, 265
- Bardeen, Press & Teukolsky (1972), ApJ 178, 347
- Tejeda et al. (2017), MNRAS 469, 4483 [arXiv:1701.00303]
- Liptai et al. (2019), MNRAS 487, 4790 [arXiv:1910.10154]
"""

import numpy as np
from typing import Tuple
import warnings

from tde_sph.core.interfaces import Metric
from tde_sph.metric.coordinates import (
    cartesian_to_bl_spherical,
    bl_spherical_to_cartesian,
    check_coordinate_validity,
    EPS_COORD
)


class KerrMetric(Metric):
    """
    Kerr metric for rotating black holes in Boyer-Lindquist coordinates.

    Line element (t, r, θ, φ):
        ds² = -(1 - 2Mr/Σ) dt² - (4Mra sin²θ/Σ) dt dφ
              + (Σ/Δ) dr² + Σ dθ² + (A sin²θ/Σ) dφ²

    Where:
        Δ = r² - 2Mr + a²
        Σ = r² + a² cos²θ
        A = (r² + a²)² - a² Δ sin²θ

    Key parameters:
    - a: dimensionless spin parameter a/M ∈ [0, 1] (extremal at a = M)
    - Event horizon: r₊ = M + √(M² - a²)
    - Ergosphere: r_ergo = M + √(M² - a² cos²θ)
    - ISCO: spin-dependent, prograde orbit: 1M to 9M depending on a

    Frame-dragging:
    - Non-zero g_tφ component induces Lense-Thirring precession
    - Particles at rest in BL coordinates are dragged along φ direction

    Parameters
    ----------
    mass : float, default=1.0
        Black hole mass M in geometric units (G = c = 1).
    spin : float, default=0.0
        Dimensionless spin parameter a/M ∈ [0, 1].
        - a = 0: Schwarzschild limit
        - a = 1: extremal Kerr (maximal spin)

    Attributes
    ----------
    mass : float
        Black hole mass M.
    spin : float
        Dimensionless spin a/M.
    a : float
        Spin parameter a (dimensional, in geometric units).
    r_plus : float
        Outer event horizon radius.
    r_minus : float
        Inner event horizon radius (Cauchy horizon).

    Notes
    -----
    Uses FP64 for metric computations near the horizon and ergosphere.
    """

    def __init__(self, mass: float = 1.0, spin: float = 0.0):
        """
        Initialize Kerr metric.

        Parameters
        ----------
        mass : float, default=1.0
            Black hole mass M.
        spin : float, default=0.0
            Dimensionless spin a/M ∈ [0, 1].
        """
        if mass <= 0:
            raise ValueError(f"Black hole mass must be positive, got {mass}")
        if not (0 <= spin <= 1):
            raise ValueError(f"Spin parameter must be in [0, 1], got {spin}")

        self.mass = float(mass)
        self.spin = float(spin)
        self.a = self.spin * self.mass  # Dimensional spin parameter

        # Compute horizons
        self.r_plus = self.mass + np.sqrt(self.mass**2 - self.a**2)
        self.r_minus = self.mass - np.sqrt(self.mass**2 - self.a**2)

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"Kerr (M={self.mass:.2f}, a={self.spin:.2f})"

    def event_horizon(self) -> float:
        """Return outer event horizon radius r₊."""
        return self.r_plus

    def isco_radius(self, prograde: bool = True) -> float:
        """
        Compute ISCO radius for circular equatorial orbits.

        Parameters
        ----------
        prograde : bool, default=True
            If True, compute prograde ISCO. If False, retrograde.

        Returns
        -------
        r_isco : float
            ISCO radius in geometric units.

        Notes
        -----
        Prograde ISCO formula (Bardeen, Press & Teukolsky 1972):
            r_ISCO = M (3 + Z₂ - sign * √[(3 - Z₁)(3 + Z₁ + 2Z₂)])

        Where:
            Z₁ = 1 + (1 - a²)^(1/3) [(1 + a)^(1/3) + (1 - a)^(1/3)]
            Z₂ = √(3a² + Z₁²)
            sign = +1 for prograde, -1 for retrograde

        Range: r_ISCO ∈ [M, 9M] for a ∈ [1, 0] (prograde)
        """
        M = self.mass
        a_dim = self.a  # Dimensional spin

        # Handle Schwarzschild limit
        if np.abs(a_dim) < 1e-6:
            return 6.0 * M

        # Compute Z₁ and Z₂ (Bardeen et al. 1972)
        a_norm = a_dim / M  # Normalized spin
        Z1 = 1.0 + (1.0 - a_norm**2)**(1./3.) * (
            (1.0 + a_norm)**(1./3.) + (1.0 - a_norm)**(1./3.)
        )
        Z2 = np.sqrt(3.0 * a_norm**2 + Z1**2)

        # Sign depends on orbit direction
        sign = 1.0 if prograde else -1.0

        r_isco = M * (3.0 + Z2 - sign * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))

        return r_isco

    def _compute_metric_functions(
        self,
        r: np.ndarray,
        theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute auxiliary metric functions Δ, Σ, A.

        Parameters
        ----------
        r : np.ndarray
            Radial coordinate.
        theta : np.ndarray
            Polar angle.

        Returns
        -------
        Delta : np.ndarray
            Δ = r² - 2Mr + a²
        Sigma : np.ndarray
            Σ = r² + a² cos²θ
        A_func : np.ndarray
            A = (r² + a²)² - a² Δ sin²θ
        """
        M = self.mass
        a = self.a

        Delta = r**2 - 2.0 * M * r + a**2
        Sigma = r**2 + a**2 * np.cos(theta)**2
        A_func = (r**2 + a**2)**2 - a**2 * Delta * np.sin(theta)**2

        return Delta, Sigma, A_func

    def metric_tensor(self, x: np.ndarray) -> np.ndarray:
        """
        Compute metric tensor g_μν in Boyer-Lindquist coordinates.

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial coordinates in Cartesian (x, y, z).

        Returns
        -------
        g : np.ndarray, shape (N, 4, 4) or (4, 4), dtype=float64
            Metric tensor components.

        Notes
        -----
        Non-zero components:
            g_tt = -(1 - 2Mr/Σ)
            g_tφ = -2Mra sin²θ / Σ  (frame-dragging)
            g_rr = Σ / Δ
            g_θθ = Σ
            g_φφ = A sin²θ / Σ

        Uses FP64 for precision near horizon and ergosphere.
        """
        x = np.atleast_1d(x)
        is_single = x.ndim == 1

        if x.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3, got shape {x.shape}")

        # Convert to spherical
        r, theta, phi = cartesian_to_bl_spherical(x)

        # Check validity
        check_coordinate_validity(r, theta, r_horizon=self.r_plus, warn=True)

        M = self.mass
        a = self.a

        # Compute auxiliary functions
        Delta, Sigma, A_func = self._compute_metric_functions(r, theta)

        sin_theta = np.sin(theta)
        sin2_theta = sin_theta**2

        if is_single:
            g = np.zeros((4, 4), dtype=np.float64)

            g[0, 0] = -(1.0 - 2.0 * M * r / Sigma)
            g[0, 3] = g[3, 0] = -2.0 * M * r * a * sin2_theta / Sigma  # Frame-dragging
            g[1, 1] = Sigma / Delta
            g[2, 2] = Sigma
            g[3, 3] = A_func * sin2_theta / Sigma
        else:
            n = r.shape[0]
            g = np.zeros((n, 4, 4), dtype=np.float64)

            g[:, 0, 0] = -(1.0 - 2.0 * M * r / Sigma)
            g[:, 0, 3] = g[:, 3, 0] = -2.0 * M * r * a * sin2_theta / Sigma
            g[:, 1, 1] = Sigma / Delta
            g[:, 2, 2] = Sigma
            g[:, 3, 3] = A_func * sin2_theta / Sigma

        return g

    def inverse_metric(self, x: np.ndarray) -> np.ndarray:
        """
        Compute inverse metric tensor g^μν.

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial coordinates in Cartesian.

        Returns
        -------
        g_inv : np.ndarray, shape (N, 4, 4) or (4, 4), dtype=float64
            Inverse metric tensor.

        Notes
        -----
        For Kerr, the inverse is non-trivial due to g_tφ ≠ 0.
        Computed via explicit inversion formulas:

            g^tt = -A / (Δ Σ)
            g^tφ = -2Mra / (Δ Σ)
            g^rr = Δ / Σ
            g^θθ = 1 / Σ
            g^φφ = (Δ - a² sin²θ) / (Δ Σ sin²θ)

        Reference: Misner, Thorne & Wheeler (1973), Chapter 33
        """
        x = np.atleast_1d(x)
        is_single = x.ndim == 1

        r, theta, phi = cartesian_to_bl_spherical(x)

        M = self.mass
        a = self.a

        Delta, Sigma, A_func = self._compute_metric_functions(r, theta)

        sin_theta = np.sin(theta)
        sin2_theta = sin_theta**2

        # Protect against division by zero
        sin2_theta = np.where(sin2_theta < EPS_COORD, EPS_COORD, sin2_theta)

        if is_single:
            g_inv = np.zeros((4, 4), dtype=np.float64)

            g_inv[0, 0] = -A_func / (Delta * Sigma)
            g_inv[0, 3] = g_inv[3, 0] = -2.0 * M * r * a / (Delta * Sigma)
            g_inv[1, 1] = Delta / Sigma
            g_inv[2, 2] = 1.0 / Sigma
            g_inv[3, 3] = (Delta - a**2 * sin2_theta) / (Delta * Sigma * sin2_theta)
        else:
            n = r.shape[0]
            g_inv = np.zeros((n, 4, 4), dtype=np.float64)

            g_inv[:, 0, 0] = -A_func / (Delta * Sigma)
            g_inv[:, 0, 3] = g_inv[:, 3, 0] = -2.0 * M * r * a / (Delta * Sigma)
            g_inv[:, 1, 1] = Delta / Sigma
            g_inv[:, 2, 2] = 1.0 / Sigma
            g_inv[:, 3, 3] = (Delta - a**2 * sin2_theta) / (Delta * Sigma * sin2_theta)

        return g_inv

    def christoffel_symbols(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^μ_νρ in Boyer-Lindquist coordinates.

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial coordinates in Cartesian.

        Returns
        -------
        gamma : np.ndarray, shape (N, 4, 4, 4) or (4, 4, 4), dtype=float64
            Christoffel symbols.

        Notes
        -----
        For Kerr, Christoffel symbols are significantly more complex than
        Schwarzschild due to frame-dragging and non-diagonal metric.

        We compute using:
            Γ^μ_νρ = (1/2) g^μσ (∂g_σν/∂x^ρ + ∂g_σρ/∂x^ν - ∂g_νρ/∂x^σ)

        Reference: Bardeen, Press & Teukolsky (1972), Appendix
        """
        x = np.atleast_1d(x)
        is_single = x.ndim == 1

        r, theta, phi = cartesian_to_bl_spherical(x)

        M = self.mass
        a = self.a

        Delta, Sigma, A_func = self._compute_metric_functions(r, theta)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin2_theta = sin_theta**2
        cos2_theta = cos_theta**2

        # Protect against division by zero
        sin_theta = np.where(np.abs(sin_theta) < EPS_COORD, EPS_COORD, sin_theta)

        # Partial derivatives of Δ, Σ, A
        dDelta_dr = 2.0 * r - 2.0 * M
        dSigma_dr = 2.0 * r
        dSigma_dtheta = -2.0 * a**2 * cos_theta * sin_theta
        dA_dr = 4.0 * r * (r**2 + a**2) - a**2 * dDelta_dr * sin2_theta
        dA_dtheta = -2.0 * a**2 * Delta * sin_theta * cos_theta

        if is_single:
            gamma = np.zeros((4, 4, 4), dtype=np.float64)

            # Time components (dominant for orbital dynamics)
            # Γ^t_tr and Γ^t_rt
            gamma[0, 0, 1] = gamma[0, 1, 0] = (M * (r**2 - a**2 * cos2_theta) /
                                                (Sigma * (Sigma - 2*M*r)))

            # Γ^t_tθ and Γ^t_θt
            gamma[0, 0, 2] = gamma[0, 2, 0] = (-2.0 * M * r * a**2 * sin_theta * cos_theta /
                                                (Sigma * (Sigma - 2*M*r)))

            # Γ^t_rφ (frame-dragging)
            gamma[0, 1, 3] = gamma[0, 3, 1] = (M * a * (r**2 - a**2 * cos2_theta) /
                                                (Sigma * (Sigma - 2*M*r)))

            # Γ^t_θφ (frame-dragging)
            gamma[0, 2, 3] = gamma[0, 3, 2] = (-2.0 * M * r * a * (r**2 + a**2) *
                                                sin_theta * cos_theta /
                                                (Sigma * (Sigma - 2*M*r)))

            # Radial components (complex, key for tidal forces)
            # Γ^r_tt
            gamma[1, 0, 0] = (M * Delta * (r**2 - a**2 * cos2_theta) / Sigma**3)

            # Γ^r_rr
            gamma[1, 1, 1] = ((M * (r**2 - a**2) - r * Delta) /
                              (Delta * Sigma))

            # Γ^r_θθ
            gamma[1, 2, 2] = -Delta * dSigma_dtheta / (2.0 * Sigma)

            # Γ^r_φφ
            gamma[1, 3, 3] = (-Delta * sin2_theta /
                              Sigma**2 * (r * Sigma + M * (r**2 - a**2 * cos2_theta)))

            # Γ^r_tφ (frame-dragging effect on radial motion)
            gamma[1, 0, 3] = gamma[1, 3, 0] = (M * a * Delta * (r**2 - a**2 * cos2_theta) /
                                                Sigma**3)

            # Theta components
            # Γ^θ_tt
            gamma[2, 0, 0] = (-2.0 * M * r * a**2 * sin_theta * cos_theta / Sigma**3)

            # Γ^θ_rθ
            gamma[2, 1, 2] = gamma[2, 2, 1] = dSigma_dr / (2.0 * Sigma)

            # Γ^θ_φφ
            gamma[2, 3, 3] = (-sin_theta * cos_theta / Sigma**2 *
                              (r**2 + a**2 + 2*M*r*a**2*sin2_theta/Sigma))

            # Γ^θ_tφ
            gamma[2, 0, 3] = gamma[2, 3, 0] = (-2.0 * M * r * a * sin_theta * cos_theta *
                                                (r**2 + a**2) / Sigma**3)

            # Phi components (frame-dragging)
            # Γ^φ_tr
            gamma[3, 0, 1] = gamma[3, 1, 0] = (2.0 * M * r * a / (Sigma * A_func))

            # Γ^φ_tθ
            gamma[3, 0, 2] = gamma[3, 2, 0] = (-2.0 * M * r * a * a**2 * sin_theta * cos_theta /
                                                (Sigma * A_func))

            # Γ^φ_rφ
            gamma[3, 1, 3] = gamma[3, 3, 1] = (r / Sigma +
                                                M * a**2 * sin2_theta * dSigma_dr /
                                                (Sigma * A_func))

            # Γ^φ_θφ
            gamma[3, 2, 3] = gamma[3, 3, 2] = (cos_theta / sin_theta +
                                                sin_theta * cos_theta *
                                                (2*r**2 + 2*M*r*a**2/Sigma) / Sigma)

        else:
            # Batched version (similar structure, vectorized)
            n = r.shape[0]
            gamma = np.zeros((n, 4, 4, 4), dtype=np.float64)

            # Implementation note: Full batched Kerr Christoffel symbols are lengthy.
            # For production use, consider precomputed tables or symbolic computation.
            # Here we implement key components for orbital dynamics:

            # Time-radial coupling
            gamma[:, 0, 0, 1] = gamma[:, 0, 1, 0] = (
                M * (r**2 - a**2 * cos2_theta) / (Sigma * (Sigma - 2*M*r))
            )

            # Radial components
            gamma[:, 1, 0, 0] = M * Delta * (r**2 - a**2 * cos2_theta) / Sigma**3
            gamma[:, 1, 1, 1] = (M * (r**2 - a**2) - r * Delta) / (Delta * Sigma)

            # Frame-dragging terms (essential for Kerr dynamics)
            gamma[:, 0, 1, 3] = gamma[:, 0, 3, 1] = (
                M * a * (r**2 - a**2 * cos2_theta) / (Sigma * (Sigma - 2*M*r))
            )

            # Additional components would be added here for full implementation
            # TODO: Complete all non-zero Christoffel components for batched case

        return gamma

    def geodesic_acceleration(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute spatial geodesic acceleration in Kerr spacetime.

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial position in Cartesian coordinates.
        v : np.ndarray, shape (N, 4) or (4,)
            4-velocity (u^t, u^r, u^θ, u^φ) in Boyer-Lindquist coordinates.

        Returns
        -------
        a : np.ndarray, shape (N, 3) or (3,)
            Spatial acceleration in Cartesian coordinates.

        Notes
        -----
        Geodesic equation: d²x^μ/dτ² = -Γ^μ_νρ u^ν u^ρ

        For Kerr, this includes:
        - Gravitational attraction
        - Frame-dragging (Lense-Thirring effect)
        - Spin-orbit coupling

        Reference: Tejeda et al. (2017), Eqs. 15-17
        """
        x = np.atleast_1d(x)
        v = np.atleast_1d(v)
        is_single = x.ndim == 1

        # Get Christoffel symbols and compute acceleration
        gamma = self.christoffel_symbols(x)

        # Extract 4-velocity
        u_t = v[..., 0]
        u_r = v[..., 1]
        u_theta = v[..., 2]
        u_phi = v[..., 3]

        # Compute acceleration: a^μ = -Γ^μ_νρ u^ν u^ρ
        # This is a contraction over ν, ρ indices

        if is_single:
            # a^i = -Γ^i_νρ u^ν u^ρ (sum over ν, ρ)
            u = np.array([u_t, u_r, u_theta, u_phi])
            a_sph = np.zeros(4, dtype=np.float64)

            for mu in range(4):
                for nu in range(4):
                    for rho in range(4):
                        a_sph[mu] -= gamma[mu, nu, rho] * u[nu] * u[rho]

            # Extract spatial components and convert to Cartesian
            r, theta, phi = cartesian_to_bl_spherical(x)
            a_r, a_theta, a_phi = a_sph[1], a_sph[2], a_sph[3]

        else:
            # Batched version
            u = np.stack([u_t, u_r, u_theta, u_phi], axis=-1)
            a_sph = np.zeros((x.shape[0], 4), dtype=np.float64)

            for mu in range(4):
                for nu in range(4):
                    for rho in range(4):
                        a_sph[:, mu] -= gamma[:, mu, nu, rho] * u[:, nu] * u[:, rho]

            r, theta, phi = cartesian_to_bl_spherical(x)
            a_r = a_sph[:, 1]
            a_theta = a_sph[:, 2]
            a_phi = a_sph[:, 3]

        # Transform to Cartesian coordinates
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        a_x = (a_r * sin_theta * cos_phi +
               r * a_theta * cos_theta * cos_phi -
               r * a_phi * sin_theta * sin_phi)

        a_y = (a_r * sin_theta * sin_phi +
               r * a_theta * cos_theta * sin_phi +
               r * a_phi * sin_theta * cos_phi)

        a_z = a_r * cos_theta - r * a_theta * sin_theta

        if is_single:
            a = np.array([a_x, a_y, a_z], dtype=np.float32)
        else:
            a = np.stack([a_x, a_y, a_z], axis=-1).astype(np.float32)

        return a

    def ergosphere_radius(self, theta: float) -> float:
        """
        Compute ergosphere radius at polar angle θ.

        Parameters
        ----------
        theta : float
            Polar angle.

        Returns
        -------
        r_ergo : float
            Ergosphere radius at θ.

        Notes
        -----
        Ergosphere: r_ergo(θ) = M + √(M² - a² cos²θ)

        Between r₊ and r_ergo, spacetime is dragged along φ direction
        (no static observers possible).
        """
        M = self.mass
        a = self.a
        r_ergo = M + np.sqrt(M**2 - a**2 * np.cos(theta)**2)
        return r_ergo
