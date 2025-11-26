"""
Schwarzschild metric implementation for non-rotating black holes.

This module implements the Schwarzschild solution to Einstein's equations,
describing a static, spherically symmetric black hole in Boyer-Lindquist
coordinates.

References
----------
- Schwarzschild, K. (1916), Sitzungsber. Preuss. Akad. Wiss., 189
- Misner, Thorne & Wheeler (1973) - Gravitation, Chapter 23
- Tejeda et al. (2017), MNRAS 469, 4483 [arXiv:1701.00303]
- Liptai & Price (2019), MNRAS 485, 819 [arXiv:1901.08064]
"""

import numpy as np
from typing import Tuple
import warnings

from tde_sph.core.interfaces import Metric
from tde_sph.metric.coordinates import (
    cartesian_to_bl_spherical,
    velocity_cartesian_to_bl,
    acceleration_bl_to_cartesian,
    check_coordinate_validity,
    EPS_COORD
)


class SchwarzschildMetric(Metric):
    """
    Schwarzschild metric for non-rotating black holes.

    Line element in Boyer-Lindquist spherical coordinates (t, r, θ, φ):
        ds² = -(1 - 2M/r) dt² + (1 - 2M/r)⁻¹ dr² + r² dθ² + r² sin²θ dφ²

    Metric tensor components:
        g_tt = -(1 - 2M/r)
        g_rr = (1 - 2M/r)⁻¹
        g_θθ = r²
        g_φφ = r² sin²θ

    Key features:
    - Event horizon at r_H = 2M
    - ISCO (innermost stable circular orbit) at r_ISCO = 6M
    - Coordinate singularity at r = 2M (physical singularity at r = 0)

    Parameters
    ----------
    mass : float, default=1.0
        Black hole mass M in geometric units (G = c = 1).
        In code units with M_BH = 1, mass = 1.0.

    Attributes
    ----------
    mass : float
        Black hole mass M.
    r_schwarzschild : float
        Schwarzschild radius (event horizon): r_s = 2M.
    r_isco : float
        ISCO radius: 6M for Schwarzschild.

    Notes
    -----
    Uses FP64 for metric computations to maintain precision near the horizon.
    Particle arrays can remain FP32, but metric tensors use FP64.
    """

    def __init__(self, mass: float = 1.0):
        """
        Initialize Schwarzschild metric.

        Parameters
        ----------
        mass : float, default=1.0
            Black hole mass in geometric units.
        """
        if mass <= 0:
            raise ValueError(f"Black hole mass must be positive, got {mass}")

        self.mass = float(mass)
        self.r_schwarzschild = 2.0 * self.mass
        self.r_isco = 6.0 * self.mass

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"Schwarzschild (M={self.mass:.2f})"

    def event_horizon(self) -> float:
        """Return event horizon radius r_H = 2M."""
        return self.r_schwarzschild

    def isco_radius(self) -> float:
        """Return ISCO radius r_ISCO = 6M."""
        return self.r_isco

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
        Converts Cartesian to spherical coordinates internally, then computes:
            g_00 = -(1 - 2M/r)
            g_11 = (1 - 2M/r)⁻¹
            g_22 = r²
            g_33 = r² sin²θ

        Uses FP64 for precision near the horizon.
        """
        x = np.atleast_1d(x)
        is_single = x.ndim == 1

        if x.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3, got shape {x.shape}")

        # Convert to spherical coordinates
        r, theta, phi = cartesian_to_bl_spherical(x)

        # Check validity
        check_coordinate_validity(r, theta, r_horizon=self.r_schwarzschild, warn=True)

        # Compute metric components (FP64 for precision)
        M = self.mass
        f = 1.0 - 2.0 * M / r  # Schwarzschild factor
        sin_theta = np.sin(theta)

        if is_single:
            g = np.zeros((4, 4), dtype=np.float64)
            g[0, 0] = -f
            g[1, 1] = 1.0 / f
            g[2, 2] = r**2
            g[3, 3] = r**2 * sin_theta**2
        else:
            n = r.shape[0]
            g = np.zeros((n, 4, 4), dtype=np.float64)
            g[:, 0, 0] = -f
            g[:, 1, 1] = 1.0 / f
            g[:, 2, 2] = r**2
            g[:, 3, 3] = r**2 * sin_theta**2

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
        For diagonal Schwarzschild metric:
            g^00 = -1/f = -1/(1 - 2M/r)
            g^11 = f = (1 - 2M/r)
            g^22 = 1/r²
            g^33 = 1/(r² sin²θ)
        """
        x = np.atleast_1d(x)
        is_single = x.ndim == 1

        r, theta, phi = cartesian_to_bl_spherical(x)

        M = self.mass
        f = 1.0 - 2.0 * M / r
        sin_theta = np.sin(theta)

        # Protect against division by zero
        sin_theta = np.where(np.abs(sin_theta) < EPS_COORD, EPS_COORD, sin_theta)

        if is_single:
            g_inv = np.zeros((4, 4), dtype=np.float64)
            g_inv[0, 0] = -1.0 / f
            g_inv[1, 1] = f
            g_inv[2, 2] = 1.0 / r**2
            g_inv[3, 3] = 1.0 / (r**2 * sin_theta**2)
        else:
            n = r.shape[0]
            g_inv = np.zeros((n, 4, 4), dtype=np.float64)
            g_inv[:, 0, 0] = -1.0 / f
            g_inv[:, 1, 1] = f
            g_inv[:, 2, 2] = 1.0 / r**2
            g_inv[:, 3, 3] = 1.0 / (r**2 * sin_theta**2)

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
        Non-zero components for Schwarzschild (indices: 0=t, 1=r, 2=θ, 3=φ):
            Γ^t_tr = Γ^t_rt = M/(r(r-2M))
            Γ^r_tt = M(r-2M)/r³
            Γ^r_rr = -M/(r(r-2M))
            Γ^r_θθ = -(r - 2M)
            Γ^r_φφ = -(r - 2M) sin²θ
            Γ^θ_rθ = Γ^θ_θr = 1/r
            Γ^θ_φφ = -sin(θ) cos(θ)
            Γ^φ_rφ = Γ^φ_φr = 1/r
            Γ^φ_θφ = Γ^φ_φθ = cot(θ)

        Reference: Misner, Thorne & Wheeler (1973), Box 23.1
        """
        x = np.atleast_1d(x)
        is_single = x.ndim == 1

        r, theta, phi = cartesian_to_bl_spherical(x)

        M = self.mass
        f = 1.0 - 2.0 * M / r  # 1 - 2M/r
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Protect against division by zero
        sin_theta = np.where(np.abs(sin_theta) < EPS_COORD, EPS_COORD, sin_theta)

        if is_single:
            gamma = np.zeros((4, 4, 4), dtype=np.float64)

            # Time components
            gamma[0, 0, 1] = gamma[0, 1, 0] = M / (r * (r - 2*M))

            # Radial components
            gamma[1, 0, 0] = M * (r - 2*M) / r**3
            gamma[1, 1, 1] = -M / (r * (r - 2*M))
            gamma[1, 2, 2] = -(r - 2*M)
            gamma[1, 3, 3] = -(r - 2*M) * sin_theta**2

            # Theta components
            gamma[2, 1, 2] = gamma[2, 2, 1] = 1.0 / r
            gamma[2, 3, 3] = -sin_theta * cos_theta

            # Phi components
            gamma[3, 1, 3] = gamma[3, 3, 1] = 1.0 / r
            gamma[3, 2, 3] = gamma[3, 3, 2] = cos_theta / sin_theta  # cot(θ)
        else:
            n = r.shape[0]
            gamma = np.zeros((n, 4, 4, 4), dtype=np.float64)

            # Time components
            gamma[:, 0, 0, 1] = gamma[:, 0, 1, 0] = M / (r * (r - 2*M))

            # Radial components
            gamma[:, 1, 0, 0] = M * (r - 2*M) / r**3
            gamma[:, 1, 1, 1] = -M / (r * (r - 2*M))
            gamma[:, 1, 2, 2] = -(r - 2*M)
            gamma[:, 1, 3, 3] = -(r - 2*M) * sin_theta**2

            # Theta components
            gamma[:, 2, 1, 2] = gamma[:, 2, 2, 1] = 1.0 / r
            gamma[:, 2, 3, 3] = -sin_theta * cos_theta

            # Phi components
            gamma[:, 3, 1, 3] = gamma[:, 3, 3, 1] = 1.0 / r
            gamma[:, 3, 2, 3] = gamma[:, 3, 3, 2] = cos_theta / sin_theta

        return gamma

    def geodesic_acceleration(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute spatial geodesic acceleration from Cartesian inputs.

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial position in Cartesian coordinates.
        v : np.ndarray, shape (N, 3) or (3,)
            Cartesian 3-velocity (dx/dt, dy/dt, dz/dt).

        Returns
        -------
        a : np.ndarray, shape (N, 3) or (3,)
            Spatial acceleration in Cartesian coordinates.
        """
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        is_single = x.ndim == 1
        if is_single:
            x_arr = x[np.newaxis, :]
            v_arr = v[np.newaxis, :]
        else:
            x_arr = x
            v_arr = v

        r, theta, phi = cartesian_to_bl_spherical(x_arr)
        v_r, v_theta, v_phi = velocity_cartesian_to_bl(x_arr, v_arr)

        M = self.mass
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.where(np.abs(sin_theta) < EPS_COORD, EPS_COORD, sin_theta)

        f = 1.0 - 2.0 * M / r

        # Normalize 4-velocity
        denom = -f + (1.0 / f) * v_r**2 + (r**2) * v_theta**2 + (r**2 * sin_theta**2) * v_phi**2
        denom = np.where(denom < -1e-12, denom, -1e-12)

        u_t = np.sqrt(-1.0 / denom)
        u_r = v_r * u_t
        u_theta = v_theta * u_t
        u_phi = v_phi * u_t

        # Geodesic equation components (Tejeda et al. 2017)
        # d²xⁱ/dτ² = -Γⁱⱼₖ uʲ uᵏ
        # Christoffel symbols for Schwarzschild:
        # Γʳ_tt = M(r-2M)/r³, Γʳ_rr = -M/(r(r-2M)), Γʳ_θθ = -(r-2M), Γʳ_φφ = -(r-2M)sin²θ
        a_r = -(M * (r - 2.0 * M) / r**3) * u_t**2 \
              + (M / (r * (r - 2.0 * M))) * u_r**2 \
              + (r - 2.0 * M) * u_theta**2 \
              + (r - 2.0 * M) * sin_theta**2 * u_phi**2

        a_theta = -(2.0 / r) * u_r * u_theta + sin_theta * cos_theta * u_phi**2

        a_phi = -(2.0 / r) * u_r * u_phi - (2.0 * cos_theta / sin_theta) * u_theta * u_phi

        a_cart = acceleration_bl_to_cartesian(
            r,
            theta,
            phi,
            v_r,
            v_theta,
            v_phi,
            a_r,
            a_theta,
            a_phi
        )

        if is_single:
            return a_cart[0].astype(np.float32)

        return a_cart.astype(np.float32)

    def circular_orbit_velocity(self, r: float) -> Tuple[float, float]:
        """
        Compute orbital velocity and angular frequency for circular orbit.

        Parameters
        ----------
        r : float
            Orbital radius.

        Returns
        -------
        v_phi : float
            Orbital velocity (coordinate φ-velocity).
        omega : float
            Orbital angular frequency.

        Notes
        -----
        For Schwarzschild, circular orbit at radius r:
            Ω = √(M/r³)
            u^φ = Ω / √(1 - 3M/r)

        Valid for r > 3M (circular photon orbit at r = 3M).
        """
        if r <= 3.0 * self.mass:
            warnings.warn(f"Radius {r} is at or inside photon sphere (r = 3M). "
                         "Circular orbit may be unstable.")

        M = self.mass
        omega = np.sqrt(M / r**3)

        # Specific angular momentum for circular orbit
        # L = r² dφ/dτ = √(M r) / √(1 - 3M/r)
        u_phi = omega * r / np.sqrt(1.0 - 3.0 * M / r)

        return u_phi, omega

    def effective_potential(self, r: np.ndarray, L: float) -> np.ndarray:
        """
        Compute effective potential for radial motion with angular momentum L.

        Parameters
        ----------
        r : np.ndarray
            Radial coordinate.
        L : float
            Specific angular momentum (per unit mass).

        Returns
        -------
        V_eff : np.ndarray
            Effective potential.

        Notes
        -----
        Effective potential for Schwarzschild:
            V_eff = -M/r + L²/(2r²) - M L²/r³

        Used for analyzing orbital dynamics and bound/unbound trajectories.
        Reference: Misner, Thorne & Wheeler, Chapter 25
        """
        M = self.mass
        V_eff = -M / r + L**2 / (2.0 * r**2) - M * L**2 / r**3
        return V_eff
