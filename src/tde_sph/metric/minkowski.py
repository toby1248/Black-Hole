"""
Minkowski (flat spacetime) metric implementation.

This module provides the trivial flat spacetime metric for testing and
validation purposes. In Minkowski spacetime, all Christoffel symbols
vanish and geodesics are straight lines.

References
----------
- Misner, Thorne & Wheeler (1973) - Gravitation, Chapter 2
"""

import numpy as np
from tde_sph.core.interfaces import Metric


class MinkowskiMetric(Metric):
    """
    Minkowski metric: flat spacetime in Cartesian coordinates.

    The line element is:
        ds² = -dt² + dx² + dy² + dz²

    Metric tensor:
        g_μν = diag(-1, 1, 1, 1)

    This metric is used for:
    - Testing metric infrastructure
    - Validating coordinate transformations
    - Newtonian limit verification
    - Debugging relativistic integration schemes

    All Christoffel symbols vanish, and geodesics are straight lines.

    Parameters
    ----------
    None

    Attributes
    ----------
    name : str
        Human-readable name: "Minkowski"
    """

    def __init__(self):
        """Initialize Minkowski metric."""
        pass

    @property
    def name(self) -> str:
        """Return metric name."""
        return "Minkowski"

    def metric_tensor(self, x: np.ndarray) -> np.ndarray:
        """
        Compute metric tensor g_μν = diag(-1, 1, 1, 1) at position(s) x.

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial coordinates (x, y, z). Position-independent for Minkowski.

        Returns
        -------
        g : np.ndarray, shape (N, 4, 4) or (4, 4)
            Metric tensor components.

        Notes
        -----
        For Minkowski spacetime:
            g_00 = -1  (time-time)
            g_11 = 1   (x-x)
            g_22 = 1   (y-y)
            g_33 = 1   (z-z)
            g_μν = 0   (μ ≠ ν, off-diagonal)
        """
        x = np.atleast_1d(x)
        is_single = x.ndim == 1

        if x.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3, got shape {x.shape}")

        # Determine batch shape
        if is_single:
            g = np.zeros((4, 4), dtype=np.float64)
            g[0, 0] = -1.0
            g[1, 1] = 1.0
            g[2, 2] = 1.0
            g[3, 3] = 1.0
        else:
            n_particles = x.shape[0]
            g = np.zeros((n_particles, 4, 4), dtype=np.float64)
            g[:, 0, 0] = -1.0
            g[:, 1, 1] = 1.0
            g[:, 2, 2] = 1.0
            g[:, 3, 3] = 1.0

        return g

    def inverse_metric(self, x: np.ndarray) -> np.ndarray:
        """
        Compute inverse metric tensor g^μν = diag(-1, 1, 1, 1).

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial coordinates (position-independent).

        Returns
        -------
        g_inv : np.ndarray, shape (N, 4, 4) or (4, 4)
            Inverse metric tensor (identical to g_μν for Minkowski).

        Notes
        -----
        For Minkowski, g^μν = g_μν (metric is its own inverse).
        """
        # For Minkowski, inverse is identical to the metric
        return self.metric_tensor(x)

    def christoffel_symbols(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^μ_νρ (all zero for Minkowski).

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial coordinates.

        Returns
        -------
        gamma : np.ndarray, shape (N, 4, 4, 4) or (4, 4, 4)
            Christoffel symbols (all zero).

        Notes
        -----
        In flat spacetime, all Christoffel symbols vanish:
            Γ^μ_νρ = 0 for all μ, ν, ρ
        """
        x = np.atleast_1d(x)
        is_single = x.ndim == 1

        if x.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3, got shape {x.shape}")

        if is_single:
            gamma = np.zeros((4, 4, 4), dtype=np.float64)
        else:
            n_particles = x.shape[0]
            gamma = np.zeros((n_particles, 4, 4, 4), dtype=np.float64)

        return gamma

    def geodesic_acceleration(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute geodesic acceleration (zero for Minkowski).

        In flat spacetime, geodesics are straight lines with zero acceleration.

        Parameters
        ----------
        x : np.ndarray, shape (N, 3) or (3,)
            Spatial position (x, y, z).
        v : np.ndarray, shape (N, 3) or (3,)
            Cartesian 3-velocity (dx/dt, dy/dt, dz/dt).

        Returns
        -------
        a : np.ndarray, shape (N, 3) or (3,)
            Spatial acceleration (all zeros).

        Notes
        -----
        Geodesic equation: d²x^μ/dτ² + Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ) = 0

        Since Γ^μ_νρ = 0, the acceleration is zero:
            d²x/dτ² = 0
        """
        x = np.atleast_1d(x)
        v = np.atleast_1d(v)

        is_single = x.ndim == 1

        if x.shape[-1] != 3:
            raise ValueError(f"Position last dimension must be 3, got shape {x.shape}")
        if v.shape[-1] != 3:
            raise ValueError(f"Velocity last dimension must be 3, got shape {v.shape}")

        if is_single:
            return np.zeros(3, dtype=np.float32)

        n_particles = x.shape[0]
        return np.zeros((n_particles, 3), dtype=np.float32)

    def isco_radius(self) -> float:
        """
        Return ISCO radius (undefined for flat spacetime).

        Returns
        -------
        r_isco : float
            Returns infinity (no ISCO in flat space).
        """
        return np.inf

    def event_horizon(self) -> float:
        """
        Return event horizon radius (undefined for flat spacetime).

        Returns
        -------
        r_h : float
            Returns 0 (no horizon in flat space).
        """
        return 0.0
