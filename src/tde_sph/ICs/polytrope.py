"""
Polytropic stellar model initial conditions generator.

Generates SPH particle distributions for polytropic stars by solving the
Lane-Emden equation and distributing particles according to the resulting
density profile.

Supports γ=5/3 (n=1.5, convective envelope) and γ=4/3 (n=3, radiation-dominated)
polytropes commonly used in TDE simulations as benchmark stellar models.

References
----------
- Guillochon, J. & Ramirez-Ruiz, E. (2013), ApJ, 767, 25
  "Tidal disruption of stars by supermassive black holes: the x-ray and optical view"
- Rosswog, S. et al. (2009), New Astron. Rev., 53, 78
  "SPH simulations of TDEs"
- Chandrasekhar, S. (1939), "An Introduction to the Study of Stellar Structure"
  Chapter IV: The theory of polytropes
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
from tde_sph.core.interfaces import ICGenerator, NDArrayFloat


class Polytrope(ICGenerator):
    """
    Generate initial conditions for polytropic stellar models.

    Solves the Lane-Emden equation:
        1/ξ² d/dξ(ξ² dθ/dξ) = -θ^n

    where θ = (ρ/ρ_c)^(1/n) is the normalized density, ξ is the dimensionless
    radius, and n is the polytropic index related to adiabatic index γ by:
        γ = 1 + 1/n

    Attributes
    ----------
    gamma : float
        Adiabatic index (5/3 for ideal monatomic gas, 4/3 for radiation-dominated).
    n : float
        Polytropic index n = 1/(γ-1).
    eta : float
        Smoothing length factor: h_i = η * (m_i/ρ_i)^(1/3).
    """

    # Known Lane-Emden first zeros (where θ=0, i.e., stellar surface)
    LANE_EMDEN_ZEROS = {
        0.0: 2.4495,      # n=0 (γ→∞, uniform density)
        1.0: 3.1416,      # n=1 (γ=2, isothermal-like)
        1.5: 3.6538,      # n=1.5 (γ=5/3, standard polytrope)
        2.0: 4.3529,      # n=2
        3.0: 6.8969,      # n=3 (γ=4/3, radiation-dominated)
        4.0: 14.9716,     # n=4
    }

    def __init__(
        self,
        gamma: float = 5.0 / 3.0,
        eta: float = 1.2,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize polytrope generator.

        Parameters
        ----------
        gamma : float, default 5/3
            Adiabatic index. Common values:
            - 5/3: ideal monatomic gas (convective envelope)
            - 4/3: radiation-dominated (massive stars, white dwarfs)
        eta : float, default 1.2
            SPH smoothing length factor (typically 1.2-1.5).
            h_i = eta * (m_i / rho_i)^(1/3)
        random_seed : Optional[int], default 42
            Random seed for reproducible particle placement.
            Set to None for non-reproducible random placement.
        """
        self.gamma = np.float32(gamma)
        self.n = np.float32(1.0 / (gamma - 1.0))
        self.eta = np.float32(eta)
        self.random_seed = random_seed

        # Cache Lane-Emden solution
        self._xi_grid: Optional[np.ndarray] = None
        self._theta_grid: Optional[np.ndarray] = None
        self._dtheta_dxi_grid: Optional[np.ndarray] = None
        self._xi_1: Optional[float] = None  # First zero (surface)

    def generate(
        self,
        n_particles: int,
        **kwargs
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Generate polytropic star initial conditions.

        Particles are distributed in radial shells with mass proportional to
        the local density from the Lane-Emden solution, then randomly scattered
        in angle.

        Parameters
        ----------
        n_particles : int
            Number of SPH particles (10³ to 10⁶).
        **kwargs
            Model parameters:
            - M_star : float, default 1.0 - Total stellar mass (solar masses in code units).
            - R_star : float, default 1.0 - Stellar radius (solar radii in code units).
            - position : NDArrayFloat, shape (3,), default [0,0,0] - Star center position.
            - velocity : NDArrayFloat, shape (3,), default [0,0,0] - Star center velocity.

        Returns
        -------
        positions : NDArrayFloat, shape (n_particles, 3)
            Particle positions (star initially at rest at origin, or offset by 'position').
        velocities : NDArrayFloat, shape (n_particles, 3)
            Particle velocities (zero in star rest frame, or offset by 'velocity').
        masses : NDArrayFloat, shape (n_particles,)
            Particle masses (uniform: M_star / n_particles).
        internal_energies : NDArrayFloat, shape (n_particles,)
            Specific internal energies u = P / [(γ-1) ρ].
        densities : NDArrayFloat, shape (n_particles,)
            Initial mass densities ρ(r).

        Notes
        -----
        - Uses dimensionless units: G=1, M_star=1, R_star=1 internally, then rescales.
        - Internal energy is set from polytropic relation: P = K ρ^γ => u = K ρ^(γ-1) / (γ-1).
        - Smoothing lengths are computed as h = eta * (m / rho)^(1/3).
        """
        # Extract parameters with defaults
        M_star = np.float32(kwargs.get('M_star', 1.0))
        R_star = np.float32(kwargs.get('R_star', 1.0))
        position = np.array(kwargs.get('position', [0.0, 0.0, 0.0]), dtype=np.float32)
        velocity = np.array(kwargs.get('velocity', [0.0, 0.0, 0.0]), dtype=np.float32)

        # Solve Lane-Emden equation if not cached
        if self._xi_grid is None:
            self._solve_lane_emden()

        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Generate particle positions in dimensionless units (star mass=1, radius=1)
        r_particles, theta_particles = self._sample_particles_radial(n_particles)

        # Scatter particles uniformly on sphere at each radius
        positions_norm = self._scatter_on_sphere(r_particles)

        # Compute density and pressure from Lane-Emden solution
        # θ(ξ) = (ρ/ρ_c)^(1/n), so ρ/ρ_c = θ^n
        # r_particles already in [0, 1] range (normalized by ξ₁)
        # theta_particles already interpolated from Lane-Emden solution

        # Normalized density ρ/ρ_c = θ^n
        rho_normalized = np.power(np.maximum(theta_particles, 1e-10), self.n)

        # Central density from mass constraint
        # M = 4π ∫_0^R ρ(r) r² dr = 4π ρ_c R³ * [-ξ² dθ/dξ]|_ξ₁
        # For dimensionless mass=1, radius=1:
        # ρ_c = -1 / (4π ξ₁² dθ/dξ|_ξ₁)
        dtheta_dxi_surface = np.interp(self._xi_1, self._xi_grid, self._dtheta_dxi_grid)
        rho_c_norm = np.float32(-1.0 / (4.0 * np.pi * self._xi_1**2 * dtheta_dxi_surface))

        # Density in normalized units
        rho_norm = rho_c_norm * rho_normalized

        # Polytropic constant K from central values
        # P = K ρ^γ, and at center: P_c / ρ_c^γ = K
        # For hydrostatic equilibrium: P_c = (2π/3) G ρ_c² R² (approx)
        # More precisely, use Lane-Emden normalization:
        # K = (4π G)^(1-γ) R^(2γ-2) M^(γ-2) * [normalization factor from Lane-Emden]
        # Simplified for normalized units (G=M=R=1):
        K = np.float32(1.0 / (4.0 * np.pi * (self.n + 1.0)))  # Standard normalization

        # Pressure in normalized units: P = K ρ^γ
        P_norm = K * np.power(rho_norm, self.gamma)

        # Specific internal energy: u = P / [(γ-1) ρ]
        u_norm = P_norm / ((self.gamma - 1.0) * rho_norm)

        # Rescale to physical units
        positions = positions_norm * R_star + position[np.newaxis, :]
        velocities = np.tile(velocity, (n_particles, 1)).astype(np.float32)

        # Masses: uniform mass particles
        masses = np.full(n_particles, M_star / n_particles, dtype=np.float32)

        # Density scaling: ρ_physical = ρ_normalized * (M_star / R_star³)
        densities = rho_norm * (M_star / R_star**3)

        # Internal energy scaling: u has dimensions [length²/time²]
        # u_physical = u_normalized * (G M_star / R_star)
        # In code units where G=1:
        internal_energies = u_norm * (M_star / R_star)

        return positions, velocities, masses, internal_energies, densities

    def compute_smoothing_lengths(
        self,
        masses: NDArrayFloat,
        densities: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute SPH smoothing lengths from masses and densities.

        h_i = η * (m_i / ρ_i)^(1/3)

        Parameters
        ----------
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        densities : NDArrayFloat, shape (N,)
            Particle densities.

        Returns
        -------
        smoothing_lengths : NDArrayFloat, shape (N,)
            SPH smoothing lengths.
        """
        return self.eta * np.power(masses / densities, 1.0 / 3.0).astype(np.float32)

    def _solve_lane_emden(self, n_points: int = 1000) -> None:
        """
        Solve Lane-Emden equation numerically using scipy.integrate.solve_ivp.

        The Lane-Emden equation in standard form:
            d²θ/dξ² + (2/ξ) dθ/dξ + θ^n = 0

        with boundary conditions:
            θ(0) = 1, dθ/dξ(0) = 0

        We convert to first-order system:
            y₁ = θ
            y₂ = dθ/dξ
            dy₁/dξ = y₂
            dy₂/dξ = -2/ξ * y₂ - y₁^n

        The singularity at ξ=0 is handled by starting slightly offset.

        Parameters
        ----------
        n_points : int, default 1000
            Number of grid points for solution.
        """
        # Estimate first zero from known values
        if self.n in self.LANE_EMDEN_ZEROS:
            xi_max = self.LANE_EMDEN_ZEROS[self.n] * 1.1
        else:
            # Interpolate or extrapolate
            xi_max = 10.0 * (1.0 + self.n / 3.0)

        def lane_emden_derivs(xi, y):
            """
            Derivatives for Lane-Emden equation.

            y[0] = θ
            y[1] = dθ/dξ
            """
            theta, dtheta_dxi = y

            # Handle singularity at ξ=0 and ensure θ ≥ 0
            if xi < 1e-8:
                # Near ξ=0: θ(ξ) ≈ 1 - ξ²/6 + O(ξ⁴)
                return [0.0, 0.0]

            theta_safe = max(theta, 0.0)  # Prevent negative θ

            dy1_dxi = dtheta_dxi
            dy2_dxi = -2.0 / xi * dtheta_dxi - theta_safe**self.n

            return [dy1_dxi, dy2_dxi]

        def event_theta_zero(xi, y):
            """Event to detect when θ crosses zero (stellar surface)."""
            return y[0]

        event_theta_zero.terminal = True
        event_theta_zero.direction = -1  # Detect decreasing crossing

        # Initial conditions: θ(0) = 1, dθ/dξ(0) = 0
        # Start slightly offset from ξ=0 to avoid singularity
        xi_start = 1e-6
        y0 = [1.0, 0.0]

        # Solve ODE
        xi_span = (xi_start, xi_max)
        xi_eval = np.linspace(xi_start, xi_max, n_points)

        sol = solve_ivp(
            lane_emden_derivs,
            xi_span,
            y0,
            method='RK45',
            t_eval=xi_eval,
            events=event_theta_zero,
            rtol=1e-8,
            atol=1e-10
        )

        # Extract solution
        if sol.t_events[0].size > 0:
            # Found first zero
            self._xi_1 = np.float32(sol.t_events[0][0])
            # Trim solution to first zero
            idx_surface = np.searchsorted(sol.t, self._xi_1)
            self._xi_grid = np.concatenate([[0.0], sol.t[:idx_surface], [self._xi_1]]).astype(np.float32)
            self._theta_grid = np.concatenate([[1.0], sol.y[0, :idx_surface], [0.0]]).astype(np.float32)
            self._dtheta_dxi_grid = np.concatenate([[0.0], sol.y[1, :idx_surface], [sol.y_events[0][0, 1]]]).astype(np.float32)
        else:
            # Didn't find zero within range (shouldn't happen for n ≤ 5)
            self._xi_grid = np.concatenate([[0.0], sol.t]).astype(np.float32)
            self._theta_grid = np.concatenate([[1.0], sol.y[0, :]]).astype(np.float32)
            self._dtheta_dxi_grid = np.concatenate([[0.0], sol.y[1, :]]).astype(np.float32)
            self._xi_1 = np.float32(sol.t[-1])

    def _sample_particles_radial(self, n_particles: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample particle radii from Lane-Emden density profile.

        Uses inverse transform sampling on the cumulative mass distribution.

        Parameters
        ----------
        n_particles : int
            Number of particles.

        Returns
        -------
        r_particles : np.ndarray, shape (n_particles,)
            Particle radii in normalized units [0, 1].
        theta_particles : np.ndarray, shape (n_particles,)
            Corresponding Lane-Emden θ values.
        """
        # Cumulative mass M(ξ) = 4π ∫_0^ξ ρ(r) r² dr
        # = 4π ρ_c ∫_0^ξ θ^n ξ² dξ
        # Normalized so M(ξ_1) = 1

        # Compute M(ξ) on grid using trapezoidal integration
        integrand = np.power(self._theta_grid, self.n) * self._xi_grid**2
        M_cumulative = np.zeros_like(self._xi_grid, dtype=np.float32)

        for i in range(1, len(self._xi_grid)):
            dxi = self._xi_grid[i] - self._xi_grid[i - 1]
            M_cumulative[i] = M_cumulative[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * dxi

        # Normalize to total mass = 1
        M_cumulative /= M_cumulative[-1]

        # Inverse transform sampling: sample uniform u ∈ [0, 1], find ξ such that M(ξ) = u
        u_samples = np.random.uniform(0.0, 1.0, n_particles).astype(np.float32)
        xi_particles = np.interp(u_samples, M_cumulative, self._xi_grid)
        theta_particles = np.interp(xi_particles, self._xi_grid, self._theta_grid)

        # Convert ξ to physical radius (normalized to R=1)
        # ξ ranges from 0 to ξ₁, which should map to r from 0 to R=1
        r_particles = xi_particles / self._xi_1  # Normalize: ξ₁ → 1

        return r_particles, theta_particles

    def _scatter_on_sphere(self, radii: np.ndarray) -> np.ndarray:
        """
        Scatter particles uniformly on spheres at given radii.

        Parameters
        ----------
        radii : np.ndarray, shape (N,)
            Particle radii.

        Returns
        -------
        positions : np.ndarray, shape (N, 3)
            3D Cartesian positions.
        """
        n = len(radii)

        # Uniform sampling on unit sphere using Marsaglia (1972) method
        # (more uniform than spherical coordinates)
        phi = np.random.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
        cos_theta = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
        sin_theta = np.sqrt(1.0 - cos_theta**2)

        x = radii * sin_theta * np.cos(phi)
        y = radii * sin_theta * np.sin(phi)
        z = radii * cos_theta

        return np.column_stack([x, y, z]).astype(np.float32)
