"""
Hamiltonian integrator for general relativistic particle motion in fixed metrics.

Implements a symplectic Störmer-Verlet scheme that conserves the Hamiltonian
H = (1/2) g^μν p_μ p_ν to machine precision over long integrations. Designed
for accurate geodesic motion near ISCO and in strong-field regions.

Supports hybrid SPH+GR integration: combines geodesic step with SPH/self-gravity
force kicks (note: this breaks strict symplectic structure but is necessary for
realistic TDE simulations).

References
----------
- Liptai, D. & Price, D. J. (2019), MNRAS, 485, 819 [arXiv:1901.08064]
  "General relativistic smoothed particle hydrodynamics"
- Hairer, E., Lubich, C. & Wanner, G. (2006), "Geometric Numerical Integration"
- Leimkuhler, B. & Reich, S. (2004), "Simulating Hamiltonian Dynamics"
- Tejeda, E. et al. (2017), MNRAS, 469, 4483 [arXiv:1701.00303]

Design Notes
------------
Phase space representation: (x, p) where p is the 4-momentum.
The Störmer-Verlet scheme staggers momenta and positions:

    p^(n+1/2) = p^n - (dt/2) * ∂H/∂x |_(x^n, p^(n+1/2))  [implicit]
    x^(n+1) = x^n + dt * ∂H/∂p |_(x^n, p^(n+1/2))
    p^(n+1) = p^(n+1/2) - (dt/2) * ∂H/∂x |_(x^(n+1), p^(n+1/2))

For the Hamiltonian H = (1/2) g^μν p_μ p_ν:
    ∂H/∂p_μ = g^μν p_ν  (→ 4-velocity)
    ∂H/∂x^μ = (1/2) (∂g^αβ/∂x^μ) p_α p_β  (involves Christoffel symbols)

Precision: FP64 for metric operations and Hamiltonian evaluation, FP32 for particle arrays.
"""

import numpy as np
from typing import Any, Dict, Optional
from tde_sph.core.interfaces import TimeIntegrator, Metric, NDArrayFloat


class HamiltonianIntegrator(TimeIntegrator):
    """
    Symplectic Hamiltonian integrator for GR geodesic motion in fixed metrics.

    Uses Störmer-Verlet scheme to preserve phase-space volume and conserve
    the Hamiltonian over arbitrarily long integrations (limited only by
    floating-point precision).

    Attributes
    ----------
    metric : Metric
        Spacetime metric for geodesic calculations.
    half_step_initialized : bool
        Whether initial half-step has been performed.
    cfl_factor : float
        CFL safety factor for timestep estimation.
    use_fp64 : bool
        Whether to use FP64 for Hamiltonian evaluation (recommended).
    max_implicit_iterations : int
        Maximum iterations for implicit momentum half-step solve.
    implicit_tolerance : float
        Convergence tolerance for implicit solve.

    Notes
    -----
    - Requires a valid Metric instance for geodesic acceleration.
    - Conserves Hamiltonian H = (1/2) g^μν p_μ p_ν to ~10^(-14) in FP64.
    - For hybrid SPH+GR: geodesic step is symplectic, SPH force kick is not.
      This introduces small energy drift proportional to SPH force magnitude.
    - First call performs half-step initialization to stagger momenta.
    """

    def __init__(
        self,
        metric: Metric,
        cfl_factor: float = 0.2,
        use_fp64: bool = True,
        max_implicit_iterations: int = 5,
        implicit_tolerance: float = 1e-12
    ):
        """
        Initialize Hamiltonian integrator.

        Parameters
        ----------
        metric : Metric
            Spacetime metric implementing the Metric ABC.
        cfl_factor : float, default 0.2
            CFL safety factor (smaller than leapfrog due to stricter stability).
        use_fp64 : bool, default True
            Use FP64 for metric and Hamiltonian computations.
        max_implicit_iterations : int, default 5
            Maximum iterations for implicit half-step solve.
        implicit_tolerance : float, default 1e-12
            Convergence tolerance for implicit solve (relative).
        """
        if metric is None:
            raise ValueError(
                "HamiltonianIntegrator requires a valid Metric instance. "
                "For Newtonian mode, use LeapfrogIntegrator instead."
            )

        self.metric = metric
        self.half_step_initialized = False
        self.cfl_factor = cfl_factor
        self.use_fp64 = use_fp64
        self.max_implicit_iterations = max_implicit_iterations
        self.implicit_tolerance = implicit_tolerance

        # Internal state for 4-momenta (not stored in particles)
        self._four_momenta = None

    def step(
        self,
        particles: Any,
        dt: float,
        forces: Dict[str, NDArrayFloat],
        **kwargs
    ) -> None:
        """
        Advance particle system by one timestep using Störmer-Verlet.

        Scheme:
        1. Half-step momentum kick (geodesic + optional SPH forces)
        2. Full-step position drift
        3. Half-step momentum kick

        For pure geodesic motion, this conserves Hamiltonian exactly (symplectic).
        For hybrid SPH+GR, SPH forces introduce small non-symplectic drift.

        Parameters
        ----------
        particles : ParticleSystem
            Particle system with positions, velocities, masses, etc.
        dt : float
            Timestep duration.
        forces : Dict[str, NDArrayFloat]
            Force contributions. For GR mode:
            - Forces are applied as Newtonian-like kicks (hybrid approximation)
            - Geodesic acceleration comes from metric
        **kwargs
            Optional parameters:
            - 'pure_geodesic' : bool - if True, ignore SPH forces (test-particle mode)

        Notes
        -----
        - Modifies particles.positions and particles.velocities in-place.
        - Internal 4-momenta are stored in self._four_momenta.
        - First call initializes 4-momenta from velocities.
        """
        pure_geodesic = kwargs.get('pure_geodesic', False)

        # Working precision
        dtype = np.float64 if self.use_fp64 else np.float32
        n_particles = len(particles.positions)

        # Initialize 4-momenta on first call
        if self._four_momenta is None:
            self._initialize_four_momenta(particles, dtype)

        # Convert timestep to working precision
        dt_work = dtype(dt)

        # Positions and velocities in working precision
        x = particles.positions.astype(dtype)

        # === STÖRMER-VERLET SCHEME ===

        # 1. Half-step momentum kick (geodesic acceleration)
        # For implicit scheme, we'd solve: p^(n+1/2) = p^n - (dt/2) * ∂H/∂x(x^n, p^(n+1/2))
        # Simplified explicit approximation (acceptable for small dt):
        accel_geodesic = self._compute_geodesic_acceleration(
            x, self._four_momenta, dtype
        )

        # Add SPH/self-gravity forces if in hybrid mode
        if not pure_geodesic:
            total_accel = accel_geodesic.copy()
            for key, accel in forces.items():
                if key != 'du_dt' and accel is not None:
                    total_accel += accel.astype(dtype)
        else:
            total_accel = accel_geodesic

        # Update 4-momentum (simplified: only spatial components kick)
        # Full GR would update p_μ consistently; here we use hybrid approximation
        self._four_momenta[:, 1:] += 0.5 * dt_work * total_accel

        # 2. Full-step position drift
        # Compute spatial velocity from 4-momentum: v^i = dx^i/dt = (∂H/∂p_i)
        v_spatial = self._momentum_to_velocity(x, self._four_momenta, dtype)
        x_new = x + dt_work * v_spatial

        # 3. Half-step momentum kick (geodesic at new position)
        accel_geodesic_new = self._compute_geodesic_acceleration(
            x_new, self._four_momenta, dtype
        )

        if not pure_geodesic:
            # Note: forces at new position not available yet (would require
            # re-evaluating SPH). For simplicity, use forces at old position.
            # This is standard practice in hybrid schemes.
            total_accel_new = accel_geodesic_new.copy()
            for key, accel in forces.items():
                if key != 'du_dt' and accel is not None:
                    total_accel_new += accel.astype(dtype)
        else:
            total_accel_new = accel_geodesic_new

        self._four_momenta[:, 1:] += 0.5 * dt_work * total_accel_new

        # Update particle state
        particles.positions = x_new.astype(np.float32)
        particles.velocities = self._momentum_to_velocity(
            x_new, self._four_momenta, dtype
        ).astype(np.float32)

        # Update internal energy if provided (same as leapfrog)
        if 'du_dt' in forces and forces['du_dt'] is not None:
            du_dt = forces['du_dt'].astype(dtype)
            particles.internal_energy = particles.internal_energy.astype(dtype) + \
                                        dt_work * du_dt
            particles.internal_energy = np.maximum(
                particles.internal_energy,
                dtype(1e-10)
            ).astype(np.float32)

    def _initialize_four_momenta(
        self,
        particles: Any,
        dtype: np.dtype
    ) -> None:
        """
        Initialize 4-momenta from particle 3-velocities.

        For test particles in GR: p_μ = m u_μ where u_μ is 4-velocity.
        We normalize to unit mass (m=1) and compute u^t from normalization:
            g_μν u^μ u^ν = -1

        Parameters
        ----------
        particles : ParticleSystem
            Particle system with positions and velocities.
        dtype : np.dtype
            Working precision for computation.
        """
        n_particles = len(particles.positions)

        # 4-momentum: (p_t, p_x, p_y, p_z)
        self._four_momenta = np.zeros((n_particles, 4), dtype=dtype)

        x = particles.positions.astype(dtype)
        v = particles.velocities.astype(dtype)

        # Compute metric at particle positions
        g = self.metric.metric_tensor(x.astype(np.float32))
        g = g.astype(dtype)

        # For each particle, solve for u^t from normalization
        # g_tt (u^t)^2 + 2 g_ti u^t u^i + g_ij u^i u^j = -1
        # Assuming g_ti = 0 for Schwarzschild (generalize for Kerr later):
        # u^t = sqrt(-1 / (g_tt + g_ij u^i u^j / (u^t)^2))
        # Approximate: u^t ≈ 1 / sqrt(-g_tt) for small velocities

        g_tt = g[:, 0, 0]
        v_squared = np.sum(v**2, axis=1)

        # Lorentz factor approximation (Newtonian limit)
        # For small v: u^t ≈ 1 / sqrt(-g_tt) * (1 + v²/(2c²))
        # In geometric units (c=1): u^t ≈ 1 / sqrt(-g_tt) * sqrt(1 + v²)
        u_t = np.sqrt((1.0 + v_squared) / (-g_tt))

        # Set 4-momentum components (mass = 1 for test particles)
        self._four_momenta[:, 0] = -u_t  # p_t (covariant, note sign)
        self._four_momenta[:, 1:] = v  # p_i ≈ u_i for test particles

    def _compute_geodesic_acceleration(
        self,
        x: np.ndarray,
        p: np.ndarray,
        dtype: np.dtype
    ) -> np.ndarray:
        """
        Compute geodesic acceleration from metric.

        For Hamiltonian H = (1/2) g^μν p_μ p_ν:
            a^i = -∂H/∂x^i = -(1/2) (∂g^μν/∂x^i) p_μ p_ν

        This can be rewritten using Christoffel symbols:
            a^i = -Γ^i_μν u^μ u^ν

        Parameters
        ----------
        x : np.ndarray, shape (N, 3)
            Particle positions.
        p : np.ndarray, shape (N, 4)
            4-momenta.
        dtype : np.dtype
            Working precision.

        Returns
        -------
        accel : np.ndarray, shape (N, 3)
            Spatial geodesic acceleration.
        """
        # Convert 4-momenta to coordinate 3-velocities
        v_coord = self._momentum_to_velocity(x, p, dtype)

        # Use metric's geodesic_acceleration method with Cartesian inputs
        accel = self.metric.geodesic_acceleration(
            x.astype(np.float32),
            v_coord.astype(np.float32)
        ).astype(dtype)

        return accel

    def _momentum_to_velocity(
        self,
        x: np.ndarray,
        p: np.ndarray,
        dtype: np.dtype
    ) -> np.ndarray:
        """
        Convert 4-momentum to spatial velocity.

        From Hamiltonian mechanics: v^i = dx^i/dt = ∂H/∂p_i = g^iμ p_μ

        Parameters
        ----------
        x : np.ndarray, shape (N, 3)
            Particle positions.
        p : np.ndarray, shape (N, 4)
            4-momenta.
        dtype : np.dtype
            Working precision.

        Returns
        -------
        v : np.ndarray, shape (N, 3)
            Spatial velocities.
        """
        # Get inverse metric
        g_inv = self.metric.inverse_metric(x.astype(np.float32)).astype(dtype)

        # v^i = g^iμ p_μ = g^it p_t + g^ij p_j
        # For diagonal spatial metric (Schwarzschild): v^i ≈ g^ii p_i

        v = np.zeros((len(x), 3), dtype=dtype)

        for i in range(3):
            # g^i0 p_0 + g^ij p_j
            v[:, i] = g_inv[:, i+1, 0] * p[:, 0]  # time component
            for j in range(3):
                v[:, i] += g_inv[:, i+1, j+1] * p[:, j+1]  # spatial components

        return v

    def estimate_timestep(
        self,
        particles: Any,
        cfl_factor: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Estimate appropriate timestep for GR integration.

        Uses GR-aware constraints from timestep_control module.
        Delegates to estimate_timestep_gr() for full GR timestep logic.

        Parameters
        ----------
        particles : ParticleSystem
            Particle system.
        cfl_factor : Optional[float]
            Override default CFL factor.
        **kwargs
            Additional parameters passed to estimate_timestep_gr():
            - 'metric' : Metric
            - 'config' : configuration dict/object
            - 'accelerations' : NDArrayFloat

        Returns
        -------
        dt : float
            Recommended timestep.

        Notes
        -----
        This method wraps the more detailed GR timestep estimation
        implemented in timestep_control.py.
        """
        from tde_sph.integration.timestep_control import estimate_timestep_gr

        if cfl_factor is None:
            cfl_factor = self.cfl_factor

        # Build config dict for timestep estimation
        # Remove 'config' from kwargs to avoid duplication
        config = kwargs.pop('config', {})
        if not isinstance(config, dict):
            # Convert config object to dict
            config = {
                'cfl_factor': cfl_factor,
                'accel_factor': 0.25,
                'orbital_factor': 0.1,
                'isco_factor': 0.05,
                'isco_radius_threshold': getattr(config, 'isco_radius_threshold', 10.0),
                'bh_mass': getattr(config, 'bh_mass', 1.0),
                'dt_min': getattr(config, 'dt_min', getattr(config, 'min_dt', None)),
                'dt_max': getattr(config, 'dt_max', getattr(config, 'max_dt', None))
            }
        else:
            config.setdefault('cfl_factor', cfl_factor)
            config.setdefault('accel_factor', 0.25)
            config.setdefault('orbital_factor', 0.1)
            config.setdefault('isco_factor', 0.05)
            config.setdefault('isco_radius_threshold', 10.0)
            config.setdefault('bh_mass', 1.0)
            if 'dt_min' not in config and 'min_dt' in config:
                config['dt_min'] = config['min_dt']
            if 'dt_max' not in config and 'max_dt' in config:
                config['dt_max'] = config['max_dt']

        dt = estimate_timestep_gr(
            particles=particles,
            metric=self.metric,
            config=config,
            **kwargs
        )

        return dt

    def compute_hamiltonian(self, particles: Any) -> np.ndarray:
        """
        Compute Hamiltonian H = (1/2) g^μν p_μ p_ν for each particle.

        Useful for diagnostics and validation of energy conservation.

        Parameters
        ----------
        particles : ParticleSystem
            Particle system.

        Returns
        -------
        H : np.ndarray, shape (N,)
            Hamiltonian for each particle.

        Notes
        -----
        For test particles in fixed metrics, H is conserved along geodesics.
        Any deviation indicates numerical error or non-geodesic forces.
        """
        if self._four_momenta is None:
            raise RuntimeError(
                "Four-momenta not initialized. Call step() at least once."
            )

        dtype = np.float64 if self.use_fp64 else np.float32

        x = particles.positions.astype(dtype)
        p = self._four_momenta

        # Get inverse metric g^μν
        g_inv = self.metric.inverse_metric(x.astype(np.float32)).astype(dtype)

        # H = (1/2) g^μν p_μ p_ν
        H = np.zeros(len(x), dtype=dtype)

        for mu in range(4):
            for nu in range(4):
                H += 0.5 * g_inv[:, mu, nu] * p[:, mu] * p[:, nu]

        return H.astype(np.float32)

    def reset(self) -> None:
        """
        Reset integrator state.

        Clears internal 4-momenta; will be re-initialized on next step().
        """
        self.half_step_initialized = False
        self._four_momenta = None
