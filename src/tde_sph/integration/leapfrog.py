"""
Leapfrog (kick-drift-kick) time integrator for SPH simulations.

Implements the classic leapfrog/velocity-Verlet scheme:
    v^(n+1/2) = v^(n-1/2) + a^n * dt
    x^(n+1) = x^n + v^(n+1/2) * dt

This is a second-order symplectic integrator ideal for conservative systems.
Supports REQ-004 (dynamic timesteps) and REQ-008 (energy updates).

References
----------
- Price, D. J. (2012), JCP, 231, 759 - "Smoothed particle hydrodynamics and magnetohydrodynamics"
- Springel, V. (2005), MNRAS, 364, 1105 - "The cosmological simulation code GADGET-2"
- Rosswog, S. (2009), New Astron. Rev., 53, 78 - "Astrophysical smooth particle hydrodynamics"
"""

import numpy as np
from typing import Any, Dict, Optional
from tde_sph.core.interfaces import TimeIntegrator, NDArrayFloat


class LeapfrogIntegrator(TimeIntegrator):
    """
    Leapfrog (kick-drift-kick) integrator for conservative dynamics.

    The leapfrog scheme staggers positions and velocities by half a timestep,
    providing second-order accuracy and excellent long-term energy conservation
    for Hamiltonian systems.

    Attributes
    ----------
    half_step_initialized : bool
        Whether the initial half-step kick has been performed.
    cfl_factor : float
        Default CFL safety factor for timestep estimation.
    accel_factor : float
        Safety factor for acceleration-based timestep constraint.

    Notes
    -----
    On the first call to step(), only a half-step velocity kick is performed
    to stagger v and x properly. Subsequent calls perform the full kick-drift-kick.
    """

    def __init__(
        self,
        cfl_factor: float = 0.3,
        accel_factor: float = 0.25,
        internal_energy_evolution: bool = True
    ):
        """
        Initialize leapfrog integrator.

        Parameters
        ----------
        cfl_factor : float, default 0.3
            CFL safety factor for timestep estimation (typically 0.2-0.5).
        accel_factor : float, default 0.25
            Safety factor for acceleration-based timestep (typically 0.25-0.5).
        internal_energy_evolution : bool, default True
            Whether to evolve internal energy (set False for pure N-body).
        """
        self.half_step_initialized = False
        self.cfl_factor = cfl_factor
        self.accel_factor = accel_factor
        self.internal_energy_evolution = internal_energy_evolution

    def step(
        self,
        particles: Any,
        dt: float,
        forces: Dict[str, NDArrayFloat],
        **kwargs
    ) -> None:
        """
        Advance particle system by one timestep using kick-drift-kick.

        Leapfrog scheme:
        1. Kick: v^(n+1/2) = v^(n-1/2) + a^n * dt
        2. Drift: x^(n+1) = x^n + v^(n+1/2) * dt
        3. Kick: (happens on next step with new acceleration)

        On first call, only performs half-step kick to initialize staggering.

        Parameters
        ----------
        particles : ParticleSystem
            Particle system with attributes: positions, velocities, internal_energy,
            masses, smoothing_lengths, etc.
        dt : float
            Timestep duration.
        forces : Dict[str, NDArrayFloat]
            Dictionary of force/acceleration contributions:
            - 'gravity' : NDArrayFloat, shape (N, 3) - gravitational acceleration
            - 'hydro' : NDArrayFloat, shape (N, 3) - hydrodynamic acceleration
            - 'du_dt' : NDArrayFloat, shape (N,) - rate of internal energy change
            Additional keys may be present for other physics (radiation, viscosity).
        **kwargs
            Optional parameters (e.g., for adaptive timestepping, not used here).

        Notes
        -----
        - Modifies particles.positions, particles.velocities, particles.internal_energy in-place.
        - Assumes forces dict contains at least 'gravity' and 'hydro' accelerations.
        - Internal energy evolution uses du/dt from artificial viscosity heating.
        """
        # Compute total acceleration from all force contributions
        total_accel = np.zeros_like(particles.positions, dtype=np.float32)

        for key, accel in forces.items():
            if key != 'du_dt' and accel is not None:
                # Sum all spatial acceleration components
                total_accel += accel.astype(np.float32)

        # First call: only half-step kick to stagger velocities
        if not self.half_step_initialized:
            particles.velocities = particles.velocities.astype(np.float32) + \
                                   0.5 * dt * total_accel.astype(np.float32)
            self.half_step_initialized = True
            return

        # Standard leapfrog: kick (full step) -> drift
        dt_f32 = np.float32(dt)

        # 1. Kick: v^(n+1/2) = v^(n-1/2) + a^n * dt
        particles.velocities = particles.velocities.astype(np.float32) + \
                               dt_f32 * total_accel.astype(np.float32)

        # 2. Drift: x^(n+1) = x^n + v^(n+1/2) * dt
        particles.positions = particles.positions.astype(np.float32) + \
                              dt_f32 * particles.velocities.astype(np.float32)

        # 3. Update internal energy if requested
        if self.internal_energy_evolution and 'du_dt' in forces and forces['du_dt'] is not None:
            du_dt = forces['du_dt'].astype(np.float32)
            particles.internal_energy = particles.internal_energy.astype(np.float32) + \
                                        dt_f32 * du_dt

            # Enforce minimum internal energy to prevent negative temperatures
            # (can occur due to numerical errors or extreme cooling)
            particles.internal_energy = np.maximum(
                particles.internal_energy,
                np.float32(1e-10)
            )

    def estimate_timestep(
        self,
        particles: Any,
        cfl_factor: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Estimate appropriate timestep based on CFL condition and acceleration.

        Implements multiple timestep constraints following standard SPH practice:

        1. CFL condition: dt_cfl = C * min(h / (c_s + |v|))
           Ensures sound waves and particle motion are resolved.

        2. Acceleration constraint: dt_acc = C * min(sqrt(h / |a|))
           Prevents excessive displacement under strong forces.

        3. Optional: Force change constraint (not implemented in v1).

        The minimum of all constraints is returned.

        Parameters
        ----------
        particles : ParticleSystem
            Particle system with attributes: positions, velocities, smoothing_lengths,
            sound_speeds (or density + internal_energy for EOS call), accelerations.
        cfl_factor : Optional[float]
            Override default CFL factor if provided.
        **kwargs
            Additional constraint parameters:
            - 'accelerations' : NDArrayFloat, shape (N, 3) - current total acceleration
            - 'sound_speeds' : NDArrayFloat, shape (N,) - sound speeds (if not in particles)
            - 'min_dt' : float - absolute minimum timestep floor
            - 'max_dt' : float - absolute maximum timestep ceiling

        Returns
        -------
        dt : float
            Recommended timestep (scalar, float32).

        References
        ----------
        - Price (2012), Eq. 60-62: SPH timestep criteria
        - Rosswog (2009): Acceleration and CFL constraints for TDE simulations
        """
        if cfl_factor is None:
            cfl_factor = self.cfl_factor

        # Extract or compute sound speeds
        if hasattr(particles, 'sound_speed'):
            c_s = particles.sound_speed.astype(np.float32)
        elif 'sound_speeds' in kwargs:
            c_s = kwargs['sound_speeds'].astype(np.float32)
        else:
            # Fallback: assume ideal gas with gamma=5/3
            # c_s = sqrt(gamma * P / rho) = sqrt(gamma * (gamma-1) * u)
            gamma = np.float32(5.0 / 3.0)
            c_s = np.sqrt(gamma * (gamma - 1.0) * particles.internal_energy.astype(np.float32))

        # Velocity magnitudes
        v_mag = np.linalg.norm(particles.velocities.astype(np.float32), axis=1)

        # Smoothing lengths
        h = particles.smoothing_lengths.astype(np.float32)

        # 1. CFL timestep: dt_cfl = C * h / (c_s + |v|)
        signal_speed = c_s + v_mag
        # Avoid division by zero
        signal_speed = np.maximum(signal_speed, np.float32(1e-30))
        dt_cfl = cfl_factor * h / signal_speed
        dt_cfl_min = np.min(dt_cfl)

        # 2. Acceleration timestep: dt_acc = C * sqrt(h / |a|)
        if 'accelerations' in kwargs and kwargs['accelerations'] is not None:
            accel = kwargs['accelerations'].astype(np.float32)
        elif hasattr(particles, 'accelerations'):
            accel = particles.accelerations.astype(np.float32)
        else:
            # No acceleration info available, skip this constraint
            dt_acc_min = np.float32(np.inf)
            accel = None

        if accel is not None:
            accel_mag = np.linalg.norm(accel, axis=1)
            # Avoid division by zero for particles with negligible acceleration
            accel_mag = np.maximum(accel_mag, np.float32(1e-30))
            dt_acc = self.accel_factor * np.sqrt(h / accel_mag)
            dt_acc_min = np.min(dt_acc)
        else:
            dt_acc_min = np.float32(np.inf)

        # Take minimum of all constraints
        dt_candidate = np.float32(min(dt_cfl_min, dt_acc_min))

        # Apply absolute bounds if provided
        if 'min_dt' in kwargs:
            dt_candidate = np.maximum(dt_candidate, np.float32(kwargs['min_dt']))
        if 'max_dt' in kwargs:
            dt_candidate = np.minimum(dt_candidate, np.float32(kwargs['max_dt']))

        return float(dt_candidate)

    def reset(self) -> None:
        """
        Reset integrator state (e.g., for restarting simulation).

        Call this before starting a new simulation or after modifying particle state.
        """
        self.half_step_initialized = False
