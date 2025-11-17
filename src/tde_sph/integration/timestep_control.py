"""
GR-aware timestep control for TDE simulations near strong-field regions.

Implements multiple timestep constraints to ensure stability and accuracy
in both Newtonian and general relativistic regimes:

1. CFL condition (SPH): dt = C * min(h / (c_s + |v|))
2. Acceleration constraint: dt = C * min(sqrt(h / |a|))
3. Orbital timescale constraint: dt = C * min(sqrt(r³ / M))
4. ISCO constraint (near r ~ 6M): dt = C * min(r / |v|)

The minimum of all applicable constraints is returned.

References
----------
- Liptai & Price (2019), MNRAS 485, 819 - GR timestep criteria
- Price (2012), JCP 231, 759 - SPH timestep constraints
- Rosswog (2009) - TDE simulation timesteps

Design Notes
------------
This module provides standalone functions (not classes) for timestep estimation,
following functional programming principles. Integrators call these functions
rather than implementing timestep logic directly.
"""

import numpy as np
from typing import Any, Dict, Optional, Union
from tde_sph.core.interfaces import Metric, NDArrayFloat


def estimate_timestep_gr(
    particles: Any,
    metric: Optional[Metric] = None,
    config: Optional[Union[Dict, Any]] = None,
    **kwargs
) -> float:
    """
    Estimate timestep with GR-aware constraints.

    Computes all applicable timestep constraints and returns the minimum.
    Works in both Newtonian mode (metric=None) and GR mode (metric provided).

    Parameters
    ----------
    particles : ParticleSystem
        Particle system with positions, velocities, smoothing_lengths, etc.
    metric : Optional[Metric]
        Spacetime metric for orbital/ISCO constraints. If None, uses Newtonian.
    config : Optional[Union[Dict, Any]]
        Configuration dict or object with timestep parameters:
        - cfl_factor : float, default 0.3
        - accel_factor : float, default 0.25
        - orbital_factor : float, default 0.1 (GR only)
        - isco_factor : float, default 0.05 (GR only)
        - isco_radius_threshold : float, default 10.0 (in units of M)
        - bh_mass : float, default 1.0
        - min_dt : float, optional absolute minimum
        - max_dt : float, optional absolute maximum
    **kwargs
        Additional parameters:
        - accelerations : NDArrayFloat, shape (N, 3)
        - sound_speeds : NDArrayFloat, shape (N,)

    Returns
    -------
    dt : float
        Recommended timestep (minimum of all constraints).

    Notes
    -----
    - Returns FP32 timestep for compatibility with particle arrays.
    - For GR mode, orbital and ISCO constraints are active.
    - ISCO constraint only applies to particles with r < isco_radius_threshold.
    - If all particles are far from BH, reduces to standard SPH timestep.

    Examples
    --------
    >>> # Newtonian mode
    >>> dt = estimate_timestep_gr(particles, metric=None, config={'cfl_factor': 0.3})
    >>>
    >>> # GR mode with Schwarzschild metric
    >>> from tde_sph.metric.schwarzschild import SchwarzschildMetric
    >>> metric = SchwarzschildMetric(mass=1.0)
    >>> dt = estimate_timestep_gr(particles, metric=metric, config={'isco_factor': 0.05})
    """
    # Parse configuration
    if config is None:
        config = {}

    if not isinstance(config, dict):
        # Convert config object to dict
        cfl_factor = getattr(config, 'cfl_factor', 0.3)
        accel_factor = getattr(config, 'accel_factor', 0.25)
        orbital_factor = getattr(config, 'orbital_factor', 0.1)
        isco_factor = getattr(config, 'isco_factor', 0.05)
        isco_radius_threshold = getattr(config, 'isco_radius_threshold', 10.0)
        bh_mass = getattr(config, 'bh_mass', 1.0)
        min_dt = getattr(config, 'min_dt', None)
        max_dt = getattr(config, 'max_dt', None)
    else:
        cfl_factor = config.get('cfl_factor', 0.3)
        accel_factor = config.get('accel_factor', 0.25)
        orbital_factor = config.get('orbital_factor', 0.1)
        isco_factor = config.get('isco_factor', 0.05)
        isco_radius_threshold = config.get('isco_radius_threshold', 10.0)
        bh_mass = config.get('bh_mass', 1.0)
        min_dt = config.get('min_dt', None)
        max_dt = config.get('max_dt', None)

    # Working precision
    dtype = np.float32

    # === 1. CFL CONSTRAINT ===
    dt_cfl = _estimate_cfl_timestep(
        particles=particles,
        cfl_factor=cfl_factor,
        dtype=dtype,
        **kwargs
    )

    # === 2. ACCELERATION CONSTRAINT ===
    dt_acc = _estimate_acceleration_timestep(
        particles=particles,
        accel_factor=accel_factor,
        dtype=dtype,
        **kwargs
    )

    # === 3. ORBITAL CONSTRAINT (GR mode) ===
    if metric is not None:
        dt_orb = _estimate_orbital_timestep(
            particles=particles,
            bh_mass=bh_mass,
            orbital_factor=orbital_factor,
            dtype=dtype
        )
    else:
        dt_orb = np.float32(np.inf)

    # === 4. ISCO CONSTRAINT (GR mode, near BH) ===
    if metric is not None:
        dt_isco = _estimate_isco_timestep(
            particles=particles,
            bh_mass=bh_mass,
            isco_factor=isco_factor,
            isco_radius_threshold=isco_radius_threshold,
            dtype=dtype
        )
    else:
        dt_isco = np.float32(np.inf)

    # === TAKE MINIMUM ===
    dt_candidate = np.float32(min(dt_cfl, dt_acc, dt_orb, dt_isco))

    # Apply absolute bounds
    if min_dt is not None:
        dt_candidate = np.maximum(dt_candidate, np.float32(min_dt))
    if max_dt is not None:
        dt_candidate = np.minimum(dt_candidate, np.float32(max_dt))

    # Validate result
    if not np.isfinite(dt_candidate) or dt_candidate <= 0:
        raise RuntimeError(
            f"Invalid timestep computed: dt={dt_candidate}. "
            f"Constraints: CFL={dt_cfl}, acc={dt_acc}, orb={dt_orb}, ISCO={dt_isco}"
        )

    return float(dt_candidate)


def _estimate_cfl_timestep(
    particles: Any,
    cfl_factor: float,
    dtype: np.dtype,
    **kwargs
) -> float:
    """
    CFL timestep: dt = C * min(h / (c_s + |v|))

    Parameters
    ----------
    particles : ParticleSystem
    cfl_factor : float
    dtype : np.dtype
    **kwargs : may contain 'sound_speeds'

    Returns
    -------
    dt_cfl : float
    """
    # Sound speeds
    if 'sound_speeds' in kwargs and kwargs['sound_speeds'] is not None:
        c_s = kwargs['sound_speeds'].astype(dtype)
    elif hasattr(particles, 'sound_speeds'):
        c_s = particles.sound_speeds.astype(dtype)
    else:
        # Fallback: compute from internal energy (ideal gas, gamma=5/3)
        gamma = np.float32(5.0 / 3.0)
        c_s = np.sqrt(gamma * (gamma - 1.0) * particles.internal_energy.astype(dtype))

    # Velocity magnitudes
    v_mag = np.linalg.norm(particles.velocities.astype(dtype), axis=1)

    # Smoothing lengths
    h = particles.smoothing_lengths.astype(dtype)

    # Signal speed
    signal_speed = c_s + v_mag
    signal_speed = np.maximum(signal_speed, dtype(1e-30))  # Avoid division by zero

    # CFL timestep
    dt_cfl = cfl_factor * h / signal_speed
    dt_cfl_min = np.min(dt_cfl)

    return float(dt_cfl_min)


def _estimate_acceleration_timestep(
    particles: Any,
    accel_factor: float,
    dtype: np.dtype,
    **kwargs
) -> float:
    """
    Acceleration timestep: dt = C * min(sqrt(h / |a|))

    Parameters
    ----------
    particles : ParticleSystem
    accel_factor : float
    dtype : np.dtype
    **kwargs : may contain 'accelerations'

    Returns
    -------
    dt_acc : float
    """
    # Get accelerations
    if 'accelerations' in kwargs and kwargs['accelerations'] is not None:
        accel = kwargs['accelerations'].astype(dtype)
    elif hasattr(particles, 'accelerations'):
        accel = particles.accelerations.astype(dtype)
    else:
        # No acceleration available, return inf (constraint not active)
        return float(np.inf)

    # Acceleration magnitudes
    accel_mag = np.linalg.norm(accel, axis=1)
    accel_mag = np.maximum(accel_mag, dtype(1e-30))  # Avoid division by zero

    # Smoothing lengths
    h = particles.smoothing_lengths.astype(dtype)

    # Acceleration timestep
    dt_acc = accel_factor * np.sqrt(h / accel_mag)
    dt_acc_min = np.min(dt_acc)

    return float(dt_acc_min)


def _estimate_orbital_timestep(
    particles: Any,
    bh_mass: float,
    orbital_factor: float,
    dtype: np.dtype
) -> float:
    """
    Orbital timescale constraint: dt = C * min(sqrt(r³ / M))

    Based on Keplerian orbital period T = 2π sqrt(r³ / GM).
    In geometric units (G=1, M_BH normalized), T = 2π sqrt(r³ / M).

    Parameters
    ----------
    particles : ParticleSystem
    bh_mass : float
        Black hole mass in code units.
    orbital_factor : float
        Safety factor (typically 0.05-0.1).
    dtype : np.dtype

    Returns
    -------
    dt_orb : float
    """
    # Radial distances from BH (assumed at origin)
    r = np.linalg.norm(particles.positions.astype(dtype), axis=1)
    r = np.maximum(r, dtype(1e-30))  # Avoid division by zero

    # Orbital timescale: T = 2π sqrt(r³ / M)
    # Timestep: dt = C * T / (2π) = C * sqrt(r³ / M)
    t_orb = np.sqrt(r**3 / bh_mass)
    dt_orb = orbital_factor * t_orb
    dt_orb_min = np.min(dt_orb)

    return float(dt_orb_min)


def _estimate_isco_timestep(
    particles: Any,
    bh_mass: float,
    isco_factor: float,
    isco_radius_threshold: float,
    dtype: np.dtype
) -> float:
    """
    ISCO constraint for particles near innermost stable circular orbit.

    dt = C * (r / |v|) for r < isco_radius_threshold

    This ensures particles don't move more than a small fraction of their
    orbital radius per timestep when in strong-field regions.

    Parameters
    ----------
    particles : ParticleSystem
    bh_mass : float
        Black hole mass in code units.
    isco_factor : float
        Safety factor (typically 0.05 or smaller).
    isco_radius_threshold : float
        Apply constraint only for r < this threshold (in units of M).
    dtype : np.dtype

    Returns
    -------
    dt_isco : float
    """
    # Radial distances
    r = np.linalg.norm(particles.positions.astype(dtype), axis=1)

    # Velocity magnitudes
    v_mag = np.linalg.norm(particles.velocities.astype(dtype), axis=1)
    v_mag = np.maximum(v_mag, dtype(1e-30))  # Avoid division by zero

    # Apply constraint only to particles within threshold
    r_threshold = isco_radius_threshold * bh_mass
    mask = r < r_threshold

    if not np.any(mask):
        # No particles near ISCO
        return float(np.inf)

    # ISCO timestep: dt = C * r / |v|
    dt_isco_particles = isco_factor * r[mask] / v_mag[mask]
    dt_isco_min = np.min(dt_isco_particles)

    return float(dt_isco_min)


def get_timestep_diagnostics(
    particles: Any,
    metric: Optional[Metric] = None,
    config: Optional[Union[Dict, Any]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute all timestep constraints and return as dict for diagnostics.

    Useful for logging which constraint is limiting the timestep.

    Parameters
    ----------
    particles : ParticleSystem
    metric : Optional[Metric]
    config : Optional[Union[Dict, Any]]
    **kwargs

    Returns
    -------
    diagnostics : Dict[str, float]
        Dictionary with keys:
        - 'dt_cfl' : CFL constraint
        - 'dt_acc' : Acceleration constraint
        - 'dt_orb' : Orbital constraint (GR mode)
        - 'dt_isco' : ISCO constraint (GR mode)
        - 'dt_total' : Minimum (actual timestep)
        - 'limiting_constraint' : str, which constraint is limiting

    Examples
    --------
    >>> diag = get_timestep_diagnostics(particles, metric, config)
    >>> print(f"Timestep limited by: {diag['limiting_constraint']}")
    >>> print(f"dt = {diag['dt_total']:.3e} (CFL: {diag['dt_cfl']:.3e})")
    """
    # Parse config
    if config is None:
        config = {}

    if not isinstance(config, dict):
        cfl_factor = getattr(config, 'cfl_factor', 0.3)
        accel_factor = getattr(config, 'accel_factor', 0.25)
        orbital_factor = getattr(config, 'orbital_factor', 0.1)
        isco_factor = getattr(config, 'isco_factor', 0.05)
        isco_radius_threshold = getattr(config, 'isco_radius_threshold', 10.0)
        bh_mass = getattr(config, 'bh_mass', 1.0)
    else:
        cfl_factor = config.get('cfl_factor', 0.3)
        accel_factor = config.get('accel_factor', 0.25)
        orbital_factor = config.get('orbital_factor', 0.1)
        isco_factor = config.get('isco_factor', 0.05)
        isco_radius_threshold = config.get('isco_radius_threshold', 10.0)
        bh_mass = config.get('bh_mass', 1.0)

    dtype = np.float32

    # Compute all constraints
    dt_cfl = _estimate_cfl_timestep(particles, cfl_factor, dtype, **kwargs)
    dt_acc = _estimate_acceleration_timestep(particles, accel_factor, dtype, **kwargs)

    if metric is not None:
        dt_orb = _estimate_orbital_timestep(particles, bh_mass, orbital_factor, dtype)
        dt_isco = _estimate_isco_timestep(
            particles, bh_mass, isco_factor, isco_radius_threshold, dtype
        )
    else:
        dt_orb = float(np.inf)
        dt_isco = float(np.inf)

    # Find minimum
    constraints = {
        'CFL': dt_cfl,
        'acceleration': dt_acc,
        'orbital': dt_orb,
        'ISCO': dt_isco
    }

    dt_total = min(constraints.values())
    limiting_constraint = min(constraints, key=constraints.get)

    return {
        'dt_cfl': dt_cfl,
        'dt_acc': dt_acc,
        'dt_orb': dt_orb,
        'dt_isco': dt_isco,
        'dt_total': dt_total,
        'limiting_constraint': limiting_constraint
    }
