"""
Coordinate transformation utilities for spacetime metrics.

This module provides transformations between Cartesian (x, y, z) and
Boyer-Lindquist spherical coordinates (r, θ, φ), as well as velocity
transformations and singularity regularization.

References
----------
- Misner, Thorne & Wheeler (1973) - Gravitation
- Tejeda et al. (2017), MNRAS 469, 4483 [arXiv:1701.00303]
- Liptai & Price (2019), MNRAS 485, 819 [arXiv:1901.08064]
"""

import numpy as np
from typing import Tuple, Union
import warnings


# Small epsilon for singularity regularization
EPS_COORD = 1e-10


def cartesian_to_bl_spherical(
    x: np.ndarray,
    regularize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to Boyer-Lindquist spherical coordinates.

    Boyer-Lindquist coordinates reduce to standard spherical coordinates
    for Schwarzschild (a=0) and flat space.

    Parameters
    ----------
    x : np.ndarray, shape (..., 3) or (3,)
        Cartesian coordinates (x, y, z).
    regularize : bool, default=True
        If True, regularize coordinate singularities at poles and origin.

    Returns
    -------
    r : np.ndarray, shape (...) or scalar
        Radial coordinate.
    theta : np.ndarray, shape (...) or scalar
        Polar angle θ ∈ [0, π].
    phi : np.ndarray, shape (...) or scalar
        Azimuthal angle φ ∈ [0, 2π).

    Notes
    -----
    For Boyer-Lindquist coordinates with spin parameter a:
        x² + y² + z² = r² + a² - a² sin²θ

    For Schwarzschild (a=0), this reduces to standard spherical:
        r² = x² + y² + z²

    Singularities:
    - θ = 0, π (poles): φ ill-defined
    - r = 0 (origin): θ, φ ill-defined

    These are regularized by clamping to small finite values if regularize=True.
    """
    x = np.atleast_1d(x)
    is_single = x.ndim == 1

    if x.shape[-1] != 3:
        raise ValueError(f"Last dimension must be 3 (x, y, z), got shape {x.shape}")

    # Extract components
    x_cart = x[..., 0]
    y_cart = x[..., 1]
    z_cart = x[..., 2]

    # Compute r (for Schwarzschild/flat space, a=0)
    # TODO: Generalize for Kerr when a != 0
    r = np.sqrt(x_cart**2 + y_cart**2 + z_cart**2)

    if regularize:
        r = np.where(r < EPS_COORD, EPS_COORD, r)

    # Compute θ ∈ [0, π]
    cos_theta = z_cart / r
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Numerical safety
    theta = np.arccos(cos_theta)

    if regularize:
        # Avoid exact poles
        theta = np.clip(theta, EPS_COORD, np.pi - EPS_COORD)

    # Compute φ ∈ [0, 2π)
    phi = np.arctan2(y_cart, x_cart)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)

    # If input was single vector, return scalars
    if is_single:
        r = r.item() if r.ndim == 0 or r.size == 1 else r
        theta = theta.item() if theta.ndim == 0 or theta.size == 1 else theta
        phi = phi.item() if phi.ndim == 0 or phi.size == 1 else phi

    return r, theta, phi


def bl_spherical_to_cartesian(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray
) -> np.ndarray:
    """
    Convert Boyer-Lindquist spherical coordinates to Cartesian.

    Parameters
    ----------
    r : np.ndarray, shape (...) or scalar
        Radial coordinate.
    theta : np.ndarray, shape (...) or scalar
        Polar angle θ ∈ [0, π].
    phi : np.ndarray, shape (...) or scalar
        Azimuthal angle φ ∈ [0, 2π).

    Returns
    -------
    x : np.ndarray, shape (..., 3) or (3,)
        Cartesian coordinates (x, y, z).

    Notes
    -----
    For Schwarzschild (a=0):
        x = r sin(θ) cos(φ)
        y = r sin(θ) sin(φ)
        z = r cos(θ)
    """
    # Check if inputs are scalars
    r_scalar = np.ndim(r) == 0
    theta_scalar = np.ndim(theta) == 0
    phi_scalar = np.ndim(phi) == 0
    all_scalar = r_scalar and theta_scalar and phi_scalar

    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    # Ensure consistent broadcasting
    r, theta, phi = np.broadcast_arrays(r, theta, phi)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta

    result = np.stack([x, y, z], axis=-1)

    # If all inputs were scalars, return 1D array
    if all_scalar:
        result = result.squeeze()

    return result


def velocity_cartesian_to_bl(
    x: np.ndarray,
    v_cart: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform velocity from Cartesian to Boyer-Lindquist coordinates.

    Parameters
    ----------
    x : np.ndarray, shape (..., 3)
        Position in Cartesian coordinates.
    v_cart : np.ndarray, shape (..., 3)
        Velocity in Cartesian coordinates (dx/dt, dy/dt, dz/dt).

    Returns
    -------
    v_r : np.ndarray, shape (...)
        Radial velocity dr/dt.
    v_theta : np.ndarray, shape (...)
        Polar velocity dθ/dt.
    v_phi : np.ndarray, shape (...)
        Azimuthal velocity dφ/dt.

    Notes
    -----
    Computed using chain rule:
        dr/dt = ∂r/∂x dx/dt + ∂r/∂y dy/dt + ∂r/∂z dz/dt
    and similarly for θ, φ.

    For Schwarzschild (a=0):
        ∂r/∂x = x/r, ∂r/∂y = y/r, ∂r/∂z = z/r
        ∂θ/∂x = xz/(r² √(x²+y²)), etc.
        ∂φ/∂x = -y/(x²+y²), ∂φ/∂y = x/(x²+y²)
    """
    x = np.atleast_1d(x)
    v_cart = np.atleast_1d(v_cart)

    if x.shape != v_cart.shape:
        raise ValueError(f"Position and velocity shapes must match: {x.shape} vs {v_cart.shape}")

    x_c, y_c, z_c = x[..., 0], x[..., 1], x[..., 2]
    vx, vy, vz = v_cart[..., 0], v_cart[..., 1], v_cart[..., 2]

    # Compute r and regularize
    r = np.sqrt(x_c**2 + y_c**2 + z_c**2)
    r = np.where(r < EPS_COORD, EPS_COORD, r)

    rho = np.sqrt(x_c**2 + y_c**2)  # Cylindrical radius
    rho = np.where(rho < EPS_COORD, EPS_COORD, rho)

    # dr/dt = (x vx + y vy + z vz) / r
    v_r = (x_c * vx + y_c * vy + z_c * vz) / r

    # dθ/dt = (z/r) * (x vx + y vy) / rho² - (ρ/r) * vz / r
    # Simplified: (z (x vx + y vy) - ρ² vz) / (r² ρ)
    v_theta = (z_c * (x_c * vx + y_c * vy) - rho**2 * vz) / (r**2 * rho)

    # dφ/dt = (x vy - y vx) / rho²
    v_phi = (x_c * vy - y_c * vx) / rho**2

    return v_r, v_theta, v_phi


def velocity_bl_to_cartesian(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    v_r: np.ndarray,
    v_theta: np.ndarray,
    v_phi: np.ndarray
) -> np.ndarray:
    """
    Transform velocity from Boyer-Lindquist to Cartesian coordinates.

    Parameters
    ----------
    r, theta, phi : np.ndarray
        Position in spherical coordinates.
    v_r, v_theta, v_phi : np.ndarray
        Velocity in spherical coordinates.

    Returns
    -------
    v_cart : np.ndarray, shape (..., 3)
        Velocity in Cartesian coordinates.

    Notes
    -----
    For Schwarzschild (a=0):
        vx = vr sin(θ) cos(φ) + r vθ cos(θ) cos(φ) - r vφ sin(θ) sin(φ)
        vy = vr sin(θ) sin(φ) + r vθ cos(θ) sin(φ) + r vφ sin(θ) cos(φ)
        vz = vr cos(θ) - r vθ sin(θ)
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    vx = (v_r * sin_theta * cos_phi +
          r * v_theta * cos_theta * cos_phi -
          r * v_phi * sin_theta * sin_phi)

    vy = (v_r * sin_theta * sin_phi +
          r * v_theta * cos_theta * sin_phi +
          r * v_phi * sin_theta * cos_phi)

    vz = v_r * cos_theta - r * v_theta * sin_theta

    return np.stack([vx, vy, vz], axis=-1)


def check_coordinate_validity(
    r: Union[np.ndarray, float],
    theta: Union[np.ndarray, float],
    r_horizon: float = 2.0,
    warn: bool = True
) -> Union[np.ndarray, bool]:
    """
    Check if coordinates are in valid regions and optionally warn.

    Parameters
    ----------
    r : np.ndarray or float
        Radial coordinate.
    theta : np.ndarray or float
        Polar angle.
    r_horizon : float, default=2.0
        Event horizon radius (2M for Schwarzschild).
    warn : bool, default=True
        If True, emit warnings for invalid regions.

    Returns
    -------
    valid : np.ndarray or bool
        Boolean mask indicating valid coordinates.

    Notes
    -----
    Invalid regions:
    - r < r_horizon (inside event horizon)
    - θ very close to 0 or π (coordinate singularity at poles)
    """
    # Convert to arrays for uniform handling
    r_arr = np.atleast_1d(r)
    theta_arr = np.atleast_1d(theta)
    is_scalar = np.ndim(r) == 0

    valid = np.ones_like(r_arr, dtype=bool)

    # Check horizon
    inside_horizon = r_arr < r_horizon
    if np.any(inside_horizon) and warn:
        n_inside = np.sum(inside_horizon)
        warnings.warn(
            f"{n_inside} particle(s) inside event horizon (r < {r_horizon:.2f}). "
            "Metric may be singular."
        )
    valid = valid & ~inside_horizon

    # Check poles
    at_poles = (theta_arr < EPS_COORD) | (theta_arr > np.pi - EPS_COORD)
    if np.any(at_poles) and warn:
        n_poles = np.sum(at_poles)
        warnings.warn(
            f"{n_poles} particle(s) near coordinate poles (θ ≈ 0 or π). "
            "Metric components may be ill-defined."
        )

    # Return scalar if input was scalar
    if is_scalar:
        return bool(valid.item())

    return valid


def regularize_near_singularity(
    r: np.ndarray,
    theta: np.ndarray,
    r_min: float = 1e-3,
    theta_margin: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regularize coordinates near singularities by clamping to safe values.

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate.
    theta : np.ndarray
        Polar angle.
    r_min : float, default=1e-3
        Minimum allowed r.
    theta_margin : float, default=1e-3
        Margin from poles (θ ∈ [margin, π-margin]).

    Returns
    -------
    r_reg, theta_reg : np.ndarray
        Regularized coordinates.

    Notes
    -----
    This is a safety measure for numerical stability. Particles that
    actually reach these regions should be handled by the simulation
    logic (e.g., removed or treated specially).
    """
    r_reg = np.maximum(r, r_min)
    theta_reg = np.clip(theta, theta_margin, np.pi - theta_margin)

    return r_reg, theta_reg
