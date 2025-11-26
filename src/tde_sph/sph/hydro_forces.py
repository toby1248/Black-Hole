"""
SPH hydrodynamic force computation.

This module implements the core SPH hydrodynamic forces following modern
SPH formulations (Price 2012, Monaghan 1997, 2005). Includes:
- Pressure gradient force
- Monaghan artificial viscosity for shock capturing
- Placeholder for thermal conductivity

Design follows REQ-001 (SPH core) with vectorized numpy implementation
suitable for future GPU acceleration (CUDA/Numba).
"""

import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional

# Type alias for clarity
NDArrayFloat = npt.NDArray[np.float32]


import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    prange = range
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Type alias for clarity
NDArrayFloat = npt.NDArray[np.float32]


@njit(fastmath=True)
def _kernel_gradient_cubic_spline_numba(r_vec, h, dim=3):
    """
    Compute gradient of cubic spline kernel ∇W(r, h).
    Inline implementation for Numba.
    """
    r = np.sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2])
    
    if r == 0.0:
        return np.zeros(3, dtype=np.float32)
        
    q = r / h
    
    # Sigma for 3D
    sigma = 1.0 / np.pi
    
    # dw/dq
    dw = 0.0
    if q < 1.0:
        dw = -3.0 * q + 2.25 * q * q
    elif q < 2.0:
        dw = -0.75 * (2.0 - q) * (2.0 - q)
        
    # dW/dr = (sigma / h^(d+1)) * dw
    # grad W = (dW/dr) * (r_vec / r)
    
    factor = (sigma / (h**(dim + 1))) * dw / r
    
    return np.array([factor * r_vec[0], factor * r_vec[1], factor * r_vec[2]], dtype=np.float32)


@njit(parallel=True, fastmath=True)
def _compute_hydro_numba(
    positions, velocities, masses, densities, pressures, sound_speeds, smoothing_lengths,
    neighbour_indices, neighbour_offsets,
    alpha, beta, eta
):
    """Numba implementation of hydro acceleration."""
    N = len(positions)
    accel = np.zeros((N, 3), dtype=np.float32)
    du_dt = np.zeros(N, dtype=np.float32)
    
    for i in prange(N):
        # Load particle i data
        pos_i = positions[i]
        vel_i = velocities[i]
        rho_i = densities[i]
        P_i = pressures[i]
        cs_i = sound_speeds[i]
        h_i = smoothing_lengths[i]
        
        # Precompute pressure term
        P_rho2_i = P_i / (rho_i * rho_i)
        
        start = neighbour_offsets[i]
        end = neighbour_offsets[i+1]
        
        ax = 0.0
        ay = 0.0
        az = 0.0
        du = 0.0
        
        for k in range(start, end):
            j = neighbour_indices[k]
            
            # Load particle j data
            pos_j = positions[j]
            vel_j = velocities[j]
            m_j = masses[j]
            rho_j = densities[j]
            P_j = pressures[j]
            cs_j = sound_speeds[j]
            h_j = smoothing_lengths[j]
            
            # Relative vectors
            dx = pos_i[0] - pos_j[0]
            dy = pos_i[1] - pos_j[1]
            dz = pos_i[2] - pos_j[2]
            
            dvx = vel_i[0] - vel_j[0]
            dvy = vel_i[1] - vel_j[1]
            dvz = vel_i[2] - vel_j[2]
            
            r2 = dx*dx + dy*dy + dz*dz
            r = np.sqrt(r2)
            
            # Kernel gradient
            # Use h_i for consistency with Python version (though symmetrizing h is better)
            # Inline gradient computation to avoid function call overhead
            # _kernel_gradient_cubic_spline_numba(r_vec, h)
            
            # Inline kernel gradient logic
            if r > 0:
                q = r / h_i
                sigma = 1.0 / np.pi # 3D
                dw = 0.0
                if q < 1.0:
                    dw = -3.0 * q + 2.25 * q * q
                elif q < 2.0:
                    dw = -0.75 * (2.0 - q) * (2.0 - q)
                
                grad_factor = (sigma / (h_i**4)) * dw / r
                grad_Wx = grad_factor * dx
                grad_Wy = grad_factor * dy
                grad_Wz = grad_factor * dz
            else:
                grad_Wx = 0.0
                grad_Wy = 0.0
                grad_Wz = 0.0

            # Pressure term
            P_rho2_j = P_j / (rho_j * rho_j)
            pressure_term = P_rho2_i + P_rho2_j
            
            # Viscosity
            v_dot_r = dvx*dx + dvy*dy + dvz*dz
            visc_term = 0.0
            
            if v_dot_r < 0:
                h_ij = 0.5 * (h_i + h_j)
                mu_ij = (h_ij * v_dot_r) / (r2 + eta*eta * h_ij*h_ij)
                
                c_ij = 0.5 * (cs_i + cs_j)
                rho_ij = 0.5 * (rho_i + rho_j)
                
                visc_term = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / rho_ij
                
            # Total force factor
            # F_ij = - m_j * (pressure + viscosity) * grad_W
            force_factor = -m_j * (pressure_term + visc_term)
            
            ax += force_factor * grad_Wx
            ay += force_factor * grad_Wy
            az += force_factor * grad_Wz
            
            # Energy
            # du/dt = 0.5 * m_j * (pressure + viscosity) * v_ij . grad_W
            v_dot_gradW = dvx*grad_Wx + dvy*grad_Wy + dvz*grad_Wz
            du += 0.5 * m_j * (pressure_term + visc_term) * v_dot_gradW
            
        accel[i, 0] = ax
        accel[i, 1] = ay
        accel[i, 2] = az
        du_dt[i] = du
        
    return accel, du_dt


def compute_hydro_acceleration(
    positions: NDArrayFloat,
    velocities: NDArrayFloat,
    masses: NDArrayFloat,
    densities: NDArrayFloat,
    pressures: NDArrayFloat,
    sound_speeds: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    neighbour_lists: List[npt.NDArray[np.int32]],
    kernel_gradient_func: callable,
    alpha: float = 1.0,
    beta: float = 2.0,
    eta: float = 0.1
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Compute SPH hydrodynamic acceleration and internal energy rate.

    Implements the standard SPH momentum and energy equations with
    artificial viscosity (Monaghan 1997, Price 2012).
    """
    n_particles = positions.shape[0]
    
    # Try Numba implementation first
    if HAS_NUMBA:
        # Convert neighbour lists to flat arrays
        # This overhead is small compared to the O(N*N_neigh) computation
        counts = np.array([len(l) for l in neighbour_lists], dtype=np.int32)
        offsets = np.zeros(len(counts) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        
        # Concatenate is fast
        if len(neighbour_lists) > 0:
             # Check if list is not empty of empty arrays
             total_neighbours = offsets[-1]
             if total_neighbours > 0:
                 indices = np.concatenate(neighbour_lists).astype(np.int32)
                 
                 return _compute_hydro_numba(
                     positions.astype(np.float32),
                     velocities.astype(np.float32),
                     masses.astype(np.float32),
                     densities.astype(np.float32),
                     pressures.astype(np.float32),
                     sound_speeds.astype(np.float32),
                     smoothing_lengths.astype(np.float32),
                     indices,
                     offsets,
                     alpha,
                     beta,
                     eta
                 )
             else:
                 # No neighbours, return zeros
                 return np.zeros((n_particles, 3), dtype=np.float32), np.zeros(n_particles, dtype=np.float32)
    
    # Fallback to original implementation
    accel = np.zeros((n_particles, 3), dtype=np.float32)
    du_dt = np.zeros(n_particles, dtype=np.float32)

    # Avoid division by zero
    densities = np.maximum(densities, 1e-10)

    for i in range(n_particles):
        neighbours = neighbour_lists[i]
        if len(neighbours) == 0:
            continue

        # Extract neighbour properties
        r_j = positions[neighbours]  # Shape (n_neigh, 3)
        v_j = velocities[neighbours]
        m_j = masses[neighbours]
        rho_j = densities[neighbours]
        P_j = pressures[neighbours]
        cs_j = sound_speeds[neighbours]
        h_j = smoothing_lengths[neighbours]

        # Particle i properties
        r_i = positions[i]
        v_i = velocities[i]
        rho_i = densities[i]
        P_i = pressures[i]
        cs_i = sound_speeds[i]
        h_i = smoothing_lengths[i]

        # Relative positions and velocities
        r_ij = r_i - r_j  # Shape (n_neigh, 3)
        v_ij = v_i - v_j  # Shape (n_neigh, 3)

        # Distances
        r_ij_mag = np.linalg.norm(r_ij, axis=1, keepdims=True)  # Shape (n_neigh, 1)
        r_ij_mag = np.maximum(r_ij_mag, 1e-10)  # Avoid division by zero

        # Average smoothing length (symmetric interaction)
        h_ij = 0.5 * (h_i + h_j)

        # Kernel gradient: ∇W_ij = ∇W(r_ij, h_i)
        # Note: Using h_i for consistency (can symmetrize later)
        grad_W_ij = kernel_gradient_func(r_ij, h_i)  # Shape (n_neigh, 3)

        # === Pressure gradient force ===
        # Standard SPH pressure gradient: -∑_j m_j (P_i/ρ_i² + P_j/ρ_j²) ∇W_ij
        pressure_factor = P_i / rho_i**2 + P_j / rho_j**2  # Shape (n_neigh,)
        pressure_accel = -np.sum(
            m_j[:, np.newaxis] * pressure_factor[:, np.newaxis] * grad_W_ij,
            axis=0
        )

        # === Artificial viscosity ===
        # Compute v_ij · r_ij
        v_dot_r = np.sum(v_ij * r_ij, axis=1)  # Shape (n_neigh,)

        # Only apply viscosity when particles approach (v_ij · r_ij < 0)
        approaching = v_dot_r < 0

        # Initialize viscosity term
        Pi_ij = np.zeros(len(neighbours), dtype=np.float32)

        if np.any(approaching):
            # Average quantities
            c_ij = 0.5 * (cs_i + cs_j[approaching])
            rho_ij = 0.5 * (rho_i + rho_j[approaching])
            h_ij_appr = h_ij[approaching]

            # Signal velocity (Monaghan 1997, Eq. 2.4)
            r_ij_mag_appr = r_ij_mag[approaching].squeeze()
            mu_ij = (h_ij_appr * v_dot_r[approaching]) / (
                r_ij_mag_appr**2 + eta**2 * h_ij_appr**2
            )

            # Artificial viscosity (Monaghan 1997, Eq. 2.3)
            Pi_ij[approaching] = (-alpha * c_ij * mu_ij + beta * mu_ij**2) / rho_ij

        # Viscosity acceleration contribution
        viscosity_accel = np.sum(
            m_j[:, np.newaxis] * Pi_ij[:, np.newaxis] * grad_W_ij,
            axis=0
        )

        # Total acceleration
        accel[i] = pressure_accel + viscosity_accel

        # === Energy equation ===
        # du_i/dt = (1/2) ∑_j m_j (P_i/ρ_i² + P_j/ρ_j² + Π_ij) v_ij · ∇W_ij
        # Note: Factor of 1/2 for symmetry (each pair contributes to both particles)
        energy_factor = pressure_factor + Pi_ij
        v_dot_gradW = np.sum(v_ij * grad_W_ij, axis=1)  # Shape (n_neigh,)
        du_dt[i] = 0.5 * np.sum(m_j * energy_factor * v_dot_gradW)

    return accel, du_dt


def compute_viscosity_timestep(
    smoothing_lengths: NDArrayFloat,
    sound_speeds: NDArrayFloat,
    alpha: float = 1.0
) -> float:
    """
    Compute viscosity-limited timestep.

    Following Monaghan (1989), the viscosity constraint is:
        dt ≤ min_i (C_cour × h_i / (α c_i))

    Parameters
    ----------
    smoothing_lengths : NDArrayFloat, shape (N,)
        Smoothing lengths.
    sound_speeds : NDArrayFloat, shape (N,)
        Sound speeds.
    alpha : float, optional
        Viscosity parameter (default 1.0).

    Returns
    -------
    dt_visc : float
        Viscosity-limited timestep.

    Notes
    -----
    This is typically combined with the Courant condition and acceleration
    constraint in the main timestep estimator.
    """
    # Avoid division by zero
    cs_safe = np.maximum(sound_speeds, 1e-10)

    # Viscosity timescale
    dt_visc = np.min(smoothing_lengths / (alpha * cs_safe))

    return dt_visc


def compute_thermal_conductivity(
    positions: NDArrayFloat,
    masses: NDArrayFloat,
    densities: NDArrayFloat,
    internal_energies: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    neighbour_lists: List[npt.NDArray[np.int32]],
    kernel_gradient_func: callable,
    kappa: float = 0.0
) -> NDArrayFloat:
    """
    Compute thermal energy diffusion (thermal conductivity).

    Implements a simple SPH thermal conductivity term to smooth
    temperature discontinuities (optional, placeholder for REQ-011).

    Energy diffusion:
        du_i/dt |_cond = ∑_j (4 κ m_j / (ρ_i ρ_j (ρ_i + ρ_j))) ×
                         (u_i - u_j) r_ij · ∇W_ij / r_ij²

    Parameters
    ----------
    positions : NDArrayFloat, shape (N, 3)
        Particle positions.
    masses : NDArrayFloat, shape (N,)
        Particle masses.
    densities : NDArrayFloat, shape (N,)
        Particle densities.
    internal_energies : NDArrayFloat, shape (N,)
        Specific internal energies.
    smoothing_lengths : NDArrayFloat, shape (N,)
        Smoothing lengths.
    neighbour_lists : List[NDArray[int32]]
        Neighbour indices.
    kernel_gradient_func : callable
        Kernel gradient function.
    kappa : float, optional
        Thermal conductivity coefficient (default 0.0 = off).

    Returns
    -------
    du_dt_cond : NDArrayFloat, shape (N,)
        Rate of change of internal energy due to conduction.

    Notes
    -----
    If kappa = 0, returns zeros (no conduction).
    For non-zero kappa, implements a simple Laplacian-like diffusion
    (Price 2012, Section 8.1; Cleary & Monaghan 1999).

    This is a placeholder for more sophisticated energy transport
    methods (flux-limited diffusion, etc.) in TASK-032.

    References
    ----------
    Cleary, P. W., & Monaghan, J. J. (1999), "Conduction modelling
    using smoothed particle hydrodynamics", Journal of Computational
    Physics, 148, 227.
    """
    if kappa == 0.0:
        return np.zeros(len(positions), dtype=np.float32)

    n_particles = positions.shape[0]
    du_dt_cond = np.zeros(n_particles, dtype=np.float32)

    # Avoid division by zero
    densities = np.maximum(densities, 1e-10)

    for i in range(n_particles):
        neighbours = neighbour_lists[i]
        if len(neighbours) == 0:
            continue

        # Neighbour properties
        r_j = positions[neighbours]
        m_j = masses[neighbours]
        rho_j = densities[neighbours]
        u_j = internal_energies[neighbours]

        # Particle i properties
        r_i = positions[i]
        rho_i = densities[i]
        u_i = internal_energies[i]
        h_i = smoothing_lengths[i]

        # Relative positions
        r_ij = r_i - r_j
        r_ij_mag_sq = np.sum(r_ij**2, axis=1)
        r_ij_mag_sq = np.maximum(r_ij_mag_sq, 1e-20)

        # Kernel gradient
        grad_W_ij = kernel_gradient_func(r_ij, h_i)

        # r_ij · ∇W_ij
        r_dot_gradW = np.sum(r_ij * grad_W_ij, axis=1)

        # Conduction term (Price 2012, Eq. 125)
        rho_sum = rho_i + rho_j
        conductivity_factor = (4.0 * kappa * m_j) / (rho_i * rho_j * rho_sum)
        energy_diff = u_i - u_j

        du_dt_cond[i] = np.sum(
            conductivity_factor * energy_diff * r_dot_gradW / r_ij_mag_sq
        )

    return du_dt_cond


# TODO (Future GPU optimization):
# - Port compute_hydro_acceleration to CUDA/Numba kernel
# - Use shared memory for neighbour data
# - Parallelize over particles with thread blocks
# - Benchmark performance gain vs. CPU vectorized numpy
#
# Example structure:
# @cuda.jit
# def hydro_kernel_cuda(positions, velocities, ..., accel_out, du_dt_out):
#     """CUDA kernel for hydrodynamic forces."""
#     i = cuda.grid(1)
#     if i < n_particles:
#         # Load particle i data
#         # Loop over neighbours (from precomputed lists)
#         # Compute forces and accumulate
#         # Write to output arrays
#         pass
