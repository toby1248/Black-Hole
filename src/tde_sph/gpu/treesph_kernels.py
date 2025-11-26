"""
TreeSPH GPU kernels using octree-based neighbour search.

Provides O(N log N) density and hydro force computation using
pre-computed neighbour lists from the octree.
"""
import numpy as np
import cupy as cp

# Kernel normalization constants (σ_3D for cubic spline)
KERNEL_NORM_3D = 1.0 / np.pi  # Cubic spline in 3D

TREESPH_CUDA_SOURCE = r'''
#define PI 3.14159265359f
#define KERNEL_NORM (1.0f / PI)

__device__ float cubic_spline_kernel(float q) {
    // q = r/h, already normalized
    if (q > 2.0f) {
        return 0.0f;
    } else if (q > 1.0f) {
        float v = 2.0f - q;
        return KERNEL_NORM * 0.25f * v * v * v;
    } else {
        return KERNEL_NORM * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    }
}

__device__ float cubic_spline_gradient(float q) {
    // Returns dW/dq (still need to divide by h to get dW/dr)
    if (q > 2.0f) {
        return 0.0f;
    } else if (q > 1.0f) {
        float v = 2.0f - q;
        return KERNEL_NORM * (-0.75f * v * v);
    } else {
        return KERNEL_NORM * (-3.0f * q + 2.25f * q * q);
    }
}

extern "C" __global__
void compute_density_from_neighbours(
    const float* positions,
    const float* masses,
    const float* smoothing_lengths,
    const int* neighbour_lists,
    const int* neighbour_counts,
    float* densities,
    int n_particles,
    int max_neighbours
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    
    float hi = smoothing_lengths[i];
    float hi_safe = fmaxf(hi, 1e-6f);
    float inv_h = 1.0f / hi_safe;
    float inv_h3 = inv_h * inv_h * inv_h;
    float xi = positions[i * 3 + 0];
    float yi = positions[i * 3 + 1];
    float zi = positions[i * 3 + 2];
    
    // Self-contribution W(0, h_i) = sigma / h_i^3 (Price 2012, Eq. 13)
    float rho = masses[i] * (KERNEL_NORM * inv_h3);
    int n_neigh = neighbour_counts[i];
    
    // Loop over neighbours
    for (int k = 0; k < n_neigh; k++) {
        int j = neighbour_lists[i * max_neighbours + k];
        if (j < 0 || j >= n_particles || j == i) continue;
        
        float xj = positions[j * 3 + 0];
        float yj = positions[j * 3 + 1];
        float zj = positions[j * 3 + 2];
        float mj = masses[j];
        
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        float q = r * inv_h;
        
        // Cubic spline kernel
        float W = cubic_spline_kernel(q) * inv_h3;
        rho += mj * W;
    }
    
    densities[i] = fmaxf(rho, 1e-10f);  // Prevent zero density
}

extern "C" __global__
void compute_hydro_from_neighbours(
    const float* positions,
    const float* velocities,
    const float* masses,
    const float* smoothing_lengths,
    const float* densities,
    const float* pressures,
    const float* sound_speeds,
    const int* neighbour_lists,
    const int* neighbour_counts,
    float* accelerations,
    float* du_dt,
    int n_particles,
    int max_neighbours,
    float alpha_visc,
    float beta_visc
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    
    float hi = smoothing_lengths[i];
    float xi = positions[i * 3 + 0];
    float yi = positions[i * 3 + 1];
    float zi = positions[i * 3 + 2];
    float vxi = velocities[i * 3 + 0];
    float vyi = velocities[i * 3 + 1];
    float vzi = velocities[i * 3 + 2];
    float mi = masses[i];
    
    // Apply safety floors to prevent NaN/inf
    float rhoi = fmaxf(densities[i], 1e-10f);
    float Pi = fmaxf(pressures[i], 0.0f);  // Pressure can't be negative
    float csi = fmaxf(sound_speeds[i], 1e-6f);  // Prevent zero sound speed
    
    // Check for NaN in input (emergency fallback)
    if (!isfinite(rhoi) || !isfinite(Pi) || !isfinite(csi)) {
        accelerations[i * 3 + 0] = 0.0f;
        accelerations[i * 3 + 1] = 0.0f;
        accelerations[i * 3 + 2] = 0.0f;
        du_dt[i] = 0.0f;
        return;
    }
    
    double ax = 0.0, ay = 0.0, az = 0.0;
    double dudt = 0.0;
    
    double P_rhoi2 = (double)Pi / ((double)rhoi * (double)rhoi);
    
    int n_neigh = neighbour_counts[i];
    
    // Loop over neighbours
    for (int k = 0; k < n_neigh; k++) {
        int j = neighbour_lists[i * max_neighbours + k];
        if (j < 0 || j >= n_particles || j == i) continue;
        
        float xj = positions[j * 3 + 0];
        float yj = positions[j * 3 + 1];
        float zj = positions[j * 3 + 2];
        float vxj = velocities[j * 3 + 0];
        float vyj = velocities[j * 3 + 1];
        float vzj = velocities[j * 3 + 2];
        float mj = masses[j];
        float hj = smoothing_lengths[j];
        
        // Apply safety floors to neighbour properties
        float rhoj = fmaxf(densities[j], 1e-10f);
        float Pj = fmaxf(pressures[j], 0.0f);
        float csj = fmaxf(sound_speeds[j], 1e-6f);
        
        // Check for NaN in neighbour data
        if (!isfinite(rhoj) || !isfinite(Pj) || !isfinite(csj)) {
            continue;  // Skip this neighbour
        }
        
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (r < 1e-10f) continue;
        
        float dvx = vxi - vxj;
        float dvy = vyi - vyj;
        float dvz = vzi - vzj;
        
        // Average smoothing length for gradient and viscosity
        float h_avg = 0.5f * (hi + hj);
        float q = r / h_avg;
        
        // Cubic spline kernel gradient: dW/dr = (dW/dq) * (dq/dr) = (dW/dq) / h
        float dWdq = cubic_spline_gradient(q);
        float dWdr = dWdq / (h_avg * h_avg * h_avg * h_avg);
        
        float dWdx = dWdr * dx / r;
        float dWdy = dWdr * dy / r;
        float dWdz = dWdr * dz / r;
        
        // Pressure force (symmetrized SPH)
        double P_rhoj2 = (double)Pj / ((double)rhoj * (double)rhoj);
        double P_term = P_rhoi2 + P_rhoj2;
        
        // Artificial viscosity
        double visc = 0.0;
        float v_dot_r = dvx*dx + dvy*dy + dvz*dz;
        if (v_dot_r < 0.0f) {
            double rho_avg = 0.5 * ((double)rhoi + (double)rhoj);
            double cs_avg = 0.5 * ((double)csi + (double)csj);
            double mu = (double)h_avg * (double)v_dot_r / 
                       ((double)r * (double)r + 0.01 * (double)h_avg * (double)h_avg);
            visc = (-(double)alpha_visc * cs_avg * mu + (double)beta_visc * mu * mu) / rho_avg;
        }
        
        double force_term = (double)mj * (P_term + visc);
        
        ax -= force_term * (double)dWdx;
        ay -= force_term * (double)dWdy;
        az -= force_term * (double)dWdz;
        
        // Energy equation
        dudt += 0.5 * force_term * ((double)dvx * (double)dWdx + 
                                     (double)dvy * (double)dWdy + 
                                     (double)dvz * (double)dWdz);
    }
    
    // Final safety check: replace any NaN/inf with zero
    if (!isfinite(ax)) ax = 0.0;
    if (!isfinite(ay)) ay = 0.0;
    if (!isfinite(az)) az = 0.0;
    if (!isfinite(dudt)) dudt = 0.0;
    
    accelerations[i * 3 + 0] = (float)ax;
    accelerations[i * 3 + 1] = (float)ay;
    accelerations[i * 3 + 2] = (float)az;
    du_dt[i] = (float)dudt;
}
'''

# Compile kernels
_treesph_module = cp.RawModule(code=TREESPH_CUDA_SOURCE)
_density_kernel = _treesph_module.get_function('compute_density_from_neighbours')
_hydro_kernel = _treesph_module.get_function('compute_hydro_from_neighbours')


def compute_density_treesph(
    positions: cp.ndarray,
    masses: cp.ndarray,
    smoothing_lengths: cp.ndarray,
    neighbour_lists: cp.ndarray,
    neighbour_counts: cp.ndarray
) -> cp.ndarray:
    """
    Compute SPH density using octree neighbour lists.
    
    Parameters
    ----------
    positions : cp.ndarray, shape (N, 3)
        Particle positions on GPU
    masses : cp.ndarray, shape (N,)
        Particle masses on GPU
    smoothing_lengths : cp.ndarray, shape (N,)
        Smoothing lengths on GPU
    neighbour_lists : cp.ndarray, shape (N, max_neighbours)
        Neighbour indices from octree
    neighbour_counts : cp.ndarray, shape (N,)
        Number of neighbours for each particle
        
    Returns
    -------
    densities : cp.ndarray, shape (N,)
        Particle densities on GPU
    """
    n = positions.shape[0]
    max_neighbours = neighbour_lists.shape[1]
    
    densities = cp.zeros(n, dtype=cp.float32)
    
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    _density_kernel(
        (grid_size,), (block_size,),
        (positions, masses, smoothing_lengths,
         neighbour_lists, neighbour_counts, densities,
         n, max_neighbours)
    )
    
    return densities


def compute_hydro_treesph(
    positions: cp.ndarray,
    velocities: cp.ndarray,
    masses: cp.ndarray,
    smoothing_lengths: cp.ndarray,
    densities: cp.ndarray,
    pressures: cp.ndarray,
    sound_speeds: cp.ndarray,
    neighbour_lists: cp.ndarray,
    neighbour_counts: cp.ndarray,
    alpha: float = 1.0,
    beta: float = 2.0
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute SPH hydrodynamic forces using octree neighbour lists.
    
    Parameters
    ----------
    positions : cp.ndarray, shape (N, 3)
        Particle positions on GPU
    velocities : cp.ndarray, shape (N, 3)
        Particle velocities on GPU
    masses : cp.ndarray, shape (N,)
        Particle masses on GPU
    smoothing_lengths : cp.ndarray, shape (N,)
        Smoothing lengths on GPU
    densities : cp.ndarray, shape (N,)
        Particle densities on GPU
    pressures : cp.ndarray, shape (N,)
        Particle pressures on GPU
    sound_speeds : cp.ndarray, shape (N,)
        Sound speeds on GPU
    neighbour_lists : cp.ndarray, shape (N, max_neighbours)
        Neighbour indices from octree
    neighbour_counts : cp.ndarray, shape (N,)
        Number of neighbours for each particle
    alpha : float
        Artificial viscosity alpha parameter
    beta : float
        Artificial viscosity beta parameter
        
    Returns
    -------
    accelerations : cp.ndarray, shape (N, 3)
        Hydrodynamic accelerations on GPU
    du_dt : cp.ndarray, shape (N,)
        Internal energy rates on GPU
    """
    n = positions.shape[0]
    max_neighbours = neighbour_lists.shape[1]
    
    accelerations = cp.zeros((n, 3), dtype=cp.float32)
    du_dt = cp.zeros(n, dtype=cp.float32)
    
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    _hydro_kernel(
        (grid_size,), (block_size,),
        (positions, velocities, masses, smoothing_lengths,
         densities, pressures, sound_speeds,
         neighbour_lists, neighbour_counts,
         accelerations, du_dt,
         n, max_neighbours, cp.float32(alpha), cp.float32(beta))
    )
    
    return accelerations, du_dt
