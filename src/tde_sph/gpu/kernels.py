import numpy as np
import cupy as cp

CUDA_SOURCE = r'''
#define TPB 128
#define PI 3.14159265359f

__device__ float cubic_spline_kernel(float r, float h) {
    float q = r / h;
    float sigma = 1.0f / (PI * h * h * h);
    
    if (q >= 2.0f) {
        return 0.0f;
    } else if (q >= 1.0f) {
        float v = 2.0f - q;
        return sigma * 0.25f * v * v * v;
    } else {
        return sigma * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    }
}

__device__ float cubic_spline_gradient(float r, float h) {
    float q = r / h;
    float sigma = 1.0f / (PI * h * h * h);
    float factor = sigma / h;
    
    if (q >= 2.0f) {
        return 0.0f;
    } else if (q >= 1.0f) {
        float v = 2.0f - q;
        return factor * (-0.75f * v * v);
    } else {
        return factor * (-3.0f * q + 2.25f * q * q);
    }
}

extern "C" __global__
void compute_gravity_bruteforce(const float* pos, const float* mass, const float* h, float* acc, float G, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_pos[TPB][3];
    __shared__ float shared_mass[TPB];
    __shared__ float shared_h[TPB];
    
    float my_pos[3];
    float my_h = 0.0f;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    
    if (idx < N) {
        my_pos[0] = pos[idx * 3 + 0];
        my_pos[1] = pos[idx * 3 + 1];
        my_pos[2] = pos[idx * 3 + 2];
        my_h = h[idx];
    }
    
    for (int tile = 0; tile < (N + TPB - 1) / TPB; tile++) {
        int tile_idx = tile * TPB + threadIdx.x;
        
        if (tile_idx < N) {
            shared_pos[threadIdx.x][0] = pos[tile_idx * 3 + 0];
            shared_pos[threadIdx.x][1] = pos[tile_idx * 3 + 1];
            shared_pos[threadIdx.x][2] = pos[tile_idx * 3 + 2];
            shared_mass[threadIdx.x] = mass[tile_idx];
            shared_h[threadIdx.x] = h[tile_idx];
        } else {
            shared_mass[threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        if (idx < N) {
            #pragma unroll
            for (int j = 0; j < TPB; j++) {
                if (shared_mass[j] > 0.0f) {
                    double dx = (double)shared_pos[j][0] - (double)my_pos[0];
                    double dy = (double)shared_pos[j][1] - (double)my_pos[1];
                    double dz = (double)shared_pos[j][2] - (double)my_pos[2];
                    
                    double r2 = dx*dx + dy*dy + dz*dz;
                    
                    double epsilon = ((double)my_h + (double)shared_h[j]) * 0.5;
                    double epsilon2 = epsilon * epsilon;
                    
                    double r2_soft = r2 + epsilon2;
                    double inv_r3 = rsqrt(r2_soft * r2_soft * r2_soft);
                    
                    double f = (double)G * (double)shared_mass[j] * inv_r3;
                    ax += f * dx;
                    ay += f * dy;
                    az += f * dz;
                }
            }
        }
        
        __syncthreads();
    }
    
    if (idx < N) {
        acc[idx * 3 + 0] = (float)ax;
        acc[idx * 3 + 1] = (float)ay;
        acc[idx * 3 + 2] = (float)az;
    }
}

extern "C" __global__
void compute_density(const float* pos, const float* mass, const float* h, float* density, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_pos[TPB][3];
    __shared__ float shared_mass[TPB];
    
    float my_pos[3];
    float my_h = 0.0f;
    float rho = 0.0f;
    
    if (idx < N) {
        my_pos[0] = pos[idx * 3 + 0];
        my_pos[1] = pos[idx * 3 + 1];
        my_pos[2] = pos[idx * 3 + 2];
        my_h = h[idx];
        
        // Self contribution
        float sigma = 1.0f / (PI * my_h * my_h * my_h);
        rho += mass[idx] * sigma;
    }
    
    for (int tile = 0; tile < (N + TPB - 1) / TPB; tile++) {
        int tile_idx = tile * TPB + threadIdx.x;
        
        if (tile_idx < N) {
            shared_pos[threadIdx.x][0] = pos[tile_idx * 3 + 0];
            shared_pos[threadIdx.x][1] = pos[tile_idx * 3 + 1];
            shared_pos[threadIdx.x][2] = pos[tile_idx * 3 + 2];
            shared_mass[threadIdx.x] = mass[tile_idx];
        } else {
            shared_mass[threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        if (idx < N) {
            #pragma unroll
            for (int j = 0; j < TPB; j++) {
                if (shared_mass[j] > 0.0f) {
                    float dx = my_pos[0] - shared_pos[j][0];
                    float dy = my_pos[1] - shared_pos[j][1];
                    float dz = my_pos[2] - shared_pos[j][2];
                    float r2 = dx*dx + dy*dy + dz*dz;
                    
                    if (r2 > 0.0f && r2 < 4.0f * my_h * my_h) {
                        float r = sqrtf(r2);
                        float w = cubic_spline_kernel(r, my_h);
                        rho += shared_mass[j] * w;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    if (idx < N) {
        density[idx] = rho;
    }
}

extern "C" __global__
void compute_hydro(const float* pos, const float* vel, const float* mass, const float* h, 
                  const float* rho, const float* pressure, const float* cs,
                  float* acc_hydro, float* du_dt, float alpha, float beta, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_pos[TPB][3];
    __shared__ float shared_vel[TPB][3];
    __shared__ float shared_mass[TPB];
    __shared__ float shared_h[TPB];
    __shared__ float shared_rho[TPB];
    __shared__ float shared_p[TPB];
    __shared__ float shared_cs[TPB];
    
    float my_pos[3];
    float my_vel[3];
    float my_h, my_rho, my_p, my_cs;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    double dudt = 0.0;
    
    if (idx < N) {
        my_pos[0] = pos[idx * 3 + 0];
        my_pos[1] = pos[idx * 3 + 1];
        my_pos[2] = pos[idx * 3 + 2];
        my_vel[0] = vel[idx * 3 + 0];
        my_vel[1] = vel[idx * 3 + 1];
        my_vel[2] = vel[idx * 3 + 2];
        my_h = h[idx];
        my_rho = rho[idx];
        my_p = pressure[idx];
        my_cs = cs[idx];
    }
    
    double p_rho2_i = (double)my_p / ((double)my_rho * (double)my_rho);
    
    for (int tile = 0; tile < (N + TPB - 1) / TPB; tile++) {
        int tile_idx = tile * TPB + threadIdx.x;
        
        if (tile_idx < N) {
            shared_pos[threadIdx.x][0] = pos[tile_idx * 3 + 0];
            shared_pos[threadIdx.x][1] = pos[tile_idx * 3 + 1];
            shared_pos[threadIdx.x][2] = pos[tile_idx * 3 + 2];
            shared_vel[threadIdx.x][0] = vel[tile_idx * 3 + 0];
            shared_vel[threadIdx.x][1] = vel[tile_idx * 3 + 1];
            shared_vel[threadIdx.x][2] = vel[tile_idx * 3 + 2];
            shared_mass[threadIdx.x] = mass[tile_idx];
            shared_h[threadIdx.x] = h[tile_idx];
            shared_rho[threadIdx.x] = rho[tile_idx];
            shared_p[threadIdx.x] = pressure[tile_idx];
            shared_cs[threadIdx.x] = cs[tile_idx];
        } else {
            shared_mass[threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        if (idx < N) {
            #pragma unroll
            for (int j = 0; j < TPB; j++) {
                if (shared_mass[j] > 0.0f) {
                    double dx = (double)my_pos[0] - (double)shared_pos[j][0];
                    double dy = (double)my_pos[1] - (double)shared_pos[j][1];
                    double dz = (double)my_pos[2] - (double)shared_pos[j][2];
                    double r2 = dx*dx + dy*dy + dz*dz;
                    
                    double hj = shared_h[j];
                    // Use my_h to match CPU implementation (Gather formulation)
                    // double h_avg = 0.5 * ((double)my_h + hj);
                    double h_use = (double)my_h;
                    
                    if (r2 > 0.0 && r2 < 4.0 * h_use * h_use) {
                        double r = sqrt(r2);
                        double dwdr = (double)cubic_spline_gradient((float)r, (float)h_use);
                        
                        double rhoj = shared_rho[j];
                        double pj = shared_p[j];
                        double p_rho2_j = pj / (rhoj * rhoj);
                        
                        double vx_ij = (double)my_vel[0] - (double)shared_vel[j][0];
                        double vy_ij = (double)my_vel[1] - (double)shared_vel[j][1];
                        double vz_ij = (double)my_vel[2] - (double)shared_vel[j][2];
                        
                        double v_dot_r = vx_ij*dx + vy_ij*dy + vz_ij*dz;
                        
                        double visc = 0.0;
                        if (v_dot_r < 0.0) {
                            // Viscosity usually uses h_avg or h_ij
                            double h_avg = 0.5 * ((double)my_h + hj);
                            double mu = h_avg * v_dot_r / (r2 + 0.01 * h_avg * h_avg);
                            double cj = shared_cs[j];
                            double c_avg = 0.5 * ((double)my_cs + cj);
                            double rho_avg = 0.5 * ((double)my_rho + rhoj);
                            
                            visc = (-(double)alpha * c_avg * mu + (double)beta * mu * mu) / rho_avg;
                        }
                        
                        double term = p_rho2_i + p_rho2_j + visc;
                        double fac = -(double)shared_mass[j] * term * dwdr / r;
                        
                        ax += fac * dx;
                        ay += fac * dy;
                        az += fac * dz;
                        
                        dudt += 0.5 * (double)shared_mass[j] * term * v_dot_r * dwdr / r;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    if (idx < N) {
        acc_hydro[idx * 3 + 0] = (float)ax;
        acc_hydro[idx * 3 + 1] = (float)ay;
        acc_hydro[idx * 3 + 2] = (float)az;
        du_dt[idx] = (float)dudt;
    }
}

extern "C" __global__
void count_neighbours(const float* pos, const float* h, int* counts, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_pos[TPB][3];
    __shared__ float shared_h[TPB];
    
    float my_pos[3];
    float my_h = 0.0f;
    int count = 0;
    
    if (idx < N) {
        my_pos[0] = pos[idx * 3 + 0];
        my_pos[1] = pos[idx * 3 + 1];
        my_pos[2] = pos[idx * 3 + 2];
        my_h = h[idx];
    }
    
    for (int tile = 0; tile < (N + TPB - 1) / TPB; tile++) {
        int tile_idx = tile * TPB + threadIdx.x;
        
        if (tile_idx < N) {
            shared_pos[threadIdx.x][0] = pos[tile_idx * 3 + 0];
            shared_pos[threadIdx.x][1] = pos[tile_idx * 3 + 1];
            shared_pos[threadIdx.x][2] = pos[tile_idx * 3 + 2];
            shared_h[threadIdx.x] = h[tile_idx];
        }
        
        __syncthreads();
        
        if (idx < N) {
            #pragma unroll
            for (int j = 0; j < TPB; j++) {
                if (tile * TPB + j < N) {
                    // Skip self
                    if (tile * TPB + j == idx) continue;

                    float dx = my_pos[0] - shared_pos[j][0];
                    float dy = my_pos[1] - shared_pos[j][1];
                    float dz = my_pos[2] - shared_pos[j][2];
                    float r2 = dx*dx + dy*dy + dz*dz;
                    
                    // Symmetric neighbour criterion: r < 2 * max(h_i, h_j)
                    float hj = shared_h[j];
                    float h_max = (my_h > hj) ? my_h : hj;
                    float dist_max = 2.0f * h_max;
                    
                    if (r2 < dist_max * dist_max) {
                        count++;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    if (idx < N) {
        counts[idx] = count;
    }
}
'''

# Compile module
module = cp.RawModule(code=CUDA_SOURCE)
compute_gravity_bruteforce_kernel = module.get_function('compute_gravity_bruteforce')
compute_density_kernel = module.get_function('compute_density')
compute_hydro_kernel = module.get_function('compute_hydro')
count_neighbours_kernel = module.get_function('count_neighbours')


def _estimate_gpu_smoothing_bounds(pos, h, min_scale: float = 1e-2, max_scale: float = 32.0):
    """Compute adaptive smoothing-length bounds directly on the GPU arrays."""
    valid_mask = cp.logical_and(cp.isfinite(h), h > 0.0)
    if not cp.any(valid_mask):
        return 1e-6, 1.0

    h_valid = h[valid_mask]
    base_min = float(cp.min(h_valid).get())
    base_max = float(cp.max(h_valid).get())

    centroid = cp.mean(pos, axis=0)
    offsets = pos - centroid
    extent = float(cp.max(cp.linalg.norm(offsets, axis=1)).get())
    if not np.isfinite(extent):
        extent = 0.0

    h_min_bound = max(base_min * min_scale, 1e-2)
    h_max_candidates = [base_max * max_scale, h_min_bound * 10.0]
    if extent > 0.0:
        h_max_candidates.append(extent)
    h_max_bound = max(h_max_candidates)
    if h_max_bound <= h_min_bound:
        h_max_bound = h_min_bound * 10.0

    return float(h_min_bound), float(h_max_bound)

# Wrappers
def compute_gravity_bruteforce_gpu(pos, mass, h, acc, G):
    N = pos.shape[0]
    block = (128,)
    grid = ((N + 127) // 128,)
    compute_gravity_bruteforce_kernel(grid, block, (pos, mass, h, acc, cp.float32(G), cp.int32(N)))

def compute_density_gpu(pos, mass, h, density):
    N = pos.shape[0]
    block = (128,)
    grid = ((N + 127) // 128,)
    compute_density_kernel(grid, block, (pos, mass, h, density, cp.int32(N)))

def compute_hydro_gpu(pos, vel, mass, h, rho, pressure, cs, acc_hydro, du_dt, alpha, beta):
    N = pos.shape[0]
    block = (128,)
    grid = ((N + 127) // 128,)
    compute_hydro_kernel(grid, block, (
        pos, vel, mass, h, rho, pressure, cs, 
        acc_hydro, du_dt, cp.float32(alpha), cp.float32(beta), cp.int32(N)
    ))

def update_smoothing_lengths_gpu(pos, h, target_neighbours=50, tolerance=0.05, max_iter=10):
    """
    Host-side wrapper for adaptive smoothing length update on GPU.
    Mirrors the CPU implementation by enforcing dynamic bounds derived
    from the instantaneous h-distribution and spatial extent.
    """
    N = pos.shape[0]
    block = (128,)
    grid = ((N + 127) // 128,)
    
    counts = cp.zeros(N, dtype=cp.int32)
    h_min_bound, h_max_bound = _estimate_gpu_smoothing_bounds(pos, h)
    
    for i in range(max_iter):
        count_neighbours_kernel(grid, block, (pos, h, counts, cp.int32(N)))
        
        n_actual = cp.maximum(counts, 1.0)
        ratio = n_actual / target_neighbours
        # Standard update (no damping to match CPU)
        factor = ratio ** (-1.0/3.0)
        
        h *= factor
        cp.clip(h, h_min_bound, h_max_bound, out=h)
        
        error = cp.abs(n_actual - target_neighbours) / target_neighbours
        max_error = cp.max(error)
        
        if max_error < tolerance:
            break
            
    return h

