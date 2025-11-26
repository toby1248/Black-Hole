import time
import numpy as np
import cupy as cp
import argparse

try:
    import nvmath
    HAS_NVMATH = True
except ImportError:
    HAS_NVMATH = False

def run_experiment(N=16777216, dtype_compute=cp.float32, dtype_accum=cp.float32, use_nvmath=False):
    print(f"Initializing {N} particles...")
    
    # Generate random data in FP32
    pos_host = np.random.rand(N, 3).astype(np.float32)
    mass_host = np.random.rand(N).astype(np.float32)
    
    # Transfer to GPU
    pos = cp.asarray(pos_host)
    mass = cp.asarray(mass_host)
    
    print(f"Running force computation with compute={dtype_compute}, accum={dtype_accum}, nvmath={use_nvmath}...")
    
    # Warmup
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()
    
    # 1. Compute squared norms |r|^2
    # Shape (N, 1)
    r2_vec = cp.sum(pos**2, axis=1, keepdims=True)
    
    # 2. Compute distance matrix D^2 = |r_i|^2 + |r_j|^2 - 2 r_i . r_j
    # We use matrix multiplication for the dot product
    # P (N, 3) x P^T (3, N) -> (N, N)
    
    # Cast to compute type (e.g. BF16)
    pos_compute = pos.astype(dtype_compute)
    
    # Matrix multiply: -2 * pos * pos.T
    # This is the heavy lifting where Tensor Cores should kick in for BF16/FP16
    if use_nvmath and HAS_NVMATH:
        # nvmath.linalg.advanced.matmul(a, b)
        # Note: nvmath might expect specific layout or types
        # For now, let's assume it handles cupy arrays
        dot_product = nvmath.linalg.advanced.matmul(pos_compute, pos_compute.T)
    else:
        dot_product = cp.matmul(pos_compute, pos_compute.T)
    
    # Reconstruct distance squared in accumulator precision (FP32)
    # D^2 = r2_vec + r2_vec.T - 2 * dot_product
    # Note: dot_product is in compute_dtype, we cast back to accum_dtype
    dist_sq = r2_vec.astype(dtype_accum) + r2_vec.T.astype(dtype_accum) - 2.0 * dot_product.astype(dtype_accum)
    
    # Softening
    epsilon = 1e-2
    dist_sq += epsilon**2
    
    # Clip to avoid negative values from precision errors
    dist_sq = cp.maximum(dist_sq, 1e-10)
    
    # Inverse cube distance
    # inv_dist_3 = (dist_sq)**(-1.5)
    # Use rsqrt for speed
    inv_dist = 1.0 / cp.sqrt(dist_sq)
    inv_dist_3 = inv_dist * inv_dist * inv_dist
    
    # Force calculation
    # F_i = Sum_j G * m_j * (r_j - r_i) / |r_ij|^3
    # F_i = G * [ Sum_j (m_j * inv_dist_3_ij * r_j) - r_i * Sum_j (m_j * inv_dist_3_ij) ]
    
    # Term 1: M_j = m_j * inv_dist_3_ij (N, N)
    M_j = mass.astype(dtype_accum)[None, :] * inv_dist_3
    
    # Sum_j M_j * r_j
    # (N, N) * (N, 3) -> (N, 3)
    # This is another matrix multiply if we view M_j as weights
    # But M_j is N*N, r_j is N*3. 
    # Actually, we can't easily use matmul here because M_j is element-wise derived.
    # But we can do: F_term1 = M_j @ pos
    
    # Cast M_j to compute type for second matmul? Or keep FP32?
    # If we want tensor cores, we need BF16.
    M_j_compute = M_j.astype(dtype_compute)
    
    if use_nvmath and HAS_NVMATH:
        term1 = nvmath.linalg.advanced.matmul(M_j_compute, pos_compute).astype(dtype_accum)
    else:
        term1 = cp.matmul(M_j_compute, pos_compute).astype(dtype_accum)
    
    # Term 2: r_i * Sum_j M_j
    sum_M_j = cp.sum(M_j, axis=1, keepdims=True) # (N, 1)
    term2 = pos.astype(dtype_accum) * sum_M_j
    
    acc = term1 - term2
    
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    elapsed = end_time - start_time
    flops = 2 * N * N * 3 # Rough estimate for the matmuls
    tflops = (flops / elapsed) / 1e12
    
    print(f"Time: {elapsed:.4f} s")
    print(f"Estimated TFLOPS (Matrix ops): {tflops:.2f}")
    
    return elapsed, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=16777216) # Start smaller than 65k for safety
    args = parser.parse_args()
    
    N = args.N
    
    print("=== Baseline FP32 ===")
    try:
        t32, acc32 = run_experiment(N, cp.float32, cp.float32)
    except Exception as e:
        print(f"FP32 failed: {e}")
        t32 = None

    print("\n=== Experimental BF16 (Accum FP32) ===")
    try:
        # Check for bfloat16 support
        if hasattr(cp, 'bfloat16'):
            dtype = cp.bfloat16
            print("Using cupy.bfloat16 (real BF16)...")
        else:
            dtype = cp.float16
            print("Using float16 (simulating BF16 behavior for tensor cores)...")
            print("WARNING: float16 has limited range (max ~65504). Gravity forces may overflow!")
        
        t16, acc16 = run_experiment(N, dtype, cp.float32, use_nvmath=HAS_NVMATH)
        
        if t32 is not None:
            print(f"\nSpeedup: {t32/t16:.2f}x")
            
            # Compare accuracy
            # Relative error
            diff = cp.linalg.norm(acc32 - acc16)
            norm = cp.linalg.norm(acc32)
            rel_err = diff / norm
            print(f"Relative Error: {rel_err:.2e}")
            
    except Exception as e:
        print(f"BF16/FP16 failed: {e}")

if __name__ == "__main__":
    main()
