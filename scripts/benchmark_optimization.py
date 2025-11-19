
import time
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from tde_sph.gravity.newtonian import NewtonianGravity
from tde_sph.sph.neighbours_cpu import find_neighbours_bruteforce, compute_density_summation
from tde_sph.sph.hydro_forces import compute_hydro_acceleration
from tde_sph.sph.kernels import CubicSplineKernel

def benchmark():
    N = 2000
    print(f"Benchmarking with N={N} particles...")
    
    positions = np.random.randn(N, 3).astype(np.float32)
    velocities = np.zeros((N, 3), dtype=np.float32)
    masses = np.ones(N, dtype=np.float32)
    smoothing_lengths = np.ones(N, dtype=np.float32) * 0.5
    densities = np.ones(N, dtype=np.float32)
    pressures = np.ones(N, dtype=np.float32)
    sound_speeds = np.ones(N, dtype=np.float32)

    gravity = NewtonianGravity()
    kernel = CubicSplineKernel()

    # Warmup
    print("Warming up JIT...")
    _ = gravity.compute_acceleration(positions[:100], masses[:100], smoothing_lengths[:100])
    neighbours_warmup, _ = find_neighbours_bruteforce(positions[:100], smoothing_lengths[:100])
    _ = compute_hydro_acceleration(
        positions[:100], velocities[:100], masses[:100], densities[:100],
        pressures[:100], sound_speeds[:100], smoothing_lengths[:100],
        neighbours_warmup, kernel.kernel_gradient
    )
    _ = compute_density_summation(positions[:100], masses[:100], smoothing_lengths[:100], neighbours_warmup, kernel.kernel)

    # Benchmark Gravity
    print("Benchmarking Gravity...")
    start = time.time()
    accel = gravity.compute_acceleration(positions, masses, smoothing_lengths)
    end = time.time()
    print(f"Gravity (Newtonian) time: {end - start:.4f} s")

    # Benchmark Neighbours
    print("Benchmarking Neighbours...")
    start = time.time()
    neighbours, _ = find_neighbours_bruteforce(positions, smoothing_lengths)
    end = time.time()
    print(f"Neighbour search time: {end - start:.4f} s")
    
    # Benchmark Density
    print("Benchmarking Density...")
    start = time.time()
    rho = compute_density_summation(positions, masses, smoothing_lengths, neighbours, kernel.kernel)
    end = time.time()
    print(f"Density summation time: {end - start:.4f} s")
    
    # Benchmark Hydro
    print("Benchmarking Hydro...")
    start = time.time()
    hydro_accel, du_dt = compute_hydro_acceleration(
        positions, velocities, masses, densities,
        pressures, sound_speeds, smoothing_lengths,
        neighbours, kernel.kernel_gradient
    )
    end = time.time()
    print(f"Hydro forces time: {end - start:.4f} s")

    try:
        import numba
        print(f"Numba is available: {numba.__version__}")
    except ImportError:
        print("Numba is NOT available")

if __name__ == "__main__":
    benchmark()
