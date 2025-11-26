"""
Test TreeSPH GPU implementation with multiple particles.

Verifies that the octree-based neighbour search and TreeSPH
kernels work correctly for density and hydro forces.
"""
import numpy as np
from tde_sph.core.simulation import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import RelativisticGravitySolver
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator

print("="*70)
print("TEST: TreeSPH GPU Implementation")
print("="*70)

# Configuration
M_bh = 1.0
G = 1.0
n_particles = 100

# Create particle cloud at r~10M
np.random.seed(42)
r_base = 10.0
positions = np.random.randn(n_particles, 3).astype(np.float32) * 0.5 + r_base
velocities = np.random.randn(n_particles, 3).astype(np.float32) * 0.1
masses = np.ones(n_particles, dtype=np.float32) * 1e-4
smoothing_lengths = np.ones(n_particles, dtype=np.float32) * 0.5

# Create particle system
particles = ParticleSystem(n_particles=n_particles, smoothing_length=smoothing_lengths)
particles.positions = positions
particles.velocities = velocities
particles.masses = masses
particles.smoothing_lengths = smoothing_lengths

# Initialize thermodynamics
particles.density = np.ones(n_particles, dtype=np.float32)
particles.internal_energy = np.ones(n_particles, dtype=np.float32) * 1.0
particles.pressure = np.ones(n_particles, dtype=np.float32)
particles.sound_speed = np.ones(n_particles, dtype=np.float32)
particles.temperature = np.ones(n_particles, dtype=np.float32)

# Create components
gravity_solver = RelativisticGravitySolver(G=G, bh_mass=M_bh)
eos = IdealGas(gamma=5.0/3.0)
integrator = LeapfrogIntegrator()

# Create simulation in NEWTONIAN mode
config = SimulationConfig(
    mode="Newtonian",
    bh_mass=M_bh,
    t_end=0.01,
    dt_initial=0.001,
    verbose=True
)

sim = Simulation(
    particles=particles,
    gravity_solver=gravity_solver,
    eos=eos,
    integrator=integrator,
    config=config,
    metric=None
)

print(f"\nInitial state:")
print(f"  N particles: {n_particles}")
print(f"  Position range: [{np.min(np.linalg.norm(positions, axis=1)):.2f}, {np.max(np.linalg.norm(positions, axis=1)):.2f}] M")

# ====================================================================
# Test CPU pipeline
# ====================================================================
print("\n" + "="*70)
print("Testing CPU pipeline...")
print("="*70)

import time
sim.use_gpu = False

t0 = time.time()
forces_cpu = sim.compute_forces()
t_cpu = time.time() - t0

print(f"\nCPU time: {t_cpu*1000:.2f} ms")
print(f"Gravity acceleration range: [{np.min(np.linalg.norm(forces_cpu['gravity'], axis=1)):.6f}, {np.max(np.linalg.norm(forces_cpu['gravity'], axis=1)):.6f}]")
print(f"Hydro acceleration range: [{np.min(np.linalg.norm(forces_cpu['hydro'], axis=1)):.6f}, {np.max(np.linalg.norm(forces_cpu['hydro'], axis=1)):.6f}]")

# ====================================================================
# Test GPU pipeline with TreeSPH
# ====================================================================
print("\n" + "="*70)
print("Testing GPU pipeline with TreeSPH...")
print("="*70)

try:
    from tde_sph.gpu import HAS_CUDA
    if HAS_CUDA:
        # Reset particle state
        particles.positions = positions.copy()
        particles.velocities = velocities.copy()
        particles.density = np.ones(n_particles, dtype=np.float32)
        particles.internal_energy = np.ones(n_particles, dtype=np.float32) * 1.0
        particles.pressure = np.ones(n_particles, dtype=np.float32)
        particles.sound_speed = np.ones(n_particles, dtype=np.float32)
        particles.temperature = np.ones(n_particles, dtype=np.float32)
        
        # Enable GPU
        sim.use_gpu = True
        from tde_sph.gpu import GPUManager
        sim.gpu_manager = GPUManager(sim.particles)
        
        t0 = time.time()
        forces_gpu = sim.compute_forces()
        t_gpu = time.time() - t0
        
        print(f"\nGPU time: {t_gpu*1000:.2f} ms")
        print(f"Speedup: {t_cpu/t_gpu:.2f}x")
        
        print(f"\nGPU Gravity acceleration range: [{np.min(np.linalg.norm(forces_gpu['gravity'], axis=1)):.6f}, {np.max(np.linalg.norm(forces_gpu['gravity'], axis=1)):.6f}]")
        print(f"GPU Hydro acceleration range: [{np.min(np.linalg.norm(forces_gpu['hydro'], axis=1)):.6f}, {np.max(np.linalg.norm(forces_gpu['hydro'], axis=1)):.6f}]")
        
        # Compare CPU vs GPU
        grav_diff = np.linalg.norm(forces_gpu['gravity'] - forces_cpu['gravity']) / np.linalg.norm(forces_cpu['gravity'])
        hydro_diff = np.linalg.norm(forces_gpu['hydro'] - forces_cpu['hydro']) / (np.linalg.norm(forces_cpu['hydro']) + 1e-10)
        
        print(f"\nCPU vs GPU relative difference:")
        print(f"  Gravity: {grav_diff:.2%}")
        print(f"  Hydro: {hydro_diff:.2%}")
        
        # Check GPU memory usage
        mem_stats = sim.gpu_manager.get_gpu_memory_usage()
        print(f"\nGPU Memory Usage:")
        print(f"  Particle data: {mem_stats['particle_data_mb']:.2f} MB")
        print(f"  Octree cached: {mem_stats['octree_cached']}")
        neighbour_cache_key = 'neighbours_cached' if 'neighbours_cached' in mem_stats else 'neighbors_cached'
        print(f"  Neighbours cached: {mem_stats[neighbour_cache_key]}")
        if 'octree_mb' in mem_stats:
            print(f"  Octree memory: {mem_stats['octree_mb']:.2f} MB")
        
        # Validation
        if grav_diff < 0.01 and hydro_diff < 0.1:
            print("\n[OK] TreeSPH GPU implementation is working correctly!")
            success = True
        else:
            print(f"\n[WARNING] Large differences between CPU and GPU")
            success = False
    else:
        print("GPU not available")
        success = None
except Exception as e:
    print(f"GPU test failed: {e}")
    import traceback
    traceback.print_exc()
    success = False

# ====================================================================
# Summary
# ====================================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
if success:
    print("[OK] TreeSPH GPU implementation validated!")
elif success is None:
    print("GPU not available, CPU test passed")
else:
    print("[ERROR] Test failed")
