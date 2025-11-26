"""Test that black hole gravity is being computed in both CPU and GPU pipelines."""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tde_sph.sph import ParticleSystem
from tde_sph.gravity import RelativisticGravitySolver
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator
from tde_sph.metric import SchwarzschildMetric
from tde_sph.core import Simulation, SimulationConfig

print("="*70)
print("TEST: Black Hole Gravity in Physics Pipeline")
print("="*70)

# Create a simple test setup: single particle at r=10M
n_particles = 1
positions = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
velocities = np.array([[0.0, 0.3, 0.0]], dtype=np.float32)  # Some velocity for geodesic
masses = np.ones(n_particles, dtype=np.float32)
smoothing_lengths = np.ones(n_particles, dtype=np.float32) * 0.5
internal_energy = np.ones(n_particles, dtype=np.float32)

particles = ParticleSystem(
    n_particles=n_particles,
    positions=positions,
    velocities=velocities,
    masses=masses,
    smoothing_length=smoothing_lengths,
    internal_energy=internal_energy
)

# Setup GR mode with Schwarzschild metric
bh_mass = 1.0
metric = SchwarzschildMetric(mass=bh_mass)

# Create gravity solver
gravity_solver = RelativisticGravitySolver(G=1.0, bh_mass=bh_mass)

# Create other components
eos = IdealGas(gamma=5./3.)
integrator = LeapfrogIntegrator()

# Configure simulation in GR mode
config = SimulationConfig(
    mode="GR",
    metric_type="schwarzschild",
    bh_mass=bh_mass,
    t_end=0.1,
    dt_initial=0.01,
    snapshot_interval=0.1,
    verbose=True
)

# Create simulation
sim = Simulation(
    particles=particles,
    gravity_solver=gravity_solver,
    eos=eos,
    integrator=integrator,
    config=config,
    metric=metric
)

print("\nInitial state:")
print(f"  Position: {particles.positions[0]}")
print(f"  Velocity: {particles.velocities[0]}")
print(f"  Distance from BH: {np.linalg.norm(particles.positions[0]):.3f} M")

# Compute forces
print("\n" + "="*70)
print("Computing forces (CPU pipeline)...")
print("="*70)
forces = sim.compute_forces()

print(f"\nGravity acceleration: {forces['gravity'][0]}")
print(f"Hydro acceleration: {forces['hydro'][0]}")
print(f"Total acceleration: {forces['total'][0]}")

# Check that gravity is non-zero (BH gravity should be present)
gravity_mag = np.linalg.norm(forces['gravity'][0])
print(f"\nGravity magnitude: {gravity_mag:.6f}")

if gravity_mag > 1e-6:
    print("[OK] BLACK HOLE GRAVITY IS WORKING!")
    print(f"  Expected: BH gravity toward origin (negative x-direction)")
    print(f"  Got: {forces['gravity'][0]}")
    
    # Check direction - should be roughly toward BH at origin
    r_hat = particles.positions[0] / np.linalg.norm(particles.positions[0])
    a_hat = forces['gravity'][0] / gravity_mag
    dot_product = np.dot(r_hat, a_hat)
    
    print(f"  Direction check: dot(r_hat, a_hat) = {dot_product:.3f}")
    if dot_product < -0.5:  # Should be pointing toward BH (opposite to r)
        print("  ✓ Direction is correct (toward BH)")
    else:
        print(f"  ⚠ Direction might be wrong (expected < -0.5)")
else:
    print("✗ BLACK HOLE GRAVITY IS ZERO!")
    print("  This means BH gravity is NOT being computed.")

# Test with GPU if available
print("\n" + "="*70)
print("Checking GPU pipeline...")
print("="*70)

try:
    from tde_sph.gpu import HAS_CUDA
    if HAS_CUDA:
        import cupy as cp
        
        # Reset particles
        particles.positions = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        particles.velocities = np.array([[0.0, 0.3, 0.0]], dtype=np.float32)
        
        # Enable GPU
        sim.use_gpu = True
        from tde_sph.gpu import GPUManager
        sim.gpu_manager = GPUManager(sim.particles)
        
        print("GPU available, testing GPU pipeline...")
        forces_gpu = sim.compute_forces()
        
        print(f"\nGPU Gravity acceleration: {forces_gpu['gravity'][0]}")
        print(f"GPU Hydro acceleration: {forces_gpu['hydro'][0]}")
        print(f"GPU Total acceleration: {forces_gpu['total'][0]}")
        
        gravity_mag_gpu = np.linalg.norm(forces_gpu['gravity'][0])
        print(f"\nGPU Gravity magnitude: {gravity_mag_gpu:.6f}")
        
        if gravity_mag_gpu > 1e-6:
            print("✓ BLACK HOLE GRAVITY IS WORKING ON GPU!")
        else:
            print("✗ BLACK HOLE GRAVITY IS ZERO ON GPU!")
    else:
        print("GPU not available (CUDA not found)")
except ImportError as e:
    print(f"GPU not available: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
