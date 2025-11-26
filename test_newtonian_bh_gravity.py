"""
Test Newtonian BH gravity in both CPU and GPU pipelines.

Verifies that:
1. CPU pipeline computes Newtonian BH gravity when metric=None
2. GPU pipeline computes Newtonian BH gravity when metric=None
3. Both pipelines produce identical results
4. Acceleration magnitude and direction are correct
"""
import numpy as np
from tde_sph.core.simulation import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import RelativisticGravitySolver
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator

print("="*70)
print("TEST: Newtonian BH Gravity in Physics Pipeline")
print("="*70)

# Configuration
M_bh = 1.0
G = 1.0
n_particles = 1

# Create single test particle at r=10M
positions = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # At rest
masses = np.array([1e-6], dtype=np.float32)  # Low mass to avoid self-gravity
smoothing_lengths = np.array([0.1], dtype=np.float32)

# Create particle system
particles = ParticleSystem(
    n_particles=n_particles,
    smoothing_length=smoothing_lengths  # Pass array, not scalar
)
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

# Create simulation in NEWTONIAN mode (no metric)
config = SimulationConfig(
    mode="Newtonian",
    metric_type="schwarzschild",  # Not used in Newtonian mode
    bh_mass=M_bh,
    t_end=0.01,
    dt_initial=0.001,
    verbose=False
)

sim = Simulation(
    particles=particles,
    gravity_solver=gravity_solver,
    eos=eos,
    integrator=integrator,
    config=config,
    metric=None  # Newtonian mode - no metric!
)

print(f"\nInitial state:")
print(f"  Position: {particles.positions[0]}")
print(f"  Velocity: {particles.velocities[0]} (AT REST)")
print(f"  Distance from BH: {np.linalg.norm(particles.positions[0]):.3f} M")
print(f"  Mode: {config.mode}")
print(f"  Metric: {sim.metric}")

# Expected Newtonian acceleration
r = np.linalg.norm(positions[0])
a_expected_mag = G * M_bh / r**2  # Magnitude (positive)
print(f"\nExpected Newtonian acceleration:")
print(f"  |a| = G*M/r² = {G}*{M_bh}/{r}² = {a_expected_mag:.6f}")

# ====================================================================
# Test CPU pipeline
# ====================================================================
print("\n" + "="*70)
print("Testing CPU pipeline (Newtonian mode)...")
print("="*70)

# Disable GPU for CPU test
sim.use_gpu = False

forces = sim.compute_forces()

print(f"\nGravity acceleration: {forces['gravity'][0]}")
print(f"Hydro acceleration: {forces['hydro'][0]}")
print(f"Total acceleration: {forces['total'][0]}")

# Check magnitude
gravity_mag = np.linalg.norm(forces['gravity'][0])
print(f"\nGravity magnitude: {gravity_mag:.6f}")
print(f"Expected magnitude: {a_expected_mag:.6f}")
print(f"Relative error: {abs(gravity_mag - a_expected_mag) / a_expected_mag:.2%}")

# Check direction (should point toward BH at origin)
r_hat = particles.positions[0] / np.linalg.norm(particles.positions[0])
a_hat = forces['gravity'][0] / gravity_mag
dot_product = np.dot(r_hat, a_hat)

print(f"\nDirection check:")
print(f"  r_hat (radial outward): {r_hat}")
print(f"  a_hat (accel direction): {a_hat}")
print(f"  dot(r_hat, a_hat) = {dot_product:.3f}")

if dot_product < -0.9:
    print("  [OK] Acceleration points toward BH (attractive)")
    cpu_success = True
else:
    print("  [ERROR] Acceleration direction is wrong!")
    cpu_success = False

if abs(gravity_mag - a_expected_mag) / a_expected_mag < 0.01:
    print("  [OK] Magnitude matches Newtonian expectation")
else:
    print("  [ERROR] Magnitude does not match!")
    cpu_success = False

# ====================================================================
# Test GPU pipeline
# ====================================================================
print("\n" + "="*70)
print("Testing GPU pipeline (Newtonian mode)...")
print("="*70)

try:
    from tde_sph.gpu import HAS_CUDA
    if HAS_CUDA:
        # Reset particle state
        particles.positions = positions.copy()
        particles.velocities = velocities.copy()
        
        # Enable GPU
        sim.use_gpu = True
        from tde_sph.gpu import GPUManager
        sim.gpu_manager = GPUManager(sim.particles)
        
        print("GPU available, testing GPU pipeline...")
        forces_gpu = sim.compute_forces()
        
        print(f"\nGPU Gravity acceleration: {forces_gpu['gravity'][0]}")
        print(f"GPU Hydro acceleration: {forces_gpu['hydro'][0]}")
        print(f"GPU Total acceleration: {forces_gpu['total'][0]}")
        
        # Check magnitude
        gpu_gravity_mag = np.linalg.norm(forces_gpu['gravity'][0])
        print(f"\nGPU Gravity magnitude: {gpu_gravity_mag:.6f}")
        print(f"Expected magnitude: {a_expected_mag:.6f}")
        print(f"Relative error: {abs(gpu_gravity_mag - a_expected_mag) / a_expected_mag:.2%}")
        
        # Check direction
        a_hat_gpu = forces_gpu['gravity'][0] / gpu_gravity_mag
        dot_product_gpu = np.dot(r_hat, a_hat_gpu)
        
        print(f"\nDirection check:")
        print(f"  dot(r_hat, a_hat) = {dot_product_gpu:.3f}")
        
        if dot_product_gpu < -0.9:
            print("  [OK] GPU acceleration points toward BH")
            gpu_success = True
        else:
            print("  [ERROR] GPU acceleration direction is wrong!")
            gpu_success = False
        
        if abs(gpu_gravity_mag - a_expected_mag) / a_expected_mag < 0.01:
            print("  [OK] GPU magnitude matches Newtonian expectation")
        else:
            print("  [ERROR] GPU magnitude does not match!")
            gpu_success = False
        
        # Compare CPU vs GPU
        diff = np.linalg.norm(forces_gpu['gravity'][0] - forces['gravity'][0])
        print(f"\nCPU vs GPU difference: {diff:.2e}")
        if diff < 1e-5:
            print("  [OK] CPU and GPU results match!")
        else:
            print(f"  [WARNING] CPU and GPU differ by {diff:.2e}")
    else:
        print("GPU not available, skipping GPU test")
        gpu_success = None
except ImportError:
    print("GPU module not available, skipping GPU test")
    gpu_success = None

# ====================================================================
# Summary
# ====================================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"CPU Pipeline: {'PASS' if cpu_success else 'FAIL'}")
if gpu_success is not None:
    print(f"GPU Pipeline: {'PASS' if gpu_success else 'FAIL'}")
else:
    print("GPU Pipeline: SKIPPED")

if cpu_success and (gpu_success is None or gpu_success):
    print("\n[OK] NEWTONIAN BH GRAVITY IS WORKING!")
else:
    print("\n[ERROR] NEWTONIAN BH GRAVITY HAS ISSUES!")
