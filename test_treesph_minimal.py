"""
Minimal test to debug TreeSPH hydro kernel directly.
"""
import numpy as np
from tde_sph.core.simulation import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import RelativisticGravitySolver
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator

# Simple 2-particle test
n = 2
positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)  # 1M separation
velocities = np.zeros((n, 3), dtype=np.float32)
masses = np.ones(n, dtype=np.float32) * 1e-4
smoothing_lengths = np.ones(n, dtype=np.float32) * 0.5  # h=0.5, so r < 2h

# Create particles
particles = ParticleSystem(n_particles=n, smoothing_length=smoothing_lengths)
particles.positions = positions
particles.velocities = velocities
particles.masses = masses

# Initialize thermodynamics with actual values
particles.density = np.array([1.0, 1.0], dtype=np.float32)
particles.internal_energy = np.ones(n, dtype=np.float32) * 1.0
particles.pressure = np.array([1.0, 1.0], dtype=np.float32)
particles.sound_speed = np.array([1.0, 1.0], dtype=np.float32)
particles.temperature = np.ones(n, dtype=np.float32)

# Create components
gravity_solver = RelativisticGravitySolver(G=1.0, bh_mass=1.0)
eos = IdealGas(gamma=5.0/3.0)
integrator = LeapfrogIntegrator()

config = SimulationConfig(
    mode="Newtonian",
    bh_mass=1.0,
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
    metric=None
)

print("2-Particle Test: TreeSPH Hydro")
print("="*70)
print(f"Particle 0: pos={positions[0]}, P={particles.pressure[0]:.6f}")
print(f"Particle 1: pos={positions[1]}, P={particles.pressure[1]:.6f}")
print(f"Separation: {np.linalg.norm(positions[1] - positions[0]):.6f}")
print(f"Smoothing length: {smoothing_lengths[0]:.6f}")
print(f"Support radius: 2h = {2*smoothing_lengths[0]:.6f}")

# Test CPU
print("\n" + "="*70)
print("CPU PIPELINE")
print("="*70)
sim.use_gpu = False
forces_cpu = sim.compute_forces()

print(f"Particle 0 hydro accel: {forces_cpu['hydro'][0]}")
print(f"Particle 1 hydro accel: {forces_cpu['hydro'][1]}")
print(f"Magnitudes: {np.linalg.norm(forces_cpu['hydro'], axis=1)}")

# Test GPU
try:
    from tde_sph.gpu import HAS_CUDA
    if HAS_CUDA:
        print("\n" + "="*70)
        print("GPU PIPELINE")
        print("="*70)
        
        # Reset state
        particles.positions = positions.copy()
        particles.velocities = velocities.copy()
        particles.density = np.array([1.0, 1.0], dtype=np.float32)
        particles.pressure = np.array([1.0, 1.0], dtype=np.float32)
        particles.sound_speed = np.array([1.0, 1.0], dtype=np.float32)
        
        sim.use_gpu = True
        from tde_sph.gpu import GPUManager
        sim.gpu_manager = GPUManager(sim.particles)
        
        forces_gpu = sim.compute_forces()
        
        print(f"Particle 0 hydro accel: {forces_gpu['hydro'][0]}")
        print(f"Particle 1 hydro accel: {forces_gpu['hydro'][1]}")
        print(f"Magnitudes: {np.linalg.norm(forces_gpu['hydro'], axis=1)}")
        
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        diff = np.linalg.norm(forces_gpu['hydro'] - forces_cpu['hydro']) / (np.linalg.norm(forces_cpu['hydro']) + 1e-10)
        print(f"Relative difference: {diff:.2%}")
        
        if diff < 0.1:
            print("[OK] CPU and GPU match!")
        else:
            print("[WARNING] Large difference!")
    else:
        print("\nGPU not available")
except Exception as e:
    print(f"\nGPU test failed: {e}")
    import traceback
    traceback.print_exc()
