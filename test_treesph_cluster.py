"""
Test TreeSPH with particles guaranteed to have neighbours.
"""
import numpy as np
from tde_sph.core.simulation import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import RelativisticGravitySolver
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator

# Create a small cluster of particles
n = 10
# Place particles in a small cube at origin
positions = np.random.randn(n, 3).astype(np.float32) * 0.3 + 10.0  # Tight cluster at r=10
velocities = np.zeros((n, 3), dtype=np.float32)
masses = np.ones(n, dtype=np.float32) * 1e-4
smoothing_lengths = np.ones(n, dtype=np.float32) * 1.0  # Large h to ensure neighbours

# Create particle system
particles = ParticleSystem(n_particles=n, smoothing_length=smoothing_lengths)
particles.positions = positions
particles.velocities = velocities
particles.masses = masses

# Initialize thermodynamics
particles.density = np.ones(n, dtype=np.float32)
particles.internal_energy = np.ones(n, dtype=np.float32) * 1.0
particles.pressure = np.ones(n, dtype=np.float32) * 1.0
particles.sound_speed = np.ones(n, dtype=np.float32) * 1.0
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

print("TreeSPH Test with Dense Particle Cluster")
print("="*70)
print(f"N particles: {n}")
print(f"Position range: [{np.min(np.linalg.norm(positions, axis=1)):.2f}, {np.max(np.linalg.norm(positions, axis=1)):.2f}]")
print(f"Smoothing length: {smoothing_lengths[0]:.2f}")

# CPU test
print("\n" + "="*70)
print("CPU")
print("="*70)
sim.use_gpu = False
forces_cpu = sim.compute_forces()
print(f"Hydro magnitude range: [{np.min(np.linalg.norm(forces_cpu['hydro'], axis=1)):.6f}, {np.max(np.linalg.norm(forces_cpu['hydro'], axis=1)):.6f}]")

# GPU test
try:
    from tde_sph.gpu import HAS_CUDA
    if HAS_CUDA:
        print("\n" + "="*70)
        print("GPU")
        print("="*70)
        
        # Reset
        particles.positions = positions.copy()
        particles.velocities = velocities.copy()
        particles.density = np.ones(n, dtype=np.float32)
        particles.internal_energy = np.ones(n, dtype=np.float32) * 1.0
        particles.pressure = np.ones(n, dtype=np.float32) * 1.0
        particles.sound_speed = np.ones(n, dtype=np.float32) * 1.0
        
        sim.use_gpu = True
        from tde_sph.gpu import GPUManager
        sim.gpu_manager = GPUManager(sim.particles)
        
        forces_gpu = sim.compute_forces()
        print(f"Hydro magnitude range: [{np.min(np.linalg.norm(forces_gpu['hydro'], axis=1)):.6f}, {np.max(np.linalg.norm(forces_gpu['hydro'], axis=1)):.6f}]")
        
        # Compare
        hydro_diff = np.linalg.norm(forces_gpu['hydro'] - forces_cpu['hydro']) / (np.linalg.norm(forces_cpu['hydro']) + 1e-10)
        print(f"\nRelative difference: {hydro_diff:.2%}")
        
        if hydro_diff < 0.1:
            print("[OK] TreeSPH working!")
        else:
            print("[WARNING] Large difference")
    else:
        print("\nGPU not available")
except Exception as e:
    print(f"\nGPU test failed: {e}")
    import traceback
    traceback.print_exc()
