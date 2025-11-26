"""
Demonstrate Newtonian vs GR black hole gravity at large radius.

At large radii (r >> 6M), GR should reduce to Newtonian gravity.
This test verifies that both modes produce similar results far from the BH.
"""
import numpy as np
from tde_sph.core.simulation import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import RelativisticGravitySolver
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator
from tde_sph.metric import SchwarzschildMetric

print("="*70)
print("Newtonian vs GR Comparison at Large Radius")
print("="*70)

# Configuration
M_bh = 1.0
G = 1.0
r_test = 100.0  # Large radius (100M >> 6M)

# Create test particle
positions = np.array([[r_test, 0.0, 0.0]], dtype=np.float32)
velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
masses = np.array([1e-6], dtype=np.float32)
smoothing_lengths = np.array([0.1], dtype=np.float32)

def setup_simulation(mode="Newtonian", metric=None):
    """Create a simulation in the specified mode."""
    particles = ParticleSystem(n_particles=1, smoothing_length=smoothing_lengths)
    particles.positions = positions.copy()
    particles.velocities = velocities.copy()
    particles.masses = masses
    particles.smoothing_lengths = smoothing_lengths
    particles.density = np.ones(1, dtype=np.float32)
    particles.internal_energy = np.ones(1, dtype=np.float32)
    particles.pressure = np.ones(1, dtype=np.float32)
    particles.sound_speed = np.ones(1, dtype=np.float32)
    particles.temperature = np.ones(1, dtype=np.float32)
    
    config = SimulationConfig(
        mode=mode,
        bh_mass=M_bh,
        t_end=0.01,
        verbose=False
    )
    
    sim = Simulation(
        particles=particles,
        gravity_solver=RelativisticGravitySolver(G=G, bh_mass=M_bh),
        eos=IdealGas(gamma=5.0/3.0),
        integrator=LeapfrogIntegrator(),
        config=config,
        metric=metric
    )
    sim.use_gpu = False  # Use CPU for comparison
    return sim

# Test Newtonian mode
print(f"\nTest radius: r = {r_test:.1f}M")
print(f"Expected Newtonian: |a| = G*M/r² = {G * M_bh / r_test**2:.6e}")

print("\n" + "="*70)
print("NEWTONIAN MODE")
print("="*70)
sim_newtonian = setup_simulation(mode="Newtonian", metric=None)
forces_newtonian = sim_newtonian.compute_forces()
a_newtonian = forces_newtonian['gravity'][0]
print(f"Acceleration: {a_newtonian}")
print(f"Magnitude: {np.linalg.norm(a_newtonian):.6e}")

print("\n" + "="*70)
print("GR MODE (Schwarzschild)")
print("="*70)
metric = SchwarzschildMetric(mass=M_bh)
sim_gr = setup_simulation(mode="GR", metric=metric)
forces_gr = sim_gr.compute_forces()
a_gr = forces_gr['gravity'][0]
print(f"Acceleration: {a_gr}")
print(f"Magnitude: {np.linalg.norm(a_gr):.6e}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
diff = np.linalg.norm(a_gr - a_newtonian)
rel_diff = diff / np.linalg.norm(a_newtonian)
print(f"Difference: {diff:.6e}")
print(f"Relative difference: {rel_diff:.2%}")

if rel_diff < 0.05:  # 5% tolerance
    print(f"\n[OK] GR reduces to Newtonian at r={r_test}M (within 5%)")
else:
    print(f"\n[WARNING] GR and Newtonian differ by {rel_diff:.2%} at r={r_test}M")

print("\nNote: Small differences are expected due to:")
print("  - Post-Newtonian corrections in GR (O(M/r) ~ 1%)")
print("  - Numerical precision differences")
