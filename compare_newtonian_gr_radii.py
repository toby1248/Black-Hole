"""
Compare Newtonian vs GR gravity at different radii.
Shows post-Newtonian corrections becoming significant closer to BH.
"""
import numpy as np
from tde_sph.core.simulation import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import RelativisticGravitySolver
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator
from tde_sph.metric import SchwarzschildMetric

M_bh = 1.0
G = 1.0

# Test at different radii
radii = [100.0, 50.0, 20.0, 10.0, 6.0]

print("="*70)
print("Newtonian vs GR: Post-Newtonian Corrections")
print("="*70)
print(f"\n{'r [M]':>8} | {'Newtonian':>12} | {'GR':>12} | {'Rel. Diff':>10}")
print("-"*70)

for r_test in radii:
    # Setup
    positions = np.array([[r_test, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    masses = np.array([1e-6], dtype=np.float32)
    smoothing_lengths = np.array([0.1], dtype=np.float32)
    
    # Newtonian
    particles_n = ParticleSystem(n_particles=1, smoothing_length=smoothing_lengths)
    particles_n.positions = positions.copy()
    particles_n.velocities = velocities.copy()
    particles_n.masses = masses
    particles_n.smoothing_lengths = smoothing_lengths
    particles_n.density = np.ones(1, dtype=np.float32)
    particles_n.internal_energy = np.ones(1, dtype=np.float32)
    particles_n.pressure = np.ones(1, dtype=np.float32)
    particles_n.sound_speed = np.ones(1, dtype=np.float32)
    particles_n.temperature = np.ones(1, dtype=np.float32)
    
    sim_n = Simulation(
        particles=particles_n,
        gravity_solver=RelativisticGravitySolver(G=G, bh_mass=M_bh),
        eos=IdealGas(gamma=5.0/3.0),
        integrator=LeapfrogIntegrator(),
        config=SimulationConfig(mode="Newtonian", bh_mass=M_bh, verbose=False),
        metric=None
    )
    sim_n.use_gpu = False
    forces_n = sim_n.compute_forces()
    a_n = np.linalg.norm(forces_n['gravity'][0])
    
    # GR
    particles_gr = ParticleSystem(n_particles=1, smoothing_length=smoothing_lengths)
    particles_gr.positions = positions.copy()
    particles_gr.velocities = velocities.copy()
    particles_gr.masses = masses
    particles_gr.smoothing_lengths = smoothing_lengths
    particles_gr.density = np.ones(1, dtype=np.float32)
    particles_gr.internal_energy = np.ones(1, dtype=np.float32)
    particles_gr.pressure = np.ones(1, dtype=np.float32)
    particles_gr.sound_speed = np.ones(1, dtype=np.float32)
    particles_gr.temperature = np.ones(1, dtype=np.float32)
    
    metric = SchwarzschildMetric(mass=M_bh)
    sim_gr = Simulation(
        particles=particles_gr,
        gravity_solver=RelativisticGravitySolver(G=G, bh_mass=M_bh),
        eos=IdealGas(gamma=5.0/3.0),
        integrator=LeapfrogIntegrator(),
        config=SimulationConfig(mode="GR", bh_mass=M_bh, verbose=False),
        metric=metric
    )
    sim_gr.use_gpu = False
    forces_gr = sim_gr.compute_forces()
    a_gr = np.linalg.norm(forces_gr['gravity'][0])
    
    # Compare
    rel_diff = abs(a_gr - a_n) / a_n
    
    print(f"{r_test:8.1f} | {a_n:12.6e} | {a_gr:12.6e} | {rel_diff:9.2%}")

print("-"*70)
print("\nPost-Newtonian corrections scale as O(M/r):")
print("  r = 100M: ~1% correction")
print("  r = 10M:  ~10% correction")
print("  r = 6M:   ~17% correction (near ISCO)")
