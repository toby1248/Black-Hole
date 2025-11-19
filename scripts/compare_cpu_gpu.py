
import sys
import time
from pathlib import Path
import numpy as np
import copy

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from tde_sph.core import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import NewtonianGravity
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator
from tde_sph.ICs import Polytrope

def run_test(use_gpu, particles_init):
    print(f"Running test with GPU={use_gpu}...")
    
    # Deep copy particles to ensure same ICs
    particles = copy.deepcopy(particles_init)
    
    gravity = NewtonianGravity()
    eos = IdealGas(gamma=5.0/3.0)
    integrator = LeapfrogIntegrator(cfl_factor=0.3)
    
    config = SimulationConfig(
        t_end=0.1,
        dt_initial=0.001,
        snapshot_interval=0.05,
        output_dir=f"compare_output_{'gpu' if use_gpu else 'cpu'}",
        verbose=False
    )
    
    sim = Simulation(
        particles=particles,
        gravity_solver=gravity,
        eos=eos,
        integrator=integrator,
        config=config
    )
    
    # Force mode
    sim.use_gpu = use_gpu
    if use_gpu and sim.gpu_manager is None:
        print("Error: GPU requested but not available/initialized.")
        return None
        
    energies = []
    
    start_time = time.time()
    steps = 10
    for i in range(steps):
        sim.step()
        drift = (sim.state.total_energy - sim.state.initial_energy) / sim.state.initial_energy
        energies.append(drift)
        print(f"  Step {i}: Drift={drift:.2e}")
        
    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.4f}s")
    
    return energies

def main():
    N = 1000
    print(f"Generating {N} particles...")
    polytrope = Polytrope(gamma=5.0/3.0)
    pos, vel, mass, u, rho = polytrope.generate(n_particles=N)
    
    particles_init = ParticleSystem(
        n_particles=N,
        positions=pos,
        velocities=vel,
        masses=mass,
        internal_energy=u,
    )
    # Compute initial h
    h = polytrope.compute_smoothing_lengths(mass, rho)
    particles_init.smoothing_lengths = h
    
    print("\n--- CPU Run ---")
    cpu_drifts = run_test(False, particles_init)
    
    print("\n--- GPU Run ---")
    gpu_drifts = run_test(True, particles_init)
    
    print("\n--- Comparison ---")
    print(f"Step | CPU Drift | GPU Drift | Diff")
    for i in range(len(cpu_drifts)):
        diff = abs(cpu_drifts[i] - gpu_drifts[i])
        print(f"{i:4d} | {cpu_drifts[i]:.2e} | {gpu_drifts[i]:.2e} | {diff:.2e}")

if __name__ == "__main__":
    main()
