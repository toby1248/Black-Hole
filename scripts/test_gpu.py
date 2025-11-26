
import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from tde_sph.core import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import NewtonianGravity
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator
from tde_sph.ICs import Polytrope

def test_gpu():
    print("Testing GPU acceleration...")
    
    # 1. Setup
    N = 5000
    print(f"Generating {N} particles...")
    polytrope = Polytrope(gamma=5.0/3.0)
    pos, vel, mass, u, rho = polytrope.generate(n_particles=N)
    
    particles = ParticleSystem(
        n_particles=N,
        positions=pos,
        velocities=vel,
        masses=mass,
        internal_energy=u,
    )
    
    # Compute initial h on CPU
    h = polytrope.compute_smoothing_lengths(mass, rho)
    particles.smoothing_lengths = h
    
    gravity = NewtonianGravity()
    eos = IdealGas(gamma=5.0/3.0)
    integrator = LeapfrogIntegrator(cfl_factor=0.3)
    
    config = SimulationConfig(
        t_end=0.1,
        dt_initial=0.001,
        snapshot_interval=0.05,
        output_dir="gpu_test_output",
        verbose=True
    )
    
    sim = Simulation(
        particles=particles,
        gravity_solver=gravity,
        eos=eos,
        integrator=integrator,
        config=config
    )
    
    if sim.use_gpu:
        print("SUCCESS: GPU enabled in Simulation!")
    else:
        print("WARNING: GPU NOT enabled in Simulation. Check CUDA/CuPy installation.")
        
    print("Starting simulation loop...")
    start_time = time.time()
    
    steps = 10
    for i in range(steps):
        t0 = time.time()
        sim.step()
        t1 = time.time()
        
        drift = (sim.state.total_energy - sim.state.initial_energy) / sim.state.initial_energy
        print(f"Step {i}: {t1-t0:.4f}s | E_tot={sim.state.total_energy:.6e} | Drift={drift:.2e}")
        
    total_time = time.time() - start_time
    print(f"Total time for {steps} steps: {total_time:.4f}s")
    print(f"Average time per step: {total_time/steps:.4f}s")

if __name__ == "__main__":
    test_gpu()
