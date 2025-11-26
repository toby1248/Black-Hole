
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

def debug_run():
    print("DEBUG: Starting debug run...")
    
    # 1. Setup
    N = 5000
    print(f"DEBUG: Generating {N} particles...")
    polytrope = Polytrope(gamma=5.0/3.0)
    pos, vel, mass, u, rho = polytrope.generate(n_particles=N)
    
    print("DEBUG: Creating ParticleSystem...")
    particles = ParticleSystem(
        n_particles=N,
        positions=pos,
        velocities=vel,
        masses=mass,
        internal_energy=u,
    )
    
    print("DEBUG: Computing smoothing lengths...")
    h = polytrope.compute_smoothing_lengths(mass, rho)
    particles.smoothing_lengths = h
    
    print("DEBUG: Initializing modules...")
    gravity = NewtonianGravity()
    eos = IdealGas(gamma=5.0/3.0)
    integrator = LeapfrogIntegrator(cfl_factor=0.3)
    
    config = SimulationConfig(
        t_end=0.1,
        dt_initial=0.001,
        snapshot_interval=0.05,
        output_dir="debug_output",
        verbose=True
    )
    
    sim = Simulation(
        particles=particles,
        gravity_solver=gravity,
        eos=eos,
        integrator=integrator,
        config=config
    )
    
    print("DEBUG: Starting sim.run()...")
    
    # Manually run the loop to see where it hangs
    sim._log("=" * 60)
    sim._log("Starting simulation")
    sim._log("=" * 60)

    print("DEBUG: Writing initial snapshot...")
    sim.write_snapshot()
    print("DEBUG: Initial snapshot written.")

    print("DEBUG: Computing initial energies...")
    energies = sim.compute_energies()
    sim.state.initial_energy = energies['total']
    sim._log(f"Initial energy: {sim.state.initial_energy:.6e}")
    print("DEBUG: Initial energies computed.")

    while sim.state.time < sim.config.t_end:
        print(f"DEBUG: Step {sim.state.step} start. t={sim.state.time:.4f}, dt={sim.state.dt:.4e}")
        
        # Step breakdown
        print("DEBUG: Calling sim.step()...")
        
        # Inside step():
        # forces = self.compute_forces()
        print("  DEBUG: compute_forces() start...")
        
        # 1. Neighbours
        print("    DEBUG: find_neighbours_bruteforce...")
        from tde_sph.sph.neighbours_cpu import find_neighbours_bruteforce
        t0 = time.time()
        neighbour_lists, _ = find_neighbours_bruteforce(
            sim.particles.positions,
            sim.particles.smoothing_lengths
        )
        print(f"    DEBUG: find_neighbours_bruteforce done in {time.time()-t0:.4f}s")
        
        # 2. Density
        print("    DEBUG: compute_density_summation...")
        from tde_sph.sph.neighbours_cpu import compute_density_summation
        t0 = time.time()
        sim.particles.density = compute_density_summation(
            sim.particles.positions,
            sim.particles.masses,
            sim.particles.smoothing_lengths,
            neighbour_lists,
            kernel_func=sim.kernel.kernel
        )
        print(f"    DEBUG: compute_density_summation done in {time.time()-t0:.4f}s")
        
        # 3. Update h
        print("    DEBUG: update_smoothing_lengths...")
        from tde_sph.sph.neighbours_cpu import update_smoothing_lengths
        t0 = time.time()
        sim.particles.smoothing_lengths = update_smoothing_lengths(
            sim.particles.positions,
            sim.particles.masses,
            sim.particles.smoothing_lengths
        )
        print(f"    DEBUG: update_smoothing_lengths done in {time.time()-t0:.4f}s")
        
        # 4. Thermodynamics
        print("    DEBUG: update_thermodynamics...")
        sim.update_thermodynamics()
        
        # 5. Gravity
        print("    DEBUG: gravity_solver.compute_acceleration...")
        t0 = time.time()
        a_grav = sim.gravity_solver.compute_acceleration(
            sim.particles.positions,
            sim.particles.masses,
            sim.particles.smoothing_lengths,
            metric=sim.metric
        )
        print(f"    DEBUG: gravity done in {time.time()-t0:.4f}s")
        
        # 6. Hydro
        print("    DEBUG: compute_hydro_acceleration...")
        from tde_sph.sph.hydro_forces import compute_hydro_acceleration
        t0 = time.time()
        a_hydro, du_dt_hydro = compute_hydro_acceleration(
            sim.particles.positions,
            sim.particles.velocities,
            sim.particles.masses,
            sim.particles.density,
            sim.particles.pressure,
            sim.particles.sound_speed,
            sim.particles.smoothing_lengths,
            neighbour_lists,
            kernel_gradient_func=sim.kernel.kernel_gradient,
            alpha=sim.config.artificial_viscosity_alpha,
            beta=sim.config.artificial_viscosity_beta,
        )
        print(f"    DEBUG: hydro done in {time.time()-t0:.4f}s")
        
        forces = {
            'gravity': a_grav,
            'hydro': a_hydro,
            'total': a_grav + a_hydro,
            'du_dt': du_dt_hydro,
        }
        print("  DEBUG: compute_forces() done.")

        # Advance particles
        print("  DEBUG: integrator.step()...")
        sim.integrator.step(
            sim.particles,
            sim.state.dt,
            forces
        )
        
        # Update time
        sim.state.time += sim.state.dt
        sim.state.step += 1
        
        # Estimate next timestep
        print("  DEBUG: integrator.estimate_timestep()...")
        sim.state.dt = sim.integrator.estimate_timestep(
            sim.particles,
            cfl_factor=sim.config.cfl_factor,
            accelerations=forces['total']
        )
        print(f"  DEBUG: New dt={sim.state.dt:.4e}")
        
        # Update energy diagnostics
        energies = sim.compute_energies()
        sim.state.kinetic_energy = energies['kinetic']
        sim.state.potential_energy = energies['potential']
        sim.state.internal_energy = energies['internal']
        sim.state.total_energy = energies['total']
        
        print("DEBUG: Step done.")
        
        # Periodic logging
        if sim.state.step % int(sim.config.log_interval / sim.state.dt + 1) == 0:
            drift = 0.0
            if sim.state.initial_energy != 0:
                drift = (sim.state.total_energy - sim.state.initial_energy) / sim.state.initial_energy

            sim._log(
                f"Step {sim.state.step:6d}  "
                f"t={sim.state.time:.4f}  "
                f"dt={sim.state.dt:.2e}  "
                f"E_tot={sim.state.total_energy:.6e}  "
                f"ΔE/E={drift:.2e}"
            )

        # Periodic snapshots
        if sim.state.time - sim.state.last_snapshot_time >= sim.config.snapshot_interval:
            print("DEBUG: Writing snapshot...")
            sim.write_snapshot()

        # Check energy conservation
        sim.check_energy_conservation()

        # Safety: prevent runaway
        if sim.state.step > 10:
            print("DEBUG: Stopping early for debug.")
            break

    print("DEBUG: Simulation loop finished.")

if __name__ == "__main__":
    debug_run()
