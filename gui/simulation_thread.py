"""
Background thread for running TDE-SPH simulations (TASK-100).

Handles:
- Simulation initialization from configuration
- Main simulation loop execution
- Progress reporting via signals
- Error handling
"""

import sys
from pathlib import Path
import traceback
import numpy as np
from typing import Dict, Any

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    try:
        from PyQt5.QtCore import QThread, pyqtSignal
    except ImportError:
        # Mock for testing without PyQt
        class QThread:
            def __init__(self): pass
            def start(self): self.run()
            def wait(self): pass
        
        class pyqtSignal:
            def __init__(self, *args): pass
            def emit(self, *args): pass
            def connect(self, func): pass

# Ensure src is in sys.path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# TDE-SPH imports
try:
    from tde_sph.core import Simulation, SimulationConfig
    from tde_sph.sph import ParticleSystem
    from tde_sph.gravity import NewtonianGravity, BarnesHutGravity
    from tde_sph.eos import IdealGas
    from tde_sph.integration import LeapfrogIntegrator
    from tde_sph.ICs import Polytrope
except ImportError:
    # Fallback if tde_sph not in path (e.g. during isolated testing)
    Simulation = None

class SimulationThread(QThread):
    """
    Worker thread for running the simulation without blocking the GUI.
    """
    
    # Signals
    progress_updated = pyqtSignal(float, int, dict, dict)  # time, step, energies, stats
    simulation_finished = pyqtSignal()
    simulation_stopped = pyqtSignal()
    simulation_error = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__()
        self.config_dict = config_dict
        self.running = False
        self.paused = False
        self.simulation = None

    def run(self):
        """Main execution loop."""
        if Simulation is None:
            self.simulation_error.emit("Could not import tde_sph modules. Check PYTHONPATH.")
            return

        try:
            self.running = True
            self.log_message.emit("Initializing simulation...")
            
            # Initialize simulation
            self.simulation = self._initialize_simulation(self.config_dict)
            self.log_message.emit(f"Initialized {self.simulation.config.mode} simulation with {self.simulation.particles.n_particles} particles")
            
            # Initial state report
            self._report_progress()
            
            # Main loop
            while self.running and self.simulation.state.time < self.simulation.config.t_end:
                if self.paused:
                    self.msleep(100)
                    continue
                
                # Step simulation
                self.simulation.step()
                
                # Report progress periodically (e.g. every 10 steps or if time passed)
                # For GUI responsiveness, we might want to limit signal emission frequency
                if self.simulation.state.step % 10 == 0:
                    self._report_progress()
                
                # Check for snapshots
                if self.simulation.state.time - self.simulation.state.last_snapshot_time >= self.simulation.config.snapshot_interval:
                    self.simulation.write_snapshot()
                    self.log_message.emit(f"Snapshot written at t={self.simulation.state.time:.2f}")

            if self.running:
                self.log_message.emit("Simulation completed successfully")
                self.simulation_finished.emit()
            else:
                self.log_message.emit("Simulation stopped by user")
                self.simulation_stopped.emit()

        except Exception as e:
            error_msg = f"Simulation error: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg)
            self.simulation_error.emit(str(e))
            self.running = False

    def stop(self):
        """Request simulation stop."""
        self.running = False

    def pause(self):
        """Pause simulation."""
        self.paused = True

    def resume(self):
        """Resume simulation."""
        self.paused = False

    def _initialize_simulation(self, config_dict: Dict[str, Any]) -> 'Simulation':
        """
        Create Simulation object from configuration dictionary.
        """
        # Extract parameters
        sim_params = config_dict.get('simulation', {})
        bh_params = config_dict.get('black_hole', {})
        star_params = config_dict.get('star', {})
        orbit_params = config_dict.get('orbit', {})
        part_params = config_dict.get('particles', {})
        int_params = config_dict.get('integration', {})
        phys_params = config_dict.get('physics', {})

        # 1. Generate Initial Conditions (Polytrope)
        n_particles = part_params.get('count', 1000)
        gamma = 5.0/3.0 if phys_params.get('eos') == 'ideal_gas' else 1.4
        
        polytrope = Polytrope(gamma=gamma, random_seed=config_dict.get('seed', 42))
        
        pos, vel, mass, u, rho = polytrope.generate(
            n_particles=n_particles,
            M_star=star_params.get('mass', 1.0),
            R_star=star_params.get('radius', 1.0),
            position=np.zeros(3, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32)
        )
        
        # 2. Setup Orbit
        # Simplified orbit setup (parabolic approximation)
        bh_mass = bh_params.get('mass', 1e6)
        periapsis = orbit_params.get('pericentre', 10.0) # In Rg or Rt? Assuming Rt for now based on run_simulation.py
        
        # Calculate Tidal Radius
        R_t = star_params.get('radius', 1.0) * (bh_mass / star_params.get('mass', 1.0))**(1/3)
        
        # If periapsis is given in Rg (Gravitational radii), convert? 
        # run_simulation.py uses units of Rt. Let's assume the config value is in Rt for consistency with run_simulation.py
        # But wait, config usually specifies physical units or code units.
        # Let's assume the input is in units of Rt if not specified.
        
        r_p = periapsis * R_t
        r_init = 3.0 * r_p
        v_init = np.sqrt(2.0 * bh_mass / r_init) # G=1
        
        # Shift to orbit
        pos[:, 0] -= r_init
        vel[:, 1] += v_init
        
        # 3. Create Particle System
        particles = ParticleSystem(
            n_particles=n_particles,
            positions=pos,
            velocities=vel,
            masses=mass,
            internal_energy=u
        )
        particles.smoothing_lengths = polytrope.compute_smoothing_lengths(mass, rho)
        
        # 4. Physics Modules
        gravity_type = phys_params.get('gravity', 'newtonian')
        if gravity_type == 'barnes_hut':
            gravity = BarnesHutGravity(theta=phys_params.get('theta', 0.5))
        else:
            gravity = NewtonianGravity() # TODO: Support GR based on config
            
        eos = IdealGas(gamma=gamma)
        integrator = LeapfrogIntegrator(cfl_factor=0.3)
        
        # 5. Simulation Config
        sim_config = SimulationConfig(
            t_start=0.0,
            t_end=int_params.get('t_end', 10.0),
            dt_initial=int_params.get('timestep', 0.001),
            snapshot_interval=int_params.get('output_interval', 0.1),
            output_dir=sim_params.get('output_dir', 'outputs'),
            mode=sim_params.get('mode', 'Newtonian'),
            bh_mass=bh_mass,
            verbose=True
        )
        
        return Simulation(
            particles=particles,
            gravity_solver=gravity,
            eos=eos,
            integrator=integrator,
            config=sim_config
        )

    def _report_progress(self):
        """Emit progress signals."""
        if not self.simulation:
            return
            
        # Calculate energies
        energies = {
            'kinetic': self.simulation.state.kinetic_energy,
            'potential': self.simulation.state.potential_energy,
            'internal': self.simulation.state.internal_energy,
            'total': self.simulation.state.total_energy,
            'error': 0.0
        }
        
        if self.simulation.state.initial_energy and self.simulation.state.initial_energy != 0:
            energies['error'] = (self.simulation.state.total_energy - self.simulation.state.initial_energy) / self.simulation.state.initial_energy

        # Calculate stats
        stats = {
            'n_particles': self.simulation.particles.n_particles,
            'total_mass': np.sum(self.simulation.particles.masses),
            'total_energy': self.simulation.state.total_energy,
            'kinetic_energy': self.simulation.state.kinetic_energy,
            'potential_energy': self.simulation.state.potential_energy,
            'internal_energy': self.simulation.state.internal_energy,
            'mean_density': np.mean(self.simulation.particles.density),
            'max_density': np.max(self.simulation.particles.density),
            'mean_temperature': np.mean(self.simulation.particles.temperature) if hasattr(self.simulation.particles, 'temperature') else 0.0,
            'max_temperature': np.max(self.simulation.particles.temperature) if hasattr(self.simulation.particles, 'temperature') else 0.0
        }
        
        self.progress_updated.emit(
            self.simulation.state.time,
            self.simulation.state.step,
            energies,
            stats
        )
