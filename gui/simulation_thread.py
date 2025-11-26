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
    from tde_sph.sph import (
        ParticleSystem,
        CubicSplineKernel,
        compute_density_summation,
        find_neighbours_bruteforce,
        update_smoothing_lengths,
    )
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
    progress_updated = pyqtSignal(float, int, dict)  # time, step, progress_info (lightweight)
    live_data_updated = pyqtSignal(float, int, dict, dict)  # time, step, energies, stats (expensive)
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
        self.detailed_diagnostics_enabled = True
        self.live_data_interval = 10  # Steps between expensive live data updates
        self.data_display_visible = True  # Whether live data display is visible

    def run(self):
        """Main execution loop."""
        if Simulation is None:
            self.simulation_error.emit("Could not import tde_sph modules. Check PYTHONPATH.")
            return

        try:
            self.running = True
            self.log_message.emit("Initializing simulation...")
            
            # Log acceleration capabilities
            self._log_acceleration_capabilities()
            
            # Initialize simulation
            self.simulation = self._initialize_simulation(self.config_dict)
            self.log_message.emit(f"Initialized {self.simulation.config.mode} simulation with {self.simulation.particles.n_particles} particles")
            
            # Initial state report
            self._report_progress_only()
            self._report_live_data()
            
            # Main loop
            while self.running and self.simulation.state.time < self.simulation.config.t_end:
                if self.paused:
                    self.msleep(100)
                    continue
                
                # Step simulation
                self.simulation.step()
                
                # Always report basic progress (lightweight - every step)
                # Get averaged timestep timing for this reporting period
                avg_timestep_timing = self.simulation.get_averaged_timestep_timing()
                #self._report_progress_only(avg_timestep_timing)
                
                # Report expensive live data only when needed
                if self.data_display_visible and self.simulation.state.step % self.live_data_interval == 0:
                    self._report_live_data(avg_timestep_timing)
                
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
    
    def set_detailed_diagnostics(self, enabled: bool):
        """Enable/disable expensive diagnostic calculations (percentiles, BH potential)."""
        self.detailed_diagnostics_enabled = enabled
        self.log_message.emit(f"Detailed diagnostics: {'enabled' if enabled else 'disabled'}")
    
    def set_live_data_interval(self, interval: int):
        """Set how many steps between live data reports."""
        self.live_data_interval = max(1, interval)
        self.log_message.emit(f"Live data interval: every {interval} steps")
    
    def set_data_display_visible(self, visible: bool):
        """Notify thread whether data display widget is visible."""
        self.data_display_visible = visible
    
    def _log_acceleration_capabilities(self):
        """Log available acceleration capabilities (CUDA, Numba)."""
        capabilities = []
        
        # Check for CUDA
        try:
            import cupy as cp
            has_cuda = True
            capabilities.append(f"✓ CUDA available (CuPy version {cp.__version__})")
        except ImportError:
            has_cuda = False
            capabilities.append("✗ CUDA not available (CuPy not installed)")
        
        # Check for Numba
        try:
            import numba
            has_numba = True
            capabilities.append(f"✓ Numba available (version {numba.__version__})")
        except ImportError:
            has_numba = False
            capabilities.append("✗ Numba not available")
        
        # Check which neighbour search will be used
        if has_numba:
            capabilities.append("→ Using Numba-accelerated neighbour search")
        else:
            capabilities.append("→ Using fallback CPU neighbour search (slower)")
        
        # Log all capabilities
        self.log_message.emit("=== Acceleration Capabilities ===")
        for cap in capabilities:
            self.log_message.emit(cap)
        self.log_message.emit("================================")

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


        def _get_float(params: Dict[str, Any], key: str, default: float) -> float:
            value = params.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                self.log_message.emit(f"Parameter '{key}' invalid; using default {default}")
                return float(default)


        # Debug logging
        bh_mass = _get_float(bh_params, 'mass', 1e6)
        self.log_message.emit(f"Config types - BH mass: {type(bh_params.get('mass'))}, Star mass: {type(star_params.get('mass'))}")

        # 1. Generate Initial Conditions (Polytrope)
        gamma = self._infer_gamma(star_params, phys_params)
        polytrope = Polytrope(gamma=gamma, random_seed=config_dict.get('seed', 42))
        n_particles = int(part_params.get('count', 1000))
        pos, vel, mass, u, rho = polytrope.generate(
            n_particles=n_particles,
            M_star=star_params.get('mass', 1.0),
            R_star=star_params.get('radius', 1.0),
            position=np.zeros(3, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32)
        )
        h = polytrope.compute_smoothing_lengths(mass, rho)

        # CRITICAL FIX: Add small random thermal velocities to prevent explosion
        # Without this, all particles have identical velocity → they spread out
        # → lose neighbors → density→0 → pressure→∞ → explosion
        # Thermal velocity scale: v_th ~ sqrt(P/rho) ~ sqrt(u * (gamma-1))
        # Use ~1% of sound speed to avoid disrupting star structure
        thermal_velocity_scale = 0.01 * np.sqrt((gamma - 1.0) * np.mean(u))
        vel += np.random.randn(*vel.shape).astype(np.float32) * thermal_velocity_scale
        self.log_message.emit(f"Added thermal velocities: v_th ~ {thermal_velocity_scale:.3e}")

        if phys_params.get('pre_relax_polytrope', True):
            self.log_message.emit("Pre-relaxing polytrope initial conditions...")
            pos, vel, u, rho, h = self._pre_relax_polytrope(
                pos, vel, mass, rho, h, gamma, polytrope.eta
            )



        star_mass = _get_float(star_params, 'mass', 1.0)
        star_radius = _get_float(star_params, 'radius', 1.0)
        bh_mass = _get_float(bh_params, 'mass', 1e6)
        
        # Orbital parameters
        periapsis = _get_float(orbit_params, 'pericentre', 10.0)  # In R_t
        eccentricity = _get_float(orbit_params, 'eccentricity', 0.95)
        starting_distance = _get_float(orbit_params, 'starting_distance', 3.0)  # In units of periapsis
        
        # Calculate Tidal Radius
        R_t = star_radius * (bh_mass / star_mass)**(1/3)
        
        # Periapsis distance in code units
        r_p = periapsis * R_t
        
        # Starting position (in units of periapsis, approaching from negative x)
        r_init = starting_distance * r_p
        
        # For a Newtonian orbit: angular momentum L and energy E
        # At periapsis: v_p = sqrt(G M (1 + e) / r_p), all tangential
        # At distance r: v^2 = G M ((2/r) - ((1-e^2)/(r_p(1+e))))
        # Velocity components depend on position in orbit
        
        # Specific orbital energy (per unit mass): E = -G M / (2 a)
        # Semi-major axis: a = r_p / (1 - e)
        a = r_p / (1.0 - eccentricity) if eccentricity < 1.0 else None
        
        if a is not None and a > 0:
            # Elliptical orbit
            # Specific energy
            E_orb = -bh_mass / (2.0 * a)  # G=1
            # At distance r, velocity magnitude from energy conservation
            v_mag = np.sqrt(2.0 * (E_orb + bh_mass / r_init))
        else:
            # Parabolic/hyperbolic - use vis-viva with e >= 1
            # For parabolic: E = 0, v = sqrt(2 G M / r)
            # For hyperbolic: use v_inf or assume minimally unbound
            v_mag = np.sqrt(2.0 * bh_mass / r_init)
        
        # Angular momentum at periapsis
        L = r_p * np.sqrt(bh_mass * (1.0 + eccentricity) / r_p)  # G=1
        
        # At distance r, tangential velocity component
        v_tangential = L / r_init
        
        # Radial velocity component (negative = approaching)
        v_radial_sq = v_mag**2 - v_tangential**2
        v_radial = -np.sqrt(max(v_radial_sq, 0.0))  # Negative = inward
        
        # CRITICAL FIX: Scale velocities to be sub-relativistic
        # In code units with c=1, velocities must be << 1 to be physical
        # Typical orbital velocity at several R_g should be v ~ 0.1-0.3 c
        # Current Newtonian formula gives v ~ sqrt(GM/r) which can be >> c
        #v_scale = 0.1  # Scale to ~0.1c (10% speed of light)
        #v_mag *= v_scale
        #v_radial *= v_scale
        #v_tangential *= v_scale
        
        self.log_message.emit(f"Velocity scaling applied: factor={v_scale} to keep v << c")
        
        # Position: star at -x direction, approaching from the left
        pos[:, 0] -= r_init
        pos[:, 1] += 0.0
        pos[:, 2] += 0.0
        
        # Velocity: radial (+x, toward BH) and tangential (-y for prograde orbit in +z)
        # For position at -x, tangential velocity must be -y to give L in +z direction
        vel[:, 0] += -v_radial  # Approaching = positive x direction
        vel[:, 1] += -v_tangential  # Tangential in -y for +z angular momentum
        vel[:, 2] += 0.0
        
        self.log_message.emit(f"Orbit setup: r_p={r_p:.2e}, e={eccentricity:.3f}, r_init={r_init:.2e}")
        self.log_message.emit(f"Initial velocity: v_r={-v_radial:.3e}, v_t={v_tangential:.3e}, |v|={v_mag:.3e}")
        
        # 3. Create Particle System
        particles = ParticleSystem(
            n_particles=n_particles,
            positions=pos,
            velocities=vel,
            masses=mass,
            internal_energy=u,
            smoothing_length=h
        )
        particles.density = rho
        
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

    def _infer_gamma(self, star_params: Dict[str, Any], phys_params: Dict[str, Any]) -> float:
        """
        Infer adiabatic index from config, preferring explicit polytropic index.
        """
        # Explicit gamma override
        for key in ('gamma', 'adiabatic_index'):
            if key in star_params:
                try:
                    return float(star_params[key])
                except (TypeError, ValueError):
                    self.log_message.emit(f"Invalid {key} in star config; falling back to defaults")

        # Polytropic index -> gamma = 1 + 1/n
        n_value = star_params.get('polytropic_index')
        if n_value is not None:
            try:
                n_float = float(n_value)
                if n_float > 0:
                    return float(1.0 + 1.0 / n_float)
            except (TypeError, ValueError):
                self.log_message.emit("Invalid polytropic_index; using EOS default gamma")

        eos_name = str(phys_params.get('eos', 'ideal_gas')).lower()
        if eos_name == 'ideal_gas':
            return 5.0 / 3.0

        return float(phys_params.get('gamma', 5.0 / 3.0))

    def _pre_relax_polytrope(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        density: np.ndarray,
        smoothing_lengths: np.ndarray,
        gamma: float,
        eta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Light pre-relaxation: recentre, refresh h/ρ, and reset u to polytropic relation.
        """
        pos = positions.astype(np.float32, copy=True)
        vel = velocities.astype(np.float32, copy=True)
        rho = density.astype(np.float32, copy=True)
        h = smoothing_lengths.astype(np.float32, copy=True)

        total_mass = float(np.sum(masses))
        if total_mass > 0.0:
            com = np.sum(pos * masses[:, None], axis=0) / total_mass
            pos -= com
            v_com = np.sum(vel * masses[:, None], axis=0) / total_mass
            vel -= v_com

        rho_safe = np.maximum(rho, 1e-10)
        h = eta * np.power(masses / rho_safe, 1.0 / 3.0).astype(np.float32)

        # For manageable particle counts, do a quick density/h refresh using SPH sums
        if len(pos) <= 200000:
            try:
                h = update_smoothing_lengths(
                    pos, masses, h, max_iterations=3, tolerance=0.1
                )
                kernel = CubicSplineKernel()
                neighbour_lists, _ = find_neighbours_bruteforce(pos, h)
                rho = compute_density_summation(
                    pos, masses, h, neighbour_lists, kernel.kernel
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.log_message.emit(f"Pre-relaxation density update skipped: {exc}")
        else:
            self.log_message.emit(
                f"Skipping neighbour-based pre-relaxation for N={len(pos)} (too large)"
            )

        n_index = 1.0 / (gamma - 1.0)
        K = np.float32(1.0 / (4.0 * np.pi * (n_index + 1.0)))
        rho_safe = np.maximum(rho, 1e-10)
        u = K * np.power(rho_safe, gamma - 1.0) / (gamma - 1.0)

        # Damp any residual motions before orbital boost is applied
        vel *= 0.0

        return (
            pos.astype(np.float32),
            vel.astype(np.float32),
            u.astype(np.float32),
            rho.astype(np.float32),
            h.astype(np.float32),
        )

    def _report_progress_only(self, avg_timestep_timing: float = 0.0):
        """Emit lightweight progress update (every step)."""
        if not self.simulation:
            return
        
        # Convert timing from seconds to milliseconds for display
        progress_info = {
            'total_steps': 0,  # Will be set by control panel from config
            'total_time': self.simulation.config.t_end,
            'dt': self.simulation.state.dt,
            'timing_ms': {
                'compute_forces': self.simulation.state.timing_compute_forces,
                'gravity': self.simulation.state.timing_gravity,
                'sph_density': self.simulation.state.timing_sph_density,
                'smoothing_lengths': self.simulation.state.timing_smoothing_lengths,
                'sph_pressure': self.simulation.state.timing_sph_pressure,
                'integration': self.simulation.state.timing_integration,
                'energy_computation': self.simulation.state.timing_energy_computation,
                'timestep_estimation': avg_timestep_timing,  # Use averaged value
                'thermodynamics': self.simulation.state.timing_thermodynamics,
                'gpu_transfer': self.simulation.state.timing_gpu_transfer,
                'io_overhead': self.simulation.state.timing_io,
                'other': self.simulation.state.timing_other,
                'total': self.simulation.state.timing_total
            }
        }
        
        self.progress_updated.emit(
            self.simulation.state.time,
            self.simulation.state.step,
            progress_info
        )
    
    def _report_live_data(self, avg_timestep_timing: float = 0.0):
        """Emit expensive live data update (periodic)."""
        if not self.simulation:
            return
        
        particles = self.simulation.particles
        
        # Calculate distances from BH (needed for several metrics)
        r_from_bh = np.linalg.norm(particles.positions, axis=1)
        median_distance = float(np.median(r_from_bh))
        
        # Calculate BH potential contribution if gravity solver supports it
        E_pot_bh = 0.0
        if hasattr(self.simulation.gravity_solver, 'bh_mass'):
            bh_mass = self.simulation.gravity_solver.bh_mass
            if bh_mass > 0:
                # BH potential: -G M_BH / r
                r_safe = np.maximum(r_from_bh, 1e-6)
                E_pot_bh = -np.sum(particles.masses * bh_mass / r_safe)
            
        # Calculate energies
        energies = {
            'kinetic': self.simulation.state.kinetic_energy,
            'potential': self.simulation.state.potential_energy,
            'potential_bh': float(E_pot_bh),
            'internal': self.simulation.state.internal_energy,
            'total': self.simulation.state.total_energy,
            'error': 0.0
        }
        
        if self.simulation.state.initial_energy and self.simulation.state.initial_energy != 0:
            energies['error'] = (self.simulation.state.total_energy - self.simulation.state.initial_energy) / self.simulation.state.initial_energy

        # Expensive diagnostics (conditional)
        distance_percentiles = [0.0] * 7
        energy_percentiles = [0.0] * 7
        
        if self.detailed_diagnostics_enabled:
            # Calculate percentiles for distance
            distance_percentiles = np.percentile(r_from_bh, [1, 10, 25, 50, 75, 90, 99])
            
            # Per-particle specific energy (kinetic + potential per unit mass)
            # Use Barnes-Hut if available (O(N log N)), otherwise fallback to direct summation
            v_sq = np.sum(particles.velocities**2, axis=1)
            E_kin_spec = 0.5 * v_sq
            
            # Use Barnes-Hut for potential calculation (same as compute_energies)
            try:
                from tde_sph.gravity.barnes_hut import BarnesHutGravity
                
                if isinstance(self.simulation.gravity_solver, BarnesHutGravity):
                    E_pot_spec = self.simulation.gravity_solver.compute_potential(
                        particles.positions,
                        particles.masses,
                        particles.smoothing_lengths
                    )
                else:
                    # Create temporary Barnes-Hut solver for O(N log N) performance
                    G = getattr(self.simulation.gravity_solver, 'G', 1.0)
                    bh_solver = BarnesHutGravity(G=G, theta=0.5)
                    E_pot_spec = bh_solver.compute_potential(
                        particles.positions,
                        particles.masses,
                        particles.smoothing_lengths
                    )
            except (ImportError, Exception):
                # Fallback to direct summation (expensive but accurate)
                E_pot_spec = self.simulation.gravity_solver.compute_potential(
                    particles.positions,
                    particles.masses,
                    particles.smoothing_lengths
                )
            
            E_tot_spec = E_kin_spec + E_pot_spec
            energy_percentiles = np.percentile(E_tot_spec, [1, 10, 25, 50, 75, 90, 99])
        
        # Calculate stats
        stats = {
            'n_particles': particles.n_particles,
            'total_mass': np.sum(particles.masses),
            'total_energy': self.simulation.state.total_energy,
            'kinetic_energy': self.simulation.state.kinetic_energy,
            'potential_energy': self.simulation.state.potential_energy,
            'internal_energy': self.simulation.state.internal_energy,
            'mean_density': np.mean(particles.density),
            'max_density': np.max(particles.density),
            'mean_temperature': float(np.mean(particles.temperature)),
            'max_temperature': float(np.max(particles.temperature)),
            'mean_velocity_magnitude': float(np.mean(particles.velocity_magnitude)),
            'max_velocity_magnitude': float(np.max(particles.velocity_magnitude)),
            'median_distance_from_bh': median_distance,
            'distance_percentiles': distance_percentiles.tolist() if self.detailed_diagnostics_enabled else distance_percentiles,
            'energy_percentiles': energy_percentiles.tolist() if self.detailed_diagnostics_enabled else energy_percentiles,
            'timings': {
                'compute_forces': self.simulation.state.timing_compute_forces,
                'gravity': self.simulation.state.timing_gravity,
                'sph_density': self.simulation.state.timing_sph_density,
                'smoothing_lengths': self.simulation.state.timing_smoothing_lengths,
                'sph_pressure': self.simulation.state.timing_sph_pressure,
                'integration': self.simulation.state.timing_integration,
                'energy_computation': self.simulation.state.timing_energy_computation,
                'timestep_estimation': avg_timestep_timing,  # Use averaged value
                'thermodynamics': self.simulation.state.timing_thermodynamics,
                'gpu_transfer': self.simulation.state.timing_gpu_transfer,
                'io_overhead': self.simulation.state.timing_io,
                'other': self.simulation.state.timing_other,
                'total': self.simulation.state.timing_total
            }
        }
        
        self.live_data_updated.emit(
            self.simulation.state.time,
            self.simulation.state.step,
            energies,
            stats
        )
