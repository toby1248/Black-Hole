"""
Simulation orchestrator for TDE-SPH framework.

This module implements the main Simulation class that coordinates all physics
modules to evolve a tidal disruption event. It follows the "system + component"
architecture pattern (PAT-001).

Design:
- Simulation orchestrates pluggable components (Metric, GravitySolver, EOS, etc.)
- Each component is swappable via dependency injection
- Supports both Newtonian and GR modes via configuration
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from pathlib import Path
import time as time_module

from tde_sph.core.interfaces import (
    Metric,
    GravitySolver,
    EOS,
    RadiationModel,
    TimeIntegrator,
)
from tde_sph.sph import (
    ParticleSystem,
    CubicSplineKernel,
    find_neighbours_bruteforce,
    compute_density_summation,
    update_smoothing_lengths,
    compute_hydro_acceleration,
)


NDArrayFloat = npt.NDArray[np.float32]


@dataclass
class SimulationConfig:
    """
    Configuration for TDE-SPH simulation.

    Supports GUD-002: separation of configuration from code.
    """
    # Time evolution
    t_start: float = 0.0
    t_end: float = 10.0
    dt_initial: float = 0.001
    cfl_factor: float = 0.3

    # I/O
    output_dir: str = "output"
    snapshot_interval: float = 0.1
    log_interval: float = 0.01

    # Physics mode
    mode: str = "Newtonian"  # "Newtonian" or "GR"

    # Black hole parameters (for GR mode)
    bh_mass: float = 1.0  # Dimensionless (G=c=M_BH=1)
    bh_spin: float = 0.0  # a/M, range [0, 1]

    # SPH parameters
    neighbour_search_method: str = "bruteforce"  # "bruteforce" or "tree" (future)
    smoothing_length_eta: float = 1.2
    artificial_viscosity_alpha: float = 1.0
    artificial_viscosity_beta: float = 2.0

    # Energy tracking
    energy_tolerance: float = 0.01  # Fractional energy drift tolerance

    # Misc
    random_seed: Optional[int] = 42
    verbose: bool = True


@dataclass
class SimulationState:
    """
    Current state of the simulation.
    """
    time: float = 0.0
    step: int = 0
    dt: float = 0.001

    # Energy tracking (REQ-009)
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    internal_energy: float = 0.0
    total_energy: float = 0.0
    initial_energy: Optional[float] = None

    # Timing
    wall_time_start: float = field(default_factory=time_module.time)
    wall_time_elapsed: float = 0.0

    # Snapshots
    last_snapshot_time: float = 0.0
    snapshot_count: int = 0


class Simulation:
    """
    Main simulation orchestrator for TDE-SPH.

    Coordinates all physics modules to evolve a stellar tidal disruption event
    in either Newtonian or general relativistic spacetime.

    Architecture (PAT-001):
        Simulation orchestrates:
        - ParticleSystem (particle data)
        - GravitySolver (Newtonian or hybrid GR)
        - EOS (thermodynamics)
        - TimeIntegrator (leapfrog, Hamiltonian, etc.)
        - Optional: Metric, RadiationModel
        - I/O via HDF5Writer

    Usage:
        >>> from tde_sph.gravity import NewtonianGravity
        >>> from tde_sph.eos import IdealGas
        >>> from tde_sph.integration import LeapfrogIntegrator
        >>>
        >>> config = SimulationConfig(t_end=1.0, mode="Newtonian")
        >>> sim = Simulation(
        ...     particles=my_particles,
        ...     gravity_solver=NewtonianGravity(),
        ...     eos=IdealGas(gamma=5.0/3.0),
        ...     integrator=LeapfrogIntegrator(),
        ...     config=config
        ... )
        >>> sim.run()

    References:
        - Price (2012) - SPH framework
        - Liptai & Price (2019) - GRSPH architecture
        - Tejeda et al. (2017) - Hybrid GR+Newtonian approach
    """

    def __init__(
        self,
        particles: ParticleSystem,
        gravity_solver: GravitySolver,
        eos: EOS,
        integrator: TimeIntegrator,
        config: Optional[SimulationConfig] = None,
        metric: Optional[Metric] = None,
        radiation_model: Optional[RadiationModel] = None,
    ):
        """
        Initialize simulation.

        Parameters
        ----------
        particles : ParticleSystem
            Initial particle configuration.
        gravity_solver : GravitySolver
            Gravity solver (Newtonian or relativistic).
        eos : EOS
            Equation of state.
        integrator : TimeIntegrator
            Time integration scheme.
        config : Optional[SimulationConfig]
            Simulation configuration. If None, uses defaults.
        metric : Optional[Metric]
            Spacetime metric (required for GR mode).
        radiation_model : Optional[RadiationModel]
            Radiation/cooling model (optional).
        """
        self.particles = particles
        self.gravity_solver = gravity_solver
        self.eos = eos
        self.integrator = integrator
        self.metric = metric
        self.radiation_model = radiation_model

        self.config = config or SimulationConfig()
        self.state = SimulationState()

        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SPH kernel
        self.kernel = CubicSplineKernel()

        # Validate configuration
        self._validate_config()

        # Initialize state
        self.state.time = self.config.t_start
        self.state.dt = self.config.dt_initial

        # Log initialization
        if self.config.verbose:
            self._log(f"Initialized {self.config.mode} TDE-SPH simulation")
            self._log(f"  Particles: {len(self.particles)}")
            self._log(f"  Output: {self.output_dir}")

    def _validate_config(self):
        """Validate configuration and component compatibility."""
        if self.config.mode == "GR" and self.metric is None:
            raise ValueError("GR mode requires a Metric instance")

        if self.config.mode not in ["Newtonian", "GR"]:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.config.verbose:
            print(f"[{self.state.time:.4f}] {message}")

    def compute_energies(self) -> Dict[str, float]:
        """
        Compute kinetic, potential, internal, and total energies.

        Implements REQ-009: energy accounting.

        Returns
        -------
        energies : Dict[str, float]
            Dictionary with 'kinetic', 'potential', 'internal', 'total' energies.
        """
        # Kinetic energy: 0.5 * Σ m_i |v_i|²
        v_mag_sq = np.sum(self.particles.velocities**2, axis=1)
        E_kin = 0.5 * np.sum(self.particles.masses * v_mag_sq)

        # Potential energy from gravity solver
        phi = self.gravity_solver.compute_potential(
            self.particles.positions,
            self.particles.masses,
            self.particles.smoothing_lengths
        )
        E_pot = 0.5 * np.sum(self.particles.masses * phi)  # Factor of 1/2 to avoid double-counting

        # Internal (thermal) energy: Σ m_i u_i
        E_int = np.sum(self.particles.masses * self.particles.internal_energy)

        # Total energy
        E_tot = E_kin + E_pot + E_int

        return {
            'kinetic': float(E_kin),
            'potential': float(E_pot),
            'internal': float(E_int),
            'total': float(E_tot),
        }

    def update_thermodynamics(self):
        """
        Update thermodynamic quantities (pressure, sound speed, temperature) from EOS.
        """
        self.particles.pressure = self.eos.pressure(
            self.particles.density,
            self.particles.internal_energy
        )
        self.particles.sound_speed = self.eos.sound_speed(
            self.particles.density,
            self.particles.internal_energy
        )
        # Temperature is optional but useful for diagnostics
        temperature = self.eos.temperature(
            self.particles.density,
            self.particles.internal_energy
        )
        # Store in particles if it has a temperature attribute
        if hasattr(self.particles, 'temperature'):
            self.particles.temperature = temperature

    def compute_forces(self) -> Dict[str, NDArrayFloat]:
        """
        Compute all forces acting on particles.

        Returns
        -------
        forces : Dict[str, NDArrayFloat]
            Dictionary with 'gravity', 'hydro', 'total' accelerations (N, 3).
        """
        # 1. Find neighbours and compute densities
        neighbour_data = find_neighbours_bruteforce(
            self.particles.positions,
            self.particles.smoothing_lengths,
            kernel=self.kernel
        )

        # Update densities
        self.particles.density = compute_density_summation(
            self.particles.positions,
            self.particles.masses,
            self.particles.smoothing_lengths,
            neighbour_data,
            kernel=self.kernel
        )

        # Update smoothing lengths (adaptive h)
        self.particles.smoothing_lengths = update_smoothing_lengths(
            self.particles.masses,
            self.particles.density,
            eta=self.config.smoothing_length_eta
        )

        # 2. Update thermodynamics (P, c_s from EOS)
        self.update_thermodynamics()

        # 3. Gravity
        a_grav = self.gravity_solver.compute_acceleration(
            self.particles.positions,
            self.particles.masses,
            self.particles.smoothing_lengths,
            metric=self.metric  # None for Newtonian
        )

        # 4. SPH hydrodynamics
        a_hydro, du_dt_hydro = compute_hydro_acceleration(
            self.particles.positions,
            self.particles.velocities,
            self.particles.masses,
            self.particles.density,
            self.particles.pressure,
            self.particles.internal_energy,
            self.particles.sound_speed,
            self.particles.smoothing_lengths,
            neighbour_data,
            kernel=self.kernel,
            alpha=self.config.artificial_viscosity_alpha,
            beta=self.config.artificial_viscosity_beta,
        )

        # Total acceleration
        a_total = a_grav + a_hydro

        return {
            'gravity': a_grav,
            'hydro': a_hydro,
            'total': a_total,
            'du_dt': du_dt_hydro,  # For energy update
        }

    def step(self):
        """
        Advance simulation by one timestep.
        """
        # Compute forces
        forces = self.compute_forces()

        # Advance particles
        self.integrator.step(
            self.particles,
            self.state.dt,
            forces
        )

        # Update time
        self.state.time += self.state.dt
        self.state.step += 1

        # Estimate next timestep
        self.state.dt = self.integrator.estimate_timestep(
            self.particles,
            cfl_factor=self.config.cfl_factor,
            accelerations=forces['total']
        )

        # Update energy diagnostics
        energies = self.compute_energies()
        self.state.kinetic_energy = energies['kinetic']
        self.state.potential_energy = energies['potential']
        self.state.internal_energy = energies['internal']
        self.state.total_energy = energies['total']

        if self.state.initial_energy is None:
            self.state.initial_energy = self.state.total_energy

    def write_snapshot(self):
        """
        Write current state to HDF5 snapshot.
        """
        from tde_sph.io import write_snapshot

        filename = self.output_dir / f"snapshot_{self.state.snapshot_count:04d}.h5"

        metadata = {
            'time': self.state.time,
            'step': self.state.step,
            'dt': self.state.dt,
            'mode': self.config.mode,
            'bh_mass': self.config.bh_mass,
            'bh_spin': self.config.bh_spin,
            'kinetic_energy': self.state.kinetic_energy,
            'potential_energy': self.state.potential_energy,
            'internal_energy': self.state.internal_energy,
            'total_energy': self.state.total_energy,
        }

        write_snapshot(str(filename), self.particles, self.state.time, metadata)

        self.state.last_snapshot_time = self.state.time
        self.state.snapshot_count += 1

        if self.config.verbose:
            self._log(f"Snapshot {self.state.snapshot_count} -> {filename.name}")

    def check_energy_conservation(self) -> bool:
        """
        Check if energy is conserved within tolerance.

        Returns
        -------
        conserved : bool
            True if energy drift is within tolerance.
        """
        if self.state.initial_energy is None or self.state.initial_energy == 0:
            return True

        drift = abs(self.state.total_energy - self.state.initial_energy) / abs(self.state.initial_energy)

        if drift > self.config.energy_tolerance:
            self._log(f"WARNING: Energy drift {drift:.2%} exceeds tolerance {self.config.energy_tolerance:.2%}")
            return False

        return True

    def run(self):
        """
        Run the simulation from t_start to t_end.
        """
        self._log("=" * 60)
        self._log("Starting simulation")
        self._log("=" * 60)

        # Initial snapshot
        self.write_snapshot()

        # Initial energies
        energies = self.compute_energies()
        self.state.initial_energy = energies['total']
        self._log(f"Initial energy: {self.state.initial_energy:.6e}")

        # Main loop
        while self.state.time < self.config.t_end:
            # Step
            self.step()

            # Periodic logging
            if self.state.step % int(self.config.log_interval / self.state.dt + 1) == 0:
                drift = 0.0
                if self.state.initial_energy != 0:
                    drift = (self.state.total_energy - self.state.initial_energy) / self.state.initial_energy

                self._log(
                    f"Step {self.state.step:6d}  "
                    f"t={self.state.time:.4f}  "
                    f"dt={self.state.dt:.2e}  "
                    f"E_tot={self.state.total_energy:.6e}  "
                    f"ΔE/E={drift:.2e}"
                )

            # Periodic snapshots
            if self.state.time - self.state.last_snapshot_time >= self.config.snapshot_interval:
                self.write_snapshot()

            # Check energy conservation
            self.check_energy_conservation()

            # Safety: prevent runaway
            if self.state.step > 1e7:
                self._log("WARNING: Maximum step count reached")
                break

        # Final snapshot
        self.write_snapshot()

        # Summary
        self.state.wall_time_elapsed = time_module.time() - self.state.wall_time_start
        self._log("=" * 60)
        self._log("Simulation complete")
        self._log(f"  Steps: {self.state.step}")
        self._log(f"  Final time: {self.state.time:.4f}")
        self._log(f"  Wall time: {self.state.wall_time_elapsed:.2f} s")
        self._log(f"  Snapshots: {self.state.snapshot_count}")
        self._log("=" * 60)
