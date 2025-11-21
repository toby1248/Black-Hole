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
import warnings
import numpy as np
import numpy.typing as npt
from pathlib import Path
import time as time_module
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

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


class SimulationConfig(BaseModel):
    """
    Configuration for TDE-SPH simulation with Pydantic validation.

    Supports both Newtonian and GR modes with comprehensive validation
    to ensure parameter consistency (GUD-002, TASK-015).

    Attributes
    ----------
    mode : str
        Simulation mode: "Newtonian" or "GR"
    metric_type : str
        Spacetime metric type: "minkowski", "schwarzschild", or "kerr"
    bh_mass : float
        Black hole mass in code units (G=c=M_BH=1)
    bh_spin : float
        Dimensionless black hole spin parameter a/M ∈ [0, 1]
    use_hamiltonian_integrator : bool
        Use Hamiltonian integrator for GR orbits near ISCO
    isco_radius_threshold : float
        Radius threshold for switching to Hamiltonian integrator (in M)
    cfl_factor_gr : float
        CFL factor for GR mode (stricter than Newtonian)
    orbital_timestep_factor : float
        Timestep as fraction of orbital period
    coordinate_system : str
        Coordinate system: "cartesian" or "boyer-lindquist"
    use_fp64_for_metric : bool
        Use FP64 precision for metric computations near horizon
    """

    # Time evolution
    t_start: float = Field(default=0.0, ge=0.0, description="Start time")
    t_end: float = Field(default=10.0, gt=0.0, description="End time")
    dt_initial: float = Field(default=0.001, gt=0.0, description="Initial timestep")
    cfl_factor: float = Field(default=0.3, gt=0.0, le=1.0, description="CFL safety factor")

    # I/O
    output_dir: str = Field(default="output", description="Output directory path")
    snapshot_interval: float = Field(default=0.1, gt=0.0, description="Snapshot output interval")
    log_interval: float = Field(default=0.01, gt=0.0, description="Log output interval")

    # Physics mode (TASK-015)
    mode: str = Field(
        default="Newtonian",
        description="Simulation mode: 'Newtonian' or 'GR'"
    )

    # Black hole parameters
    bh_mass: float = Field(
        default=1.0,
        gt=0.0,
        description="Black hole mass in code units (G=c=M_BH=1)"
    )
    bh_spin: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dimensionless black hole spin a/M ∈ [0, 1]"
    )

    # Metric specification (TASK-015)
    metric_type: str = Field(
        default="schwarzschild",
        description="Metric type: 'minkowski', 'schwarzschild', or 'kerr'"
    )

    # GR integration control (TASK-015)
    use_hamiltonian_integrator: bool = Field(
        default=False,
        description="Use Hamiltonian integrator for GR orbits near ISCO"
    )
    isco_radius_threshold: float = Field(
        default=10.0,
        gt=0.0,
        description="Radius threshold (in M) for Hamiltonian integrator"
    )

    # GR-specific timestep control
    cfl_factor_gr: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Stricter CFL factor for GR mode"
    )
    orbital_timestep_factor: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Timestep as fraction of orbital period"
    )

    # Coordinate system
    coordinate_system: str = Field(
        default="cartesian",
        description="Coordinate system: 'cartesian' or 'boyer-lindquist'"
    )

    # Precision control
    use_fp64_for_metric: bool = Field(
        default=True,
        description="Use FP64 precision for metric computations near horizon"
    )

    # SPH parameters
    neighbour_search_method: str = Field(
        default="bruteforce",
        description="Neighbour search method: 'bruteforce' or 'tree'"
    )
    smoothing_length_eta: float = Field(
        default=1.2,
        gt=0.0,
        description="Smoothing length parameter η"
    )
    artificial_viscosity_alpha: float = Field(
        default=1.0,
        ge=0.0,
        description="Artificial viscosity α parameter"
    )
    artificial_viscosity_beta: float = Field(
        default=2.0,
        ge=0.0,
        description="Artificial viscosity β parameter"
    )

    # Energy tracking
    energy_tolerance: float = Field(
        default=0.01,
        gt=0.0,
        description="Fractional energy drift tolerance"
    )

    # Misc
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    verbose: bool = Field(
        default=True,
        description="Enable verbose logging"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Raise error on unknown fields
    )

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate simulation mode."""
        valid_modes = ["Newtonian", "GR"]
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{v}'")
        return v

    @field_validator('metric_type')
    @classmethod
    def validate_metric_type(cls, v: str) -> str:
        """Validate metric type."""
        valid_metrics = ["minkowski", "schwarzschild", "kerr"]
        if v not in valid_metrics:
            raise ValueError(f"metric_type must be one of {valid_metrics}, got '{v}'")
        return v

    @field_validator('coordinate_system')
    @classmethod
    def validate_coordinate_system(cls, v: str) -> str:
        """Validate coordinate system."""
        valid_systems = ["cartesian", "boyer-lindquist"]
        if v not in valid_systems:
            raise ValueError(f"coordinate_system must be one of {valid_systems}, got '{v}'")
        return v

    @field_validator('neighbour_search_method')
    @classmethod
    def validate_neighbour_search(cls, v: str) -> str:
        """Validate neighbour search method."""
        valid_methods = ["bruteforce", "tree"]
        if v not in valid_methods:
            raise ValueError(f"neighbour_search_method must be one of {valid_methods}, got '{v}'")
        return v

    @model_validator(mode='after')
    def validate_configuration_consistency(self):
        """
        Cross-field validation to ensure parameter consistency.

        Implements validation rules from TASK-015:
        1. GR mode requires metric_type
        2. Kerr metric should have non-zero spin
        3. Spin bounds [0, 1]
        4. Hamiltonian integrator only valid in GR mode
        5. ISCO threshold sensible
        """
        # Rule 1: GR mode requires metric (always has one, but check it's not Minkowski by accident)
        if self.mode == "GR" and self.metric_type == "minkowski":
            warnings.warn(
                "GR mode with Minkowski metric is equivalent to flat spacetime. "
                "Consider using 'Newtonian' mode or 'schwarzschild'/'kerr' metric."
            )

        # Rule 2: Kerr with zero spin is equivalent to Schwarzschild
        if self.metric_type == "kerr" and self.bh_spin == 0.0:
            warnings.warn(
                "Kerr metric with a=0 is equivalent to Schwarzschild. "
                "Consider setting metric_type='schwarzschild' or bh_spin > 0."
            )

        # Rule 3: Schwarzschild with non-zero spin is inconsistent
        if self.metric_type == "schwarzschild" and self.bh_spin > 0.0:
            warnings.warn(
                f"Schwarzschild metric with non-zero spin (a={self.bh_spin}) is inconsistent. "
                "Spin will be ignored. Consider using metric_type='kerr'."
            )

        # Rule 4: Hamiltonian integrator requires GR mode
        if self.use_hamiltonian_integrator and self.mode != "GR":
            raise ValueError(
                "Hamiltonian integrator (use_hamiltonian_integrator=True) "
                "only valid in GR mode, but mode='{self.mode}'"
            )

        # Rule 5: ISCO threshold warning
        if self.isco_radius_threshold < 3.0:
            warnings.warn(
                f"ISCO threshold {self.isco_radius_threshold}M is below photon orbit (3M). "
                "This may cause timestep issues or particle capture."
            )

        # Rule 6: t_end > t_start
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must be greater than t_start ({self.t_start})")

        return self


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
            self._log(f"  Particles: {self.particles.n_particles}")
            self._log(f"  Output: {self.output_dir}")
            if self.config.mode == "GR":
                self._log(f"  Metric: {self.metric.name if self.metric else 'None'}")
                self._log(f"  BH mass: {self.config.bh_mass} (code units)")
                self._log(f"  BH spin: {self.config.bh_spin}")
                self._log(f"  Hamiltonian integrator: {self.config.use_hamiltonian_integrator}")
                if self.config.use_hamiltonian_integrator:
                    self._log(f"  ISCO threshold: {self.config.isco_radius_threshold} M")

    def _validate_config(self):
        """
        Validate configuration and component compatibility.

        Implements TASK-015b validation requirements.
        """
        # GR mode requires a metric
        if self.config.mode == "GR" and self.metric is None:
            raise ValueError(
                "GR mode requires a Metric instance. "
                "Pass a Metric object (Schwarzschild, Kerr, etc.) to Simulation.__init__"
            )

        # Newtonian mode should not have a metric (or it will be ignored)
        if self.config.mode == "Newtonian" and self.metric is not None:
            warnings.warn(
                "Newtonian mode with a Metric instance: metric will be ignored. "
                "Set mode='GR' to use relativistic dynamics."
            )

        # Warn if Hamiltonian integrator requested but metric is None
        if self.config.use_hamiltonian_integrator and self.metric is None:
            raise ValueError(
                "Hamiltonian integrator requires a Metric instance for GR orbits"
            )

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
        self.particles.temperature = self.eos.temperature(
            self.particles.density,
            self.particles.internal_energy
        )

    def compute_forces(self) -> Dict[str, NDArrayFloat]:
        """
        Compute all forces acting on particles.

        Returns
        -------
        forces : Dict[str, NDArrayFloat]
            Dictionary with 'gravity', 'hydro', 'total' accelerations (N, 3).
        """
        # 1. Find neighbours and compute densities
        neighbour_lists, _ = find_neighbours_bruteforce(
            self.particles.positions,
            self.particles.smoothing_lengths
        )

        # Update densities
        self.particles.density = compute_density_summation(
            self.particles.positions,
            self.particles.masses,
            self.particles.smoothing_lengths,
            neighbour_lists,
            kernel_func=self.kernel.kernel
        )

        # Update smoothing lengths (adaptive h)
        self.particles.smoothing_lengths = update_smoothing_lengths(
            self.particles.positions,
            self.particles.masses,
            self.particles.smoothing_lengths
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
            self.particles.sound_speed,
            self.particles.smoothing_lengths,
            neighbour_lists,
            kernel_gradient_func=self.kernel.kernel_gradient,
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

        # Estimate next timestep (use GR-specific CFL factor if in GR mode)
        cfl_factor = self.config.cfl_factor_gr if self.config.mode == "GR" else self.config.cfl_factor
        self.state.dt = self.integrator.estimate_timestep(
            self.particles,
            cfl_factor=cfl_factor,
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
            'metric_type': self.config.metric_type,
            'bh_mass': self.config.bh_mass,
            'bh_spin': self.config.bh_spin,
            'coordinate_system': self.config.coordinate_system,
            'use_hamiltonian_integrator': self.config.use_hamiltonian_integrator,
            'kinetic_energy': self.state.kinetic_energy,
            'potential_energy': self.state.potential_energy,
            'internal_energy': self.state.internal_energy,
            'total_energy': self.state.total_energy,
        }

        # Convert ParticleSystem to dict of arrays
        particle_data = {
            'positions': self.particles.positions,
            'velocities': self.particles.velocities,
            'masses': self.particles.masses,
            'density': self.particles.density,
            'internal_energy': self.particles.internal_energy,
            'smoothing_length': self.particles.smoothing_length,
            'pressure': self.particles.pressure,
            'sound_speed': self.particles.sound_speed,
            'temperature': self.particles.temperature,
        }

        write_snapshot(str(filename), particle_data, self.state.time, metadata)

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
