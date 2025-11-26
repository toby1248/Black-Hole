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

from typing import Dict, Optional
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
    find_neighbours_octree,
    compute_density_summation,
    update_smoothing_lengths,
    compute_hydro_acceleration,
)
# GPU imports
try:
    from tde_sph.gpu import (
        HAS_CUDA,
        GPUManager,
        compute_density_gpu,
        compute_hydro_gpu,
        update_smoothing_lengths_gpu
    )
except ImportError:
    HAS_CUDA = False
    GPUManager = None


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
    dt_min: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Absolute minimum allowable timestep"
    )
    dt_max: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Absolute maximum allowable timestep"
    )
    dt_change_limit: float = Field(
        default=4.0,
        ge=1.0,
        description="Maximum factor by which dt may change between consecutive steps"
    )
    cfl_factor: float = Field(default=1.0, gt=0.0, le=1.0, description="CFL safety factor")

    # I/O
    output_dir: str = Field(default="outputs/default_run", description="Output directory path")
    snapshot_interval: float = Field(default=0.01, gt=0.0, description="Snapshot output interval")
    log_interval: float = Field(default=0.001, gt=0.0, description="Log output interval")

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
        default=0.02,
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

        # Timestep bounds and limits
        if self.dt_min is None:
            default_min = max(self.dt_initial * 1e-2, 1e-12)
            object.__setattr__(self, 'dt_min', default_min)
        if self.dt_max is None:
            object.__setattr__(self, 'dt_max', self.dt_initial * 1e2)

        if self.dt_max <= self.dt_min:
            raise ValueError(
                f"dt_max ({self.dt_max}) must be greater than dt_min ({self.dt_min})"
            )

        if not (self.dt_min <= self.dt_initial <= self.dt_max):
            raise ValueError(
                "dt_initial must lie within [dt_min, dt_max]; "
                f"got dt_initial={self.dt_initial}, dt_min={self.dt_min}, dt_max={self.dt_max}"
            )

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
    last_dt_candidate: float = 0.0
    last_dt_limiter: str = "none"

    # Timing diagnostics
    timing_gravity: float = 0.0
    timing_sph_density: float = 0.0
    timing_smoothing_lengths: float = 0.0  # Adaptive h updates
    timing_sph_pressure: float = 0.0
    timing_integration: float = 0.0
    timing_energy_computation: float = 0.0  # Renamed from timing_diagnostics
    timing_timestep_estimation: float = 0.0  # Averaged over steps between UI updates
    timing_thermodynamics: float = 0.0  # EOS updates (P, cs, T)
    timing_gpu_transfer: float = 0.0  # CPU<->GPU data movement
    timing_compute_forces: float = 0.0  # Total time for compute_forces()
    timing_io: float = 0.0
    timing_other: float = 0.0  # Remaining overhead
    timing_total: float = 0.0
    
    # Timing accumulation for averaging
    _timestep_est_accumulator: float = 0.0  # Sum of timestep estimation timings
    _timestep_est_count: int = 0  # Number of steps accumulated

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
        use_gpu: Optional[bool] = None,
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
        use_gpu : Optional[bool], default None
            Whether to use GPU acceleration if available. If None, defaults to True when CUDA is detected.
        """
        self.particles = particles
        self.gravity_solver = gravity_solver
        self.eos = eos
        self.integrator = integrator
        self.metric = metric
        self.radiation_model = radiation_model

        self.config = config or SimulationConfig()
        self.state = SimulationState()

        # Ensure derived diagnostics start from consistent state
        self._update_velocity_magnitude()

        # Initialize adaptive smoothing length manager for performance
        from ..sph.smoothing_adaptive import AdaptiveSmoothingManager
        self.smoothing_manager = AdaptiveSmoothingManager(
            eta=1.2,
            update_interval=10,  # Update every 10 steps
            density_change_threshold=0.2  # Update if density changes >20%
        )

        # Ensure gravity solver is aware of configured BH parameters if supported
        self._configure_gravity_solver()

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

        # Ensure smoothing lengths are self-consistent before first snapshot
        if self.particles.n_particles > 0:
            self._initialize_smoothing_lengths()

        # Initialize state
        self.state.time = self.config.t_start
        self.state.dt = self.config.dt_initial

        # GPU Setup - Default to enabled when CUDA is detected
        if use_gpu is None:
            use_gpu = HAS_CUDA  # Auto-detect: enable GPU by default when CUDA available
        
        self.use_gpu = use_gpu and HAS_CUDA
        self.gpu_manager = None
        if self.use_gpu:
            try:
                self.gpu_manager = GPUManager(self.particles)
                self._log("GPU acceleration enabled (CUDA detected and initialized)")
            except Exception as e:
                self._log(f"GPU initialization failed: {e}")
                self.use_gpu = False
        elif HAS_CUDA and not use_gpu:
            self._log("CUDA detected but GPU acceleration explicitly disabled")
        elif not HAS_CUDA:
            self._log("GPU acceleration unavailable (CUDA not detected)")

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

    def _configure_gravity_solver(self) -> None:
        """Synchronize config metadata (e.g., BH mass) with gravity solver."""
        if not hasattr(self, 'gravity_solver'):
            return

        solver = self.gravity_solver
        target_mass = float(getattr(self.config, 'bh_mass', 0.0))

        if hasattr(solver, 'bh_mass'):
            solver_mass = float(getattr(solver, 'bh_mass', 0.0))
            if not np.isclose(solver_mass, target_mass):
                setattr(solver, 'bh_mass', np.float32(target_mass))
                if getattr(self.config, 'verbose', False):
                    self._log(
                        "Gravity solver bh_mass updated to match config: "
                        f"{target_mass:.3e}"
                    )

    def _initialize_smoothing_lengths(self) -> None:
        """
        Fast initial smoothing-length solve using GPU or octree when available.

        Falls back to the legacy O(N^2) routine only for modest particle counts
        to avoid prohibitive startup times at N ≥ 1e5.
        """
        n_particles = self.particles.n_particles
        if n_particles == 0:
            return

        # Prefer GPU path (O(N) with cached neighbour counts)
        if self.use_gpu and 'update_smoothing_lengths_gpu' in globals():
            try:
                self.particles.smoothing_lengths = update_smoothing_lengths_gpu(
                    self.particles.positions,
                    self.particles.smoothing_lengths
                ).astype(np.float32)
                if self.config.verbose:
                    h0 = self.particles.smoothing_lengths
                    self._log(
                        "Initial smoothing-lengths (GPU): "
                        f"[{float(np.min(h0)):.3e}, {float(np.max(h0)):.3e}]"
                    )
                return
            except Exception as exc:  # pragma: no cover - defensive
                if self.config.verbose:
                    self._log(f"GPU smoothing-length init failed; falling back: {exc}")

        # Octree-based path for moderate N (avoids O(N^2) brute force)
        if self.config.neighbour_search_method == "tree" and n_particles <= 200_000:
            try:
                from ..gravity.barnes_hut import BarnesHutGravity

                theta = getattr(self.gravity_solver, 'theta', 0.5)
                bh_solver = BarnesHutGravity(theta=theta)
                # Build tree once (compute_acceleration populates tree_data)
                bh_solver.compute_acceleration(
                    self.particles.positions.astype(np.float32),
                    self.particles.masses.astype(np.float32),
                    self.particles.smoothing_lengths.astype(np.float32),
                    metric=None
                )
                tree_data = bh_solver.get_tree_data()
                if tree_data is not None:
                    h_new = self._smoothing_from_tree(
                        self.particles.positions,
                        self.particles.masses,
                        self.particles.smoothing_lengths,
                        tree_data
                    )
                    self.particles.smoothing_lengths = h_new
                    if self.config.verbose:
                        self._log(
                            "Initial smoothing-lengths (TreeSPH): "
                            f"[{float(np.min(h_new)):.3e}, {float(np.max(h_new)):.3e}]"
                        )
                    return
            except Exception as exc:  # pragma: no cover - defensive
                if self.config.verbose:
                    self._log(f"Tree-based smoothing init failed; fallback: {exc}")

        # Legacy brute-force only for smaller systems (avoid runaway O(N^2))
        if n_particles <= 50_000:
            self.particles.smoothing_lengths = update_smoothing_lengths(
                self.particles.positions,
                self.particles.masses,
                self.particles.smoothing_lengths
            )
            if self.config.verbose:
                h0 = self.particles.smoothing_lengths
                self._log(
                    "Initial smoothing-length range: "
                    f"[{float(np.min(h0)):.3e}, {float(np.max(h0)):.3e}]"
                )
        else:
            # Keep provided h to avoid O(N^2) startup; downstream adaptive manager will refine.
            if self.config.verbose:
                h0 = self.particles.smoothing_lengths
                self._log(
                    "Skipping costly initial smoothing-length solve "
                    f"(N={n_particles:,}); using provided h range "
                    f"[{float(np.min(h0)):.3e}, {float(np.max(h0)):.3e}]"
                )

    def _smoothing_from_tree(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        h_initial: NDArrayFloat,
        tree_data: Dict[str, np.ndarray],
        target_neighbours: int = 50,
        max_iterations: int = 3,
        tolerance: float = 0.05,
    ) -> NDArrayFloat:
        """Iteratively adjust h using octree neighbour counts (O(N log N))."""
        h_new = np.asarray(h_initial, dtype=np.float32).copy()
        if tree_data is None:
            return h_new

        for _ in range(max_iterations):
            neighbour_lists, _ = find_neighbours_octree(
                positions.astype(np.float32),
                h_new,
                tree_data
            )
            counts = np.array([len(nb) for nb in neighbour_lists], dtype=np.float32)
            counts = np.maximum(counts, 1.0)
            ratio = counts / float(target_neighbours)
            h_new *= np.power(ratio, -1.0 / 3.0, dtype=np.float32)

            frac_err = np.max(np.abs(counts - target_neighbours) / target_neighbours)
            if frac_err < tolerance:
                break

        return h_new.astype(np.float32)

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
        
        Uses O(N log N) Barnes-Hut tree for potential energy calculation when available,
        falling back to direct O(N²) summation from gravity solver otherwise.

        Returns
        -------
        energies : Dict[str, float]
            Dictionary with 'kinetic', 'potential', 'internal', 'total' energies.
        """
        # Kinetic energy: 0.5 * Σ m_i |v_i|²
        v_mag_sq = np.sum(self.particles.velocities**2, axis=1)
        E_kin = 0.5 * np.sum(self.particles.masses * v_mag_sq)

        # Potential energy - Use Barnes-Hut O(N log N) if available for performance
        try:
            from ..gravity.barnes_hut import BarnesHutGravity
            
            # Check if we can use Barnes-Hut (either current solver is BH or we can create one)
            if isinstance(self.gravity_solver, BarnesHutGravity):
                # Use existing Barnes-Hut solver
                phi = self.gravity_solver.compute_potential(
                    self.particles.positions,
                    self.particles.masses,
                    self.particles.smoothing_lengths
                )
            else:
                # Create temporary Barnes-Hut solver for energy calculation only
                # Use same G as current gravity solver
                G = getattr(self.gravity_solver, 'G', 1.0)
                bh_solver = BarnesHutGravity(G=G, theta=0.5)
                phi = bh_solver.compute_potential(
                    self.particles.positions,
                    self.particles.masses,
                    self.particles.smoothing_lengths
                )
        except (ImportError, Exception):
            # Fallback to direct summation from current gravity solver
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
        # Apply density floor to prevent issues
        density_safe = np.maximum(self.particles.density, 1e-20)
        
        self.particles.pressure = self.eos.pressure(
            density_safe,
            self.particles.internal_energy
        )
        # Ensure pressure is non-negative
        self.particles.pressure = np.maximum(self.particles.pressure, 0.0)
        
        self.particles.sound_speed = self.eos.sound_speed(
            density_safe,
            self.particles.internal_energy
        )
        # Apply sound speed floor to prevent zero/NaN
        self.particles.sound_speed = np.maximum(self.particles.sound_speed, 1e-6)
        
        # Check for NaN in sound speed (shouldn't happen with floors, but be safe)
        if not np.all(np.isfinite(self.particles.sound_speed)):
            nan_count = np.sum(~np.isfinite(self.particles.sound_speed))
            self._log(f"WARNING: {nan_count} NaN sound speeds detected, replacing with minimum")
            self.particles.sound_speed = np.where(
                np.isfinite(self.particles.sound_speed),
                self.particles.sound_speed,
                1e-6
            )
        
        # Temperature is tracked for diagnostics and I/O (T.1)
        temperature = self.eos.temperature(
            density_safe,
            self.particles.internal_energy
        )
        self.particles.temperature = temperature

    def _compute_radiation_du_dt(self) -> Optional[NDArrayFloat]:
        """
        Compute radiative cooling/heating rate du/dt using attached radiation model.
        """
        if self.radiation_model is None:
            return None

        try:
            rho = np.asarray(self.particles.density, dtype=np.float32)
            T = np.asarray(self.particles.temperature, dtype=np.float32)
            u = np.asarray(self.particles.internal_energy, dtype=np.float32)
            m = np.asarray(self.particles.masses, dtype=np.float32)
            h = np.asarray(self.particles.smoothing_lengths, dtype=np.float32)

            rad = self.radiation_model

            if hasattr(rad, "compute_net_rates"):
                rates = rad.compute_net_rates(
                    density=rho,
                    temperature=T,
                    internal_energy=u,
                    masses=m,
                    smoothing_lengths=h,
                )
                return np.asarray(rates.du_dt_net, dtype=np.float32)

            if hasattr(rad, "compute_cooling_rate"):
                return np.asarray(
                    rad.compute_cooling_rate(
                        density=rho,
                        temperature=T,
                        internal_energy=u,
                        masses=m,
                        smoothing_lengths=h,
                    ),
                    dtype=np.float32,
                )

            if hasattr(rad, "cooling_rate"):
                return np.asarray(
                    rad.cooling_rate(
                        density=rho,
                        temperature=T,
                        internal_energy=u,
                        masses=m,
                        smoothing_lengths=h,
                    ),
                    dtype=np.float32,
                )
        except Exception as exc:  # pragma: no cover - defensive
            if self.config.verbose:
                self._log(f"Radiation cooling failed, ignoring this step: {exc}")

        return None

    def _update_velocity_magnitude(self) -> None:
        """Cache |v| for diagnostics and snapshot output."""
        velocities = np.asarray(self.particles.velocities, dtype=np.float32)
        self.particles.velocity_magnitude = np.linalg.norm(velocities, axis=1).astype(np.float32)

    def compute_forces(self) -> Dict[str, NDArrayFloat]:
        """
        Compute all forces acting on particles.

        Returns
        -------
        forces : Dict[str, NDArrayFloat]
            Dictionary with 'gravity', 'hydro', 'total' accelerations (N, 3).
        """
        t0_forces_total = time_module.time()
        
        if self.use_gpu:
            result = self._compute_forces_gpu()
            self.state.timing_compute_forces = time_module.time() - t0_forces_total
            return result

        # 1. Gravity (computed first to build octree for TreeSPH)
        t0_grav = time_module.time()
        # Only pass velocities for GR mode (geodesic acceleration)
        if self.config.mode == "GR":
            a_grav = self.gravity_solver.compute_acceleration(
                self.particles.positions,
                self.particles.masses,
                self.particles.smoothing_lengths,
                metric=self.metric,
                velocities=self.particles.velocities
            )
        else:
            a_grav = self.gravity_solver.compute_acceleration(
                self.particles.positions,
                self.particles.masses,
                self.particles.smoothing_lengths,
                metric=self.metric
            )
        self.state.timing_gravity = time_module.time() - t0_grav

        # 2. Find neighbours - use TreeSPH if octree is available
        t0_density = time_module.time()
        
        # Check if we can use octree neighbour search (TreeSPH approach)
        tree_data = None
        try:
            from ..gravity.barnes_hut import BarnesHutGravity
            from ..gravity.barnes_hut_gpu import BarnesHutGravityGPU
            if isinstance(self.gravity_solver, (BarnesHutGravity, BarnesHutGravityGPU)):
                tree_data = self.gravity_solver.get_tree_data()
        except ImportError:
            pass
        
        if tree_data is not None and tree_data.get('gpu', False):
            # Use GPU octree neighbour search (10x+ speedup)
            from ..sph.neighbours_gpu import find_neighbours_octree_gpu_integrated
            neighbour_lists, _ = find_neighbours_octree_gpu_integrated(
                self.particles.positions,
                self.particles.smoothing_lengths,
                tree_data,
                support_radius=2.0,
                max_neighbours=64
            )
        elif tree_data is not None:
            # Use CPU octree neighbour search
            neighbour_lists, _ = find_neighbours_octree(
                self.particles.positions,
                self.particles.smoothing_lengths,
                tree_data
            )
        else:
            # Fall back to O(N^2) brute force
            neighbour_lists, _ = find_neighbours_bruteforce(
                self.particles.positions,
                self.particles.smoothing_lengths
            )

        # 3. Update densities using the latest neighbour topology
        self.particles.density = compute_density_summation(
            self.particles.positions,
            self.particles.masses,
            self.particles.smoothing_lengths,
            neighbour_lists,
            kernel_func=self.kernel.kernel
        )
        self.state.timing_sph_density = time_module.time() - t0_density

        # 4. Adaptive smoothing length update (uses density, very fast O(N))
        t0_h = time_module.time()
        h_new, did_update = self.smoothing_manager.update(
            self.particles.masses,
            self.particles.density,
            self.particles.smoothing_lengths,
            self.state.step
        )
        self.particles.smoothing_lengths = h_new
        self.state.timing_smoothing_lengths = time_module.time() - t0_h

        if self.config.verbose and did_update:
            h_vals = self.particles.smoothing_lengths
            h_min = float(np.min(h_vals))
            h_max = float(np.max(h_vals))
            near_floor = int(np.count_nonzero(h_vals <= 1e-5))
            self._log(
                "Adaptive h updated -> range: "
                f"[{h_min:.3e}, {h_max:.3e}] (<=1e-5: {near_floor})"
            )

        # 5. Update thermodynamics (P, c_s from EOS)
        t0_thermo = time_module.time()
        self.update_thermodynamics()
        self.state.timing_thermodynamics = time_module.time() - t0_thermo

        # 6. SPH hydrodynamics
        t0_pressure = time_module.time()
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
        self.state.timing_sph_pressure = time_module.time() - t0_pressure

        # Radiation cooling/heating (additive to hydro du/dt)
        du_dt_total = du_dt_hydro
        if self.radiation_model is not None:
            du_dt_rad = self._compute_radiation_du_dt()
            if du_dt_rad is not None:
                du_dt_total = du_dt_hydro + du_dt_rad

        # Total acceleration
        a_total = a_grav + a_hydro
        if not np.all(np.isfinite(a_total)):
            self._log("ERROR: Accelerations contain NaN or inf")
            raise ValueError("Invalid accelerations")
        
        # Total compute_forces time
        self.state.timing_compute_forces = time_module.time() - t0_forces_total

        return {
            'gravity': a_grav,
            'hydro': a_hydro,
            'total': a_total,
            'du_dt': du_dt_total,  # For energy update
        }

    def _compute_forces_gpu(self) -> Dict[str, NDArrayFloat]:
        """Compute forces using GPU acceleration with TreeSPH (octree-based neighbours)."""
        import cupy as cp
        from tde_sph.gpu import compute_density_treesph, compute_hydro_treesph
        
        # Only update positions and velocities if they changed
        # (don't re-transfer data that's already on GPU)
        if not hasattr(self.gpu_manager, 'data_on_gpu') or not self.gpu_manager.data_on_gpu:
            t0_transfer = time_module.time()
            self.gpu_manager.sync_to_device(self.particles)
            self.state.timing_gpu_transfer = time_module.time() - t0_transfer
        else:
            # Just update what changed from CPU side (if anything)
            t0_transfer = time_module.time()
            # Typically only positions and velocities change
            self.gpu_manager.update_from_cpu(self.particles, fields=['positions', 'velocities'])
            self.state.timing_gpu_transfer = time_module.time() - t0_transfer
        
        # 1. Build/reuse octree for gravity and neighbour search
        theta = self.config.__dict__.get('barnes_hut_theta', 0.5)
        if hasattr(self.gravity_solver, 'theta'):
            theta = self.gravity_solver.theta
        self.gpu_manager.build_octree(theta=theta)  # Smart rebuilding - only when needed
        
        # 2. Compute neighbour lists ONCE (reused for both density and hydro)
        support_radius = 2.0  # Standard SPH support radius
        max_neighbours = 64
        neighbour_lists, neighbour_counts = self.gpu_manager.compute_neighbours(
            support_radius=support_radius,
            max_neighbours=max_neighbours
        )
        neighbour_lists_host = None
        if compute_density_treesph is None:
            neighbour_lists_host, _ = self.gpu_manager.get_neighbour_lists_host()
            if neighbour_lists_host is None:
                raise RuntimeError("GPU neighbour lists not available for CPU fallback")

        # 4. Compute Density using TreeSPH (octree neighbours)
        t0_density = time_module.time()
        if compute_density_treesph is not None:
            self.gpu_manager.rho = compute_density_treesph(
                self.gpu_manager.pos,
                self.gpu_manager.mass,
                self.gpu_manager.h,
                neighbour_lists,
                neighbour_counts
            )
            t0_transfer = time_module.time()
            density_cpu = cp.asnumpy(self.gpu_manager.rho)
            self.state.timing_gpu_transfer += time_module.time() - t0_transfer
        else:
            density_cpu = compute_density_summation(
                self.particles.positions,
                self.particles.masses,
                self.particles.smoothing_lengths,
                neighbour_lists_host,
                kernel_func=self.kernel.kernel
            )

            t0_transfer = time_module.time()
            self.gpu_manager.rho = cp.asarray(density_cpu, dtype=cp.float32)
            self.state.timing_gpu_transfer += time_module.time() - t0_transfer

        self.particles.density = density_cpu
        self.state.timing_sph_density = time_module.time() - t0_density
        density_cpu = self.particles.density

        # 4b. Adaptive smoothing length update (reuse CPU smoothing manager)
        t0_h = time_module.time()
        h_new, did_update = self.smoothing_manager.update(
            self.particles.masses,
            self.particles.density,
            self.particles.smoothing_lengths,
            self.state.step
        )
        self.particles.smoothing_lengths = h_new
        self.state.timing_smoothing_lengths = time_module.time() - t0_h

        if did_update:
            t0_transfer = time_module.time()
            self.gpu_manager.h = cp.asarray(h_new, dtype=cp.float32)
            self.gpu_manager.invalidate_neighbours()
            self.state.timing_gpu_transfer += time_module.time() - t0_transfer

            if self.config.verbose:
                h_vals = self.particles.smoothing_lengths
                h_min = float(np.min(h_vals))
                h_max = float(np.max(h_vals))
                near_floor = int(np.count_nonzero(h_vals <= 1e-5))
                self._log(
                    "Adaptive h updated -> range: "
                    f"[{h_min:.3e}, {h_max:.3e}] (<=1e-5: {near_floor})"
                )
        
        t0_thermo = time_module.time()
        self.update_thermodynamics()
        self.state.timing_thermodynamics = time_module.time() - t0_thermo
        
        # Transfer P, cs back to GPU
        t0_transfer = time_module.time()
        self.gpu_manager.pressure = cp.asarray(self.particles.pressure, dtype=cp.float32)
        self.gpu_manager.cs = cp.asarray(self.particles.sound_speed, dtype=cp.float32)
        self.state.timing_gpu_transfer += time_module.time() - t0_transfer
        
        # 6. Compute Gravity using Barnes-Hut octree
        t0_grav = time_module.time()
        G = 1.0
        if hasattr(self.gravity_solver, 'G'):
            G = float(self.gravity_solver.G)
        
        epsilon = 0.1  # Softening parameter
        if hasattr(self.gravity_solver, 'epsilon'):
            epsilon = float(self.gravity_solver.epsilon)
            
        # Use octree for self-gravity calculation (particle-particle)
        octree = self.gpu_manager.get_octree()
        forces_gpu = octree.compute_gravity(
            self.gpu_manager.pos,
            self.gpu_manager.mass,
            self.gpu_manager.h,
            G=G,
            epsilon=epsilon
        )
        
        # Convert forces to accelerations (a = F/m)
        self.gpu_manager.acc_grav = forces_gpu / self.gpu_manager.mass[:, cp.newaxis]
        
        # Add black hole gravity (GR or Newtonian)
        if self.metric is not None:
            # GR mode: Use geodesic acceleration from metric
            # Transfer positions and velocities to CPU for BH gravity computation
            pos_cpu = cp.asnumpy(self.gpu_manager.pos)
            vel_cpu = cp.asnumpy(self.gpu_manager.vel)
            
            # Compute BH geodesic acceleration using metric
            # Note: geodesic_acceleration expects Cartesian 3-velocity, not 4-velocity
            a_bh = self.metric.geodesic_acceleration(
                pos_cpu.astype(np.float64),
                vel_cpu.astype(np.float64)
            )
            
            # Add adaptive softening near BH to prevent extreme accelerations
            r_bh = np.linalg.norm(pos_cpu, axis=1)
            bh_softening = np.maximum(epsilon, 0.01 * r_bh)  # Adaptive softening
            a_bh_magnitude = np.linalg.norm(a_bh, axis=1, keepdims=True)
            a_bh_max = 1e6  # Cap extreme accelerations
            a_bh_scale = np.minimum(a_bh_max / (a_bh_magnitude + 1e-10), 1.0)
            a_bh *= a_bh_scale
            
            # Add to total gravity acceleration on GPU
            self.gpu_manager.acc_grav += cp.asarray(a_bh.astype(np.float32))
        else:
            # Newtonian mode: Use simple point-mass gravity a = -G M_BH r / r³
            # Get BH mass and G from gravity solver
            bh_mass = getattr(self.gravity_solver, 'bh_mass', self.config.bh_mass)
            G = getattr(self.gravity_solver, 'G', 1.0)
            
            # Compute on GPU with adaptive softening near BH
            pos = self.gpu_manager.pos
            r = cp.sqrt(cp.sum(pos**2, axis=1, keepdims=True))  # Distance from origin
            r_safe = cp.maximum(r, 1e-6)  # Avoid division by zero
            
            # Adaptive softening: increase near BH to prevent extreme forces
            adaptive_epsilon = cp.maximum(epsilon, 0.01 * r_safe)
            
            # Newtonian BH acceleration with softening: a = -G M_BH r / (r² + ε²)^{3/2}
            r_soft = cp.sqrt(r_safe**2 + adaptive_epsilon**2)
            a_bh_gpu = -G * bh_mass * pos / (r_soft**3)
            
            # Cap extreme accelerations to prevent numerical instability
            a_bh_magnitude = cp.linalg.norm(a_bh_gpu, axis=1, keepdims=True)
            max_accel = 1e4  # Maximum allowed acceleration
            scale_factor = cp.minimum(max_accel / (a_bh_magnitude + 1e-10), 1.0)
            a_bh_gpu *= scale_factor
            
            self.gpu_manager.acc_grav += a_bh_gpu
        
        self.state.timing_gravity = time_module.time() - t0_grav
        
        # 7. Compute Hydro forces using same neighbour lists (TreeSPH)
        t0_pressure = time_module.time()
        
        if compute_hydro_treesph is not None:
            # Use optimized TreeSPH kernel with cached neighbours
            self.gpu_manager.acc_hydro, self.gpu_manager.du_dt = compute_hydro_treesph(
                self.gpu_manager.pos,
                self.gpu_manager.vel,
                self.gpu_manager.mass,
                self.gpu_manager.h,
                self.gpu_manager.rho,
                self.gpu_manager.pressure,
                self.gpu_manager.cs,
                neighbour_lists,
                neighbour_counts,
                alpha=self.config.artificial_viscosity_alpha,
                beta=self.config.artificial_viscosity_beta
            )
        else:
            # Fallback to brute force
            from tde_sph.gpu import compute_hydro_gpu
            self.gpu_manager.acc_hydro.fill(0.0)
            self.gpu_manager.du_dt.fill(0.0)
            compute_hydro_gpu(
                self.gpu_manager.pos,
                self.gpu_manager.vel,
                self.gpu_manager.mass,
                self.gpu_manager.h,
                self.gpu_manager.rho,
                self.gpu_manager.pressure,
                self.gpu_manager.cs,
                self.gpu_manager.acc_hydro,
                self.gpu_manager.du_dt,
                self.config.artificial_viscosity_alpha,
                self.config.artificial_viscosity_beta
            )
        self.state.timing_sph_pressure = time_module.time() - t0_pressure
        
        # 8. Transfer results back to CPU only at the end
        t0_transfer = time_module.time()
        a_grav = cp.asnumpy(self.gpu_manager.acc_grav)
        a_hydro = cp.asnumpy(self.gpu_manager.acc_hydro)
        du_dt = cp.asnumpy(self.gpu_manager.du_dt)
        #self.state.timing_gpu_transfer += time_module.time() - t0_transfer
        
        # Mark that data is on GPU for next step
        self.gpu_manager.data_on_gpu = True
        
        # Check for NaN/inf in accelerations
        if not np.all(np.isfinite(a_grav)) or not np.all(np.isfinite(a_hydro)):
            self._log("ERROR: GPU accelerations contain NaN or inf")
            self._log(f"a_grav finite: {np.all(np.isfinite(a_grav))}, a_hydro finite: {np.all(np.isfinite(a_hydro))}")
            
            # Detailed diagnostics
            if not np.all(np.isfinite(a_hydro)):
                # Check input data that went into hydro computation
                self._log(f"Hydro inputs:")
                self._log(f"  density range: [{float(np.min(density_cpu)):.3e}, {float(np.max(density_cpu)):.3e}]")
                self._log(f"  density zeros: {int(np.sum(density_cpu == 0.0))}")
                self._log(f"  pressure range: [{float(np.min(self.particles.pressure)):.3e}, {float(np.max(self.particles.pressure)):.3e}]")
                self._log(f"  sound_speed range: [{float(np.min(self.particles.sound_speed)):.3e}, {float(np.max(self.particles.sound_speed)):.3e}]")
                self._log(f"  neighbour_counts range: [{int(cp.min(neighbour_counts))}, {int(cp.max(neighbour_counts))}]")
                
                # Check which particles have NaN
                nan_mask = ~np.isfinite(a_hydro).any(axis=1)
                nan_count = int(np.sum(nan_mask))
                self._log(f"  particles with NaN a_hydro: {nan_count}/{len(a_hydro)}")
                
                if nan_count > 0 and nan_count < 10:
                    nan_indices = np.where(nan_mask)[0]
                    self._log(f"  NaN particle indices: {nan_indices.tolist()}")
                    for idx in nan_indices[:3]:
                        self._log(f"    Particle {idx}: rho={density_cpu[idx]:.3e}, P={self.particles.pressure[idx]:.3e}, "
                            f"cs={self.particles.sound_speed[idx]:.3e}, neighbours={int(neighbour_counts[idx])}")
            
            raise ValueError("Invalid GPU accelerations")
        
        a_total = a_grav + a_hydro
        if not np.all(np.isfinite(a_total)):
            self._log("ERROR: Total accelerations contain NaN or inf")
            raise ValueError("Invalid total accelerations")

        # Radiation cooling/heating on CPU (density/temperature already on host)
        du_dt_total = du_dt
        if self.radiation_model is not None:
            du_dt_rad = self._compute_radiation_du_dt()
            if du_dt_rad is not None:
                du_dt_total = du_dt + du_dt_rad

        return {
            'gravity': a_grav,
            'hydro': a_hydro,
            'total': a_total,
            'du_dt': du_dt_total
        }

    def _enforce_timestep_limits(self, candidate_dt: float) -> float:
        """Clamp timestep proposals using absolute and per-step limits."""
        dt_min = float(self.config.dt_min)
        dt_max = float(self.config.dt_max)
        change_limit = max(float(self.config.dt_change_limit), 1.0)
        prev_dt = max(float(self.state.dt), dt_min)

        lower_change = prev_dt / change_limit
        upper_change = prev_dt * change_limit

        effective_min = max(dt_min, lower_change)
        effective_max = min(dt_max, upper_change)

        limited_dt = min(max(candidate_dt, effective_min), effective_max)

        reasons = []
        eps = 1e-12
        if candidate_dt < dt_min - eps:
            reasons.append('dt_min')
        if candidate_dt > dt_max + eps:
            reasons.append('dt_max')
        if candidate_dt < lower_change - eps:
            reasons.append('decrease_limit')
        if candidate_dt > upper_change + eps:
            reasons.append('increase_limit')

        reason_str = 'none' if not reasons else ','.join(sorted(set(reasons)))

        self.state.last_dt_candidate = candidate_dt
        self.state.last_dt_limiter = reason_str

        if reason_str != 'none' and self.config.verbose:
            self._log(
                "dt clamp applied (%s): proposed %.3e -> %.3e"
                % (reason_str, candidate_dt, limited_dt)
            )

        return limited_dt

    def step(self):
        """
        Advance simulation by one timestep.
        """
        t0_step = time_module.time()
        
        # Compute forces (this internally times gravity, SPH density, SPH pressure)
        forces = self.compute_forces()

        # Advance particles
        t0_integrate = time_module.time()
        self.integrator.step(
            self.particles,
            self.state.dt,
            forces
        )

        # Update derived diagnostics after state advance
        self._update_velocity_magnitude()
        self.state.timing_integration = time_module.time() - t0_integrate

        # Update time
        self.state.time += self.state.dt
        if not np.isfinite(self.state.time):
            self._log(f"ERROR: Time became NaN, dt={self.state.dt}")
            raise ValueError("Time became NaN")
        self.state.step += 1

        # Estimate next timestep (use GR-specific CFL factor if in GR mode)
        t0_dt_est = time_module.time()
        cfl_factor = self.config.cfl_factor_gr if self.config.mode == "GR" else self.config.cfl_factor
        proposed_dt = self.integrator.estimate_timestep(
            self.particles,
            cfl_factor=cfl_factor,
            accelerations=forces['total'],
            min_dt=self.config.dt_min,
            max_dt=self.config.dt_max,
            metric=self.metric,
            config=self.config
        )
        self.state.dt = self._enforce_timestep_limits(proposed_dt)
        if not np.isfinite(self.state.dt) or self.state.dt <= 0:
            self._log(f"ERROR: Invalid timestep dt={self.state.dt}, proposed={proposed_dt}")
            raise ValueError(f"Timestep became invalid: dt={self.state.dt}")
        
        # Accumulate timestep estimation timing for averaging
        dt_est_time = time_module.time() - t0_dt_est
        self.state._timestep_est_accumulator += dt_est_time
        self.state._timestep_est_count += 1

        # Energy computation moved to write_snapshot() for performance
        # (only compute when actually writing snapshots)
        
        # Total step time
        self.state.timing_total = time_module.time() - t0_step
        
        # Calculate "other" time (remaining overhead)
        accounted_time = (
            self.state.timing_compute_forces +
            self.state.timing_integration +
            self.state.timing_energy_computation +
            self.state.timing_timestep_estimation +
            self.state.timing_io
        )
        self.state.timing_other = max(0.0, self.state.timing_total - accounted_time)

    def write_snapshot(self):
        """
        Write current state to HDF5 snapshot.
        """
        from tde_sph.io import write_snapshot
        
        # Compute energies when writing snapshot (not every step)
        t0_energy = time_module.time()
        energies = self.compute_energies()
        self.state.kinetic_energy = energies['kinetic']
        self.state.potential_energy = energies['potential']
        self.state.internal_energy = energies['internal']
        self.state.total_energy = energies['total']
        self.state.timing_energy_computation = (time_module.time() - t0_energy) * (self.state.dt / self.config.snapshot_interval)
        
        if self.state.initial_energy is None:
            self.state.initial_energy = self.state.total_energy

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
            'velocity_magnitude': self.particles.velocity_magnitude,
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

    def get_averaged_timestep_timing(self) -> float:
        """
        Get averaged timestep estimation timing since last call and reset accumulator.
        
        Returns
        -------
        float
            Average timestep estimation time in seconds, or 0.0 if no steps accumulated.
        """
        if self.state._timestep_est_count == 0:
            return 0.0
        
        avg_time = self.state._timestep_est_accumulator / self.state._timestep_est_count
        
        # Reset accumulators
        self.state._timestep_est_accumulator = 0.0
        self.state._timestep_est_count = 0
        
        return avg_time

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
