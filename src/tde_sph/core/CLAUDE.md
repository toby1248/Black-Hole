# CLAUDE Instructions — core module

Role: Implement and maintain high-level simulation orchestration and shared interfaces.

## Phase 1 Status (Complete)
- ✅ Abstract base classes defined in `interfaces.py`
- ✅ `Simulation` orchestrator coordinating all modules
- ✅ `SimulationConfig` and `SimulationState` dataclasses
- ✅ Energy diagnostics (kinetic, potential, internal)
- ✅ Newtonian-only mode functional

## Phase 2 Goals: GR–Newtonian Mode Toggle

### GR–Newtonian Interface Design

The core module must support **seamless switching** between Newtonian and GR modes via configuration, without breaking existing Newtonian functionality.

#### Configuration Interface (TASK-015)

`SimulationConfig` must support:
```python
mode: str = "Newtonian"  # "Newtonian" or "GR"
metric_type: str = "schwarzschild"  # "minkowski", "schwarzschild", "kerr"
bh_mass: float = 1.0  # M_BH in code units (G=c=M_BH=1)
bh_spin: float = 0.0  # Dimensionless spin a/M ∈ [0, 1]
use_hamiltonian_integrator: bool = False  # True for GR near ISCO
isco_radius_threshold: float = 10.0  # Switch to Hamiltonian inside this radius
```

#### Module Coordination

**Simulation orchestrator must:**
1. **Conditionally instantiate** `Metric` based on `config.mode`:
   - `mode="Newtonian"` → `metric=None`
   - `mode="GR"` → instantiate appropriate `Metric` subclass

2. **Pass metric to solvers** that need it:
   - `GravitySolver.compute_acceleration(..., metric=metric)`
   - `TimeIntegrator` may switch algorithms based on presence of metric

3. **Validate configuration**:
   - Ensure `metric` is provided when `mode="GR"`
   - Warn if `bh_spin > 0` but `metric_type != "kerr"`

4. **Energy diagnostics in GR**:
   - Track *conserved* energy (Hamiltonian) for GR orbits
   - Log proper vs coordinate time
   - Document that Newtonian energy formulae differ from GR

#### Precision Requirements (Phase 2 Update)

- **Default: FP32** for particle data and most computations
- **FP64 where necessary**: metric tensor inversions, Christoffel symbols near horizons
- **Precision-agnostic code preferred**: use `dtype` parameters, avoid hardcoded `float32`
- **GPU compatibility**: structure for mixed-precision tensor ops

#### Architectural Risks

**RISK-CORE-001**: Energy non-conservation in hybrid GR+Newtonian
- **Mitigation**: Clearly document approximations; validate against pure Newtonian and test-particle GR limits

**RISK-CORE-002**: Unit confusion (geometric vs CGS)
- **Mitigation**: Enforce G=c=M_BH=1 internally; provide conversion utilities; log unit system in snapshots

**RISK-CORE-003**: Mode-switching bugs
- **Mitigation**: Extensive unit tests for both modes; regression tests must pass for Newtonian after GR changes

### Phase 2 Tasks for Core Module Agents

**TASK-015a**: Extend `SimulationConfig` with GR parameters
- Add fields: `mode`, `metric_type`, `bh_mass`, `bh_spin`, `use_hamiltonian_integrator`, `isco_radius_threshold`
- Validate combinations (e.g., GR mode requires metric)
- Document units and conventions

**TASK-015b**: Update `Simulation` orchestrator
- Conditionally instantiate `Metric` based on config
- Pass metric to `GravitySolver` and `TimeIntegrator`
- Add mode-aware energy diagnostics
- Log GR-specific quantities (proper time, ISCO violations, etc.)

**TASK-015c**: Enhance interfaces for GR
- Ensure `Metric`, `GravitySolver`, `TimeIntegrator` ABCs support both modes
- Add optional `metric` parameter where needed
- Document expected behavior in Newtonian vs GR modes

**TASK-015d**: Backward compatibility
- All existing Newtonian examples/tests must pass unchanged
- Default config remains Newtonian
- GR features opt-in via explicit config

## Phase 3 Tasks: Energy Diagnostics & Thermodynamic Tracking

### TASK-022: Implement `energy_diagnostics.py`

**Goal**: Comprehensive energy accounting for TDE simulations, tracking all energy components and their evolution.

**Implementation Requirements**:

1. **EnergyDiagnostics Class**:
   - Compute kinetic energy: `E_kin = Σ(½ m v²)`
   - Compute potential energy:
     - BH gravitational potential (Newtonian or GR-corrected)
     - Self-gravity potential energy
   - Compute internal energy:
     - Total: `E_int = Σ(m u)`
     - Gas component (if using RadiationGas EOS)
     - Radiation component (if using RadiationGas EOS)
   - Total energy and conservation tracking

2. **Thermodynamic State Tracking**:
   - Global quantities: total mass, angular momentum, linear momentum
   - Temperature statistics: min, max, mean, median
   - Pressure statistics
   - Entropy proxies (for adiabatic validation)

3. **Time-Series Storage**:
   - Dictionary or dataclass to accumulate diagnostics over time
   - Methods: `compute_current()`, `append_to_history()`, `get_time_series()`
   - Export to structured format (pandas-compatible dict or numpy arrays)

4. **GR-Specific Diagnostics** (if metric provided):
   - Hamiltonian / conserved energy per particle
   - Proper time vs coordinate time
   - Particles inside ISCO count/mass fraction

**Interface**:
```python
class EnergyDiagnostics:
    def compute(self, particles, gravity_solver, eos, metric=None) -> Dict[str, float]
    def compute_kinetic_energy(self, particles) -> float
    def compute_potential_energy(self, particles, gravity_solver, metric=None) -> float
    def compute_internal_energy(self, particles, eos) -> Dict[str, float]  # {total, gas, radiation}
    def compute_global_quantities(self, particles) -> Dict[str, Any]
```

**Cross-Module Expectations**:
- Will use `EOS.pressure()`, `EOS.temperature()` from `eos/`
- Will query `RadiationModel.luminosity()` from `radiation/` if available
- `Simulation` will call this after each timestep
- Results passed to `io/diagnostics.py` for logging

**Testing Requirements**:
- Energy conservation in adiabatic Newtonian runs (< 1% drift)
- Correct energy partitioning with RadiationGas EOS
- GR corrections near ISCO

## DO

- Focus ONLY on `tde_sph/core`
- Define clean, minimal interfaces for metrics, gravity, EOS, SPH, radiation, integrators, IO, visualization
- Keep `Simulation` and configuration handling thin and composable
- Coordinate with other modules via interfaces, not concrete implementations
- Support both Newtonian and GR modes transparently
- Use precision-agnostic code (FP32 default, FP64 where needed)

## DO NOT

- Implement physics details (SPH kernels, GR, EOS, etc.)
- Modify files in other subpackages except when explicitly instructed
- Open or modify anything under the `prompts/` folder
- Break existing Newtonian functionality
