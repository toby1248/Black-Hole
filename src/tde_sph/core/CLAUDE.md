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

## Phase 3 Goals: Energy Diagnostics & Radiation Integration

### TASK-022: Global Energy Bookkeeping

**Objective**: Implement `energy_diagnostics.py` for comprehensive energy tracking.

**Energy Components to Track**:
1. **Kinetic**: E_kin = Σ (1/2) m v²
   - GR mode: use 4-velocity and proper formulation
2. **Potential (BH gravity)**: E_pot_BH = Σ m Φ_BH(r)
   - Newtonian: -GM/r
   - GR: effective potential from metric
3. **Potential (self-gravity)**: E_pot_self = (1/2) Σᵢ Σⱼ m_i m_j / |r_i - r_j|
4. **Internal (thermal)**: E_int = Σ m u
5. **Internal (radiation)**: E_rad = Σ m u_rad (if separated in EOS)
6. **Radiated (cumulative)**: E_radiated = ∫ L dt
7. **Total**: E_tot = E_kin + E_pot + E_int - E_radiated

**Implementation Requirements**:
1. Create `EnergyDiagnostics` class
2. Provide methods:
   - `compute_all_energies(particles, state, config)` → dict of energies
   - `compute_energy_conservation_error()` → ΔE/E₀
   - `log_energy_history(time, energies)` → time series
3. Handle both Newtonian and GR modes
4. Compute GR Hamiltonian (conserved energy) correctly
5. Track energy errors and flag violations

**Output Format**:
- Per-snapshot energy breakdown (all components)
- Time series: E_kin(t), E_pot(t), E_int(t), E_rad(t), E_tot(t)
- Conservation error tracking: ΔE(t)/E₀

**Tests**:
- Adiabatic run: E_tot constant to < 0.1%
- Isolated star: E_tot conserved (no BH)
- Newtonian vs GR energy definitions consistent in weak field

### TASK-015e: Integrate Radiation & Energy Diagnostics

**Objective**: Update `Simulation` orchestrator to integrate Phase 3 modules.

**Changes Required**:
1. Add `radiation_model` to simulation state
2. Call `radiation.compute_cooling_rate()` during evolution
3. Apply cooling to internal energy: u += (du/dt) × dt
4. Update energy diagnostics every snapshot
5. Add config options for radiation on/off, cooling timescale limits
6. Validate backward compatibility (radiation off = Phase 2 behavior)

## DO

- Focus ONLY on `tde_sph/core`
- Define clean, minimal interfaces for metrics, gravity, EOS, SPH, radiation, integrators, IO, visualization
- Keep `Simulation` and configuration handling thin and composable
- Coordinate with other modules via interfaces, not concrete implementations
- Support both Newtonian and GR modes transparently
- Use precision-agnostic code (FP32 default, FP64 where needed)
- Add comprehensive energy diagnostics
- Integrate radiation cooling seamlessly

## DO NOT

- Implement physics details (SPH kernels, GR, EOS, etc.)
- Modify files in other subpackages except when explicitly instructed
- Open or modify anything under the `prompts/` folder
- Break existing Newtonian functionality
- Break backward compatibility with Phase 1 & 2
