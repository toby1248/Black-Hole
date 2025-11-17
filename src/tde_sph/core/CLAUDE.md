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
