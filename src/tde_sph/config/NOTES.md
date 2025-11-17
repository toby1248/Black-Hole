# Phase 2 Implementation Notes — Config Module

## Status

- [x] SimulationConfig extensions (TASK-015-config-a) — COMPLETED 2025-11-17
- [x] YAML/JSON loaders (TASK-015-config-b) — COMPLETED 2025-11-17
- [x] Command-line overrides (TASK-015-config-c) — COMPLETED 2025-11-17
- [x] Example config templates (TASK-015-config-d) — COMPLETED 2025-11-17

## Implementation Decisions

### Config Loader Architecture

**File**: `tde_sph/config/loaders.py`

**Functions implemented**:
1. `load_config(filename, **overrides)` — Main entry point
   - Supports `.yaml`, `.yml`, `.json` files
   - Auto-detects format from file extension
   - Applies command-line style overrides (`mode="GR"`, `bh_spin=0.9`)
   - Returns validated `SimulationConfig` with clear error messages

2. `save_config(config, filename)` — Config export
   - Saves SimulationConfig to YAML/JSON
   - Organizes into nested sections (simulation, black_hole, metric, etc.)
   - Human-readable output with comments preserved

3. `flatten_config(config_dict)` — Nested dict flattening
   - Converts nested YAML sections to flat SimulationConfig fields
   - Handles common aliases (e.g., `black_hole.mass` → `bh_mass`)
   - Extensible mapping system for new config sections

4. `config_from_dict(config_dict)` — Programmatic config creation

**Nested structure mapping**:
```yaml
black_hole:
  mass: 1.0     → bh_mass: 1.0
  spin: 0.7     → bh_spin: 0.7
metric:
  type: kerr    → metric_type: "kerr"
integration:
  use_hamiltonian_integrator: true  → use_hamiltonian_integrator: True
```

### Unit Conversion Strategy

**Current approach**: Code units (G=c=M_BH=1)
- All parameters in config are dimensionless
- No automatic unit conversion in Phase 2
- Config files document units in comments

**Deferred to Phase 3**:
- Physical units module (solar masses, km, seconds)
- Automatic conversion `mass: 1e6 M_sun` → `bh_mass: 1.0` (code units)
- Unit system tags in config files

**Rationale**: Avoid complexity in Phase 2; keep configs explicit about units

### Command-Line Overrides

**Implementation**: Kwargs in `load_config()`
```python
config = load_config("base.yaml", mode="GR", bh_spin=0.9)
```

**Deferred to future CLI module**:
- Argparse integration: `--mode GR --bh-spin 0.9`
- Merging priority: CLI > config file > defaults

## Configuration Examples

### Example Config Files

**Location**: `/home/user/Black-Hole/configs/`

1. **newtonian_tde.yaml**
   - Mode: Newtonian
   - BH mass: 1.0e6 (documented as solar masses, but code units)
   - Use case: Phase 1 baseline, comparison with GR

2. **schwarzschild_tde.yaml**
   - Mode: GR
   - Metric: Schwarzschild (non-spinning)
   - BH spin: 0.0
   - Hamiltonian integrator: optional
   - Use case: GR orbital dynamics without frame-dragging

3. **kerr_retrograde_tde.yaml**
   - Mode: GR
   - Metric: Kerr
   - BH spin: 0.9 (high spin, near maximal)
   - Hamiltonian integrator: true
   - ISCO threshold: 8.0 M (retrograde ISCO ~ 9M for a=0.9)
   - Stricter timestep control: `cfl_factor_gr=0.08`
   - Use case: Maximal GR effects, frame-dragging, Lense-Thirring precession

### Validated Parameter Ranges

**Black hole spin**: [0, 1]
- a=0: Schwarzschild
- a=0.5: Moderate spin
- a=0.9: High spin (kerr_retrograde_tde.yaml)
- a=0.998: Near-extremal (future test case)

**ISCO thresholds**:
- Schwarzschild: r_ISCO = 6M → threshold ~10M (conservative)
- Kerr prograde (a=0.9): r_ISCO ~ 2.3M → threshold ~5M
- Kerr retrograde (a=0.9): r_ISCO ~ 9M → threshold ~8M (used in example)

**CFL factors**:
- Newtonian: 0.3 (standard SPH)
- GR moderate: 0.1 (Schwarzschild)
- GR high-spin: 0.08 (Kerr a=0.9, stricter for geodesic accuracy)

## Issues / Blockers

**None** — All functionality implemented and tested.

**Known limitations**:
1. No automatic unit conversion (deferred to Phase 3)
2. No argparse integration (deferred to CLI module)
3. Config file schema versioning not implemented
   - Future: Add `schema_version: "2.0"` field
   - Migration utilities for older configs

**Edge cases handled**:
- Empty YAML files → default config
- Invalid metric types → clear ValidationError
- Missing required nested sections → graceful defaults
- Unknown fields → Pydantic "extra=forbid" raises error

## Reviewer Comments

(Reviewer agent will add notes here)
