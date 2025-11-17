# Phase 2 Implementation Notes — Core Module

## Status

- [x] Configuration extensions (TASK-015a) — COMPLETED 2025-11-17
- [x] Simulation orchestrator updates (TASK-015b) — COMPLETED 2025-11-17
- [x] Interface enhancements for GR (TASK-015c) — COMPLETED 2025-11-17
- [x] Backward compatibility verified (TASK-015d) — COMPLETED 2025-11-17

## Implementation Decisions

### SimulationConfig Migration to Pydantic

**Decision**: Migrated SimulationConfig from dataclass to Pydantic BaseModel

**Rationale**:
- Comprehensive field validation (ranges, enums, cross-field consistency)
- Built-in JSON/dict serialization for config I/O
- Clear error messages for invalid configurations
- Supports both backward compatibility (defaults) and new GR parameters

**Key additions**:
- `mode`: "Newtonian" | "GR" (default: "Newtonian")
- `metric_type`: "minkowski" | "schwarzschild" | "kerr" (default: "schwarzschild")
- `use_hamiltonian_integrator`: bool (default: False)
- `isco_radius_threshold`: float (default: 10.0 M)
- `cfl_factor_gr`: float (default: 0.1, stricter than Newtonian 0.3)
- `orbital_timestep_factor`: float (default: 0.05)
- `coordinate_system`: "cartesian" | "boyer-lindquist"
- `use_fp64_for_metric`: bool (default: True for GR precision)

**Validation rules implemented**:
1. GR mode with Minkowski → warning (use Newtonian instead)
2. Kerr with a=0 → warning (use Schwarzschild)
3. Schwarzschild with a>0 → warning (spin ignored)
4. Hamiltonian integrator requires GR mode → error
5. ISCO threshold < 3M → warning (photon orbit issues)
6. t_end > t_start → error
7. Spin ∈ [0, 1] → error if violated
8. CFL ∈ (0, 1] → error if violated

### Simulation Orchestrator Updates

**Changes to `Simulation` class**:
1. **Enhanced validation**: `_validate_config()` checks metric presence in GR mode
2. **GR-aware logging**: Logs metric type, BH mass/spin, Hamiltonian integrator settings
3. **Adaptive CFL**: Uses `cfl_factor_gr` in GR mode, `cfl_factor` in Newtonian
4. **Snapshot metadata**: Includes `metric_type`, `coordinate_system`, `use_hamiltonian_integrator`

**Metric instantiation**:
- **Newtonian mode**: `metric=None` (validated to raise warning if metric passed)
- **GR mode**: Requires `Metric` instance (validated to raise error if missing)
- Metric passed to `GravitySolver.compute_acceleration(metric=...)`

### Interfaces Enhancement

**No changes to ABCs** — interfaces already support optional `metric` parameter:
- `GravitySolver.compute_acceleration(..., metric=None)`
- Designed for forward compatibility with Phase 2 metric implementations

## Testing Notes

**Test suite**: `tests/test_config.py` — 23 tests, all passing

**Coverage**:
- Pydantic field validation (modes, metrics, spins, CFL factors)
- Cross-field validation (GR+Minkowski, Kerr+a=0, Hamiltonian+Newtonian)
- Warnings for inconsistent configurations
- YAML/JSON loading and saving
- Config overrides (CLI-style)
- Backward compatibility (Phase 1 defaults unchanged)

**Backward compatibility verification**:
- All 70 tests pass (23 new + 47 existing Phase 1/2 tests)
- Default config is Newtonian (unchanged from Phase 1)
- Phase 1 Newtonian examples unaffected by GR additions

**Example configs tested**:
- `configs/newtonian_tde.yaml` — loads successfully
- `configs/schwarzschild_tde.yaml` — validates GR mode
- `configs/kerr_retrograde_tde.yaml` — validates high-spin Kerr setup

## Issues / Blockers

**None** — Implementation complete and validated.

**Future considerations**:
- Unit conversions (solar masses → code units) in config loaders
  - Currently documented as TODO in loaders.py
  - Deferred to Phase 3 when physical units module is implemented
- Command-line argument parsing (argparse integration)
  - Deferred to CLI module implementation
  - Config loaders support **overrides already

## Reviewer Comments

(Reviewer agent will add notes here)
