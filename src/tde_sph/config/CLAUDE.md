# CLAUDE Instructions — configuration module

Role: Manage configuration schemas, validation, and loading for both Newtonian and GR modes.

## Phase 1 Status (Complete)
- ✅ `SimulationConfig` dataclass in `core/simulation.py`
- ✅ Basic parameters: t_start, t_end, dt_initial, output_dir
- ✅ Newtonian-only mode functional

## Phase 2 Goals: GR Mode Configuration

Extend configuration to support seamless Newtonian ↔ GR mode switching with validation.

## Configuration Schema Extensions (TASK-015)

### New Parameters for GR Mode

Add to `SimulationConfig` (or create dedicated `GRConfig` class):

```python
# Mode control
mode: str = "Newtonian"  # "Newtonian" or "GR"

# Black hole parameters
bh_mass: float = 1.0  # In code units (G=c=M_BH=1)
bh_spin: float = 0.0  # Dimensionless a/M ∈ [0, 1]

# Metric specification
metric_type: str = "schwarzschild"  # "minkowski", "schwarzschild", "kerr"

# Integration control
use_hamiltonian_integrator: bool = False  # True for GR near ISCO
isco_radius_threshold: float = 10.0  # Switch to Hamiltonian inside r < 10M

# Timestep control (GR-specific)
cfl_factor_gr: float = 0.1  # Stricter CFL for GR (default 0.3 for Newtonian)
orbital_timestep_factor: float = 0.05  # Fraction of orbital period

# Coordinate system
coordinate_system: str = "cartesian"  # "cartesian" or "boyer-lindquist"

# Precision control
use_fp64_for_metric: bool = True  # High precision for metric near horizon
```

### Validation Rules

Implement in `SimulationConfig.__post_init__()` or separate validator:

1. **GR mode requires metric**:
   ```python
   if mode == "GR" and metric_type is None:
       raise ValueError("GR mode requires metric_type to be specified")
   ```

2. **Kerr requires spin**:
   ```python
   if metric_type == "kerr" and bh_spin == 0.0:
       warnings.warn("Kerr metric with a=0 is equivalent to Schwarzschild")
   ```

3. **Spin bounds**:
   ```python
   if not (0.0 <= bh_spin <= 1.0):
       raise ValueError(f"Black hole spin must be in [0, 1], got {bh_spin}")
   ```

4. **Hamiltonian integrator requires GR**:
   ```python
   if use_hamiltonian_integrator and mode != "GR":
       raise ValueError("Hamiltonian integrator only valid in GR mode")
   ```

5. **ISCO threshold sensible**:
   ```python
   if isco_radius_threshold < 3.0:  # Photon orbit
       warnings.warn("ISCO threshold < 3M may cause timestep issues")
   ```

### YAML/JSON Loading (TASK-015-config)

Support configuration files like:

```yaml
# config_gr_tde.yaml
simulation:
  mode: "GR"
  t_end: 100.0
  snapshot_interval: 1.0

black_hole:
  mass: 1.0e6  # Solar masses (will convert to code units)
  spin: 0.7
  metric_type: "kerr"

integration:
  use_hamiltonian_integrator: true
  isco_radius_threshold: 8.0
  cfl_factor_gr: 0.1

star:
  mass: 1.0
  radius: 1.0
  gamma: 1.6667  # 5/3

orbit:
  periapsis: 1.0  # In units of tidal radius
  inclination: 30.0  # degrees
```

Implement loader:
```python
from tde_sph.config import load_config

config = load_config("config_gr_tde.yaml")
```

## Phase 2 Tasks for Config Module Agents

**TASK-015-config-a**: Extend `SimulationConfig`
- Add all GR-related fields listed above
- Implement validation in `__post_init__()`
- Document units and conventions
- Provide sensible defaults (Newtonian mode by default)

**TASK-015-config-b**: YAML/JSON loaders
- Implement `load_config(filename)` function
- Support nested dictionaries (black_hole, star, orbit, integration)
- Handle unit conversions (solar masses → code units)
- Validate on load

**TASK-015-config-c**: Command-line overrides
- Support argparse-style overrides: `--mode GR --bh-spin 0.9`
- Merge with config file (CLI overrides file)
- Document in `scripts/run_simulation.py`

**TASK-015-config-d**: Presets/templates
- Create example configs in `configs/` directory:
  - `newtonian_tde.yaml`
  - `schwarzschild_tde.yaml`
  - `kerr_retrograde_tde.yaml`
- Document parameter choices

## Architectural Risks

**RISK-CONFIG-001**: Unit system confusion
- **Context**: Users may input solar masses, code expects G=c=M_BH=1
- **Mitigation**: Provide unit conversion utilities; log effective units in output

**RISK-CONFIG-002**: Invalid parameter combinations
- **Context**: Many parameters interdependent (mode, metric, integrator)
- **Mitigation**: Comprehensive validation with clear error messages

**RISK-CONFIG-003**: Config file versioning
- **Context**: Config schema may evolve across phases
- **Mitigation**: Include schema version in config files; migration utilities

## DO

- Work ONLY inside `tde_sph/config`
- Define configuration dataclasses/schemas (extend `SimulationConfig` or create new classes)
- Implement comprehensive validation with clear errors/warnings
- Support YAML/JSON loading and command-line overrides
- Provide example config files
- Document all parameters with units and valid ranges

## DO NOT

- Hard-code physics logic or numerical defaults (belongs in physics modules)
- Implement numerical algorithms (belongs in SPH, gravity, integration, etc.)
- Open or modify anything under the `prompts/` folder
- Change default behavior to break Phase 1 Newtonian examples
