# Quick Start: Configuration System

## Load a Config File

```python
from tde_sph.config import load_config

# Load example configs
config = load_config("configs/newtonian_tde.yaml")
config = load_config("configs/schwarzschild_tde.yaml")
config = load_config("configs/kerr_retrograde_tde.yaml")

# Load with overrides
config = load_config("configs/newtonian_tde.yaml", mode="GR", bh_spin=0.7)
```

## Create a Config Programmatically

```python
from tde_sph.core.simulation import SimulationConfig

# Newtonian mode (default)
config = SimulationConfig(
    t_end=50.0,
    output_dir="output/my_newtonian_run"
)

# GR mode with Schwarzschild metric
config = SimulationConfig(
    mode="GR",
    metric_type="schwarzschild",
    bh_mass=1.0,
    t_end=100.0,
    use_hamiltonian_integrator=False,
    output_dir="output/schwarzschild_run"
)

# GR mode with Kerr metric
config = SimulationConfig(
    mode="GR",
    metric_type="kerr",
    bh_mass=1.0,
    bh_spin=0.9,
    use_hamiltonian_integrator=True,
    isco_radius_threshold=8.0,
    cfl_factor_gr=0.08,
    output_dir="output/kerr_high_spin"
)
```

## Create a Config from Dict

```python
from tde_sph.config import config_from_dict

config_dict = {
    'simulation': {
        'mode': 'GR',
        't_end': 100.0,
    },
    'black_hole': {
        'mass': 1.0,
        'spin': 0.7,
    },
    'metric': {
        'type': 'kerr',
    },
    'integration': {
        'use_hamiltonian_integrator': True,
    },
}

config = config_from_dict(config_dict)
```

## Save a Config

```python
from tde_sph.config import save_config

save_config(config, "my_config.yaml")
```

## Use with Simulation

```python
from tde_sph.core.simulation import Simulation
from tde_sph.config import load_config

# Load config
config = load_config("configs/schwarzschild_tde.yaml")

# Create simulation components
# (particles, gravity_solver, eos, integrator, metric)

# Create simulation
sim = Simulation(
    particles=particles,
    gravity_solver=gravity_solver,
    eos=eos,
    integrator=integrator,
    config=config,
    metric=metric,  # Required for GR mode, None for Newtonian
)

# Run
sim.run()
```

## Key Parameters

### Mode Selection
- `mode: "Newtonian"` — Phase 1 Newtonian gravity
- `mode: "GR"` — General relativistic dynamics

### Metric Types (for GR mode)
- `metric_type: "minkowski"` — Flat spacetime (test only)
- `metric_type: "schwarzschild"` — Non-spinning black hole
- `metric_type: "kerr"` — Spinning black hole

### Black Hole Parameters
- `bh_mass: 1.0` — In code units (G=c=M_BH=1)
- `bh_spin: 0.0` — Dimensionless a/M ∈ [0, 1]

### Integration Control
- `use_hamiltonian_integrator: true/false` — High-accuracy GR integrator
- `isco_radius_threshold: 10.0` — Switch to Hamiltonian inside this radius (in M)
- `cfl_factor: 0.3` — Newtonian timestep safety factor
- `cfl_factor_gr: 0.1` — GR timestep safety factor (stricter)

## Example YAML Config

```yaml
simulation:
  mode: "GR"
  t_start: 0.0
  t_end: 100.0
  dt_initial: 0.0005
  snapshot_interval: 1.0
  output_dir: "output/my_gr_run"

black_hole:
  mass: 1.0
  spin: 0.7

metric:
  type: "kerr"
  coordinate_system: "cartesian"

integration:
  use_hamiltonian_integrator: true
  isco_radius_threshold: 8.0
  cfl_factor_gr: 0.1

sph:
  neighbour_search_method: "bruteforce"
  smoothing_length_eta: 1.2
  artificial_viscosity_alpha: 1.0
  artificial_viscosity_beta: 2.0

misc:
  random_seed: 42
  verbose: true
```

## Validation

Configs are validated automatically:

```python
# This will raise a ValidationError
config = SimulationConfig(
    mode="Newtonian",
    use_hamiltonian_integrator=True  # Error: Hamiltonian requires GR mode
)

# This will raise a warning
config = SimulationConfig(
    metric_type="kerr",
    bh_spin=0.0  # Warning: Kerr with a=0 is Schwarzschild
)
```

## Run Demo

```bash
python examples/config_demo.py
```

## Run Tests

```bash
pytest tests/test_config.py -v
```
