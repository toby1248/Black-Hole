# Core Module API

## Class: `Simulation`
Main orchestrator.

### `__init__(particles, gravity_solver, eos, integrator, config, metric=None)`
- **Config**: `SimulationConfig` object determining mode (Newtonian/GR).

### `step()`
Advances simulation by `dt`.
1. Computes forces (Gravity + Hydro).
2. Updates time integrator.
3. Updates timestep (CFL).

### `run()`
Main loop until `t_end`.

### `write_snapshot()`
Exports current state to HDF5.

## Class: `SimulationConfig`
Pydantic model for validation.
- `mode`: "Newtonian" | "GR"
- `metric_type`: "schwarzschild" | "kerr"
- `bh_mass`, `bh_spin`
