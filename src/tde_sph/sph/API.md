# SPH Module API

## Class: `ParticleSystem`
Container for particle data.
- **Methods**:
    - `get/set_positions`, `velocities`, `masses`, `density`, `pressure`, `internal_energy`.
    - `kinetic_energy()`, `thermal_energy()`.

## Functions

### `compute_hydro_acceleration(...)`
Computes pressure gradient and viscosity forces.
- **Input**: Positions, Velocities, Masses, Density, Pressure, Sound Speed, Smoothing Lengths.
- **Output**:
    - `acc_hydro`: (N, 3) Hydrodynamic acceleration.
    - `du_dt`: (N,) Rate of change of internal energy (PdV work + viscosity heating).

### `compute_density_summation(...)`
Computes density via SPH summation.
- **Input**: Positions, Masses, Smoothing Lengths.
- **Output**: Density array (N,).

### `update_smoothing_lengths(...)`
Iteratively solves for $h$ such that $N_{neigh} \approx \text{target}$.
- **Input**: Positions, Masses, current h.
- **Output**: Updated h.
