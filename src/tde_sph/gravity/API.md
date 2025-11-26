# Gravity Module API

## Abstract Base Class: `GravitySolver`

### `compute_acceleration(positions, masses, smoothing_lengths, metric=None, velocities=None) -> NDArray[N, 3]`
Computes total gravitational acceleration.
- **Input**:
    - `positions`: (N, 3) Cartesian.
    - `masses`: (N,).
    - `smoothing_lengths`: (N,) for softening.
    - `metric`: Optional `Metric` instance.
    - `velocities`: (N, 3) Cartesian (Required if `metric` is provided).
- **Output**: (N, 3) Acceleration vector.

### `compute_potential(positions, masses, smoothing_lengths, metric=None) -> NDArray[N]`
Computes gravitational potential (Newtonian approximation).
- **Input**: Same as acceleration.
- **Output**: (N,) Potential values.

## Implementations

### `RelativisticGravitySolver(G=1.0, bh_mass=1.0)`
- **Mode**: Hybrid.
    - If `metric` is None: Uses Newtonian Point Mass + Newtonian Self-Gravity.
    - If `metric` is Set: Uses Metric Geodesic + Newtonian Self-Gravity.

### `NewtonianGravity(G=1.0)`
- **Mode**: Pure Newtonian.
- **Methods**: Computes self-gravity (N-body).
