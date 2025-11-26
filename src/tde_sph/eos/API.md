# EOS Module API

## Abstract Base Class: `EOS`

### `pressure(density, internal_energy) -> NDArray`
Computes $P(\rho, u)$.

### `sound_speed(density, internal_energy) -> NDArray`
Computes $c_s(\rho, u)$.

### `temperature(density, internal_energy) -> NDArray`
Computes $T(\rho, u)$.

## Implementations
- `IdealGas(gamma=5/3)`
- `RadiationGas` (Planned)
