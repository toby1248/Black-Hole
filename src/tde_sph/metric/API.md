# Metric Module API

## Abstract Base Class: `Metric`

### `metric_tensor(x: NDArray[N, 3]) -> NDArray[N, 4, 4]`
Computes the metric tensor $g_{\mu\nu}$ at Cartesian position `x`.
- **Input**: Cartesian coordinates $(x, y, z)$.
- **Output**: Metric tensor components in the basis associated with the metric implementation (usually BL for Kerr), OR transformed to Cartesian basis if specified. *Clarification needed in implementation: Standardize on returning components in the coordinate basis used by the solver, or document strictly that it returns BL components.*

### `geodesic_acceleration(x: NDArray[N, 3], v: NDArray[N, 3]) -> NDArray[N, 3]`
Computes the spatial acceleration $\frac{d^2x^i}{dt^2}$ (or $\frac{d^2x^i}{d\tau^2}$) for a test particle.
- **Input**: Cartesian position `x` and Cartesian 3-velocity `v`.
- **Output**: Cartesian acceleration vector.
- **Note**: Handles all internal coordinate conversions and 4-velocity normalization.

## Implementations

### `KerrMetric(mass: float, spin: float)`
- **Parameters**: `mass` (M), `spin` (a/M).
- **Methods**: Implements all ABC methods using Boyer-Lindquist coordinates internally.

### `SchwarzschildMetric(mass: float)`
- **Parameters**: `mass` (M).
- **Methods**: Implements all ABC methods (can be a special case of Kerr with a=0).

### `MinkowskiMetric()`
- **Methods**: Returns flat space metric and zero geodesic acceleration.
