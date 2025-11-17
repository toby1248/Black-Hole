# Implementation Notes: Newtonian Gravity & Ideal Gas EOS

**Date:** 2025-11-15
**Modules:** `tde_sph.gravity.newtonian`, `tde_sph.eos.ideal_gas`
**Requirements:** REQ-004, REQ-006, REQ-008

---

## Summary

Successfully implemented two core physics modules for the TDE-SPH framework:

1. **NewtonianGravity** (`src/tde_sph/gravity/newtonian.py`)
   - Newtonian self-gravity solver using direct O(N²) pairwise summation
   - Softening scale tied to SPH smoothing length
   - Implements `GravitySolver` interface

2. **IdealGas** (`src/tde_sph/eos/ideal_gas.py`)
   - Ideal gas equation of state with arbitrary adiabatic index γ
   - Supports monatomic (γ=5/3) and relativistic (γ=4/3) gases
   - Implements `EOS` interface

---

## Implementation Details

### NewtonianGravity

**Physics:**
- Softened acceleration: `a_i = -∑_j G m_j (r_i - r_j) / (|r_i - r_j|² + ε²)^(3/2)`
- Softened potential: `φ_i = -∑_j G m_j / sqrt(|r_i - r_j|² + ε²)`
- Softening length: `ε = (h_i + h_j) / 2` (average of smoothing lengths)

**Design choices:**
- Direct N² summation using NumPy broadcasting for clarity and correctness
- All computations in float32 for GPU compatibility (CON-002)
- Self-interaction (i=j) handled via `np.where` masking
- Dimensionless units with G=1 (GUD-003)

**Performance:**
- Current: O(N²), suitable for N ≲ 10⁵ particles
- Future: Tree-based solver (Barnes-Hut/FMM) for N > 10⁵ (noted in docstring)

**Edge cases handled:**
- Division by zero prevented by softening
- Self-interaction excluded from force summation
- Potential diagonal set to infinity to avoid self-interaction

### IdealGas

**Physics:**
- Pressure: `P = (γ - 1) ρ u`
- Sound speed: `c_s = sqrt(γ P / ρ)`
- Temperature: `T = (γ - 1) u μ m_p / k_B`

**Design choices:**
- Physical constants (k_B, m_p) stored in CGS units as class attributes
- Default mean molecular weight μ = 0.6 (ionized solar composition)
- All array operations in float32
- Bonus method: `internal_energy_from_temperature()` for IC generation

**Edge cases handled:**
- Negative density/energy → returns 0 for derived quantities
- Zero density → small floor (1e-30) in sound speed to prevent division by zero
- Validation for γ > 1 at initialization

**Supported configurations:**
- γ = 5/3: monatomic ideal gas (REQ-007)
- γ = 4/3: relativistic gas (REQ-007)
- Arbitrary γ for custom polytropes

---

## Validation Results

All tests in `validate_implementations.py` pass successfully:

### NewtonianGravity Tests
- ✓ Two-particle system: symmetric accelerations and potentials
- ✓ 100-particle system: reasonable force/potential ranges
- ✓ Interface compliance: inherits from `GravitySolver`

### IdealGas Tests
- ✓ Pressure formula: P = (γ-1)ρu verified numerically
- ✓ Multiple gamma values: γ=5/3 and γ=4/3 work correctly
- ✓ Temperature round-trip: T → u → T' with T ≈ T'
- ✓ Edge cases: zero/negative values handled gracefully
- ✓ Interface compliance: inherits from `EOS`

### Example Output
```
Single particle test (γ=5/3):
  Density: 1.00e+00
  Internal energy: 1.00e+12
  Pressure: 6.67e+11
  Sound speed: 1.05e+06
  Temperature: 4.85e+03 K
```

---

## Usage Examples

### Basic Usage

```python
from tde_sph.gravity import NewtonianGravity
from tde_sph.eos import IdealGas
import numpy as np

# Initialize solvers
gravity = NewtonianGravity(G=1.0)
eos = IdealGas(gamma=5.0/3.0, mean_molecular_weight=0.6)

# Particle data (N particles)
positions = np.random.randn(1000, 3).astype(np.float32)
masses = np.ones(1000, dtype=np.float32)
smoothing_lengths = 0.1 * np.ones(1000, dtype=np.float32)
densities = np.ones(1000, dtype=np.float32)
internal_energies = 1e12 * np.ones(1000, dtype=np.float32)

# Compute gravitational forces
accel = gravity.compute_acceleration(positions, masses, smoothing_lengths)
potential = gravity.compute_potential(positions, masses, smoothing_lengths)

# Compute thermodynamic quantities
pressure = eos.pressure(densities, internal_energies)
sound_speed = eos.sound_speed(densities, internal_energies)
temperature = eos.temperature(densities, internal_energies)
```

### Different Adiabatic Indices

```python
# Monatomic gas (main-sequence star)
eos_monatomic = IdealGas(gamma=5.0/3.0)

# Relativistic gas (radiation-dominated)
eos_relativistic = IdealGas(gamma=4.0/3.0)

# Custom polytrope
eos_custom = IdealGas(gamma=1.4)  # Diatomic-like
```

### Setting Initial Conditions from Temperature

```python
# Define temperature profile
temperatures = 1e4 * np.ones(N, dtype=np.float32)  # 10^4 K

# Convert to internal energy
internal_energies = eos.internal_energy_from_temperature(temperatures)

# Verify round-trip
reconstructed_T = eos.temperature(densities, internal_energies)
assert np.allclose(temperatures, reconstructed_T)
```

---

## Integration with Framework

Both modules follow the established architecture patterns:

1. **Interface Compliance (GUD-001)**
   - `NewtonianGravity` implements `GravitySolver` ABC
   - `IdealGas` implements `EOS` ABC
   - All methods match interface signatures exactly

2. **Dimensionless Units (GUD-003)**
   - Gravity: G = 1 by default
   - EOS: Physical constants in CGS for temperature only
   - All other quantities dimensionless

3. **Float32 (CON-002)**
   - All arrays and computations use np.float32
   - Compatible with GPU acceleration (future CUDA kernels)

4. **Modular Design (GUD-001)**
   - Self-contained modules with no cross-dependencies
   - Easy to swap for alternative implementations
   - Clear separation of physics and numerics

---

## Next Steps

### Immediate
- [x] Newtonian gravity solver (TASK-004)
- [x] Ideal gas EOS (TASK-006)

### Phase 1 Remaining
- [ ] SPH particle container and neighbour search (TASK-003)
- [ ] SPH hydrodynamic forces (TASK-005)
- [ ] Initial conditions generator (polytrope) (TASK-007)
- [ ] Time integrator (leapfrog) (TASK-008)

### Future Optimizations
- **Gravity:** Tree-based solver for N > 10⁵ particles
- **Gravity:** GPU CUDA kernel for direct summation
- **EOS:** Add radiation pressure (gas + radiation EOS)
- **Tests:** Formal pytest test suite in `tests/` directory

---

## Physics References

**Gravity:**
- Price & Monaghan (2007) - SPH self-gravity implementations
- Tejeda et al. (2017) - Newtonian self-gravity in hybrid GR framework
  [arXiv:1701.00303]

**Thermodynamics:**
- Kippenhahn & Weigert - Stellar Structure and Evolution
- Price (2012) - SPH review (J. Comp. Phys. 231, 759)

---

## Notes

- Both implementations prioritize **clarity and correctness** over performance
- Direct O(N²) gravity is intentional for Phase 1 baseline
- Edge cases (zero/negative values) handled conservatively
- All code fully documented with physics references in docstrings
- Validation script (`validate_implementations.py`) demonstrates correctness
