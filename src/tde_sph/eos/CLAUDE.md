# CLAUDE Instructions — EOS module

Role: Provide equation-of-state (EOS) and thermodynamic closures.

## Phase 1 & 2 Status (Complete)
- ✅ `IdealGas` EOS implemented with arbitrary adiabatic index γ
- ✅ Full interface implementation (pressure, sound_speed, temperature)
- ✅ Support for γ=5/3 (monatomic) and γ=4/3 (relativistic) gases
- ✅ Precision-agnostic FP32 implementation

## Phase 3 Tasks: Gas + Radiation Pressure EOS

### TASK-021: Implement `radiation_gas.py`

**Goal**: Extend thermodynamics to include radiation pressure, critical for hot TDE debris.

**Implementation Requirements**:
1. Create `RadiationGas` class implementing EOS interface
2. Combined pressure: `P = P_gas + P_rad = (γ-1)ρu + (1/3)aT⁴`
3. Self-consistent temperature iteration (u depends on both gas and radiation contributions)
4. Sound speed including radiation: `c_s² = (γP_gas + (4/3)P_rad)/ρ`
5. Energy partitioning methods: `gas_energy()`, `radiation_energy()`

**Physical Constants**:
- Radiation constant: `a = 7.5657e-15 erg/(cm³·K⁴)` in CGS
- Support dimensionless units (a scaled by code units)

**Interface Consistency**:
- Must implement all `EOS` abstract methods
- Temperature solver should handle both radiation-dominated and gas-dominated regimes
- Include convergence tolerance parameters for iterative T solution

**Testing Requirements**:
- Pure gas limit (low T) should match `IdealGas`
- Radiation-dominated limit (high T) should give P ∝ T⁴
- Smooth transition between regimes

**Cross-Module Expectations**:
- `core/energy_diagnostics.py` will query both gas and radiation energy separately
- `radiation/simple_cooling.py` will use temperature from this EOS
- Configuration must allow selection: `eos_type: "ideal_gas"` or `"radiation_gas"`

DO:
- Work ONLY inside `tde_sph/eos`.
- Implement ideal-gas and gas+radiation EOS variants.
- Provide functions/classes to map (density, internal energy, composition) to (pressure, temperature, sound speed)
- Ensure backward compatibility: existing `IdealGas` remains default

DO NOT:
- Implement gravity, metric, or SPH kernels.
- Decide timestepping; expose only thermodynamic quantities.
- Modify radiation cooling rates (that's `radiation/` module's job).
- Open or modify anything under the `prompts/` folder.
