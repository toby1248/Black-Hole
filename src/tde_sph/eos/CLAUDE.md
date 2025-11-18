# CLAUDE Instructions — EOS module

Role: Provide equation-of-state (EOS) and thermodynamic closures.

## Phase 1 & 2 Status (Complete)
- ✅ Ideal gas EOS implemented in `ideal_gas.py`
- ✅ Support for adiabatic index γ
- ✅ Basic pressure, temperature, sound speed calculations

## Phase 3 Goals: Radiation Pressure & Advanced Thermodynamics

### TASK-021: Combined Gas + Radiation Pressure EOS

**Objective**: Implement `radiation_gas.py` with radiation pressure for optically thick gas.

**Physics**:
- Total pressure: P = P_gas + P_rad = (γ - 1) u ρ + (1/3) a T⁴
- Radiation constant: a = 4σ/c = 7.5657e-16 erg cm⁻³ K⁻⁴
- Consistent internal energy and temperature handling
- Support both gas-dominated and radiation-dominated regimes

**Implementation Requirements**:
1. Create `RadiationGasEOS` class inheriting from `EOS` interface
2. Provide methods:
   - `compute_pressure(rho, u, **kwargs)` → pressure
   - `compute_temperature(rho, u, **kwargs)` → temperature
   - `compute_sound_speed(rho, u, **kwargs)` → c_s
   - `compute_internal_energy(rho, T, **kwargs)` → u (iterative solve)
3. Handle mixed gas+radiation regime with Newton-Raphson for T(ρ,u)
4. Document regime transitions and numerical stability
5. FP32 default, FP64 for temperature iterations if needed

**Tests**:
- Gas-dominated limit matches ideal gas
- Radiation-dominated limit follows Stefan-Boltzmann
- Temperature-energy consistency
- Smooth transitions between regimes
- Validate against analytic stellar structure

**Cross-module Dependencies**:
- Used by `sph/hydro_forces.py` for pressure forces
- Used by `radiation/simple_cooling.py` for temperature
- Integrated in `core/simulation.py` via config

DO:
- Work ONLY inside `tde_sph/eos`.
- Implement ideal-gas and gas+radiation EOS variants.
- Provide functions/classes to map (density, internal energy, composition) to (pressure, temperature, sound speed)
- Add comprehensive docstrings with physics equations
- Include unit tests for all regimes

DO NOT:
- Implement gravity, metric, or SPH kernels.
- Decide timestepping; expose only thermodynamic quantities.
- Open or modify anything under the `prompts/` folder.
- Modify files outside `tde_sph/eos/`
