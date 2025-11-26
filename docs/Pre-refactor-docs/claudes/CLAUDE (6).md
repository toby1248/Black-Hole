# CLAUDE Instructions — radiation module

Role: Model radiative cooling, diffusion and luminosity diagnostics.

## Phase 1 & 2 Status
- ✅ Module structure created
- ❌ No radiation physics implemented yet

## Phase 3 Goals: Simple Cooling & Luminosity

### TASK-023: Simple Radiative Cooling / Luminosity Model

**Objective**: Implement `simple_cooling.py` with local cooling functions and luminosity proxies.

**Physics Options** (implement 2-3):
1. **Optically thin cooling**: L = Λ(ρ, T) × ρ² × V (bremsstrahlung, line cooling)
2. **Optically thick diffusion**: dE/dt = -∇·(κ∇T) approximated as local radiative diffusion
3. **Blackbody escape**: L ≈ σ T⁴ × A_eff (photosphere approximation)
4. **Viscous heating proxy**: track artificial viscosity dissipation as heating source

**Implementation Requirements**:
1. Create `SimpleCoolingModel` class inheriting from `RadiationModel` interface
2. Provide methods:
   - `compute_cooling_rate(particles, **kwargs)` → du/dt per particle
   - `compute_luminosity(particles, **kwargs)` → total L, L(r), L(energy)
   - `compute_heating_rate(particles, **kwargs)` → viscous heating
3. Support multiple cooling modes via config flags
4. Track cumulative radiated energy
5. Provide luminosity vs time and radius diagnostics

**Luminosity Diagnostics**:
- Integrate local dissipation/cooling over particles
- Fallback rate proxy: Ṁ(t) = dM/dt crossing r_fallback
- Bolometric luminosity vs time
- Spectral energy distribution (simple bins)

**Tests**:
- Cooling timescale validation
- Energy conservation (internal + radiated = constant)
- Luminosity scales correctly with temperature/density
- Adiabatic limit (cooling off) conserves energy

**Cross-module Dependencies**:
- Requires `eos/radiation_gas.py` for temperature
- Called by `core/simulation.py` during evolution
- Outputs logged by `io/diagnostics.py`

DO:
- Work ONLY inside `tde_sph/radiation`.
- Implement simple cooling laws and, later, approximate diffusion / FLD schemes.
- Provide luminosity and energy-loss rate estimates that can be queried by the core simulation.
- Document cooling prescriptions and regimes clearly
- Add unit tests for cooling rates and energy conservation

DO NOT:
- Implement hydrodynamic forces or gravity.
- Hard-code unit conversions or global configuration logic.
- Open or modify anything under the `prompts/` folder.
- Modify files outside `tde_sph/radiation/`
