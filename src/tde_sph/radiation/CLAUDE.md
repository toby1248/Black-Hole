# CLAUDE Instructions — radiation module

Role: Model radiative cooling, diffusion and luminosity diagnostics.

## Phase 3 Tasks: Simple Radiative Cooling & Luminosity

### TASK-023: Implement `simple_cooling.py`

**Goal**: Provide basic radiative cooling and luminosity estimation for TDE simulations.

**Implementation Requirements**:

1. **SimpleCooling Class** (implements `RadiationModel` interface):
   - Cooling rate: `du/dt = -Λ(ρ, T)`
   - Multiple cooling models selectable via config:
     - **Free-free (bremsstrahlung)**: `Λ_ff ∝ ρ² T^(1/2)` (optically thin)
     - **Blackbody**: `Λ_bb = σ_SB T⁴` (optically thick)
     - **Power-law**: `Λ = Λ₀ (T/T₀)^β` (simple parameterization)
   - Cooling timescale estimation: `t_cool = u / |du/dt|`

2. **Luminosity Calculation**:
   - Total luminosity: `L = Σ(m_i |du/dt|_i)` (sum over all particles)
   - Effective temperature from total luminosity
   - Radiated energy accounting

3. **Opacity-Based Regimes** (optional, simple):
   - Optical depth proxy: `τ ≈ κ ρ h` (h = smoothing length)
   - Switch between optically thick/thin cooling
   - Default: simple uniform opacity `κ = κ₀`

4. **Safety Features**:
   - Cooling timestep limiter: `Δt_cool = min(t_cool) * safety_factor`
   - Subcycling or implicit update for fast cooling
   - Prevent negative internal energy
   - Temperature floor (e.g., 100 K)

**Physical Constants** (CGS):
- Stefan-Boltzmann: `σ_SB = 5.67e-5 erg/(cm²·s·K⁴)`
- Electron scattering opacity: `κ_es = 0.34 cm²/g`
- Free-free cooling coefficient (hydrogen plasma)

**Interface Implementation**:
```python
class SimpleCooling(RadiationModel):
    def cooling_rate(self, density, temperature, internal_energy, **kwargs) -> NDArrayFloat
    def luminosity(self, density, temperature, internal_energy, masses, **kwargs) -> float
    def cooling_timescale(self, density, temperature, internal_energy, **kwargs) -> NDArrayFloat
    def apply_cooling(self, particles, eos, dt, **kwargs) -> NDArrayFloat  # returns Δu
```

**Configuration Parameters**:
- `cooling_model`: "free_free" | "blackbody" | "power_law" | "none"
- `cooling_timescale_factor`: safety factor for timestep (default: 0.1)
- `temperature_floor`: minimum temperature in K (default: 100)
- `opacity`: uniform opacity in cm²/g (default: 0.34)

**Cross-Module Expectations**:
- Uses `EOS.temperature()` from `eos/` module
- Called by `Simulation` or `TimeIntegrator` to update internal energies
- Luminosity logged by `io/diagnostics.py`
- Should work with both `IdealGas` and `RadiationGas` EOS

**Testing Requirements**:
- Cooling timescale matches analytic estimates
- Energy conservation: radiated energy = lost internal energy
- No negative internal energies
- Reasonable fallback luminosity curves for TDE (compare to literature)

DO:
- Work ONLY inside `tde_sph/radiation`.
- Implement simple cooling laws and, later, approximate diffusion / FLD schemes.
- Provide luminosity and energy-loss rate estimates that can be queried by the core simulation.

DO NOT:
- Implement hydrodynamic forces or gravity.
- Hard-code unit conversions or global configuration logic.
- Modify EOS thermodynamics (query, don't change).
- Open or modify anything under the `prompts/` folder.
