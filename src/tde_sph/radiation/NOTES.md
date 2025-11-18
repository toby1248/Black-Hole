# Radiation Module — Implementation Notes

## Phase 3 Implementation

### SimpleCooling Model
**Status**: ✅ COMPLETE

**Key Design Decisions**:
- Three cooling models: free-free, blackbody, power-law (+ 'none')
- Explicit timestep update with energy floor limiter
- Temperature floor: 100 K default (configurable)
- Optical depth proxy: τ ≈ κ ρ h (not yet used in model selection)

**Physical Implementations**:
- **Free-free**: Λ_ff = 1.4e-27 ρ² T^(1/2) [erg/(cm³·s)]
  - Specific rate: du/dt = -Λ_ff/ρ [erg/(g·s)]
  - Valid for optically thin gas
- **Blackbody**: Λ_bb = σ_SB T⁴ [erg/(cm²·s)]
  - Surface area estimate: A ≈ 4πh²
  - Volume-averaged cooling
- **Power-law**: Λ = Λ₀ (T/T₀)^β
  - Parameterized cooling for flexibility

**Safety Features**:
1. Temperature floor enforcement (zero cooling below T_floor)
2. Energy floor: retain at least 1% of current internal energy
3. Cooling timescale: t_cool = u / |du/dt|
4. Suggested timestep: Δt = 0.1 * min(t_cool)

**Methods Implemented**:
- `cooling_rate()`: Compute du/dt [erg/(g·s)]
- `luminosity()`: Total L = Σ(m_i |du/dt|_i) [erg/s]
- `cooling_timescale()`: Per-particle t_cool
- `apply_cooling()`: Safe energy update with limiters
- `suggested_timestep()`: Cooling-based dt constraint

**Integration with EOS**:
- Works with both IdealGas and RadiationGas
- Queries temperature via EOS.temperature()
- Optional internal_energy_from_temperature() for floor enforcement

## Testing Notes
- Cooling timescale validation
- Energy conservation tracking
- Luminosity curve comparison with TDE literature
- Temperature floor stability
