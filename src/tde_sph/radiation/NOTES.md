# Radiation Module — Development Notes

## Phase 3 Implementation Notes

### Simple Cooling Model (TASK-023)
- **Implementation status**: ✅ COMPLETE
- **Module**: `tde_sph/radiation/simple_cooling.py` (585 lines)
- **Tests**: `tests/test_simple_cooling.py` (23 tests, 100% pass rate)
- **Dependencies**: Independent (uses temperatures from EOS when provided)

#### Implementation Details
**Physics Modes Implemented**:
1. **Bremsstrahlung cooling** (optically thin): Λ_ff ≈ 1.4×10⁻²⁷ √T erg cm³ s⁻¹
2. **Radiative diffusion** (optically thick): t_diff ~ (3 κ ρ h²) / (16 σ T³)
3. **Blackbody photosphere**: L ≈ 4π R² σ T⁴ with R ~ h
4. **Viscous heating**: Tracks artificial viscosity dissipation
5. **Adiabatic mode**: No cooling for testing

**Key Features**:
- Cooling timescale limits to prevent runaway cooling: |du/dt| ≤ u / t_cool_min
- Cumulative radiated energy tracking: E_rad = ∫ L(t) dt
- Luminosity diagnostics: L_bol, L_mean, L_max
- Mode-agnostic interface for easy integration
- Float64 precision for T⁴ calculations to avoid overflow

**Physical Constants** (CGS):
- Stefan-Boltzmann: σ = 5.670374×10⁻⁵ erg cm⁻² s⁻¹ K⁻⁴
- Radiation constant: a = 7.5657×10⁻¹⁶ erg cm⁻³ K⁻⁴
- Electron scattering opacity: κ = 0.4 cm²/g (for diffusion)

#### Test Coverage
- ✅ All 4 cooling modes (bremsstrahlung, diffusion, blackbody, none)
- ✅ Physics scaling: √T for bremsstrahlung, T⁴ for blackbody, h² for area
- ✅ Cooling timescale limits prevent runaway
- ✅ Viscous heating integration
- ✅ Net rates (cooling + heating)
- ✅ Luminosity computation and cumulative energy tracking
- ✅ Parameter validation and error handling
- ✅ Array operations for various sizes (1 to 1000 particles)

#### Known Limitations
- **Simplified cooling functions**: Not full radiative transfer
  - Bremsstrahlung: Approximate power-law (real cooling curves are tabulated)
  - Diffusion: Local approximation (full FLD requires flux computation)
  - Blackbody: Uses h as photosphere radius (needs optical depth τ calculation)
- **No spectral information**: Only bolometric luminosity (future: energy bins)
- **Float32 overflow**: Fixed with float64 for T⁴ calculations
- **Production readiness**: Suitable for TDE simulations with T ~ 10⁴–10⁸ K

#### Integration Notes
- **Usage**: `SimpleCoolingModel(cooling_mode="bremsstrahlung")`
- **Required inputs**: density, temperature, internal_energy, masses
- **Optional inputs**: smoothing_lengths (diffusion/blackbody), viscous_dissipation (heating)
- **Returns**: CoolingRates dataclass with du_dt_cooling, du_dt_heating, du_dt_net, L_total

#### Next Steps
- TASK-024: Integrate luminosity into diagnostics I/O
- Future: Add tabulated cooling curves (e.g., Sutherland & Dopita)
- Future: Implement flux-limited diffusion (FLD) for radiative transfer
- Future: Add spectral energy distribution (SED) binning

### Agent Work Log
**2025-11-18**: Implemented SimpleCoolingModel with 4 cooling modes, 23 comprehensive tests (100% pass), fixed float32 overflow in blackbody cooling. Ready for integration.

---
