# CLAUDE Instructions — IO module

Role: Handle simulation I/O, snapshot storage and diagnostics.

## Phase 1 & 2 Status (Complete)
- ✅ HDF5 snapshot writing in `hdf5.py`
- ✅ Particle data export (positions, velocities, masses, etc.)
- ✅ Basic metadata storage

## Phase 3 Goals: Diagnostic Outputs & Light Curves

### TASK-024: Light Curve & Fallback Rate Diagnostics

**Objective**: Implement `diagnostics.py` for comprehensive diagnostic outputs.

**Diagnostic Types**:

1. **Fallback Rate**:
   - Ṁ_fb(t) = mass crossing r_fallback per unit time
   - Track mass vs specific energy distribution
   - Estimate return time distribution

2. **Luminosity vs Time** (Light Curve):
   - L_bol(t) from `radiation.compute_luminosity()`
   - L(E) spectral distribution (simple energy bins)
   - Peak luminosity and time

3. **Energy Evolution**:
   - E_kin(t), E_pot(t), E_int(t), E_rad(t), E_tot(t)
   - From `core/energy_diagnostics.py`
   - Conservation error tracking

4. **Orbital Elements**:
   - Per-particle: apocenter, pericenter, eccentricity, inclination
   - Mass-weighted distributions
   - Evolution of debris cloud shape

5. **Thermodynamic Profiles**:
   - ρ(r), T(r), P(r), u(r) radial profiles
   - Entropy distribution
   - Shock identification (entropy jumps)

**Output Formats**:
- **HDF5 time series**: `/diagnostics/light_curve`, `/diagnostics/fallback_rate`, `/diagnostics/energies`
- **CSV summaries**: `diagnostics/light_curve.csv`, `diagnostics/fallback.csv`
- **Snapshot metadata**: include diagnostic summary in each snapshot

**Implementation Requirements**:
1. Create `DiagnosticsWriter` class
2. Provide methods:
   - `write_light_curve(time, luminosity, **kwargs)`
   - `write_fallback_rate(time, M_dot, **kwargs)`
   - `write_energy_evolution(time, energies, **kwargs)`
   - `write_orbital_elements(particles, **kwargs)`
   - `write_radial_profiles(particles, bins, **kwargs)`
3. Support incremental writing (append to time series)
4. Provide summary statistics (peak L, total radiated E, etc.)

**Tests**:
- Diagnostic output format validity
- Time series monotonicity
- Energy conservation cross-check
- Fallback rate units and normalization

**Cross-module Dependencies**:
- Reads from `core/energy_diagnostics.py`
- Reads from `radiation/simple_cooling.py`
- Integrated in `core/simulation.py` I/O loop

DO:
- Work ONLY inside `tde_sph/io`.
- Implement snapshot writing (HDF5/Parquet) and diagnostics output (energies, fallback rates, etc.).
- Maintain stable schemas to keep external tools working.
- Add comprehensive diagnostic outputs for Phase 3
- Document all output formats clearly

DO NOT:
- Implement physics calculations.
- Change visualisation logic (use visualisation module hooks instead).
- Open or modify anything under the `prompts/` folder.
- Modify files outside `tde_sph/io/`
