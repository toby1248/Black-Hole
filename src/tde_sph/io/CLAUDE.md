# CLAUDE Instructions — IO module

Role: Handle simulation I/O, snapshot storage and diagnostics.

## Phase 1 & 2 Status (Complete)
- ✅ `HDF5Writer` for snapshot storage with compression
- ✅ Particle data schema with metadata support
- ✅ Read/write utilities and convenience functions

## Phase 3 Tasks: Diagnostic Outputs & Light Curves

### TASK-024: Implement `diagnostics.py`

**Goal**: Provide structured diagnostic outputs for energy evolution, fallback rates, and luminosity light curves.

**Implementation Requirements**:

1. **DiagnosticsWriter Class**:
   - Time-series outputs in CSV, HDF5, or JSON format
   - Columns: time, E_kinetic, E_potential, E_internal (gas, radiation, total), E_total, luminosity, fallback_rate, etc.
   - Automatic directory creation and file management
   - Append mode for long simulations

2. **Energy Time-Series Logging**:
   - Interface with `core/energy_diagnostics.py`
   - Log all energy components at each output
   - Compute and log energy conservation metrics (ΔE/E₀)

3. **Fallback Rate Estimation**:
   - **Simple proxy**: mass flux through radius shells
   - Count/mass of particles with `r < r_capture` and `E < 0` (bound)
   - Time derivative: `dM/dt = d/dt (Σ m_i for bound particles)`
   - Binning in radius to track debris stream return

4. **Luminosity Light Curves**:
   - Query `RadiationModel.luminosity()` each timestep
   - Store time-series: `(time, L_total, L_components...)`
   - Optionally bin by radius or temperature ranges
   - Peak luminosity tracking

5. **Thermodynamic Profiles**:
   - Radial profiles: ρ(r), T(r), P(r), u(r)
   - Angular momentum profiles
   - Velocity dispersion

6. **Output Formats**:
   - **CSV**: Human-readable, easy plotting (default for light curves)
   - **HDF5**: Efficient for large time-series, grouped by diagnostic type
   - **JSON**: Metadata and summary statistics

**Interface**:
```python
class DiagnosticsWriter:
    def __init__(self, output_dir, format='csv', **kwargs)
    def write_energy_diagnostic(self, time, energy_dict)
    def write_luminosity(self, time, luminosity, **components)
    def write_fallback_rate(self, time, fallback_rate, bound_mass)
    def write_radial_profile(self, time, radii, quantities_dict)
    def finalize()  # Close files, write metadata
```

**Cross-Module Expectations**:
- Receives data from `core/energy_diagnostics.py`
- Receives luminosity from `radiation/simple_cooling.py`
- Called by `Simulation.run()` at diagnostic output cadence
- Configuration: `diagnostic_output_interval`, `diagnostic_format`

**File Organization**:
```
output/
  diagnostics/
    energy.csv          # Time-series energy components
    luminosity.csv      # Light curve
    fallback.csv        # Fallback rate vs time
    profiles/
      profile_0000.h5   # Radial profiles at different times
    metadata.json       # Run metadata
```

**Testing Requirements**:
- Correct energy sum: E_kin + E_pot + E_int = E_total
- CSV format readable by pandas
- HDF5 format compatible with existing readers

DO:
- Work ONLY inside `tde_sph/io`.
- Implement snapshot writing (HDF5/Parquet) and diagnostics output (energies, fallback rates, etc.).
- Maintain stable schemas to keep external tools working.

DO NOT:
- Implement physics calculations (compute nothing, only log what's provided).
- Change visualisation logic (use visualisation module hooks instead).
- Open or modify anything under the `prompts/` folder.
