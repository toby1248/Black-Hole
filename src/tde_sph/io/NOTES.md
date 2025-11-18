# IO Module — Implementation Notes

## Phase 1 & 2 Implementation

### HDF5Writer (COMPLETE)
- ✅ Compressed snapshot storage
- ✅ Metadata support
- ✅ Read/write utilities
- ✅ Batch snapshot processing

## Phase 3 Implementation Notes

### DiagnosticsWriter
**Status**: ✅ COMPLETE

**Key Design Decisions**:
- CSV default format (pandas-compatible, immediate flush)
- HDF5 optional with buffering and extensible datasets
- JSON format for metadata and summary statistics
- Separate files for energy, luminosity, fallback rate

**File Schema**:
```
output/diagnostics/
  energy.csv          # Time-series: E_kin, E_pot, E_int, E_total, conservation
  luminosity.csv      # Time-series: L_total, [components]
  fallback.csv        # Time-series: dM/dt, M_bound
  profiles/
    profile_NNNNNN.h5 # Radial profiles at different times
  metadata.json       # Run metadata, creation/finalize times
```

**CSV Format**:
- Headers written on initialization
- Immediate flush after each write
- Append mode supported
- NaN for missing values

**HDF5 Format**:
- Grouped datasets: /energy/, /luminosity/, /fallback/
- Extensible datasets (maxshape=(None,))
- Compression: gzip
- Buffering for performance (default: write immediately)

**Methods Implemented**:
- `write_energy_diagnostic(time, energy_dict)`: Log all energy components
- `write_luminosity(time, L_total, **components)`: Log luminosity
- `write_fallback_rate(time, dM_dt, M_bound)`: Log fallback
- `write_radial_profile(time, radii, quantities)`: HDF5 profiles
- `finalize()`: Close files, flush buffers, write metadata
- Context manager support: `with DiagnosticsWriter(...) as dw:`

**Utility Functions**:
- `compute_fallback_rate(particles, bh_mass, r_capture)`: Compute bound/captured mass
  - Bound condition: E = ½v² - GM/r < 0
  - Returns: bound_mass, captured_mass, N_bound, N_captured

## Testing Notes
- CSV pandas compatibility
- Energy sum validation
- HDF5 schema stability
- Append mode correctness
