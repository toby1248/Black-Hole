# GUI Enhancements Summary

## Overview
This document describes the comprehensive GUI enhancements implemented for the TDE-SPH simulation package, including improved live data visualization, performance diagnostics, and physically accurate orbit initialization.

## Completed Features

### 1. Black Hole Potential Energy Display
**Location**: `gui/simulation_thread.py`, `gui/data_display.py`

**Implementation**:
- Separated BH potential energy from self-gravity potential
- Calculation: `potential_bh = -G * M_BH * sum(m_i / r_i)`
- Uses particle distances from BH (`r_from_bh`) computed during gravity solve
- Displayed in live statistics widget as separate row

**Purpose**: Enables tracking of gravitational binding energy between star particles and the black hole, essential for understanding tidal disruption dynamics.

### 2. Median Distance from Black Hole
**Location**: `gui/simulation_thread.py`, `gui/data_display.py`

**Implementation**:
- Computed using `np.median(r_from_bh)` where r_from_bh is array of particle distances
- Updated every progress report interval
- Displayed in statistics table with proper scientific notation

**Purpose**: Provides quick assessment of star-BH separation and disruption progress. Median is more robust than mean for distributions with outliers (e.g., escaped particles).

### 3. Percentile Evolution Plots
**Location**: `gui/data_display.py` - `PercentilePlotWidget` class

**Implementation**:
- New tab "Percentiles" with two subplots:
  - **Distance from BH percentiles**: Tracks 10th, 25th, 50th (median), 75th, 90th percentiles
  - **Total energy percentiles**: Same percentile breakdown for particle energies
- Visualization: Shaded regions for 10-90% (light) and 25-75% (darker), median as solid line
- Time-series data with automatic y-axis scaling

**Purpose**: 
- Reveals particle distribution evolution during disruption
- Distance percentiles show tidal tail formation and spreading
- Energy percentiles identify bound vs. unbound material

**Usage**:
```python
# In simulation thread:
distance_percentiles = np.percentile(r_from_bh, [10, 25, 50, 75, 90])
energy_percentiles = np.percentile(total_energies, [10, 25, 50, 75, 90])
```

### 4. Timing Diagnostics
**Location**: `src/tde_sph/core/simulation.py`, `gui/data_display.py` - `TimingDiagnosticsWidget`

**Implementation**:
- Instrumented `Simulation.compute_forces()` and `Simulation.step()` methods
- Tracks timing for:
  - Gravity solve
  - SPH density computation
  - SPH pressure forces
  - Integration (position/velocity update)
  - Diagnostics/output
  - I/O overhead
  - Total timestep duration
- GUI displays in 7-row table with millisecond precision
- Performance summary shows steps/second

**Timing Fields** (in `SimulationState`):
```python
timing_gravity: float = 0.0
timing_sph_density: float = 0.0
timing_sph_pressure: float = 0.0
timing_integration: float = 0.0
timing_diagnostics: float = 0.0
timing_io: float = 0.0
timing_total: float = 0.0
```

**Purpose**: 
- Identifies computational bottlenecks
- Guides optimization efforts
- Validates GPU acceleration effectiveness

### 5. Orbital Mechanics Setup
**Location**: `gui/simulation_thread.py` - `_initialize_simulation()` method

**Implementation**:
- Uses Keplerian orbit formulas to compute initial conditions
- Parameters:
  - `eccentricity` (e): 0 = circular, 0 < e < 1 = elliptical, e = 1 = parabolic, e > 1 = hyperbolic
  - `pericentre` (r_p): Closest approach in units of tidal radius
  - `starting_distance`: Initial position as multiple of periapsis

**Physics**:
```python
# Semi-major axis (elliptical orbits)
a = r_p / (1.0 - e)

# Specific orbital energy
E_orb = -G * M_BH / (2 * a)

# Velocity magnitude from vis-viva equation
v = sqrt(2 * (E_orb + G * M_BH / r))

# Angular momentum (conserved)
L = r_p * sqrt(G * M_BH * (1 + e) / r_p)

# Velocity decomposition
v_tangential = L / r
v_radial = sqrt(v^2 - v_tangential^2)
```

**Coordinate System**:
- Star positioned at (-r_init, 0, 0) (left of BH)
- Velocity: (+v_radial, -v_tangential, 0)
  - Radial component points toward BH (+x direction)
  - Tangential component in -y direction
  - Produces angular momentum in +z direction (right-hand rule)

**Validation**:
- Four test cases in `tests/test_orbit_setup.py`:
  1. Circular orbit (e=0)
  2. Elliptical orbit (0 < e < 1)
  3. Parabolic orbit (e=1)
  4. Angular momentum direction verification
- All tests verify energy conservation and proper velocity decomposition

**Purpose**: Enables exploration of different encounter geometries and their effect on disruption outcomes.

## Data Flow Architecture

```
Simulation (src/tde_sph/core/simulation.py)
    │
    ├─ Timing instrumentation
    │   ├─ time.time() before/after each module
    │   └─ Store in SimulationState.timing_*
    │
    └─ State snapshot
        │
        ▼
SimulationThread (gui/simulation_thread.py)
    │
    ├─ Orbit initialization (once)
    │   └─ Compute position/velocity from e, r_p, starting_distance
    │
    ├─ Progress reporting (every N steps)
    │   ├─ Compute BH potential
    │   ├─ Calculate median distance
    │   ├─ Generate percentiles (distance & energy)
    │   ├─ Package timing data
    │   └─ Emit progress signal
    │
    └─ progress.emit(step, t, dt, stats)
        │
        ▼
MainWindow (gui/main_window.py)
    │
    └─ Route signal to display
        │
        ▼
DataDisplayWidget (gui/data_display.py)
    │
    ├─ StatisticsWidget: Display BH potential, median distance
    ├─ EnergyPlotWidget: Energy evolution charts
    ├─ PercentilePlotWidget: Distance & energy percentiles
    └─ TimingDiagnosticsWidget: Module execution times
```

## Configuration Updates

The orbit setup now requires/supports these parameters:

```yaml
orbit:
  pericentre: 4.0              # In units of tidal radius
  eccentricity: 0.95           # Orbital eccentricity (0-1 typical)
  starting_distance: 3.0       # In units of periapsis
```

**Defaults** (if not specified):
- `eccentricity`: 0.95 (highly eccentric, typical for TDEs)
- `starting_distance`: 3.0 (starts 3× periapsis distance)

## Testing

### Orbit Setup Tests
**File**: `tests/test_orbit_setup.py`

**Coverage**:
- Circular orbits (e=0)
- Elliptical orbits (0 < e < 1)
- Parabolic orbits (e=1)
- Angular momentum conservation
- Energy conservation

**Status**: All 4 tests passing ✓

### Config Tests
**Files**: `tests/test_config.py`, `configs/newtonian_tde.yaml`

**Fixes**:
- Corrected `neighbour_search_method` from `"newtonian"` to `"bruteforce"`

**Status**: 28/28 tests passing ✓

## Known Issues

### Barnes-Hut Tree
**Issue**: Numba compilation error in `tests/test_barnes_hut.py`
```
ValueError: The truth value of an array with more than one element is ambiguous
```

**Impact**: Tree-based gravity solver tests fail
**Workaround**: Use `neighbour_search_method: "bruteforce"` in configs
**Status**: Requires Numba debugging (separate from GUI enhancements)

### GUI Tests
**Issue**: PyQt6/PyQt5 not installed in test environment
**Status**: 32 GUI tests skipped (expected in headless environments)
**Solution**: Install PyQt6 for interactive testing: `uv add --dev PyQt6`

## Performance Considerations

### Timing Overhead
- Timing instrumentation adds ~microseconds per measurement
- Negligible compared to actual computation (milliseconds)
- Can be disabled by commenting out `time.time()` calls if needed

### Percentile Computation
- O(n log n) due to sorting in `np.percentile()`
- For 10k particles: ~0.1 ms overhead
- Acceptable for progress reporting (not in inner loop)

### GUI Update Rate
- Progress signals emitted every `log_interval` (typically 0.005 time units)
- Matplotlib redraws can be expensive for high-frequency updates
- Recommended: `log_interval >= 0.005` to avoid GUI sluggishness

## Usage Examples

### Running with New Features
```bash
# Launch GUI
cd gui
python main_window.py

# Or via run script
run.bat

# Load config with orbit parameters
# File -> Open -> select configs/newtonian_tde.yaml

# Modify eccentricity/starting_distance in config editor
# Run simulation and observe:
# - Live Data -> Percentiles tab for distribution evolution
# - Diagnostics -> Timing for performance profiling
# - Statistics panel for BH potential and median distance
```

### Programmatic Access
```python
from gui.simulation_thread import SimulationThread
from tde_sph.config import load_config

# Load config
config = load_config("configs/newtonian_tde.yaml")

# Override orbit parameters
config.orbit_pericentre = 5.0
config.orbit_eccentricity = 0.8
config.orbit_starting_distance = 4.0

# Create thread
thread = SimulationThread(config)

# Connect to progress signal
def on_progress(step, t, dt, stats):
    print(f"Step {step}: t={t:.4f}")
    print(f"  Median distance: {stats['median_distance_from_bh']:.2e}")
    print(f"  BH potential: {stats['energies']['potential_bh']:.2e}")
    print(f"  Gravity time: {stats['timings']['gravity']:.3f} ms")

thread.progress.connect(on_progress)
thread.start()
```

## Future Enhancements

### Suggested Additions
1. **Orbital Element Tracking**: Compute semi-major axis, eccentricity evolution for individual particles
2. **Phase Space Plots**: Position-velocity diagrams in 2D projections
3. **Radial Profile Plots**: Density, pressure vs. distance from BH
4. **GPU Timing**: Extend timing instrumentation to `_compute_forces_gpu()`
5. **Export Functionality**: Save percentile/timing data to CSV for offline analysis

### Configuration Extensions
```yaml
visualization:
  show_percentile_plots: true
  show_timing_diagnostics: true
  update_interval: 0.01  # Seconds between GUI updates
  export_percentiles: true
  export_timing: true
```

## References

### Orbital Mechanics
- Vis-viva equation: Murray & Dermott (1999) "Solar System Dynamics"
- Tidal disruption orbits: Rees (1988) Nature 333, 523

### Implementation Files
- `gui/simulation_thread.py`: Lines 180-235 (orbit setup), 280-350 (progress reporting)
- `gui/data_display.py`: Lines 1-100 (PercentilePlotWidget), 100-150 (TimingDiagnosticsWidget)
- `src/tde_sph/core/simulation.py`: Lines 180-220 (timing fields), 350-400 (instrumentation)
- `tests/test_orbit_setup.py`: Validation suite for orbital mechanics

## Changelog

### Version: GUI Enhancements (2025-01-XX)
- ✅ Added BH potential energy calculation and display
- ✅ Added median distance from BH tracking
- ✅ Implemented percentile evolution plots (distance & energy)
- ✅ Added timing diagnostics for performance profiling
- ✅ Implemented Keplerian orbit setup with eccentricity/periapsis
- ✅ Fixed angular momentum direction in orbit initialization
- ✅ Created comprehensive test suite for orbit setup
- ✅ Fixed config validation error in `newtonian_tde.yaml`

### Contributors
- Primary development: AI Agent (Beast mode)
- Testing & validation: Automated test suite
- Physics validation: Keplerian orbit mechanics

---
**Document Status**: Complete
**Last Updated**: 2025-01-XX
**Next Review**: After user testing and feedback
