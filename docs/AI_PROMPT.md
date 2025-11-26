# AI_PROMPT.md - GUI Enhancement and Debugging Task

## 🎯 Purpose

Transform the TDE-SPH simulator's GUI applications (desktop PyQt6 and web Three.js) from prototype state to production-ready visualization tools. The ultimate goal is to create a **fully functional, debugged, and visually rich** simulation interface that maximizes the RTX 4090's ray-tracing capabilities for volumetric rendering while providing comprehensive diagnostic data for scientific analysis.

**Business/Technical Value:**
- Enable real-time monitoring and debugging of relativistic SPH simulations
- Leverage RT hardware for volumetric lighting visualization (web app)
- Provide scientists with comprehensive diagnostic data for validation and analysis
- Create a shippable product for ultrawide monitor workflows (desktop app)

**Success Definition:**
- All GUI buttons are fully implemented (no placeholders)
- Desktop app displays comprehensive diagnostic/debug data optimized for ultrawide displays
- Web app renders particle data with WebGL-based volumetric lighting effects
- All temperature data bugs and similar issues are identified and fixed
- Both GUIs are stable, tested, and production-ready

---

## 📁 Environment & Codebase Context

### Tech Stack
- **Language:** Python 3.9+ (desktop app), JavaScript (web app)
- **Desktop GUI Framework:** PyQt6/PyQt5 (cross-compatible)
- **Web Visualization:** Three.js r128 with WebGL
- **Backend:** Flask (planned for web server mode)
- **Physics Engine:** Custom SPH framework with CUDA/GPU acceleration
- **Data Format:** HDF5 for snapshots
- **Build Tools:** pip, npm (minimal web dependencies)
- **Platform:** Windows 10/11 with RTX 4090 GPU, ultrawide monitor

### Project Structure

```
Black-Hole/
├── gui/                          # Desktop PyQt6 application
│   ├── main_window.py           # Main application window with tabs
│   ├── config_editor.py         # YAML configuration editor
│   ├── control_panel.py         # Simulation control (start/stop/pause)
│   ├── data_display.py          # Live data plots (matplotlib embedded)
│   ├── simulation_thread.py     # Background simulation execution
│   └── web_viewer.py            # Embedded web viewer widget
│
├── web/                          # Web-based 3D visualizer
│   ├── index.html               # Main HTML structure
│   ├── css/style.css            # Styling
│   └── js/
│       ├── app.js               # Main application logic
│       ├── visualizer.js        # Three.js rendering engine
│       └── data_loader.js       # HDF5/JSON data loading
│
├── src/tde_sph/                 # Core physics engine
│   ├── core/
│   │   ├── simulation.py        # Simulation orchestrator
│   │   └── energy_diagnostics.py # Energy conservation tracking
│   ├── sph/
│   │   ├── particles.py         # ParticleSystem class
│   │   └── kernels.py           # SPH kernel functions
│   ├── eos/
│   │   └── ideal_gas.py         # Equation of state (temperature calculations)
│   ├── metric/
│   │   └── coordinates.py       # Coordinate transformations
│   ├── io/
│   │   ├── hdf5.py              # HDF5 snapshot I/O
│   │   └── diagnostics.py       # Diagnostic output
│   └── visualization/
│       ├── plotly_3d.py         # 3D plotting
│       ├── pyqtgraph_3d.py      # PyQtGraph integration
│       └── interpolation.py     # Point cloud to volumetric conversion
│
├── configs/                      # YAML configuration files
│   ├── newtonian_tde.yaml
│   └── schwarzschild_tde.yaml
│
└── outputs/                      # Simulation output directory
    └── [snapshots in HDF5 format]
```

### Architecture Pattern

**Desktop App (PyQt6):**
- **Main Window** (`main_window.py`): Central orchestrator with tab widget
  - Tab 1: Configuration Editor (YAML editing)
  - Tab 2: 3D Visualization (embedded web viewer)
- **Left Dock**: Control Panel (start/stop/pause, progress tracking)
- **Right Dock**: Data Display (energy plots, statistics)
- **Background Thread**: `SimulationThread` runs physics simulation without blocking GUI

**Web App (Three.js):**
- **Left Sidebar**: Data source selection, animation controls, visualization settings
- **Center Canvas**: WebGL rendering of particle point cloud
- **Planned Enhancement**: Volumetric rendering with ray-traced lighting

**Data Flow:**
1. User configures simulation via YAML editor
2. SimulationThread initializes physics modules (gravity, EOS, integrator)
3. Simulation runs in background, emitting progress signals
4. Control panel and data display update in real-time
5. Snapshots written to HDF5 files
6. Web viewer loads HDF5 data and renders with Three.js

### Key Files

**Entry Points:**
- `gui/main_window.py:781` - Desktop app entry point (`main()` function)
- `web/index.html` - Web app entry point

**Current Simulation Integration:**
- `gui/simulation_thread.py` - Background simulation execution
- `src/tde_sph/core/simulation.py` - Core simulation orchestrator
- `src/tde_sph/sph/particles.py` - ParticleSystem data structure

**Temperature Bug Location:**
- `gui/simulation_thread.py:262-263` - Temperature access using `hasattr()` check
- `src/tde_sph/sph/particles.py:114-117` - ParticleSystem does NOT initialize temperature array
- `src/tde_sph/core/simulation.py:517-523` - Temperature computation exists but conditionally stores

### Existing Conventions

**Code Style:**
- Follow PEP 8 for Python
- Type hints for function signatures
- Docstrings in NumPy style
- FP32 arrays for GPU compatibility (not FP64)
- Error handling with try/except blocks

**Naming:**
- Snake_case for functions/variables: `compute_density()`, `particle_system`
- PascalCase for classes: `ParticleSystem`, `Simulation`
- ALL_CAPS for constants: `HAS_CUDA`, `PYQT_VERSION`

**Testing Pattern:**
- Tests co-located with source files: `foo.py` → `foo.test.py` (but GUI tests may be in `tests/test_gui.py`)
- Mock external dependencies (file I/O, network, time)
- No separate `__tests__/` directory

### Current State

**What Already Exists:**

✅ **Desktop GUI (Partial):**
- Main window with tabs (config editor, 3D viewer)
- Control panel with start/stop/pause buttons (functional)
- Data display with matplotlib energy plots (functional)
- Simulation thread integration (functional but basic)
- File operations (open/save config) (functional)
- Settings persistence (window geometry) (functional)

✅ **Web GUI (Partial):**
- HTML structure with sidebar controls
- Three.js scene setup (assumed in visualizer.js)
- Data source selection UI
- Animation controls (play/pause/step) (placeholder)
- Color mapping options (placeholder)
- Camera controls (placeholder)

**What Needs Work:**

❌ **Desktop GUI Issues:**
- Diagnostics tab is empty placeholder (`data_display.py:289-294`)
- Preferences dialog not implemented (`main_window.py:726`)
- Limited diagnostic data displayed (only basic energy + stats)
- No metric unit conversion option
- Temperature data access bug (see below)
- Web viewer widget may not be fully functional

❌ **Web GUI Issues:**
- No JavaScript implementation files exist (js/ folder empty or incomplete)
- No WebGL/OpenGL volumetric rendering
- All buttons likely non-functional (need JS implementation)
- No server connection mode implemented
- No actual Three.js rendering code

❌ **Data/Logic Bugs:**
- **Temperature Bug:** `ParticleSystem` does not have `temperature` attribute initialized
  - `gui/simulation_thread.py:262-263` uses `hasattr()` fallback to 0.0
  - `src/tde_sph/core/simulation.py:517-523` computes temperature but only stores if attribute exists
  - **Root cause:** `particles.py:114-117` initializes `density`, `pressure`, `sound_speed` but NOT `temperature`
  - **Expected behavior:** Temperature should be computed and accessible like density/pressure
- **Other similar bugs:** Need investigation (per user request)

### Dependencies

**Internal Modules:**
- `tde_sph.core.simulation` - Simulation orchestrator
- `tde_sph.sph.particles` - ParticleSystem data
- `tde_sph.eos.ideal_gas` - Temperature calculations
- `tde_sph.io.hdf5` - Snapshot I/O
- `tde_sph.visualization.interpolation` - Point cloud to volumetric conversion

**External Packages (Python):**
- PyQt6 or PyQt5 (GUI framework)
- matplotlib (embedded plots)
- numpy (array operations)
- h5py (HDF5 I/O)
- yaml (config parsing)

**External Packages (Web):**
- Three.js r128 (3D rendering)
- OrbitControls.js (camera manipulation)

### Integration Points

**Desktop → Simulation:**
- `SimulationThread` creates `Simulation` object from config dict
- Signals emitted: `progress_updated(time, step, energies, stats)`
- Main window receives signals and updates UI

**Desktop → Web:**
- `WebViewerWidget` embeds web viewer (likely via QWebEngineView)
- Could load local files or connect to Flask server

**Web → Data:**
- Load HDF5 snapshots via File API or fetch from server
- Parse particle positions, velocities, densities, temperatures
- Convert to Three.js point cloud geometry

**Simulation → HDF5:**
- Snapshots written via `tde_sph.io.hdf5`
- Contains: positions, velocities, masses, densities, energies, temperatures

---

### Files to Modify

**Desktop App - Core Files:**
- `gui/data_display.py` - Expand diagnostics tab, add metric conversion
- `gui/control_panel.py` - Add debug/diagnostic data display
- `gui/main_window.py` - Implement preferences dialog, connect buttons
- `gui/simulation_thread.py` - Fix temperature access, add more diagnostic data

**Desktop App - Bug Fixes:**
- `src/tde_sph/sph/particles.py` - Add temperature attribute initialization
- Search for similar bugs in:
  - `src/tde_sph/core/simulation.py` - Check all computed quantities are stored
  - `src/tde_sph/eos/ideal_gas.py` - Verify temperature calculation correctness
  - `src/tde_sph/metric/coordinates.py` - Check coordinate transform bugs

**Web App - Full Implementation:**
- `web/js/app.js` - Main application logic, button handlers
- `web/js/visualizer.js` - Three.js scene setup, rendering loop, volumetric effects
- `web/js/data_loader.js` - HDF5/JSON loading, data parsing
- `web/css/style.css` - Ensure layout works for visualization

**Web App - New Files (if needed):**
- `web/js/shaders/volumetric.vert` - Volumetric vertex shader (optional)
- `web/js/shaders/volumetric.frag` - Volumetric fragment shader (optional)

### Files That Should NOT Be Changed

- `src/tde_sph/gravity/*.py` - Gravity solvers (out of scope)
- `src/tde_sph/integration/*.py` - Time integrators (out of scope)
- `src/tde_sph/ICs/*.py` - Initial conditions (out of scope)
- Core simulation logic unless fixing bugs

---

## ✅ Acceptance Criteria

Each criterion is independently testable and must be verified:

### 1. Desktop GUI - Diagnostic Data Enhancement

**Top Priority (User Specified):**
- [ ] **C.1** Diagnostics tab displays comprehensive particle statistics in tabular format:
  - Particle count, total mass, total energy (already present)
  - Energy components: kinetic, potential, internal, gravitational (all displayed)
  - Min/max/mean/stddev for: density, pressure, sound speed, temperature, velocity magnitude
  - Smoothing length statistics
  - Timestep information (current dt, CFL condition)
- [ ] **C.2** Diagnostics tab includes live performance metrics:
  - Wall-clock time vs simulation time
  - Steps per second
  - GPU utilization (if CUDA available)
  - Memory usage (particle arrays size)
- [ ] **C.3** Diagnostics tab shows coordinate/metric data (for GR simulations):
  - Coordinate system (Cartesian, Boyer-Lindquist)
  - Metric type (Minkowski, Schwarzschild, Kerr)
  - Black hole mass and spin
  - Particle distances from black hole (min/max/mean)
  - Number of particles within ISCO
- [ ] **C.4** Window layout optimized for ultrawide monitor:
  - Main window minimum width: 2560px (adjustable)
  - Diagnostics panel uses horizontal space efficiently (multi-column layout)
  - All text readable at default scaling

**Second Priority (User Specified):**
- [ ] **A.1** Metric unit conversion option implemented:
  - Toggle button in diagnostics panel: "Show Dimensionless / Show Physical Units"
  - When physical units selected, convert:
    - Lengths: code units → solar radii (R☉) or gravitational radii (Rg)
    - Masses: code units → solar masses (M☉)
    - Times: code units → seconds or orbital periods
    - Energies: code units → ergs
    - Temperatures: code units → Kelvin
  - Conversion factors derived from simulation config (BH mass, star mass/radius)
  - Units displayed next to values (e.g., "1.23e6 K" instead of "1.23e-3")

### 2. Desktop GUI - Placeholder Button Implementation

**Mandatory Implementation (User Priority A and B):**
- [ ] **B.1** Preferences dialog (`main_window.py:726`):
  - Opens modal dialog window
  - Settings categories: General, Visualization, Performance
  - General: Default config directory, auto-save interval
  - Visualization: Default colormap, point size, camera settings
  - Performance: GPU usage preference, thread count
  - Settings persisted via `QSettings` (already used in code)
  - Apply/Cancel/OK buttons functional
- [ ] **B.2** All menu actions functional:
  - Edit → Undo/Redo (already delegated to config editor)
  - Help → Documentation (opens browser to repo wiki)
  - Help → About (already implemented with QMessageBox)
- [ ] **B.3** Web viewer integration:
  - `web_viewer.py` widget functional
  - Embeds web app via QWebEngineView
  - Can load simulation snapshots from output directory
  - Refresh button to reload current snapshot

**Future Design Consideration (User Priority C):**
- [ ] **B.4** Architecture supports future extensions:
  - Plugin system for custom diagnostics panels (documented in code comments)
  - Extensible settings schema (use dict-based config, easy to add fields)
  - Modular widget structure (each diagnostic panel is separate QWidget subclass)

### 3. Web GUI - WebGL/OpenGL Graphical Enhancement

**Ultimate Goal - Volumetric Lighting (User Priority C):**
- [ ] **V.1** Particle data converted to volumetric representation:
  - Use `src/tde_sph/visualization/interpolation.py` to create 3D grid from point cloud
  - Grid resolution: configurable (default 128³)
  - Interpolation method: SPH kernel-weighted averaging
  - Support density, temperature, velocity magnitude fields
- [ ] **V.2** WebGL volumetric rendering shader implemented:
  - Fragment shader performs ray marching through volume
  - Transfer function maps density/temperature to color and opacity
  - Lighting model: basic directional light + ambient occlusion
  - RT hardware acceleration: leverage `RTX 4090` via WebGPU (if browser supports) or optimized WebGL
- [ ] **V.3** Volumetric rendering controls in UI:
  - Ray marching step size slider
  - Opacity scaling slider
  - Transfer function presets (density-focused, temperature-focused, hybrid)
  - Light direction controls (azimuth, elevation)
  - Toggle volumetric mode vs point cloud mode
- [ ] **V.4** Performance optimization for RTX 4090:
  - Use WebGL 2.0 or WebGPU for compute shaders
  - Tile-based rendering for large volumes
  - Level-of-detail (LOD) for distant regions
  - FPS counter displays actual frame rate (target: 60 fps)

**Note on Data Conversion:**
- If point cloud → volume conversion is too slow, implement in Python backend and precompute
- Store volumetric data in HDF5 alongside particle data
- Web app can load precomputed volumes

### 4. Web GUI - Full Button Implementation

**Data Source Panel:**
- [ ] **D.1** "Load Files" button functional:
  - Triggers HTML5 File API to select `.h5` or `.hdf5` files
  - Reads HDF5 using h5wasm or similar JS library
  - Parses particle data (positions, velocities, densities, temperatures)
  - Populates snapshot slider with loaded snapshots
  - Updates statistics table
- [ ] **D.2** "Connect to Server" button functional:
  - Validates server URL
  - Sends GET request to check server availability
  - If successful, fetches snapshot list via REST API
  - Displays connection status (connected/disconnected)
- [ ] **D.3** "Load Demo Data" button functional:
  - Generates synthetic particle data in JavaScript
  - Creates expanding sphere or tidal disruption demo
  - Useful for testing without simulation data

**Animation Control Panel:**
- [ ] **D.4** Play/Pause buttons functional:
  - Play: starts animation loop through snapshots at specified FPS
  - Pause: stops animation, maintains current snapshot
  - State toggle: play ↔ pause
- [ ] **D.5** Step forward/backward buttons functional:
  - Step forward: advance to next snapshot
  - Step backward: go to previous snapshot
  - Disabled at start/end of snapshot list
- [ ] **D.6** Snapshot slider functional:
  - Dragging slider updates current snapshot index
  - Releases trigger re-render of particle positions
  - Current index displayed as "X / Y"
  - Time display updates to show simulation time of current snapshot
- [ ] **D.7** FPS slider functional:
  - Sets animation playback speed (1-60 fps)
  - Updates interval for animation timer
  - FPS value displayed next to slider

**Visualization Panel:**
- [ ] **D.8** Color-by dropdown functional:
  - Options: density, temperature, internal_energy, velocity_magnitude, pressure, entropy
  - Changing selection recomputes particle colors
  - Updates colormap legend (if displayed)
- [ ] **D.9** Colormap dropdown functional:
  - Options: viridis, plasma, inferno, hot, cool, rainbow
  - Changes color transfer function
  - Re-renders particles with new colors
- [ ] **D.10** Point size slider functional:
  - Adjusts Three.js PointsMaterial size property
  - Range: 0.5 to 10.0 (screen-space pixels)
  - Immediate visual feedback
- [ ] **D.11** Logarithmic scale checkbox functional:
  - When checked: color mapping uses log scale for selected quantity
  - When unchecked: linear scale
  - Useful for density (spans many orders of magnitude)
- [ ] **D.12** Show black hole checkbox functional:
  - Toggles visibility of black hole sphere at origin
  - Black hole radius: Schwarzschild radius (2M) or event horizon
- [ ] **D.13** Show axes checkbox functional:
  - Toggles visibility of coordinate axes helper

**Camera Panel:**
- [ ] **D.14** Reset view button functional:
  - Resets camera to default position and orientation
  - Default: looking down z-axis at origin
- [ ] **D.15** Top/Side view buttons functional:
  - Top view: camera along +y axis, looking down at xy plane
  - Side view: camera along +x axis, looking at yz plane

**Export Panel:**
- [ ] **D.16** Screenshot button functional:
  - Captures current canvas as PNG image
  - Downloads file to user's computer
  - Filename includes timestamp and snapshot index
- [ ] **D.17** Export JSON button functional:
  - Exports current snapshot data as JSON
  - Format: `{time, particles: {positions, velocities, densities, temperatures}}`
  - Useful for external analysis

### 5. Bug Fixes - Temperature and Similar Issues

**Temperature Bug (Confirmed):**
- [ ] **T.1** ParticleSystem initializes temperature array:
  - Add `self.temperature = np.zeros(n_particles, dtype=np.float32)` in `particles.py:__init__()`
  - Add shape validation in `_validate_shapes()`
  - Add getter method `get_temperature()`
- [ ] **T.2** Simulation always computes and stores temperature:
  - Remove `hasattr()` check in `simulation.py:523`
  - Always execute `self.particles.temperature = temperature`
  - Temperature available from first step onwards
- [ ] **T.3** SimulationThread reports temperature correctly:
  - Remove `hasattr()` fallback in `simulation_thread.py:262-263`
  - Directly access `self.simulation.particles.temperature`
  - Mean and max temperature displayed in diagnostics

**Investigate Similar Bugs (User Request: "all the other bugs like it"):**
- [ ] **T.4** Verify all computed quantities are stored:
  - Check: velocity_magnitude (currently not stored)
  - Check: entropy (may be computed by EOS but not stored)
  - Check: acceleration (computed in gravity solver, possibly not stored in particles)
  - Check: du/dt (change in internal energy, needed for diagnostics)
  - For each missing quantity: add attribute to ParticleSystem, compute in Simulation
- [ ] **T.5** Fix coordinate transformation bugs:
  - Review `src/tde_sph/metric/coordinates.py`
  - Verify Cartesian ↔ Boyer-Lindquist transformations for Kerr metric
  - Test with known orbits (circular at ISCO)
  - Fix any sign errors or missing terms
- [ ] **T.6** Fix HDF5 I/O bugs:
  - Ensure all particle attributes are written to snapshots
  - Verify attribute names match between writer and reader
  - Test round-trip: write snapshot, read back, compare arrays
- [ ] **T.7** Document all fixes in commit messages and code comments

### 6. Edge Cases and Error Scenarios

**Desktop GUI:**
- [ ] **E.1** Simulation fails to start:
  - Display error message in QMessageBox
  - Log full traceback to control panel log
  - Reset UI state (buttons enabled/disabled correctly)
- [ ] **E.2** Config file is invalid YAML:
  - Show error dialog with syntax error details
  - Prevent simulation start
  - Highlight error line in config editor (if possible)
- [ ] **E.3** No snapshots available:
  - Diagnostics display "No data" placeholders
  - Web viewer shows message: "No snapshots loaded"
- [ ] **E.4** GPU not available:
  - Display warning but continue with CPU mode
  - Performance metrics show "GPU: N/A"

**Web GUI:**
- [ ] **E.5** HDF5 file is corrupted:
  - Display error message overlay
  - Log error to browser console
  - Clear previous data, reset UI state
- [ ] **E.6** Server connection fails:
  - Show connection error message
  - Retry button available
  - Fallback to local file mode
- [ ] **E.7** WebGL not supported:
  - Detect browser capability on load
  - Display fallback message: "WebGL required"
  - Suggest compatible browsers
- [ ] **E.8** Large datasets (>1M particles):
  - Implement particle downsampling for display
  - Show actual particle count vs displayed count
  - Performance degradation warning if FPS < 30

### 7. Testing and Validation

**Manual Testing Checklist:**
- [ ] **M.1** Desktop app launches without errors on Windows 10/11
- [ ] **M.2** Load example config (`configs/schwarzschild_tde.yaml`)
- [ ] **M.3** Start simulation, verify progress updates in real-time
- [ ] **M.4** All diagnostics panels populate with live data
- [ ] **M.5** Metric unit conversion toggles correctly
- [ ] **M.6** Preferences dialog saves and restores settings
- [ ] **M.7** Web app opens in Chrome/Firefox/Edge
- [ ] **M.8** Load demo data, verify particles render
- [ ] **M.9** Play animation, verify smooth playback at 30 fps
- [ ] **M.10** Change colormap, verify particle colors update
- [ ] **M.11** Toggle volumetric mode (if implemented), verify rendering
- [ ] **M.12** Export screenshot, verify image quality

**Automated Testing (Minimal):**
- [ ] **A.1** Unit tests for temperature initialization (`test_particles.py`)
- [ ] **A.2** Unit tests for metric conversion functions
- [ ] **A.3** Integration test: load config, start simulation, verify temperature in first progress update
- [ ] **A.4** (Optional) Selenium/Playwright test for web app button interactions

---

## ⚙️ Implementation Guidance

### Execution Layers

**Layer 0 (Foundation) - MUST complete first:**
1. **Fix temperature bug in ParticleSystem:**
   - Add `temperature` attribute to `particles.py:__init__()`
   - Update `_validate_shapes()`
   - This unblocks all temperature-related features
2. **Investigate and fix similar bugs:**
   - Search codebase for other missing attributes (velocity_magnitude, entropy, etc.)
   - Add to ParticleSystem and Simulation as needed
3. **Create diagnostic data collection infrastructure:**
   - Extend `SimulationThread._report_progress()` to emit comprehensive stats dict
   - Include: all particle stats, performance metrics, coordinate data

**Layer 1 (Desktop GUI Enhancement) - Can parallelize:**
1. **Diagnostics tab implementation:**
   - Create `DiagnosticsWidget` with sub-tabs for different categories
   - Particle stats, performance metrics, coordinate data
   - Use QTableWidget and matplotlib for display
2. **Metric unit conversion:**
   - Create conversion utility module: `gui/unit_conversion.py`
   - Functions: `code_to_physical(value, unit_type, config)` and reverse
   - Add toggle button to each diagnostic panel
3. **Preferences dialog:**
   - Create `gui/preferences_dialog.py` with QDialog
   - Settings categories as tab widget
   - Persist to QSettings

**Layer 2 (Web GUI Implementation) - Can parallelize with Layer 1:**
1. **JavaScript infrastructure:**
   - `app.js`: Application state manager, event handlers
   - `data_loader.js`: HDF5 parsing (use h5wasm library), demo data generator
   - `visualizer.js`: Three.js scene setup, rendering loop
2. **Button implementations:**
   - Data source panel: file loading, server connection
   - Animation controls: play/pause, step, slider
   - Visualization controls: colormap, point size, toggles
3. **Volumetric rendering (advanced):**
   - Interpolation: call Python backend or implement in JS
   - Shader development: ray marching, transfer functions
   - Performance optimization: LOD, tile rendering

**Layer 3 (Integration and Polish):**
1. **Web viewer widget integration:**
   - Ensure `WebViewerWidget` embeds web app correctly
   - Synchronize with simulation thread (auto-refresh on new snapshots)
2. **End-to-end testing:**
   - Run full simulation pipeline: configure → run → visualize
   - Test all UI interactions
   - Verify data correctness (compare GUI values to HDF5 files)
3. **Documentation and cleanup:**
   - Update docstrings
   - Add code comments for complex logic
   - Remove debug print statements

### Expected Artifacts

**Code Files:**
- **Modified:**
  - `gui/data_display.py` - Enhanced diagnostics
  - `gui/control_panel.py` - Additional debug displays
  - `gui/main_window.py` - Preferences dialog integration
  - `gui/simulation_thread.py` - Comprehensive stats reporting
  - `src/tde_sph/sph/particles.py` - Temperature and other attributes
  - `src/tde_sph/core/simulation.py` - Bug fixes for attribute storage
- **New (Desktop):**
  - `gui/preferences_dialog.py` - Settings dialog
  - `gui/unit_conversion.py` - Metric conversion utilities
  - `gui/diagnostics_widget.py` - Comprehensive diagnostics panel (optional, can extend data_display.py)
- **New (Web):**
  - `web/js/app.js` - Application logic
  - `web/js/visualizer.js` - Three.js rendering
  - `web/js/data_loader.js` - Data loading
  - `web/js/utils.js` - Utility functions (colormap, interpolation)
  - `web/js/shaders/volumetric.frag` - Volumetric fragment shader (optional)
  - `web/js/shaders/volumetric.vert` - Volumetric vertex shader (optional)

**Configuration:**
- No changes to YAML config schema expected
- `QSettings` keys extended for new preferences

**Documentation:**
- Code comments explaining new features
- Docstrings for new functions/classes
- (Optional) Update README with GUI usage instructions

### Constraints

**What NOT to do:**
- ❌ Do NOT modify core physics algorithms (gravity, SPH, integration) unless fixing bugs
- ❌ Do NOT change YAML config format (backward compatibility required)
- ❌ Do NOT add heavyweight dependencies (keep web app lightweight)
- ❌ Do NOT implement features not in acceptance criteria (stay focused)
- ❌ Do NOT use FP64 for particle data (GPU performance)

**Performance Requirements:**
- Desktop GUI: Responsive at all times (simulation on background thread)
- Web rendering: Maintain 30+ fps for <100k particles, 15+ fps for <1M particles
- Volumetric rendering: Target 30 fps at 128³ resolution on RTX 4090

**Security Considerations:**
- Web app: Validate all user input (file uploads, server URLs)
- Desktop app: Validate YAML before loading (prevent code injection)
- No eval() or exec() in Python or JavaScript

**Backward Compatibility:**
- Existing YAML configs must still load and run
- HDF5 snapshots from old code must still be readable
- Settings from previous app versions should migrate gracefully

---

## 🔍 Verification and Traceability

### Requirement Mapping

Each user requirement from the original request must be traceable to acceptance criteria:

**User Request:** "Debug and fix both the GUIs"
- **Maps to:** Acceptance criteria T.1-T.7 (bug fixes)
- **Maps to:** Acceptance criteria E.1-E.8 (error handling)
- **Verification:** All identified bugs documented and fixed, error scenarios tested

**User Request:** "Implement all the placeholder buttons"
- **Maps to:** Acceptance criteria B.1-B.4 (desktop), D.1-D.17 (web)
- **Verification:** Manual testing checklist M.1-M.12, no placeholder text remains in UI

**User Request:** "Add much more debug and diagnostic data readout utility to the desktop app"
- **Maps to:** Acceptance criteria C.1-C.4 (diagnostics), A.1 (unit conversion)
- **Verification:** Diagnostics tab shows comprehensive data, unit conversion tested

**User Request:** "Much more graphical fidelity with OpenGL to the web app"
- **Maps to:** Acceptance criteria V.1-V.4 (volumetric rendering), D.8-D.13 (visualization controls)
- **Verification:** Volumetric mode functional, visual comparison to point cloud mode

**User Clarification:** "C top priority, A second priority" (diagnostics focus)
- **Reflected in:** Layer 0 and Layer 1 prioritize desktop diagnostics
- **Layer 2 (web) can be done in parallel or deferred if time-constrained**

**User Clarification:** "Ultrawide monitor - make the window as wide as you like"
- **Maps to:** Acceptance criteria C.4 (layout optimization)
- **Verification:** Window tested at 2560px+ width, all panels visible

**User Clarification:** "Ultimate goal is volumetric lighting using RT hardware"
- **Maps to:** Acceptance criteria V.1-V.4
- **Verification:** WebGL volumetric renderer implemented, FPS measured on RTX 4090

**User Clarification:** "Investigate temperature bug and all other bugs like it"
- **Maps to:** Acceptance criteria T.1-T.7
- **Verification:** Code search for missing attributes, all found and fixed

**User Clarification:** "Design with extensibility in mind for future"
- **Maps to:** Acceptance criteria B.4 (plugin architecture notes)
- **Verification:** Code comments document extension points

### Self-Verification Checklist

Before marking task complete, verify:

- [ ] Every acceptance criterion has a corresponding implementation
- [ ] All placeholder TODOs removed from code
- [ ] All identified bugs have fix commits with clear messages
- [ ] Manual testing checklist fully executed
- [ ] No regressions in existing functionality (old configs still work)
- [ ] Code follows project conventions (snake_case, docstrings, FP32)
- [ ] Git commits are logical and well-described
- [ ] User can launch both GUIs and complete a full workflow without errors

---

## 🧠 Reasoning Boundaries

### System Coherence Over Engineering

**Prefer:**
- ✅ Extending existing widgets (`DataDisplayWidget`) over creating entirely new architectures
- ✅ Using established PyQt patterns (QTabWidget, QTableWidget, QSettings)
- ✅ Following existing Three.js examples for WebGL implementation
- ✅ Simple, direct solutions for button handlers (event listeners → state update → render)

**Avoid:**
- ❌ Over-engineered state management (no Redux/Vuex unless clearly beneficial)
- ❌ Complex dependency injection for UI components (keep it simple)
- ❌ Premature abstraction (don't create frameworks)

### Preserve Existing Logic

**Do NOT change:**
- Existing simulation logic flow (initialization → step loop → snapshot)
- Signal/slot architecture in PyQt (already well-designed)
- HDF5 schema (ensure compatibility)

**DO extend:**
- Add new signals for additional diagnostic data
- Add new attributes to ParticleSystem (with backward compatibility checks)
- Add new UI panels without breaking existing ones

### When Uncertain

**Ask rather than assume:**
- If HDF5 schema is unclear, document the assumption and add a TODO to verify
- If WebGL implementation path is ambiguous (native Three.js vs custom shaders), document both options and choose simplest first
- If performance target is unrealistic (e.g., 60 fps for 10M particles), note the limitation and propose fallback

**Fallback principles:**
- Functionality > polish (working buttons > fancy animations)
- Stability > features (robust error handling > advanced visualization)
- User needs > aesthetics (comprehensive diagnostics > beautiful UI)

### Debugging Strategy

**For temperature bug:**
1. Trace data flow: EOS computes T → Simulation stores T → GUI displays T
2. Identify break point: ParticleSystem lacks temperature attribute
3. Fix at root cause: Initialize attribute in ParticleSystem
4. Verify fix: Check all downstream consumers work

**For "other bugs like it":**
1. Search pattern: attributes accessed with `hasattr()` or try/except AttributeError
2. Check ParticleSystem.__init__() for missing initializations
3. Check Simulation update methods for computed-but-not-stored quantities
4. Document findings and fix systematically

**For volumetric rendering:**
1. Start simple: basic point cloud with Three.js Points
2. Add complexity: custom shader for better appearance
3. Final step: volumetric ray marching (may be out of scope if too complex)
4. Performance first: ensure 30 fps before adding features

---

## 📋 Testing Strategy (Minimal & Relevant)

### Philosophy
Test changed code with minimum sufficient evidence. Focus on functionality, not coverage theater.

### Testing Scope

**Unit Tests (Required):**
- **Temperature fix:**
  - Test: `ParticleSystem` initializes temperature array
  - Test: Temperature has correct shape and dtype (float32)
- **Unit conversion:**
  - Test: Code units → physical units conversion (known values)
  - Test: Round-trip conversion (code → physical → code)
- **Diagnostic data collection:**
  - Test: `_report_progress()` emits all expected keys in stats dict

**Integration Tests (Selective):**
- **Simulation → GUI:**
  - Test: Start simulation, verify first progress signal contains temperature data
  - Mock: File I/O (no actual HDF5 writes)
- **Config → Simulation:**
  - Test: Load YAML config, initialize Simulation, verify all modules created
  - Mock: Time (use fake clock)

**Manual Tests (Critical):**
- All acceptance criteria in section 7 (M.1 - M.12)
- Visual verification of volumetric rendering
- Performance profiling on RTX 4090

### Coverage Target
- 100% of **changed code** covered (feasible for small changes)
- Temperature bug fix: 100% line coverage of modified `particles.py` and `simulation.py` sections
- Unit conversion: 100% of conversion functions
- GUI code: Manual testing only (PyQt difficult to unit test)

### What NOT to Test
- Unchanged physics modules (already tested)
- Third-party libraries (Three.js, PyQt)
- Browser compatibility (manual check only)

---

## 🚀 Additional Context

### User Environment
- **Hardware:** RTX 4090 GPU, ultrawide monitor (2560px+ width)
- **OS:** Windows 10/11
- **Python:** 3.9+ (likely 3.11 based on recent code)
- **Browser:** Chrome/Firefox (WebGL 2.0 support)

### User Workflow
1. Open desktop GUI
2. Load or create YAML config
3. Start simulation
4. Monitor diagnostics in real-time
5. Open 3D viewer (web app embedded or standalone)
6. Load snapshots from output directory
7. Visualize with volumetric rendering
8. Export screenshots/data for publication

### Success Metrics
- **Functional:** All buttons work, no crashes, data displays correctly
- **Performance:** Desktop GUI responsive (<100ms input lag), Web app 30+ fps
- **Usability:** Scientist can run full pipeline without consulting documentation
- **Visual:** Volumetric rendering clearly superior to point cloud (subjective but noticeable)

### Known Limitations
- HDF5 in JavaScript: Limited libraries, may need Python backend for complex operations
- WebGL volumetric rendering: May be performance-limited for very large datasets (>512³ grid)
- RTX hardware acceleration in browser: Limited, may rely on general GPU compute

### Future Extensions (Out of Scope)
- Multi-snapshot comparison view
- Collaborative features (share simulations via cloud)
- VR/AR visualization
- Real-time simulation control (parameter tweaking mid-run)

---

## 📌 Summary

This AI prompt provides a **complete, context-rich specification** for transforming the TDE-SPH GUI applications from prototype to production-ready. The downstream agent should:

1. **Start with Layer 0**: Fix temperature bug and investigate similar issues
2. **Prioritize desktop diagnostics**: Implement comprehensive data display with unit conversion
3. **Implement all placeholder buttons**: Desktop preferences, web controls
4. **Enhance web visualization**: WebGL volumetric rendering with RT optimization
5. **Validate thoroughly**: Manual testing checklist, bug verification
6. **Document changes**: Clear commit messages, code comments

The agent has all necessary context to execute autonomously:
- Codebase structure and architecture
- Existing patterns to follow
- Specific files to modify
- Bug locations with root cause analysis
- Acceptance criteria mapping to user requirements
- Performance targets and constraints

**Expected outcome:** A fully functional, debugged, and visually rich simulation interface that leverages ultrawide displays and RTX 4090 hardware for scientific visualization.
