@dependencies [TASK0, TASK1]
# Task: Implement Comprehensive Diagnostics Tab

## Summary
Transform the placeholder diagnostics tab in the desktop GUI into a comprehensive data display showing particle statistics, performance metrics, and coordinate/metric data. This is the TOP PRIORITY feature (per user request "C top priority") and must be optimized for ultrawide monitor displays.

**Current State:**
- Diagnostics tab is empty placeholder at `gui/data_display.py:289-294`
- Only basic energy plots and minimal stats shown in existing panels

**Target State:**
- Comprehensive diagnostics with multiple sub-panels:
  - Particle Statistics: min/max/mean/stddev for all quantities
  - Performance Metrics: wall-clock time, steps/sec, GPU utilization
  - Coordinate/Metric Data: for GR simulations (BH mass, ISCO particles, etc.)
- Layout optimized for ultrawide displays (2560px+ width)
- All data updates in real-time from simulation signals

## Context Reference
**For complete environment context, see:**
- `../AI_PROMPT.md` - Contains full tech stack, architecture, coding conventions, and related code patterns

**Task-Specific Context:**
- **Files to modify:**
  - `gui/data_display.py:289-294` - Expand empty diagnostics tab placeholder
  - `gui/simulation_thread.py` - Extend `_report_progress()` to emit comprehensive stats dict
- **Pattern to follow:**
  - Reference: `gui/data_display.py:234-405` - Existing DataDisplayWidget with tabs
  - Use QTabWidget for organizing diagnostic categories
  - Use QTableWidget for structured data display
  - Use matplotlib FigureCanvas for embedded plots
- **Integration points:**
  - SimulationThread emits `progress_updated` signal with stats dict
  - Main window connects signal to diagnostics widget update method

## Complexity
Medium

## Dependencies
Depends on: [TASK0, TASK1]
Blocks: [TASK11]
Parallel with: [TASK3, TASK4, TASK5]

## Detailed Steps
1. **Extend SimulationThread data collection:**
   - In `gui/simulation_thread.py:_report_progress()`, expand stats dict to include:
     ```python
     stats = {
         # Existing basic stats
         'n_particles': len(particles.mass),
         'total_mass': np.sum(particles.mass),

         # NEW: Particle statistics (all quantities)
         'density': {'min': np.min(particles.density), 'max': np.max(particles.density),
                     'mean': np.mean(particles.density), 'std': np.std(particles.density)},
         'pressure': {...},
         'temperature': {...},
         'velocity_magnitude': {...},
         'smoothing_length': {...},

         # NEW: Performance metrics
         'wall_time': time.time() - self.start_time,
         'sim_time': current_time,
         'steps_per_sec': step_count / wall_time,
         'gpu_available': HAS_CUDA,  # From simulation config

         # NEW: Coordinate/metric data (if GR simulation)
         'metric_type': getattr(simulation, 'metric_type', 'Minkowski'),
         'bh_mass': getattr(simulation, 'black_hole_mass', None),
         'distances_from_bh': {'min': ..., 'max': ..., 'mean': ...},
         'particles_within_isco': count_within_isco,
     }
     ```

2. **Create diagnostics widget structure:**
   - In `gui/data_display.py`, replace placeholder (line 289-294) with new widget class
   - Create `DiagnosticsWidget(QWidget)` with sub-tabs:
     - Tab 1: Particle Statistics
     - Tab 2: Performance Metrics
     - Tab 3: Coordinate/Metric Data (conditionally shown if GR simulation)

3. **Implement Particle Statistics panel:**
   - Use QTableWidget with columns: Quantity | Min | Max | Mean | Std Dev
   - Rows for: density, pressure, temperature, velocity_magnitude, smoothing_length, sound_speed
   - Update values on `progress_updated` signal
   - Format numbers with scientific notation (e.g., 1.23e-4)

4. **Implement Performance Metrics panel:**
   - Use QFormLayout or QTableWidget
   - Display:
     - Wall-clock time vs simulation time (formatted as HH:MM:SS and seconds)
     - Steps per second (real-time performance)
     - GPU status: "CUDA Available" or "CPU Only"
     - Memory usage: particle array sizes (calculate from n_particles * attributes * sizeof(float32))
   - Update on each progress signal

5. **Implement Coordinate/Metric Data panel:**
   - Conditionally display if `metric_type != 'Minkowski'`
   - Show:
     - Coordinate system (Cartesian, Boyer-Lindquist)
     - Metric type (Schwarzschild, Kerr)
     - Black hole parameters (mass, spin)
     - Particle distance statistics from black hole
     - Count of particles within ISCO
   - Use QFormLayout for key-value pairs

6. **Optimize layout for ultrawide monitor:**
   - Use horizontal layouts where possible (multi-column tables)
   - Set minimum window width hint: 2560px
   - Ensure text is readable at default scaling
   - Test layout at 2560x1080 and 3440x1440 resolutions

7. **Connect to simulation signals:**
   - In main_window.py, connect `SimulationThread.progress_updated` to diagnostics widget
   - Ensure thread-safe updates (use Qt signals/slots, no direct GUI manipulation from thread)

8. **Add error handling:**
   - If stats dict is missing keys, display "N/A" instead of crashing
   - Handle case where simulation hasn't started (no data yet)

## Acceptance Criteria
- [ ] **C.1** Particle statistics displayed in tabular format:
  - [ ] Min/max/mean/stddev for: density, pressure, temperature, velocity_magnitude, smoothing_length, sound_speed
  - [ ] Energy components: kinetic, potential, internal, gravitational
  - [ ] Timestep information: current dt, CFL condition
- [ ] **C.2** Performance metrics displayed:
  - [ ] Wall-clock time vs simulation time
  - [ ] Steps per second
  - [ ] GPU status (CUDA available or CPU only)
  - [ ] Memory usage estimate
- [ ] **C.3** Coordinate/metric data displayed (for GR simulations):
  - [ ] Coordinate system and metric type
  - [ ] Black hole mass and spin
  - [ ] Particle distances from BH (min/max/mean)
  - [ ] Particles within ISCO count
- [ ] **C.4** Layout optimized for ultrawide:
  - [ ] Window minimum width 2560px
  - [ ] Multi-column layout uses horizontal space efficiently
  - [ ] All text readable at default scaling
- [ ] All data updates in real-time (every progress signal)
- [ ] No crashes if simulation not started or stats incomplete
- [ ] Unit tests for stats dict construction in SimulationThread

## Code Review Checklist
- [ ] Clear widget naming: `ParticleStatsWidget`, `PerformanceMetricsWidget`, etc.
- [ ] No blocking operations in update methods (fast UI updates)
- [ ] Thread-safe: all GUI updates via Qt signals/slots
- [ ] Error handling: missing stats keys don't crash
- [ ] Follows PyQt conventions: signals/slots, layouts, widgets
- [ ] Docstrings for all new classes and methods
- [ ] Type hints where applicable
- [ ] No hardcoded magic numbers (use named constants)

## Reasoning Trace
**Why Top Priority:**
- User explicitly stated "C top priority" (diagnostics focus)
- Desktop GUI for scientific analysis needs comprehensive data readout
- Ultrawide monitor optimization enables side-by-side comparison of data

**Design Decisions:**
- **QTableWidget over QTreeWidget:** Tabular data is naturally suited to table display
- **Sub-tabs over single scrolling view:** Organizes data by category, reduces clutter
- **Real-time updates:** Scientific users need to monitor simulation progress
- **Conditional metric panel:** Only show GR-specific data when relevant

**Layout Strategy for Ultrawide:**
- Use QSplitter to allow user adjustment of panel sizes
- Particle stats: 4-5 columns (Quantity | Min | Max | Mean | Std)
- Performance metrics: 2-column layout (Label | Value)
- Horizontal space allows more data visible without scrolling

**Performance Considerations:**
- Update frequency: every progress signal (typically every N steps)
- Avoid recomputing stats in GUI (do it in SimulationThread)
- Use batch updates (update entire table at once, not row-by-row)
