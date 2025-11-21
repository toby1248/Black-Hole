## PROMPT
Implement comprehensive diagnostics panel in the desktop GUI showing real-time particle statistics, performance metrics, and coordinate/metric data. This is the TOP PRIORITY feature (user specified "C top priority"). Replace the empty placeholder at `gui/data_display.py:289-294` with a rich multi-tab widget optimized for ultrawide monitor displays (2560px+ width).

## COMPLEXITY
Medium

## CONTEXT REFERENCE
**For complete environment context, read:**
- `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md` - Contains full tech stack, architecture, project structure, coding conventions, and related code patterns

**You MUST read AI_PROMPT.md before executing this task to understand the environment.**

## TASK-SPECIFIC CONTEXT

### Current State Analysis
**Empty placeholder location:**
- `gui/data_display.py:289-294` - Diagnostics tab with only QLabel "Diagnostics panel"

**Existing reference implementation:**
- `gui/data_display.py:234-405` - DataDisplayWidget with tabs for energy plots
- Follow this pattern: QTabWidget with sub-panels, matplotlib embedding, real-time updates

**Data source:**
- `gui/simulation_thread.py:_report_progress()` - Emits progress_updated signal with stats dict
- Currently minimal stats (particle count, total mass, energies)
- Must be extended with comprehensive statistics

### Files This Task Will Touch
**Primary modifications:**
- `gui/data_display.py:289-294` - Replace placeholder with DiagnosticsWidget implementation
- `gui/simulation_thread.py:_report_progress()` - Extend stats dict with comprehensive data

**New code (within data_display.py):**
- `class ParticleStatsWidget(QWidget)` - Particle statistics table
- `class PerformanceMetricsWidget(QWidget)` - Performance data display
- `class CoordinateDataWidget(QWidget)` - GR-specific coordinate/metric data
- `class DiagnosticsWidget(QWidget)` - Main container with sub-tabs

**Test file:**
- `gui/data_display.test.py` - Unit tests for diagnostics widget updates
- `gui/simulation_thread.test.py` - Tests for stats dict construction

### Patterns to Follow

**Widget structure (from data_display.py:234-405):**
```python
class DiagnosticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Create tab widget for organizing categories
        tabs = QTabWidget()
        tabs.addTab(ParticleStatsWidget(self), "Particle Statistics")
        tabs.addTab(PerformanceMetricsWidget(self), "Performance")
        tabs.addTab(CoordinateDataWidget(self), "Coordinates/Metric")

        layout.addWidget(tabs)

    def update_diagnostics(self, stats: dict):
        """Called on progress_updated signal"""
        # Update all sub-widgets with new stats
        pass
```

**Stats dict extension (in simulation_thread.py):**
```python
def _report_progress(self):
    stats = {
        # Existing
        'n_particles': len(self.simulation.particles.mass),
        'total_mass': np.sum(self.simulation.particles.mass),

        # NEW: All particle quantities with min/max/mean/std
        'density': self._compute_stats(self.simulation.particles.density),
        'temperature': self._compute_stats(self.simulation.particles.temperature),
        'velocity_magnitude': self._compute_stats(
            np.linalg.norm(self.simulation.particles.velocity, axis=1)
        ),
        # ... etc

        # NEW: Performance
        'wall_time': time.time() - self.start_time,
        'steps_per_sec': self.step_count / elapsed,

        # NEW: GR data (if applicable)
        'metric_type': getattr(self.simulation, 'metric_type', 'Minkowski'),
    }
    self.progress_updated.emit(current_time, step, energies, stats)

def _compute_stats(self, array):
    """Helper: compute min/max/mean/std for array"""
    return {
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
    }
```

**Table widget for particle stats:**
```python
class ParticleStatsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Create table: rows = quantities, columns = min/max/mean/std
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ['Quantity', 'Min', 'Max', 'Mean', 'Std Dev']
        )

        quantities = ['Density', 'Pressure', 'Temperature',
                     'Velocity Magnitude', 'Smoothing Length', 'Sound Speed']
        self.table.setRowCount(len(quantities))

        for i, qty in enumerate(quantities):
            self.table.setItem(i, 0, QTableWidgetItem(qty))

        layout.addWidget(self.table)

    def update_stats(self, stats: dict):
        """Update table with new statistics"""
        for i, key in enumerate(['density', 'pressure', 'temperature', ...]):
            if key in stats:
                self.table.setItem(i, 1, QTableWidgetItem(f"{stats[key]['min']:.3e}"))
                self.table.setItem(i, 2, QTableWidgetItem(f"{stats[key]['max']:.3e}"))
                # ... etc
```

### Integration Points
- **Signal/slot connection (in main_window.py):**
  ```python
  self.sim_thread.progress_updated.connect(self.diagnostics_widget.update_diagnostics)
  ```
- **Thread safety:** All GUI updates happen in main thread via signals
- **Performance:** Stats computed in SimulationThread (background), GUI just displays

### Ultrawide Layout Optimization
**Target resolutions:**
- 2560x1080 (21:9 ultrawide)
- 3440x1440 (21:9 ultrawide)

**Layout strategy:**
- Particle stats table: 5 columns fit comfortably at 2560px width
- Performance metrics: 2-column form layout (label | value)
- Use QSplitter for user-adjustable panel sizes
- Set `setMinimumWidth(2560)` on main window (or make configurable)

### Data to Display

**Particle Statistics tab:**
- Quantities: density, pressure, temperature, velocity_magnitude, smoothing_length, sound_speed
- For each: min, max, mean, std dev
- Energy components: kinetic, potential, internal, gravitational (from existing energies dict)
- Timestep info: current dt, CFL condition

**Performance Metrics tab:**
- Wall-clock time (HH:MM:SS format)
- Simulation time (seconds)
- Steps per second
- GPU status: "CUDA Available" or "CPU Only"
- Memory usage: estimate from n_particles * num_attributes * 4 bytes (FP32)

**Coordinate/Metric Data tab (conditionally shown):**
- Only display if `metric_type != 'Minkowski'`
- Coordinate system (Cartesian, Boyer-Lindquist)
- Metric type (Schwarzschild, Kerr)
- Black hole mass and spin
- Particle distances from BH (min/max/mean)
- Particles within ISCO (count and percentage)

## EXTRA DOCUMENTATION

### Testing Strategy
**Unit tests (gui/data_display.test.py):**
1. Test DiagnosticsWidget updates without crashing when stats dict has all keys
2. Test graceful handling of missing stats keys (display "N/A")
3. Test table formatting (scientific notation for small/large numbers)

**Unit tests (gui/simulation_thread.test.py):**
1. Test `_compute_stats()` helper function returns correct min/max/mean/std
2. Test stats dict includes all required keys
3. Mock simulation and verify progress signal emits comprehensive stats

**Manual testing:**
1. Launch desktop GUI
2. Load `configs/schwarzschild_tde.yaml`
3. Start simulation
4. Verify diagnostics tab populates with real-time data
5. Check all values are non-zero and reasonable
6. Test on ultrawide monitor (verify layout uses space efficiently)

### Implementation Checklist
- [ ] Extend `_report_progress()` in simulation_thread.py with comprehensive stats
- [ ] Create `_compute_stats()` helper method
- [ ] Create ParticleStatsWidget with QTableWidget
- [ ] Create PerformanceMetricsWidget
- [ ] Create CoordinateDataWidget (conditional display logic)
- [ ] Create DiagnosticsWidget container with tabs
- [ ] Connect progress_updated signal to update_diagnostics slot
- [ ] Set minimum window width for ultrawide optimization
- [ ] Add error handling for missing stats keys
- [ ] Write unit tests for stats computation and widget updates
- [ ] Manual test on ultrawide monitor

## LAYER
1 (Desktop GUI Enhancement - parallel)

## PARALLELIZATION
Parallel with: [TASK3, TASK4, TASK5]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Must use Qt signals/slots for thread-safe GUI updates (no direct manipulation from thread)
- Do NOT block main thread with heavy computation (compute stats in SimulationThread)
- Layout must be responsive and not cause horizontal scrolling at 2560px width
- Follow PyQt6 conventions (or PyQt5 if project uses that)
- Use QTableWidget (not custom painting) for simplicity
- All numeric values should use scientific notation (format: `.3e` or similar)
- Error handling: missing stats keys should display "N/A", not crash
- Test ONLY changed files (data_display.py, simulation_thread.py)
- Follow PEP 8 style and existing code conventions
