# CLAUDE Instructions — visualisation module

Role: Provide advanced 3D visualisation with PyQtGraph, export helpers, and interpolation.

## Phase 1 & 2 Status (Complete)
- ✅ Plotly 3D scatter plots in `plotly_3d.py`
- ✅ Basic particle cloud visualization
- ✅ Color mapping by density, velocity, etc.

## Phase 3 Goals: PyQtGraph Upgrade & Advanced Features

### TASK-037: PyQtGraph 3D Visualizer with Time-Scrubbing

**Objective**: Implement `pyqtgraph_3d.py` to replace Plotly with interactive, high-performance visualization.

**Features Required**:
1. **Real-time 3D rendering**:
   - Use PyQtGraph `GLViewWidget` for hardware-accelerated 3D
   - Scatter plots for particles with variable sizes/colors
   - Optional iso-surfaces for density/temperature
   - Camera controls: pan, zoom, rotate

2. **Time-scrubbing animation**:
   - Load multiple snapshots into memory
   - Timeline slider to scrub through simulation time
   - Play/pause/step controls
   - Frame rate control

3. **Color mapping**:
   - Map particle properties to colors: density, temperature, velocity, energy
   - Custom colormaps (viridis, plasma, hot, custom)
   - Color scale controls (linear, log, percentile)
   - Legend with units

4. **Transparency & smoothing**:
   - Alpha blending for overlapping particles
   - Smooth particle rendering (SPH kernel-weighted splats)
   - Adjustable particle sizes

**Implementation Requirements**:
1. Create `PyQtGraphVisualizer` class
2. Provide methods:
   - `load_snapshots(snapshot_files, **kwargs)` → load sequence
   - `render_frame(snapshot_index, color_by, **kwargs)` → display
   - `export_frame(filename, resolution, **kwargs)` → save image
3. GUI controls: sliders, buttons, dropdowns for interactive control
4. Support millions of particles efficiently (LOD, downsampling)

### TASK-101: Visualization Library & Menu System

**Objective**: Implement `viz_library.py` with comprehensive 2D/3D visualization options.

**Visualization Types**:

**3D Options**:
- Point cloud (particles)
- Iso-surfaces (density, temperature)
- Volume rendering (ray marching)
- Streamlines (velocity field)
- Vector fields (velocity arrows)

**2D Options** (projections):
- Column density maps (∫ ρ dl along axis)
- Velocity maps
- Temperature/pressure maps
- Radial profiles ρ(r), T(r), v(r)
- Energy distribution histograms

**Matplotlib Integration**:
- High-quality publication plots
- 2D slices through 3D data
- Radial profiles and time series
- Energy evolution plots

**Menu System**:
- Expose all options in GUI dropdown menus
- Presets: "Density 3D", "Temperature slice", "Velocity profile", etc.
- Save/load visualization configs

**Implementation Requirements**:
1. Create `VisualizationLibrary` class with methods for each viz type
2. Integrate with PyQtGraph GUI (TASK-100)
3. Provide matplotlib backend for publication plots
4. Support live updating during simulation

### TASK-102: Spatial & Temporal Interpolation/Smoothing

**Objective**: Implement `interpolation.py` for smooth volume rendering and export.

**Features**:
1. **Spatial interpolation**:
   - SPH kernel-weighted interpolation to regular grid
   - Adaptive grid resolution based on particle density
   - Support for 2D slices and 3D volumes

2. **Temporal interpolation**:
   - Smooth animation between snapshots
   - Linear or spline interpolation of particle positions/properties
   - Configurable interpolation frames (e.g., 10x between snapshots)

3. **Smoothing**:
   - Gaussian smoothing of field quantities
   - Outlier filtering for extreme values
   - Preserve mass/energy while smoothing

**Output**:
- Regular grids for volume rendering
- Smoothed particle data for export
- **IMPORTANT**: Keep original "clean" data untouched; write smoothed data to separate files

**Implementation Requirements**:
1. Create `InterpolationEngine` class
2. Provide methods:
   - `particles_to_grid(particles, field, resolution, **kwargs)` → 3D grid
   - `interpolate_snapshots(snap1, snap2, n_frames, **kwargs)` → intermediate frames
   - `smooth_field(particles, field, kernel_size, **kwargs)` → smoothed particles
3. Efficient GPU implementation where possible
4. Output to separate directories: `output/` (raw) vs `output/smoothed/`

DO:
- Work ONLY inside `tde_sph/visualization`.
- Create PyQtGraph 3D scatter/volume plots with time-scrubbing.
- Provide comprehensive visualization library (2D/3D, multiple backends).
- Implement spatial/temporal interpolation for smooth rendering.
- Provide utilities to export frames or grids for external rendering tools (Blender, ParaView).
- Add extensive documentation and examples

DO NOT:
- Implement core physics or I/O formats.
- Change simulation stepping or configuration.
- Open or modify anything under the `prompts/` folder.
- Modify files outside `tde_sph/visualization/`
- Overwrite original simulation data with smoothed data
