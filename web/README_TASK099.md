# TASK-099: HTML/Three.js/WebGL Browser Interface

**Status**: ✅ Completed
**Date**: 2025-11-18
**Author**: TDE-SPH Development Team

---

## Overview

The TDE-SPH Web Visualizer is a browser-based, platform-independent 3D visualization tool for exploring tidal disruption event simulations. Built with HTML5, JavaScript, and Three.js, it provides GPU-accelerated WebGL rendering of millions of SPH particles with no installation required beyond a modern web browser.

### Key Features

- **WebGL Acceleration**: Hardware-accelerated point cloud rendering via Three.js
- **Cross-Platform**: Runs in any modern browser (Chrome, Firefox, Safari, Edge)
- **No Installation**: Zero dependencies beyond a web browser
- **Interactive Controls**: Orbit camera, time-scrubbing, real-time parameter adjustments
- **Multiple Data Sources**: Local files (JSON), simulation server (WebSocket), demo data
- **Colormaps**: 6 built-in colormaps (viridis, plasma, inferno, hot, cool, rainbow)
- **Logarithmic Scaling**: Toggle log/linear color scales
- **Keyboard Shortcuts**: Space (play/pause), arrows (step), R (reset camera), Ctrl+S (screenshot)
- **Export**: Screenshot PNG, export JSON data

---

## Architecture

### File Structure

```
web/
├── index.html              # Main HTML page (200 lines)
├── css/
│   └── style.css           # Responsive CSS styling (350 lines)
├── js/
│   ├── visualizer.js       # Three.js visualization engine (550 lines)
│   ├── data_loader.js      # Data loading and streaming (400 lines)
│   └── app.js              # Application logic and UI bindings (430 lines)
└── README_TASK099.md       # This file
```

### Technology Stack

- **HTML5**: Structure and canvas element
- **CSS3**: Flexbox layout, responsive design, dark theme
- **JavaScript ES6**: Modern async/await, classes, modules
- **Three.js r128**: WebGL rendering library (via CDN)
- **OrbitControls**: Camera manipulation (via CDN)

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `index.html` | UI structure, controls, canvas |
| `style.css` | Dark theme, responsive layout, animations |
| `visualizer.js` | Three.js scene, WebGL rendering, colormaps |
| `data_loader.js` | Load JSON files, generate demo data, WebSocket |
| `app.js` | Event handling, animation loop, UI updates |

---

## Quick Start

### Option 1: Local File Server (Recommended)

**Using Python**:
```bash
cd web
python3 -m http.server 8000
# Open http://localhost:8000 in browser
```

**Using Node.js**:
```bash
cd web
npx http-server -p 8000
# Open http://localhost:8000 in browser
```

**Using PHP**:
```bash
cd web
php -S localhost:8000
# Open http://localhost:8000 in browser
```

### Option 2: Open Directly (Limited Functionality)

Due to CORS restrictions, loading local files directly won't work. Use a local server.

### Option 3: Demo Data (No Server Required if hosting on web)

1. Open `index.html` in browser via local server
2. Select "Demo Data" from Data Source dropdown
3. Click "Load Demo Data"
4. Click ▶ Play to start animation

---

## Usage Guide

### Loading Data

#### Method 1: Demo Data (Procedurally Generated)

1. Select **"Demo Data"** from Data Source dropdown
2. Click **"Load Demo Data"** button
3. Generates 50 snapshots of a Keplerian disc + eccentric debris stream
4. 12,000 particles per snapshot
5. Automatic pericentre precession

**Use Case**: Testing the interface without real simulation data.

#### Method 2: Local JSON Files

**JSON Format**:
```json
{
  "time": 10.5,
  "step": 100,
  "n_particles": 100000,
  "positions": [[x1,y1,z1], [x2,y2,z2], ...],
  "density": [rho1, rho2, ...],
  "temperature": [T1, T2, ...],
  "internal_energy": [u1, u2, ...],
  "velocity_magnitude": [v1, v2, ...],
  "pressure": [P1, P2, ...],
  "entropy": [s1, s2, ...]
}
```

**Steps**:
1. Select **"Local Files"** from Data Source
2. Click **"Choose Files"** and select one or more JSON files
3. Click **"Load Files"**
4. Files are sorted by time automatically

**Converting HDF5 to JSON** (Python):
```python
import h5py
import json
import numpy as np

def hdf5_to_json(h5_file, json_file):
    with h5py.File(h5_file, 'r') as f:
        data = {
            'time': float(f['time'][()]),
            'step': int(f['step'][()]),
            'n_particles': int(f['particles/positions'].shape[0]),
            'positions': f['particles/positions'][:].tolist(),
            'density': f['particles/density'][:].tolist(),
            'temperature': f['particles/temperature'][:].tolist(),
            'internal_energy': f['particles/internal_energy'][:].tolist(),
            'velocity_magnitude': np.linalg.norm(f['particles/velocities'][:], axis=1).tolist(),
            'pressure': f['particles/pressure'][:].tolist(),
            'entropy': f['particles/entropy'][:].tolist()
        }

    with open(json_file, 'w') as f:
        json.dump(data, f)

# Convert all snapshots
import glob
for h5_file in sorted(glob.glob('outputs/snapshot_*.h5')):
    json_file = h5_file.replace('.h5', '.json')
    hdf5_to_json(h5_file, json_file)
    print(f'Converted: {json_file}')
```

#### Method 3: Simulation Server (WebSocket)

**Requirements**: Backend server with WebSocket endpoint (see `web_server.py`)

**Steps**:
1. Select **"Simulation Server"** from Data Source
2. Enter server URL (e.g., `http://localhost:8000`)
3. Click **"Connect"**
4. Server pushes snapshots in real-time

**WebSocket Message Format**:
```json
{
  "type": "snapshot",
  "snapshot": {
    "time": 10.5,
    "step": 100,
    "n_particles": 100000,
    "positions": [[x1,y1,z1], ...],
    "density": [rho1, rho2, ...],
    ...
  }
}
```

---

### Animation Controls

| Control | Function |
|---------|----------|
| ▶ Play | Start animation loop |
| ⏸ Pause | Pause animation |
| ⏮ Step Back | Previous snapshot |
| ⏭ Step Forward | Next snapshot |
| Snapshot Slider | Jump to specific snapshot |
| FPS Slider | Adjust playback speed (1-60 FPS) |

**Keyboard Shortcuts**:
- **Space**: Play/Pause
- **← →**: Step backward/forward
- **R**: Reset camera
- **Ctrl+S**: Screenshot

---

### Visualization Options

#### Color By

Choose which physical quantity to visualize:

| Field | Description | Typical Range |
|-------|-------------|---------------|
| **Density** | Mass density (ρ) | 10⁻⁶ - 10² g/cm³ |
| **Temperature** | Gas temperature (T) | 10³ - 10⁷ K |
| **Internal Energy** | Specific internal energy (u) | 10⁻³ - 10² |
| **Velocity Magnitude** | \|v\| | 0.01 - 1.0 c |
| **Pressure** | Gas pressure (P) | 10⁻⁸ - 10² |
| **Entropy** | Specific entropy (s) | Variable |

#### Colormaps

- **Viridis**: Perceptually uniform, colorblind-friendly (default)
- **Plasma**: Purple to yellow, high contrast
- **Inferno**: Black to white through red/orange
- **Hot**: Black → red → yellow → white (temperature-like)
- **Cool**: Cyan → blue → magenta
- **Rainbow**: Full spectrum (not perceptually uniform)

#### Point Size

- Range: 0.5 - 10.0 pixels
- Default: 2.0
- Smaller values for high particle counts
- Larger values for sparse data

#### Logarithmic Scale

- **Enabled** (default): log₁₀ scaling, good for data spanning many orders of magnitude
- **Disabled**: Linear scaling, good for narrowly distributed data

#### Black Hole

- Toggle visibility of black hole sphere at origin
- Rendered as black sphere with purple emissive glow
- Radius = 1 R_g (gravitational radius)

#### Axes

- Toggle visibility of coordinate axes
- Red: X-axis, Green: Y-axis, Blue: Z-axis
- Length: 10 units

---

### Camera Controls

#### Mouse Controls (OrbitControls)

- **Left Click + Drag**: Rotate around target
- **Right Click + Drag** (or Middle Click): Pan camera
- **Scroll Wheel**: Zoom in/out
- **Damping**: Smooth camera motion with inertia

#### Preset Views

- **Reset View**: Return to default position (20, 20, 20) looking at origin
- **Top View**: View from +Z axis (looking down onto XY plane)
- **Side View**: View from +X axis (looking at YZ plane)

---

### Export

#### Screenshot

- **Button**: 📷 Screenshot
- **Keyboard**: Ctrl+S
- Captures current WebGL canvas as PNG
- Auto-downloads: `tde_sph_screenshot_<timestamp>.png`
- Resolution: Full canvas size (scales with window)

#### JSON Export

- **Button**: 💾 Export JSON
- Exports all loaded snapshots to single JSON file
- Format: Array of snapshot objects
- Auto-downloads: `tde_sph_export_<timestamp>.json`
- Use case: Share visualized data or transfer between sessions

---

## API Reference

### `Visualizer` Class

**Purpose**: Three.js visualization engine

**Constructor**:
```javascript
const visualizer = new Visualizer('canvas-id');
```

**Key Methods**:
```javascript
// Load particle data
visualizer.loadParticles(positions, colorField, colorBy);
// positions: Float32Array, shape (N, 3)
// colorField: Float32Array, length N
// colorBy: string, e.g., 'density'

// Update existing particles (efficient)
visualizer.updateParticles(positions, colorField);

// Change point size
visualizer.setPointSize(2.5);

// Change colormap
visualizer.setColormap('viridis');  // or 'plasma', 'inferno', etc.

// Toggle logarithmic scaling
visualizer.setLogScale(true);

// Toggle black hole visibility
visualizer.setShowBlackHole(true);

// Toggle axes visibility
visualizer.setShowAxes(true);

// Camera presets
visualizer.resetCamera();
visualizer.setTopView();
visualizer.setSideView();

// Export screenshot
visualizer.screenshot();
```

**Colormap Implementation**:
```javascript
applyColormap(value, colormapName)
// value: Normalized [0, 1]
// Returns: [r, g, b] in [0, 1]

// Example: Viridis approximation
viridis: [
    0.267 * (1-t) + 0.993 * t,  // R
    0.005 * (1-t) + 0.906 * t,  // G
    0.329 * (1-t) + 0.144 * t   // B
]
```

---

### `DataLoader` Class

**Purpose**: Load and manage snapshot data

**Constructor**:
```javascript
const dataLoader = new DataLoader();
```

**Key Methods**:
```javascript
// Generate demo data
dataLoader.loadDemoData();
// Returns: Array of snapshot objects

// Load local JSON files
await dataLoader.loadMultipleJSONFiles(fileList);
// fileList: FileList from <input type="file">

// Connect to simulation server
await dataLoader.connectToServer('http://localhost:8000');
// Establishes WebSocket connection

// Navigation
const snapshot = dataLoader.getCurrentSnapshot();
const snapshot = dataLoader.nextSnapshot();
const snapshot = dataLoader.previousSnapshot();
dataLoader.setCurrentIndex(10);

// Statistics
const stats = dataLoader.computeStatistics(snapshot);
// Returns: {n_particles, total_mass, rho_min, rho_max, temp_min, temp_max, total_energy}

// Export
dataLoader.exportToJSON('my_snapshots.json');
```

---

## Performance Optimization

### Particle Count Recommendations

| Particles | Performance | Recommended Point Size |
|-----------|-------------|------------------------|
| < 10,000 | Excellent | 2.0 - 5.0 |
| 10,000 - 100,000 | Good | 1.5 - 3.0 |
| 100,000 - 1M | Fair | 1.0 - 2.0 |
| > 1M | Slow | 0.5 - 1.0 |

**Tips**:
- Use logarithmic scaling for better contrast
- Reduce point size for dense datasets
- Disable black hole/axes for marginal speed gain
- Close other browser tabs to free GPU memory

### Browser Compatibility

| Browser | WebGL Support | Performance |
|---------|---------------|-------------|
| Chrome 90+ | ✅ Excellent | Best |
| Firefox 88+ | ✅ Excellent | Very Good |
| Safari 14+ | ✅ Good | Good |
| Edge 90+ | ✅ Excellent | Best |

**Requirements**:
- WebGL 1.0 or higher
- JavaScript ES6+ support
- Minimum 2 GB RAM (4 GB+ recommended for large datasets)

### Data Format Optimization

**Minimize JSON File Size**:
```javascript
// Use Float32Array toJSON (compact representation)
const positions = new Float32Array(data.positions.flat());

// Or use binary format (not human-readable)
// MessagePack, BSON, or custom binary protocol
```

**Server Streaming**:
- Send deltas (position changes) instead of full snapshots
- Use compression (gzip, brotli) for WebSocket messages
- Batch particles into chunks for progressive loading

---

## Customization

### Add Custom Colormap

**Edit `visualizer.js`**:
```javascript
applyColormap(value, colormapName) {
    switch(colormapName) {
        // ... existing colormaps

        case 'custom':
            // Your colormap logic
            const r = Math.sin(value * Math.PI);
            const g = Math.cos(value * Math.PI);
            const b = 1.0 - value;
            return [r, g, b];

        default:
            return [value, value, value];
    }
}
```

**Add to HTML dropdown**:
```html
<select id="colormap">
    <!-- ... existing options -->
    <option value="custom">Custom</option>
</select>
```

### Add Custom UI Panel

**Edit `index.html`**:
```html
<div class="panel">
    <h2>Custom Panel</h2>
    <div class="control-group">
        <button id="my-button" class="btn">My Action</button>
    </div>
</div>
```

**Bind event in `app.js`**:
```javascript
document.getElementById('my-button').addEventListener('click', () => {
    console.log('Custom button clicked');
    // Your logic here
});
```

---

## Troubleshooting

### Issue: Blank screen, no particles visible

**Causes**:
1. Data not loaded
2. Camera positioned incorrectly
3. Point size too small

**Solutions**:
- Check browser console for errors
- Click "Load Demo Data" to test
- Click "Reset View" to reposition camera
- Increase point size slider

---

### Issue: "Failed to load local files"

**Cause**: CORS policy blocks file:// protocol

**Solution**: Use local web server (see Quick Start)

---

### Issue: Performance is slow with many particles

**Solutions**:
- Reduce point size to 1.0 or less
- Downsample data (use every 10th particle)
- Close other browser tabs
- Upgrade GPU/browser

---

### Issue: WebSocket connection fails

**Causes**:
1. Server not running
2. Incorrect URL
3. Firewall blocking port

**Solutions**:
- Verify server is running: `curl http://localhost:8000`
- Check URL matches server address
- Try HTTP polling instead of WebSocket

---

### Issue: Colors look washed out

**Cause**: Linear scaling on data with narrow range

**Solutions**:
- Enable logarithmic scaling
- Try different colormap (viridis, plasma)
- Check data quality (NaN/inf values)

---

## Future Enhancements

Potential additions for future versions:

- [ ] **Volume Rendering**: 3D volumetric grids via ray marching
- [ ] **Streamlines**: Velocity field integration for flow visualization
- [ ] **Slice Planes**: Interactive 2D density slices
- [ ] **VR Support**: WebXR for immersive visualization
- [ ] **GPU Particle Physics**: Simulate particle motion in browser
- [ ] **Binary Data Format**: WASM HDF5 reader for direct file loading
- [ ] **Multi-View**: Compare side-by-side snapshots
- [ ] **Annotations**: Add labels, arrows, regions of interest
- [ ] **Video Export**: Render animation to MP4/WebM

---

## References

1. **Three.js Documentation**: https://threejs.org/docs/
2. **WebGL Specification**: https://www.khronos.org/webgl/
3. **OrbitControls**: https://threejs.org/docs/#examples/en/controls/OrbitControls
4. **Colormap Theory**: "A Better Default Colormap for Matplotlib" (2015)
5. **WebSocket API**: https://developer.mozilla.org/en-US/docs/Web/API/WebSocket

---

**TASK-099 Complete**: HTML/Three.js/WebGL browser interface provides a cross-platform, zero-install 3D visualization solution for TDE-SPH simulations.
