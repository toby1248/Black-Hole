# TDE-SPH Manual Testing Checklist

This document provides the manual testing checklist for validating all TDE-SPH GUI features.

## Prerequisites

- Python 3.8+ with PyQt6 or PyQt5
- Modern web browser (Chrome/Firefox/Edge)
- Test configuration files in `configs/`

## Desktop Application Tests (M.1 - M.6)

### M.1 Desktop App Launch
```bash
cd src/tde_sph/gui
python main_window.py
```
**Expected:** Application launches without errors, main window appears with:
- Config editor in center
- Control panel on left
- Data display on right
- Menu bar and toolbar visible

### M.2 Load Configuration
1. File → Open Config...
2. Select `configs/schwarzschild_tde.yaml`

**Expected:**
- YAML content displays in editor with syntax highlighting
- Window title shows filename
- Status bar shows "Loaded: ..."

### M.3 Start Simulation
1. Load a config file
2. Click Start Simulation or Ctrl+R

**Expected:**
- Progress bar updates in real-time
- Status shows "Running"
- Control panel buttons update (Start disabled, Stop enabled)

### M.4 Diagnostics Panels
While simulation runs, check:
- Particle statistics table populates
- Performance metrics update
- Energy plot shows values

**Expected:** All panels show live data, no "N/A" values

### M.5 Unit Conversion Toggle
1. In Diagnostics panel, check "Show Physical Units"
2. Observe values change

**Expected:**
- Values convert from code units to CGS
- Labels show appropriate units (cm, g, K, etc.)
- Toggle back shows code units

### M.6 Preferences Dialog
1. Edit → Preferences
2. Modify settings (e.g., default colormap)
3. Click OK
4. Reopen Preferences

**Expected:**
- All tabs render correctly
- Settings persist after closing and reopening
- Values are restored correctly

## Web Application Tests (M.7 - M.12)

### M.7 Web App Launch
1. Open `web/index.html` in browser
2. Open browser console (F12)

**Expected:**
- Page loads with sidebar and canvas
- No console errors
- Three.js initializes (black background with axes)

### M.8 Demo Data
1. Click "Load Demo Data"
2. Observe canvas

**Expected:**
- Particles render in 3D
- Keplerian disc visible
- Statistics panel updates

### M.9 Animation Playback
1. After loading demo data, click Play
2. Observe FPS counter

**Expected:**
- Animation plays smoothly
- FPS counter shows 30+ fps
- Particles animate over time

### M.10 Colormap Change
1. Select different colormap (e.g., Plasma)
2. Observe particle colors

**Expected:**
- Colors update immediately
- Colormap matches selection

### M.11 Volumetric Mode (Optional)
- Note: Volumetric rendering is optional feature
- If implemented, toggle between point cloud and volumetric

### M.12 Screenshot Export
1. Click Screenshot button
2. Check downloads folder

**Expected:**
- PNG file downloads
- Image shows current canvas view
- Quality is acceptable (full resolution)

## Error Handling Tests (E.1 - E.4)

### E.1 Simulation Failure
1. Modify config to cause error (e.g., invalid particle count: -1)
2. Start simulation

**Expected:**
- Error dialog appears
- Error logged to control panel
- UI resets to idle state

### E.2 Invalid YAML
1. Type invalid YAML in editor (e.g., unclosed bracket)
2. Try Simulation → Validate Config

**Expected:**
- Error dialog shows syntax error
- Editor remains functional

### E.3 No Data State
1. Open data display without loading data

**Expected:**
- Diagnostics show "No data" or placeholder
- No crashes

### E.4 GPU Not Available
1. Start simulation on system without CUDA

**Expected:**
- Warning dialog about CPU fallback
- Simulation continues using CPU
- Performance panel shows CPU mode

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Web FPS | 30+ | With 10k particles |
| Desktop responsiveness | <100ms | UI response time |
| Memory usage | <2GB | For typical simulation |

## Test Results Template

```
Date: YYYY-MM-DD
Tester:
Platform:

M.1: [ ] Pass / [ ] Fail - Notes:
M.2: [ ] Pass / [ ] Fail - Notes:
M.3: [ ] Pass / [ ] Fail - Notes:
M.4: [ ] Pass / [ ] Fail - Notes:
M.5: [ ] Pass / [ ] Fail - Notes:
M.6: [ ] Pass / [ ] Fail - Notes:
M.7: [ ] Pass / [ ] Fail - Notes:
M.8: [ ] Pass / [ ] Fail - Notes:
M.9: [ ] Pass / [ ] Fail - Notes:
M.10: [ ] Pass / [ ] Fail - Notes:
M.11: [ ] Pass / [ ] Fail / [ ] N/A - Notes:
M.12: [ ] Pass / [ ] Fail - Notes:

E.1: [ ] Pass / [ ] Fail - Notes:
E.2: [ ] Pass / [ ] Fail - Notes:
E.3: [ ] Pass / [ ] Fail - Notes:
E.4: [ ] Pass / [ ] Fail - Notes:

Issues Found:
1.
2.
```
