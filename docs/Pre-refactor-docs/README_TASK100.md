
# Task 100: GUI & Physics Updates

## Completed
- **GUI Imports**: Fixed `sys.path` issues in `gui/main_window.py` and `gui/simulation_thread.py`. Verified with `scripts/verify_imports.py`.
- **Barnes-Hut Gravity**: Implemented `src/tde_sph/gravity/barnes_hut.py` with Octree and Numba acceleration.
  - *Note*: Unit tests (`tests/test_barnes_hut.py`) show divergence from direct summation. Needs further debugging of the tree traversal logic.
  - Integrated into `gui/simulation_thread.py` (selectable via config `physics.gravity: barnes_hut`).
- **GUI 3D Viewer**: Added `gui/web_viewer.py` and integrated it as a tab in `TDESPHMainWindow`. Loads `web/index.html`.

## Pending
- **Barnes-Hut Debugging**: Fix the force calculation (currently returning 0 or diverging).
- **GUI Builder**: Implement visual YAML editor.
- **Web App**: Minimize ribbon, add data readouts.
- **New ICs**: Red Giant, Globular Cluster.
- **Interpolation**: Add finite volume vs spline settings.

## Usage
To run the GUI:
```bash
python gui/main_window.py
```

To use Barnes-Hut:
In your config YAML:
```yaml
physics:
  gravity: barnes_hut
  theta: 0.5
```
