# GUI Module API

## Components

### `MainWindow`
- Central hub. Manages `SimulationThread` and UI layout.

### `ControlPanel`
- Widgets for Start/Stop, Config editing.

### `DataDisplay`
- Plots (Matplotlib/PyQtGraph) for scalar quantities (Energy, dt).

### `WebViewer` (Optional)
- Browser widget for Plotly 3D visualizations (if used).

## Threading
- `SimulationThread`: Runs the `Simulation.step()` loop. Emits signals `on_step_completed(state)` to update GUI.
