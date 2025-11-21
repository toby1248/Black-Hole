---
goal: Refactor and Stabilize TDE-SPH Codebase
version: 1.0
date_created: 2025-11-20
status: Planned
tags: [refactor, architecture, stability, gr-sph]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan outlines the steps to refactor the `tde_sph` codebase to resolve inter-module communication issues, specifically between the Gravity and Metric modules, and to stabilize the simulation for Phase 3 physics extensions.

## 1. Requirements & Constraints

- **REQ-001**: Fix GR/Newtonian interface mismatch (Cartesian vs BL coordinates).
- **REQ-002**: Ensure strict energy conservation monitoring in the GUI.
- **REQ-003**: Do not break existing Newtonian functionality.
- **CON-001**: No direct code edits by the planning agent (Plan generation only).

## 2. Implementation Steps

### Phase 1: Interface Standardization (Critical)

- **GOAL**: Ensure `Metric` and `GravitySolver` speak the same language (Cartesian).

| Task | Description |
|------|-------------|
| TASK-001 | **Metric Module**: Update `KerrMetric.geodesic_acceleration` to accept Cartesian `x` and `v`. Implement internal conversion to BL coordinates and 4-velocity normalization. |
| TASK-002 | **Gravity Module**: Update `RelativisticGravitySolver` to pass Cartesian 3-velocity to the metric. Remove ad-hoc 4-velocity construction. |
| TASK-003 | **Tests**: Add a specific test case `test_metric_interface.py` that verifies `geodesic_acceleration` works with Cartesian inputs and returns expected values for circular orbits. |

### Phase 2: Core & SPH Cleanup

- **GOAL**: Clean up `Simulation` class and SPH backend.

| Task | Description |
|------|-------------|
| TASK-004 | **Core**: Refactor `Simulation.compute_forces` to delegate GPU logic to a `GPUBackend` class. |
| TASK-005 | **SPH**: Verify `ParticleSystem` consistency across CPU/GPU boundaries. |

### Phase 3: GUI Enhancement

- **GOAL**: Real-time debugging and diagnostics.

| Task | Description |
|------|-------------|
| TASK-006 | **GUI**: Implement Energy vs Time plot widget. |
| TASK-007 | **GUI**: Implement Timestep vs Time plot widget. |
| TASK-008 | **GUI**: Add toggle for 3D visualization to save resources. |

## 3. Alternatives

- **ALT-001**: Keep BL coordinates in `Metric` interface.
    - *Reason rejected*: SPH is fundamentally Cartesian. Forcing the solver to handle BL conversion exposes internal metric details and increases complexity in the solver.

## 4. Dependencies

- `numpy`, `scipy`, `h5py`
- `PyQt6` / `PySide6` for GUI.
- `pyqtgraph` for fast real-time plotting.

## 5. Files Affected

- `src/tde_sph/metric/kerr.py`
- `src/tde_sph/gravity/relativistic_orbit.py`
- `src/tde_sph/core/simulation.py`
- `gui/main_window.py`
- `gui/data_display.py`

## 6. Testing

- **TEST-001**: `test_metric_interface.py`: Verify Cartesian input handling.
- **TEST-002**: `test_energy_conservation.py`: Run adiabatic collapse and check energy drift < 1%.
