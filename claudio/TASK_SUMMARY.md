# Task Decomposition Summary

## Overview
AI_PROMPT.md decomposed into **13 tasks + 1 final validation task (Ω)** organized in 4 execution layers.

## Task Structure

### Layer 0: Foundation (Sequential - Must Complete First)
**TASK0** - Fix Temperature Attribute Bug
- Fix ParticleSystem temperature initialization
- Remove hasattr() checks in Simulation and SimulationThread
- Blocks: ALL other tasks (foundational bug fix)

**TASK1** - Investigate and Fix Similar Attribute Bugs
- Search for missing attributes (velocity_magnitude, entropy, acceleration, du_dt)
- Fix coordinate transformation bugs
- Fix HDF5 I/O bugs
- Depends on: TASK0
- Blocks: TASK2-TASK5 (desktop features need complete data)

---

### Layer 1: Desktop GUI Enhancement (Parallel)
**TASK2** - Implement Comprehensive Diagnostics Tab (TOP PRIORITY)
- Particle statistics panel (min/max/mean/std for all quantities)
- Performance metrics panel (wall time, steps/sec, GPU status)
- Coordinate/metric data panel (GR-specific data)
- Ultrawide layout optimization (2560px+ width)
- Depends on: TASK0, TASK1

**TASK3** - Implement Metric Unit Conversion (SECOND PRIORITY)
- Create gui/unit_conversion.py module
- Toggle: dimensionless ↔ physical units
- Conversions: length→R☉, mass→M☉, temperature→K, etc.
- Depends on: TASK0, TASK1
- Parallel with: TASK2, TASK4, TASK5

**TASK4** - Implement Preferences Dialog
- Create gui/preferences_dialog.py
- Three tabs: General, Visualization, Performance
- QSettings persistence
- Depends on: TASK0
- Parallel with: TASK2, TASK3, TASK5

**TASK5** - Implement Web Viewer Widget Integration
- Verify QWebEngineView embedding
- Load web/index.html in desktop GUI
- Refresh functionality
- Depends on: TASK0
- Parallel with: TASK2, TASK3, TASK4

---

### Layer 2: Web GUI Implementation (Parallel with Layer 1)
**TASK6** - Implement Web App JavaScript Infrastructure (Foundation)
- Create web/js/app.js (state management, event handlers)
- Create web/js/data_loader.js (HDF5/JSON loading, demo data)
- Create web/js/visualizer.js (Three.js scene setup)
- Depends on: TASK0, TASK1
- Blocks: TASK7, TASK8, TASK9

**TASK7** - Implement Web Data Source and Animation Controls
- Load Files, Connect to Server, Load Demo buttons
- Play/Pause, Step Forward/Back, Snapshot Slider, FPS Slider
- Depends on: TASK6
- Parallel with: TASK8, TASK9

**TASK8** - Implement Web Visualization and Camera Controls
- Color-by, Colormap, Point Size, Log Scale toggles
- Show Black Hole, Show Axes toggles
- Reset View, Top View, Side View buttons
- Depends on: TASK6
- Parallel with: TASK7, TASK9

**TASK9** - Implement Web Export and Volumetric Rendering
- Screenshot export (mandatory)
- JSON export (mandatory)
- Volumetric rendering (optional - ultimate goal but complex)
- Depends on: TASK6
- Parallel with: TASK7, TASK8

---

### Layer 3: Integration and Testing (Sequential)
**TASK10** - Desktop GUI Integration and Error Handling
- Integrate all desktop components
- Implement error handling (E.1-E.4: sim failure, invalid config, no snapshots, no GPU)
- Test full desktop workflow
- Depends on: TASK0-TASK5
- Blocks: TASK11, TASKΩ

**TASK11** - End-to-End Testing and Manual Validation
- Execute manual testing checklist (M.1-M.12)
- Test on Windows 10/11, ultrawide monitor, RTX 4090
- Document results and issues
- Depends on: TASK6-TASK10
- Parallel with: TASK12

**TASK12** - Automated Unit and Integration Testing
- Write unit tests for all modified code
- Temperature tests, unit conversion tests, integration tests
- >80% coverage target
- Depends on: TASK0-TASK9
- Parallel with: TASK11

---

### Layer Ω: Final System Validation (MANDATORY)
**TASKΩ** - Final System-Level Validation and Cohesion Verification
- Verify EVERY acceptance criterion from AI_PROMPT.md
- Test complete workflows end-to-end
- Performance validation (fps, input lag, memory)
- Code quality review (no placeholders, documentation complete)
- Generate final validation report
- Production readiness check
- Depends on: ALL previous tasks (TASK0-TASK12)

---

## Execution Strategy

### Phase 1: Foundation (Sequential)
1. TASK0 (temperature bug)
2. TASK1 (similar bugs)

### Phase 2: Parallel Development
**Stream A (Desktop):**
- TASK2, TASK3, TASK4, TASK5 (all parallel)

**Stream B (Web):**
- TASK6 (foundation)
- Then TASK7, TASK8, TASK9 (all parallel)

### Phase 3: Integration & Testing (Sequential)
1. TASK10 (integration)
2. TASK11 + TASK12 (parallel manual and automated testing)

### Phase 4: Final Validation (Sequential)
1. TASKΩ (system-level validation)

---

## Coverage Summary

### All Acceptance Criteria Covered
**Desktop Diagnostics (C.1-C.4, A.1):** TASK2, TASK3
**Desktop Buttons (B.1-B.4):** TASK4, TASK5, TASK10
**Web Visualization (V.1-V.4):** TASK9 (optional)
**Web Buttons (D.1-D.17):** TASK7, TASK8, TASK9
**Bug Fixes (T.1-T.7):** TASK0, TASK1
**Error Handling (E.1-E.8):** TASK10, TASK6-TASK9 (implicit)
**Testing (M.1-M.12, A.1-A.4):** TASK11, TASK12

### No Information Loss
Every requirement from AI_PROMPT.md is mapped to at least one task.
No requirements merged or skipped.

### Context Propagation
- All tasks reference AI_PROMPT.md for universal context (no duplication)
- Each task includes task-specific context (files to touch, patterns to follow)
- References are precise (file:line-range, not vague)

---

## Key Principles Applied

✅ **No Information Loss:** Every requirement covered
✅ **Context Reference Pattern:** Point to AI_PROMPT.md, don't copy
✅ **Precise References:** file:line-range for all code patterns
✅ **Self-Contained Tasks:** Each task executable in isolation
✅ **Maximum Parallelization:** Layer 1 and Layer 2 fully parallel
✅ **Final Validation:** TASKΩ ensures system cohesion

---

## Task Count: 14 Total
- Layer 0: 2 tasks (sequential)
- Layer 1: 4 tasks (parallel)
- Layer 2: 4 tasks (1 foundation + 3 parallel)
- Layer 3: 3 tasks (1 sequential + 2 parallel)
- Layer Ω: 1 task (final validation)

**Estimated Complexity:**
- Low: 4 tasks (TASK4, TASK5, TASK11, part of TASK0)
- Medium: 8 tasks (TASK1, TASK2, TASK3, TASK7, TASK8, TASK10, TASK12, TASKΩ)
- High: 2 tasks (TASK6, TASK9)

---

This decomposition ensures complete coverage, maximum parallelism, and rigorous validation while maintaining context richness throughout the task hierarchy.
