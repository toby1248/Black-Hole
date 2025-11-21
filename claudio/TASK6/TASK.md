@dependencies [TASK0, TASK1]
# Task: Implement Web App JavaScript Infrastructure

## Summary
Create the core JavaScript files for the web app: `app.js` (application state and event handlers), `data_loader.js` (HDF5/JSON loading), and `visualizer.js` (Three.js scene setup). This establishes the foundation for all web button implementations.

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **New files:** `web/js/app.js`, `web/js/data_loader.js`, `web/js/visualizer.js`
- **HTML reference:** `web/index.html:40-164` (button IDs and structure)
- **Dependencies:** Three.js r128, OrbitControls.js

## Complexity
High

## Dependencies
Depends on: [TASK0, TASK1]
Blocks: [TASK7, TASK8, TASK9]
Parallel with: [TASK2, TASK3, TASK4, TASK5]

## Detailed Steps
1. **Create app.js - Application State Manager:**
   - Global state object: currentSnapshot, snapshotList, animationPlaying, etc.
   - Event handler registration for all buttons
   - Initialize visualizer and data loader on page load
   - Connect UI controls to state updates

2. **Create data_loader.js - Data Loading:**
   - Function: loadHDF5File(file) - Use h5wasm or similar library
   - Function: loadDemoData() - Generate synthetic particle data
   - Function: parseSnapshot(data) - Extract positions, velocities, densities, temperatures
   - Handle file selection via HTML5 File API

3. **Create visualizer.js - Three.js Scene:**
   - Setup: scene, camera, renderer, controls (OrbitControls)
   - Function: createParticleSystem(data) - Create Points geometry
   - Function: updateColors(colorBy, colormap) - Update particle colors
   - Function: render() - Animation loop
   - Function: resetCamera(), setTopView(), setSideView()

4. **Connect components:**
   - app.js imports data_loader and visualizer
   - Button clicks trigger data loading → visualization update

5. **Add error handling:**
   - Display errors in UI overlay (not just console)
   - Handle WebGL not supported
   - Handle missing files gracefully

## Acceptance Criteria
- [ ] All three JS files created and functional
- [ ] Three.js scene renders (even if empty)
- [ ] Demo data button generates and displays particles
- [ ] File loading triggered via HTML5 File API
- [ ] WebGL detection with fallback message
- [ ] No console errors on page load
- [ ] Integration tested in browser

## Code Review Checklist
- [ ] ES6 modules or clear namespacing (avoid global pollution)
- [ ] Error handling for all async operations
- [ ] Memory cleanup (dispose geometries, textures)
- [ ] Clear function names and documentation
- [ ] No hardcoded magic numbers
