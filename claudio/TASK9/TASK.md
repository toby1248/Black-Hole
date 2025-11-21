@dependencies [TASK6]
# Task: Implement Web Export and Volumetric Rendering

## Summary
Implement export buttons (Screenshot, Export JSON) and optional volumetric rendering with WebGL ray marching (ultimate goal per user request). Volumetric rendering is advanced and may be deferred if time-constrained.

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **Files to modify:** `web/js/app.js`, `web/js/visualizer.js`
- **Optional new files:** `web/js/shaders/volumetric.vert`, `web/js/shaders/volumetric.frag`
- **Export:** Canvas to PNG, snapshot data to JSON
- **Volumetric (optional):** Ray marching shader, 3D grid interpolation

## Complexity
High (volumetric), Low (export)

## Dependencies
Depends on: [TASK6]
Blocks: []
Parallel with: [TASK7, TASK8]

## Acceptance Criteria
- [ ] **D.16** Screenshot button: captures canvas as PNG, downloads file
- [ ] **D.17** Export JSON button: exports current snapshot as JSON
- [ ] **V.1-V.4** (Optional) Volumetric rendering:
  - [ ] Point cloud to 3D grid conversion
  - [ ] Ray marching fragment shader
  - [ ] Transfer function controls
  - [ ] Performance: 30+ fps at 128³ resolution on RTX 4090

## Code Review Checklist
- [ ] Screenshot captures full canvas (correct dimensions)
- [ ] JSON export format documented
- [ ] Volumetric shader optimized (early ray termination, LOD)
- [ ] Toggle between point cloud and volumetric modes

**Note:** Volumetric rendering is ultimate goal but complex. Can be deferred to future enhancement if time-constrained.
