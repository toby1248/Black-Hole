@dependencies [TASK6, TASK7, TASK8, TASK9, TASK10]
# Task: End-to-End Testing and Manual Validation

## Summary
Execute comprehensive manual testing checklist (M.1-M.12 from AI_PROMPT.md) to validate all features work correctly. This is the final quality assurance step before system validation.

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **Manual tests:** M.1-M.12 from AI_PROMPT.md section 7
- **Test configs:** `configs/schwarzschild_tde.yaml`, `configs/newtonian_tde.yaml`
- **Platforms:** Windows 10/11, ultrawide monitor (2560px+), RTX 4090

## Complexity
Low (execution), but critical

## Dependencies
Depends on: [TASK6, TASK7, TASK8, TASK9, TASK10]
Blocks: [TASKW]
Parallel with: [TASK12]

## Manual Testing Checklist
- [ ] **M.1** Desktop app launches without errors
- [ ] **M.2** Load `configs/schwarzschild_tde.yaml`
- [ ] **M.3** Start simulation, verify real-time progress
- [ ] **M.4** All diagnostics panels populate with live data
- [ ] **M.5** Metric unit conversion toggles correctly
- [ ] **M.6** Preferences dialog saves and restores settings
- [ ] **M.7** Web app opens in Chrome/Firefox/Edge
- [ ] **M.8** Load demo data, verify particles render
- [ ] **M.9** Play animation, verify smooth playback at 30 fps
- [ ] **M.10** Change colormap, verify colors update
- [ ] **M.11** Toggle volumetric mode (if implemented), verify rendering
- [ ] **M.12** Export screenshot, verify image quality

## Acceptance Criteria
- [ ] All manual tests pass (M.1-M.12)
- [ ] No crashes or errors during testing
- [ ] Performance meets targets (30+ fps web, responsive desktop)
- [ ] Visual quality acceptable (screenshots, diagnostics readable)
- [ ] Document any issues found for fixing

## Code Review Checklist
- [ ] Test results documented
- [ ] Any bugs found are logged
- [ ] Performance metrics recorded
