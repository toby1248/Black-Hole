@dependencies [TASK6]
# Task: Implement Web Data Source and Animation Controls

## Summary
Implement all buttons in the Data Source panel (Load Files, Connect to Server, Load Demo) and Animation Control panel (Play/Pause, Step, Slider, FPS) from `web/index.html`.

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **Files to modify:** `web/js/app.js`, `web/js/data_loader.js`
- **Buttons:** Load Files, Connect to Server, Load Demo, Play, Pause, Step Forward/Back, Snapshot Slider, FPS Slider
- **Pattern:** Event listeners → update appState → call visualizer update

## Complexity
Medium

## Dependencies
Depends on: [TASK6]
Blocks: [TASK11]
Parallel with: [TASK8, TASK9]

## Acceptance Criteria
- [ ] **D.1** Load Files button: triggers File API, loads HDF5, populates snapshot list
- [ ] **D.2** Connect to Server button: validates URL, fetches snapshot list via REST API
- [ ] **D.3** Load Demo button: generates synthetic data, displays expanding sphere
- [ ] **D.4** Play/Pause buttons: start/stop animation loop
- [ ] **D.5** Step forward/backward: advance/rewind one snapshot
- [ ] **D.6** Snapshot slider: dragging updates current snapshot
- [ ] **D.7** FPS slider: sets animation playback speed (1-60 fps)
- [ ] All controls update visualizer in real-time
- [ ] Error handling for missing files, server errors

## Code Review Checklist
- [ ] Animation loop uses requestAnimationFrame
- [ ] Slider updates are smooth (throttled if needed)
- [ ] Server connection handles CORS
- [ ] File parsing errors display user-friendly messages
