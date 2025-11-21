@dependencies [TASK6]
# Task: Implement Web Visualization and Camera Controls

## Summary
Implement visualization panel buttons (Color-by dropdown, Colormap, Point Size, Log Scale, Show BH, Show Axes) and camera panel (Reset View, Top View, Side View) from `web/index.html`.

## Context Reference
**For complete environment context, see:** `../AI_PROMPT.md`

**Task-Specific Context:**
- **Files to modify:** `web/js/app.js`, `web/js/visualizer.js`
- **Buttons:** Color-by, Colormap, Point Size slider, Checkboxes (Log Scale, Show BH, Show Axes), Camera views
- **Pattern:** Update material properties, recompute colors, update camera position

## Complexity
Medium

## Dependencies
Depends on: [TASK6]
Blocks: []
Parallel with: [TASK7, TASK9]

## Acceptance Criteria
- [ ] **D.8** Color-by dropdown: changes particle color mapping (density, temperature, etc.)
- [ ] **D.9** Colormap dropdown: changes color transfer function (viridis, plasma, etc.)
- [ ] **D.10** Point size slider: adjusts particle size (0.5-10.0)
- [ ] **D.11** Log scale checkbox: toggles logarithmic color mapping
- [ ] **D.12** Show black hole checkbox: toggles BH sphere visibility
- [ ] **D.13** Show axes checkbox: toggles coordinate axes
- [ ] **D.14** Reset view: returns camera to default position
- [ ] **D.15** Top/Side view: sets camera to predefined positions
- [ ] All controls update display immediately

## Code Review Checklist
- [ ] Color mapping handles full value range (no clipping)
- [ ] Log scale prevents log(0) errors
- [ ] Camera transitions are smooth
- [ ] Material updates don't recreate geometry (performance)
