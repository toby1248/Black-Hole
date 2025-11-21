## PROMPT
Implement export functionality (screenshot, JSON) and optionally volumetric rendering with WebGL ray marching. Export is mandatory, volumetric is stretch goal (ultimate objective but complex).

## COMPLEXITY
High (volumetric), Low (export)

## CONTEXT REFERENCE
**Read:** `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

## TASK-SPECIFIC CONTEXT
- **Files:** `web/js/app.js`, `web/js/visualizer.js`
- **Export:** Use canvas.toDataURL() for PNG
- **Volumetric:** Ray marching shader (see AI_PROMPT.md section V.1-V.4)

### Screenshot Example
```javascript
function exportScreenshot() {
  const dataURL = renderer.domElement.toDataURL('image/png');
  const link = document.createElement('a');
  link.download = `snapshot_${Date.now()}.png`;
  link.href = dataURL;
  link.click();
}
```

### Volumetric Approach (if implementing)
1. Convert point cloud to 3D grid (use SPH kernel interpolation)
2. Create 3D texture from grid
3. Ray marching shader samples texture
4. Transfer function maps density/temperature to color/opacity

**Note:** Volumetric is complex. Prioritize export buttons first, add volumetric only if time permits.

## LAYER
2

## PARALLELIZATION
Parallel with: [TASK7, TASK8]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Export must work in Chrome/Firefox/Edge
- Volumetric rendering optional (document if not implemented)
