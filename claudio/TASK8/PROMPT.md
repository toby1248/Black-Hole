## PROMPT
Implement visualization controls (color mapping, colormap, point size, log scale, toggles) and camera controls (reset, top view, side view) for the web app. Update particle appearance and camera position based on user interaction.

## COMPLEXITY
Medium

## CONTEXT REFERENCE
**Read:** `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

## TASK-SPECIFIC CONTEXT
- **Files:** `web/js/app.js`, `web/js/visualizer.js`
- **Acceptance criteria:** D.8-D.15

### Color Mapping Example
```javascript
function updateParticleColors(data, colorBy, colormap, logScale) {
  const values = data[colorBy];  // e.g., densities
  const colors = new Float32Array(values.length * 3);
  const [vmin, vmax] = logScale ? [Math.log10(Math.min(...values)), Math.log10(Math.max(...values))] : [Math.min(...values), Math.max(...values)];

  for (let i = 0; i < values.length; i++) {
    const val = logScale ? Math.log10(values[i]) : values[i];
    const normalized = (val - vmin) / (vmax - vmin);
    const rgb = applyColormap(normalized, colormap);
    colors[i*3] = rgb[0];
    colors[i*3+1] = rgb[1];
    colors[i*3+2] = rgb[2];
  }

  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  particleMaterial.vertexColors = true;
}
```

## LAYER
2

## PARALLELIZATION
Parallel with: [TASK7, TASK9]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Optimize color updates (avoid recreating geometry)
