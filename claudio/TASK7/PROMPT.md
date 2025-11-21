## PROMPT
Implement data source controls (Load Files, Connect to Server, Load Demo) and animation controls (Play/Pause, Step, Sliders) for the web app. Connect button event handlers to data loading and visualization updates.

## COMPLEXITY
Medium

## CONTEXT REFERENCE
**Read:** `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

## TASK-SPECIFIC CONTEXT
- **Files:** `web/js/app.js`, `web/js/data_loader.js`
- **Acceptance criteria:** D.1-D.7 from AI_PROMPT.md

### Implementation Pattern
```javascript
// Animation loop
let animationTimer;
function playAnimation() {
  animationTimer = setInterval(() => {
    appState.currentIndex = (appState.currentIndex + 1) % appState.snapshots.length;
    updateVisualization();
  }, 1000 / appState.fps);
}
```

## LAYER
2

## PARALLELIZATION
Parallel with: [TASK8, TASK9]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Use requestAnimationFrame for smooth rendering
- Handle CORS for server connections
