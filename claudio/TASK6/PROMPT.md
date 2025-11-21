## PROMPT
Create the core JavaScript infrastructure for the web app: `app.js` (application state and event handlers), `data_loader.js` (HDF5/JSON loading with demo data generator), and `visualizer.js` (Three.js scene setup and rendering). This is the foundation for all web button functionality.

## COMPLEXITY
High

## CONTEXT REFERENCE
**Read:** `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

## TASK-SPECIFIC CONTEXT

### Files to Create
- `web/js/app.js` - Main app logic, state management, event handlers
- `web/js/data_loader.js` - HDF5/JSON loading, demo data generation
- `web/js/visualizer.js` - Three.js scene, rendering, camera controls

### HTML Reference
- Button IDs from `web/index.html:40-164`
- Example: `<button id="loadFiles">Load Files</button>`

### JavaScript Structure
**app.js:**
```javascript
// Global application state
const appState = {
  snapshots: [],
  currentIndex: 0,
  isPlaying: false,
  colorBy: 'density',
  colormap: 'viridis',
  // ...
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  initVisualizer();
  registerEventHandlers();
});

function registerEventHandlers() {
  document.getElementById('loadFiles').addEventListener('click', handleLoadFiles);
  document.getElementById('loadDemo').addEventListener('click', handleLoadDemo);
  // ... all other buttons
}
```

**data_loader.js:**
```javascript
async function loadHDF5File(file) {
  // Use h5wasm library or fetch + parse
  const data = await parseHDF5(file);
  return extractParticleData(data);
}

function generateDemoData() {
  // Create expanding sphere or TDE demo
  const n = 10000;
  const positions = new Float32Array(n * 3);
  // ... populate with synthetic data
  return { positions, velocities, densities, temperatures };
}
```

**visualizer.js:**
```javascript
let scene, camera, renderer, controls, particleSystem;

function initScene() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  renderer = new THREE.WebGLRenderer();
  controls = new OrbitControls(camera, renderer.domElement);
  // ...
}

function createParticleSystem(data) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  const material = new THREE.PointsMaterial({ size: 2.0 });
  particleSystem = new THREE.Points(geometry, material);
  scene.add(particleSystem);
}
```

## LAYER
2 (Web GUI Implementation)

## PARALLELIZATION
Parallel with: [TASK2, TASK3, TASK4, TASK5]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Use Three.js r128 (specified in AI_PROMPT.md)
- Handle WebGL not supported gracefully
- Minimize external dependencies
- Test in Chrome/Firefox/Edge
