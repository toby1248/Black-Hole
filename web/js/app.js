/**
 * TDE-SPH Web Application
 *
 * Main application logic connecting UI controls, data loader, and visualizer.
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-11-18
 */

// Global instances
let visualizer;
let dataLoader;
let animationFrameId = null;
let isPlaying = false;
let lastFrameTime = 0;
let targetFPS = 30;
let frameInterval = 1000 / 30;  // ms between frames

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    console.log('TDE-SPH Web Visualizer initializing...');

    // Create instances
    visualizer = new Visualizer('three-canvas');
    dataLoader = new DataLoader();

    // Bind UI controls
    bindControlEvents();

    console.log('Initialization complete. Load data to begin.');
});

function bindControlEvents() {
    // Data source controls
    document.getElementById('data-source').addEventListener('change', onDataSourceChange);
    document.getElementById('load-demo-btn').addEventListener('click', loadDemoData);
    document.getElementById('load-files-btn').addEventListener('click', loadLocalFiles);
    document.getElementById('connect-server-btn').addEventListener('click', connectToServer);

    // Animation controls
    document.getElementById('play-btn').addEventListener('click', playAnimation);
    document.getElementById('pause-btn').addEventListener('click', pauseAnimation);
    document.getElementById('step-back-btn').addEventListener('click', stepBackward);
    document.getElementById('step-fwd-btn').addEventListener('click', stepForward);
    document.getElementById('snapshot-slider').addEventListener('input', onSnapshotSliderChange);
    document.getElementById('fps-slider').addEventListener('input', onFPSChange);

    // Visualization controls
    document.getElementById('color-by').addEventListener('change', onColorByChange);
    document.getElementById('colormap').addEventListener('change', onColormapChange);
    document.getElementById('point-size').addEventListener('input', onPointSizeChange);
    document.getElementById('log-scale-checkbox').addEventListener('change', onLogScaleChange);
    document.getElementById('show-bh-checkbox').addEventListener('change', onShowBHChange);
    document.getElementById('show-axes-checkbox').addEventListener('change', onShowAxesChange);

    // Camera controls
    document.getElementById('reset-camera-btn').addEventListener('click', () => visualizer.resetCamera());
    document.getElementById('top-view-btn').addEventListener('click', () => visualizer.setTopView());
    document.getElementById('side-view-btn').addEventListener('click', () => visualizer.setSideView());

    // Export controls
    document.getElementById('screenshot-btn').addEventListener('click', () => visualizer.screenshot());
    document.getElementById('export-json-btn').addEventListener('click', exportData);
}

// ============================================================================
// Data Source Management
// ============================================================================

function onDataSourceChange() {
    const source = document.getElementById('data-source').value;

    // Show/hide appropriate controls
    document.getElementById('file-upload-group').style.display = (source === 'local') ? 'block' : 'none';
    document.getElementById('server-connect-group').style.display = (source === 'server') ? 'block' : 'none';
}

function loadDemoData() {
    setLoadingMessage('Generating demo data...');

    setTimeout(() => {
        try {
            dataLoader.loadDemoData();

            // Load first snapshot
            const snapshot = dataLoader.getCurrentSnapshot();
            displaySnapshot(snapshot);

            // Update UI
            const count = dataLoader.getSnapshotCount();
            document.getElementById('snapshot-count').textContent = count;
            document.getElementById('snapshot-slider').max = count - 1;
            document.getElementById('snapshot-slider').disabled = false;

            enableAnimationControls();
            hideLoadingMessage();

            console.log('Demo data loaded successfully');

        } catch(error) {
            showError(`Failed to load demo data: ${error.message}`);
        }
    }, 100);  // Small delay for UI update
}

function loadLocalFiles() {
    const fileInput = document.getElementById('file-upload');
    const files = fileInput.files;

    if (files.length === 0) {
        showError('No files selected');
        return;
    }

    setLoadingMessage(`Loading ${files.length} file(s)...`);

    dataLoader.loadMultipleJSONFiles(files)
        .then(() => {
            const snapshot = dataLoader.getCurrentSnapshot();
            displaySnapshot(snapshot);

            // Update UI
            const count = dataLoader.getSnapshotCount();
            document.getElementById('snapshot-count').textContent = count;
            document.getElementById('snapshot-slider').max = count - 1;
            document.getElementById('snapshot-slider').disabled = false;

            enableAnimationControls();
            hideLoadingMessage();

            console.log(`Loaded ${count} snapshots from files`);
        })
        .catch(error => {
            showError(`Failed to load files: ${error.message}`);
        });
}

function connectToServer() {
    const serverURL = document.getElementById('server-url').value;

    setLoadingMessage(`Connecting to ${serverURL}...`);

    dataLoader.connectToServer(serverURL)
        .then(() => {
            hideLoadingMessage();
            console.log('Connected to server');

            // Wait for first snapshot
            const checkInterval = setInterval(() => {
                if (dataLoader.getSnapshotCount() > 0) {
                    clearInterval(checkInterval);

                    const snapshot = dataLoader.getCurrentSnapshot();
                    displaySnapshot(snapshot);

                    enableAnimationControls();
                }
            }, 100);
        })
        .catch(error => {
            showError(`Failed to connect to server: ${error.message}`);
        });
}

// ============================================================================
// Snapshot Display
// ============================================================================

function displaySnapshot(snapshot) {
    if (!snapshot) {
        console.warn('No snapshot to display');
        return;
    }

    // Get selected color field
    const colorBy = document.getElementById('color-by').value;
    let colorField;

    switch(colorBy) {
        case 'density':
            colorField = snapshot.density;
            break;
        case 'temperature':
            colorField = snapshot.temperature;
            break;
        case 'internal_energy':
            colorField = snapshot.internal_energy;
            break;
        case 'velocity_magnitude':
            colorField = snapshot.velocity_magnitude;
            break;
        case 'pressure':
            colorField = snapshot.pressure;
            break;
        case 'entropy':
            colorField = snapshot.entropy;
            break;
        default:
            colorField = snapshot.density;
    }

    // Load particles into visualizer
    if (!visualizer.particleSystem) {
        // First load
        visualizer.loadParticles(snapshot.positions, colorField, colorBy);
    } else {
        // Update existing
        visualizer.updateParticles(snapshot.positions, colorField);
    }

    // Update time display
    document.getElementById('time-display').textContent = snapshot.time.toFixed(2);

    // Update statistics
    const stats = dataLoader.computeStatistics(snapshot);
    updateStatistics(stats);

    // Update snapshot index display
    document.getElementById('snapshot-index').textContent = dataLoader.currentIndex;
}

function updateStatistics(stats) {
    document.getElementById('stat-particles').textContent = stats.n_particles.toLocaleString();
    document.getElementById('stat-mass').textContent = stats.total_mass.toExponential(2);
    document.getElementById('stat-energy').textContent = stats.total_energy.toExponential(2);
    document.getElementById('stat-rho-min').textContent = stats.rho_min.toExponential(2);
    document.getElementById('stat-rho-max').textContent = stats.rho_max.toExponential(2);
    document.getElementById('stat-temp-min').textContent = stats.temp_min.toFixed(1);
    document.getElementById('stat-temp-max').textContent = stats.temp_max.toFixed(1);
}

// ============================================================================
// Animation Control
// ============================================================================

function playAnimation() {
    if (isPlaying) return;

    isPlaying = true;
    document.getElementById('play-btn').disabled = true;
    document.getElementById('pause-btn').disabled = false;

    // Update frame interval based on FPS slider
    targetFPS = parseInt(document.getElementById('fps-slider').value);
    frameInterval = 1000 / targetFPS;
    lastFrameTime = performance.now();

    // Start animation loop using requestAnimationFrame
    animationLoop();
}

function animationLoop(currentTime) {
    if (!isPlaying) return;

    animationFrameId = requestAnimationFrame(animationLoop);

    // Throttle to target FPS
    const elapsed = currentTime - lastFrameTime;
    if (elapsed < frameInterval) return;

    lastFrameTime = currentTime - (elapsed % frameInterval);

    const snapshot = dataLoader.nextSnapshot();

    if (snapshot) {
        displaySnapshot(snapshot);
        // Update slider
        document.getElementById('snapshot-slider').value = dataLoader.currentIndex;
    } else {
        // End of sequence
        pauseAnimation();
    }
}

function pauseAnimation() {
    if (!isPlaying) return;

    isPlaying = false;
    document.getElementById('play-btn').disabled = false;
    document.getElementById('pause-btn').disabled = true;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

function stepBackward() {
    pauseAnimation();

    const snapshot = dataLoader.previousSnapshot();
    if (snapshot) {
        displaySnapshot(snapshot);
        document.getElementById('snapshot-slider').value = dataLoader.currentIndex;
    }
}

function stepForward() {
    pauseAnimation();

    const snapshot = dataLoader.nextSnapshot();
    if (snapshot) {
        displaySnapshot(snapshot);
        document.getElementById('snapshot-slider').value = dataLoader.currentIndex;
    }
}

function onSnapshotSliderChange(event) {
    pauseAnimation();

    const index = parseInt(event.target.value);
    dataLoader.setCurrentIndex(index);

    const snapshot = dataLoader.getCurrentSnapshot();
    displaySnapshot(snapshot);
}

function onFPSChange(event) {
    const fps = parseInt(event.target.value);
    document.getElementById('fps-value').textContent = fps;

    // Update frame interval (animation loop will pick up new value)
    targetFPS = fps;
    frameInterval = 1000 / fps;
}

// ============================================================================
// Visualization Control
// ============================================================================

function onColorByChange() {
    // Reload current snapshot with new color field
    const snapshot = dataLoader.getCurrentSnapshot();
    if (snapshot) {
        displaySnapshot(snapshot);
    }
}

function onColormapChange(event) {
    const colormap = event.target.value;
    visualizer.setColormap(colormap);
}

function onPointSizeChange(event) {
    const size = parseFloat(event.target.value);
    document.getElementById('point-size-value').textContent = size.toFixed(1);
    visualizer.setPointSize(size);
}

function onLogScaleChange(event) {
    const enabled = event.target.checked;
    visualizer.setLogScale(enabled);
}

function onShowBHChange(event) {
    const show = event.target.checked;
    visualizer.setShowBlackHole(show);
}

function onShowAxesChange(event) {
    const show = event.target.checked;
    visualizer.setShowAxes(show);
}

// ============================================================================
// Export
// ============================================================================

function exportData() {
    if (!dataLoader.isLoaded) {
        showError('No data to export');
        return;
    }

    dataLoader.exportToJSON(`tde_sph_export_${Date.now()}.json`);
}

// ============================================================================
// UI Helpers
// ============================================================================

function enableAnimationControls() {
    document.getElementById('play-btn').disabled = false;
    document.getElementById('step-back-btn').disabled = false;
    document.getElementById('step-fwd-btn').disabled = false;
    document.getElementById('export-json-btn').disabled = false;
}

function setLoadingMessage(message) {
    const loadingElement = document.getElementById('loading-message');
    loadingElement.textContent = message;
    loadingElement.classList.add('loading');
}

function hideLoadingMessage() {
    const loadingElement = document.getElementById('loading-message');
    loadingElement.textContent = '';
    loadingElement.classList.remove('loading');
}

function showError(message) {
    const loadingElement = document.getElementById('loading-message');
    loadingElement.textContent = `Error: ${message}`;
    loadingElement.style.color = '#ff6b6b';

    setTimeout(() => {
        loadingElement.style.color = '';
        hideLoadingMessage();
    }, 5000);
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

document.addEventListener('keydown', (event) => {
    // Space: Play/Pause
    if (event.code === 'Space') {
        event.preventDefault();
        if (isPlaying) {
            pauseAnimation();
        } else {
            playAnimation();
        }
    }

    // Arrow left: Previous snapshot
    if (event.code === 'ArrowLeft') {
        stepBackward();
    }

    // Arrow right: Next snapshot
    if (event.code === 'ArrowRight') {
        stepForward();
    }

    // R: Reset camera
    if (event.code === 'KeyR') {
        visualizer.resetCamera();
    }

    // S: Screenshot
    if (event.code === 'KeyS' && event.ctrlKey) {
        event.preventDefault();
        visualizer.screenshot();
    }
});

console.log('TDE-SPH Web Visualizer ready!');
console.log('Keyboard shortcuts:');
console.log('  Space: Play/Pause');
console.log('  ← →: Step backward/forward');
console.log('  R: Reset camera');
console.log('  Ctrl+S: Screenshot');
