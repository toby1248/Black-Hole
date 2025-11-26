/**
 * TDE-SPH Web Application
 *
 * Main application logic connecting UI controls, data loader, and visualizer.
 * Supports both legacy 2D points and new 3D instanced mesh rendering.
 * Includes smooth interpolation for fluid playback.
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-01-25
 */

// Global instances
let visualizer = null;       // Legacy 2D Points visualizer
let visualizer3D = null;     // New 3D InstancedMesh visualizer
let activeVisualizer = null; // Currently active visualizer
let dataLoader = null;
let streamingLoader = null;  // Streaming data loader (for large datasets)
let interpolator = null;

// Streaming mode state
let streamingMode = false;
let ramCacheSizeGB = 16;     // Default 16GB RAM cache

// Animation state
let animationFrameId = null;
let isPlaying = false;
let playbackSpeed = 1.0;        // Logarithmic scale: 0.0001 to 10
let interpolationEnabled = false;
let currentPlaybackTime = 0;     // Current time for interpolated playback
let lastAnimationTime = 0;       // For delta time calculation

// Wait for Three.js to be ready
let threeJSReady = false;
let initAttempts = 0;
const MAX_INIT_ATTEMPTS = 50; // 5 seconds at 100ms intervals

window.addEventListener('threejs-ready', () => {
    threeJSReady = true;
    console.log('Three.js and post-processing modules loaded');
});

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    console.log('TDE-SPH Web Visualizer initializing...');

    // Create data loader
    dataLoader = new DataLoader();
    
    // Create interpolator
    if (typeof SplineInterpolator !== 'undefined') {
        interpolator = new SplineInterpolator();
    } else {
        console.warn('SplineInterpolator not available');
    }

    // Wait for Three.js modules to load, then create visualizers
    const waitForThreeJS = () => {
        initAttempts++;
        
        if (threeJSReady && window.THREE && window.THREE.OrbitControls) {
            try {
                initVisualizers();
                bindControlEvents();
                console.log('Initialization complete. Load data to begin.');
            } catch (error) {
                console.error('Error initializing visualizers:', error);
                setLoadingMessage('Error initializing: ' + error.message);
            }
        } else if (initAttempts < MAX_INIT_ATTEMPTS) {
            // Keep waiting
            setTimeout(waitForThreeJS, 100);
        } else {
            console.error('Three.js failed to load after ' + MAX_INIT_ATTEMPTS + ' attempts');
            setLoadingMessage('Error: Three.js failed to load. Please refresh the page.');
        }
    };
    
    setTimeout(waitForThreeJS, 100);
});

function initVisualizers() {
    // Only create the selected visualizer to avoid WebGL context conflicts
    const rendererMode = document.getElementById('renderer-mode').value;
    
    if (rendererMode === 'instanced' && typeof Visualizer3D !== 'undefined') {
        visualizer3D = new Visualizer3D('three-canvas');
        activeVisualizer = visualizer3D;
        document.getElementById('point-shape-group').style.display = 'none';
        document.getElementById('bloom-panel').style.display = '';
    } else {
        visualizer = new Visualizer('three-canvas');
        activeVisualizer = visualizer;
        document.getElementById('point-shape-group').style.display = '';
        document.getElementById('bloom-panel').style.display = 'none';
    }
}

function setActiveRenderer(mode) {
    // Dispose current visualizer to free WebGL context
    if (activeVisualizer) {
        if (activeVisualizer.dispose) {
            activeVisualizer.dispose();
        }
        activeVisualizer = null;
    }
    
    // Clear references
    visualizer = null;
    visualizer3D = null;
    
    if (mode === 'instanced' && typeof Visualizer3D !== 'undefined') {
        // Create 3D instanced mesh visualizer
        visualizer3D = new Visualizer3D('three-canvas');
        activeVisualizer = visualizer3D;
        
        // Update UI
        document.getElementById('point-shape-group').style.display = 'none';
        document.getElementById('bloom-panel').style.display = '';
        
        // Re-apply current data if available
        if (dataLoader && dataLoader.getSnapshotCount() > 0) {
            visualizer3D.setDensityBaseline(dataLoader.getDensity90thPercentile());
            const bhParams = dataLoader.getBHParams();
            if (bhParams.mass) {
                visualizer3D.setBHRadius(dataLoader.getEventHorizonRadius());
            }
            const snapshot = dataLoader.getCurrentSnapshot();
            if (snapshot) {
                displaySnapshot(snapshot);
            }
        }
    } else {
        // Create legacy 2D points visualizer
        visualizer = new Visualizer('three-canvas');
        activeVisualizer = visualizer;
        
        // Update UI
        document.getElementById('point-shape-group').style.display = '';
        document.getElementById('bloom-panel').style.display = 'none';
        
        // Re-apply current data if available
        if (dataLoader && dataLoader.getSnapshotCount() > 0) {
            const snapshot = dataLoader.getCurrentSnapshot();
            if (snapshot) {
                displaySnapshot(snapshot);
            }
        }
    }
}

function bindControlEvents() {
    // Header close button
    document.getElementById('header-close-btn').addEventListener('click', toggleHeader);
    document.getElementById('show-header-btn').addEventListener('click', toggleHeader);
    
    // Streaming mode controls
    document.getElementById('streaming-mode-checkbox').addEventListener('change', onStreamingModeChange);
    document.getElementById('ram-cache-slider').addEventListener('input', onRamCacheChange);
    document.getElementById('clear-cache-btn').addEventListener('click', clearDiskCache);
    document.getElementById('precompute-splines-btn').addEventListener('click', precomputeSplines);
    
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
    document.getElementById('interpolation-checkbox').addEventListener('change', onInterpolationChange);
    document.getElementById('playback-speed').addEventListener('input', onPlaybackSpeedChange);
    
    // Renderer mode
    document.getElementById('renderer-mode').addEventListener('change', onRendererModeChange);

    // Visualization controls
    document.getElementById('color-by').addEventListener('change', onColorByChange);
    document.getElementById('colormap').addEventListener('change', onColormapChange);
    document.getElementById('point-size').addEventListener('input', onPointSizeChange);
    document.getElementById('max-size-mult').addEventListener('input', onMaxSizeMultChange);
    document.getElementById('point-shape').addEventListener('change', onPointShapeChange);
    document.getElementById('point-opacity').addEventListener('input', onPointOpacityChange);
    document.getElementById('opacity-scale').addEventListener('input', onOpacityScaleChange);
    document.getElementById('size-by-density').addEventListener('change', onSizeByDensityChange);
    document.getElementById('log-scale-checkbox').addEventListener('change', onLogScaleChange);
    
    // Scene controls
    document.getElementById('show-bh-checkbox').addEventListener('change', onShowBHChange);
    document.getElementById('show-axes-checkbox').addEventListener('change', onShowAxesChange);
    document.getElementById('show-skybox-checkbox').addEventListener('change', onShowSkyboxChange);
    
    // Bloom controls
    document.getElementById('bloom-checkbox').addEventListener('change', onBloomChange);
    document.getElementById('bloom-strength').addEventListener('input', onBloomStrengthChange);
    document.getElementById('bloom-radius').addEventListener('input', onBloomRadiusChange);
    document.getElementById('bloom-threshold').addEventListener('input', onBloomThresholdChange);

    // Camera controls
    document.getElementById('reset-camera-btn').addEventListener('click', () => activeVisualizer.resetCamera());
    document.getElementById('top-view-btn').addEventListener('click', () => activeVisualizer.setTopView());
    document.getElementById('side-view-btn').addEventListener('click', () => activeVisualizer.setSideView());
    document.getElementById('move-speed').addEventListener('input', onMoveSpeedChange);

    // Export controls
    document.getElementById('screenshot-btn').addEventListener('click', () => activeVisualizer.screenshot());
    document.getElementById('export-json-btn').addEventListener('click', exportData);

    // Initialize label values
    updateUILabels();
}

function updateUILabels() {
    document.getElementById('point-opacity-value').textContent = parseFloat(document.getElementById('point-opacity').value).toFixed(2);
    // Format point size to show 3 decimal places for small values
    const pointSize = parseFloat(document.getElementById('point-size').value);
    document.getElementById('point-size-value').textContent = pointSize < 0.1 ? pointSize.toFixed(3) : pointSize.toFixed(2);
    document.getElementById('max-size-mult-value').textContent = parseFloat(document.getElementById('max-size-mult').value).toFixed(1) + 'x';
    document.getElementById('fps-value').textContent = document.getElementById('fps-slider').value;
    
    // Playback speed (logarithmic)
    const speedSlider = document.getElementById('playback-speed').value;
    playbackSpeed = Math.pow(10, parseFloat(speedSlider));
    document.getElementById('playback-speed-value').textContent = playbackSpeed.toFixed(4) + 'x';
    
    // Opacity scale (logarithmic)
    const opacityScaleSlider = document.getElementById('opacity-scale').value;
    const opacityScale = Math.pow(10, parseFloat(opacityScaleSlider));
    document.getElementById('opacity-scale-value').textContent = opacityScale.toFixed(3);
    
    // Bloom values
    document.getElementById('bloom-strength-value').textContent = parseFloat(document.getElementById('bloom-strength').value).toFixed(1);
    // Format bloom radius to show 3 decimal places for small values
    const bloomRadius = parseFloat(document.getElementById('bloom-radius').value);
    document.getElementById('bloom-radius-value').textContent = bloomRadius < 0.1 ? bloomRadius.toFixed(3) : bloomRadius.toFixed(2);
    document.getElementById('bloom-threshold-value').textContent = parseFloat(document.getElementById('bloom-threshold').value).toFixed(2);
    
    // Move speed
    document.getElementById('move-speed-value').textContent = parseFloat(document.getElementById('move-speed').value).toFixed(1);
}

// ============================================================================
// Data Source Management
// ============================================================================

function onDataSourceChange() {
    const source = document.getElementById('data-source').value;
    document.getElementById('file-upload-group').style.display = (source === 'local') ? 'block' : 'none';
    document.getElementById('server-connect-group').style.display = (source === 'server') ? 'block' : 'none';
}

function loadDemoData() {
    setLoadingMessage('Generating demo data...');

    setTimeout(async () => {
        try {
            dataLoader.loadDemoData();
            
            // Build splines for interpolation
            if (interpolator && interpolationEnabled) {
                setLoadingMessage('Building spline interpolation...');
                await interpolator.buildSplines(dataLoader.snapshots);
            }
            
            // Pass density baseline to visualizer (90th percentile for clamp threshold)
            if (visualizer3D) {
                visualizer3D.setDensityBaseline(dataLoader.getDensity90thPercentile());
                
                // Set BH size from metadata
                const bhParams = dataLoader.getBHParams();
                if (bhParams.mass) {
                    const r_h = dataLoader.getEventHorizonRadius();
                    visualizer3D.setBHRadius(r_h);
                    updateBHStats(bhParams, r_h);
                }
            }

            // Load first snapshot
            const snapshot = dataLoader.getCurrentSnapshot();
            displaySnapshot(snapshot);
            currentPlaybackTime = snapshot.time;

            // Update UI
            updateSnapshotUI();
            enableAnimationControls();
            hideLoadingMessage();

            console.log('Demo data loaded successfully');

        } catch(error) {
            showError(`Failed to load demo data: ${error.message}`);
        }
    }, 100);
}

function loadLocalFiles() {
    const fileInput = document.getElementById('file-upload');
    const files = fileInput.files;

    if (files.length === 0) {
        showError('No files selected');
        return;
    }

    // Use streaming loader if streaming mode is enabled
    if (streamingMode) {
        loadFilesStreaming(files);
        return;
    }

    setLoadingMessage(`Loading ${files.length} file(s)...`);
    updateProgress(0);
    
    // Reset camera initialization for new data
    if (visualizer3D) {
        visualizer3D._cameraInitialized = false;
    }

    dataLoader.loadMultipleFiles(files, updateProgress)
        .then(async () => {
            // Build splines for interpolation
            if (interpolator && interpolationEnabled) {
                setLoadingMessage('Building spline interpolation...');
                await interpolator.buildSplines(dataLoader.snapshots);
                
                // Build field splines for smooth density/temperature transitions
                setLoadingMessage('Building field splines (density, temperature)...');
                interpolator.buildFieldSplines(dataLoader.snapshots);
            }
            
            // Pass density baseline to visualizer (90th percentile for clamp threshold)
            if (visualizer3D) {
                visualizer3D.setDensityBaseline(dataLoader.getDensity90thPercentile());
                
                // Set BH size from metadata
                const bhParams = dataLoader.getBHParams();
                if (bhParams.mass) {
                    const r_h = dataLoader.getEventHorizonRadius();
                    visualizer3D.setBHRadius(r_h);
                    updateBHStats(bhParams, r_h);
                }
            }
            
            const snapshot = dataLoader.getCurrentSnapshot();
            displaySnapshot(snapshot);
            currentPlaybackTime = snapshot.time;

            updateSnapshotUI();
            enableAnimationControls();
            hideLoadingMessage();
            updateProgress(100);

            console.log(`Loaded ${dataLoader.getSnapshotCount()} snapshots from files`);
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

            const checkInterval = setInterval(() => {
                if (dataLoader.getSnapshotCount() > 0) {
                    clearInterval(checkInterval);

                    const snapshot = dataLoader.getCurrentSnapshot();
                    displaySnapshot(snapshot);
                    currentPlaybackTime = snapshot.time;

                    enableAnimationControls();
                }
            }, 100);
        })
        .catch(error => {
            showError(`Failed to connect to server: ${error.message}`);
        });
}

function updateSnapshotUI() {
    const count = dataLoader.getSnapshotCount();
    document.getElementById('snapshot-count').textContent = count;
    document.getElementById('snapshot-slider').max = count - 1;
    document.getElementById('snapshot-slider').disabled = false;
}

function updateBHStats(bhParams, r_h) {
    document.getElementById('stat-bh-mass').textContent = bhParams.mass ? bhParams.mass.toExponential(2) : '-';
    document.getElementById('stat-event-horizon').textContent = r_h ? r_h.toFixed(2) : '-';
}

// ============================================================================
// Snapshot Display
// ============================================================================

function displaySnapshot(snapshot) {
    if (!snapshot) {
        console.warn('No snapshot to display');
        return;
    }

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

    // Load particles into active visualizer
    if (activeVisualizer === visualizer3D) {
        if (!visualizer3D.particleSystem) {
            visualizer3D.loadParticles(snapshot.positions, colorField, snapshot.density, colorBy);
        } else {
            visualizer3D.updateParticles(snapshot.positions, colorField, snapshot.density);
        }
    } else {
        if (!visualizer.particleSystem) {
            visualizer.loadParticles(snapshot.positions, colorField, snapshot.density, colorBy);
        } else {
            visualizer.updateParticles(snapshot.positions, colorField, snapshot.density);
        }
    }

    // Update time display
    document.getElementById('time-display').textContent = snapshot.time.toFixed(2);

    // Update statistics
    const stats = dataLoader.computeStatistics(snapshot);
    updateStatistics(stats);

    // Update snapshot index display
    document.getElementById('snapshot-index').textContent = dataLoader.currentIndex;
}

/**
 * Display interpolated positions at a given time.
 */
function displayInterpolatedFrame(time) {
    // Check if interpolation is available (splines OR linear fallback)
    const canInterpolate = interpolator && (interpolator.isReady || interpolator.useLinearFallback);
    
    if (!canInterpolate) {
        // Fallback to nearest snapshot
        const idx = interpolator ? interpolator.findClosestSnapshotIndex(time) : 0;
        dataLoader.setCurrentIndex(idx);
        displaySnapshot(dataLoader.getCurrentSnapshot());
        return;
    }
    
    // Get interpolated positions
    const positions = interpolator.interpolate(time);
    
    if (!positions) {
        // Interpolation failed, fall back to nearest snapshot
        const idx = interpolator.findClosestSnapshotIndex(time);
        dataLoader.setCurrentIndex(idx);
        displaySnapshot(dataLoader.getCurrentSnapshot());
        return;
    }
    
    // Get other fields from nearest snapshot
    const snapIdx = interpolator.findClosestSnapshotIndex(time);
    const snapshot = dataLoader.getSnapshotByIndex(snapIdx);
    
    if (!snapshot) return;
    
    // Use interpolated fields if available, otherwise fall back to nearest snapshot
    let interpolatedDensity = null;
    let interpolatedTemperature = null;
    
    if (interpolator.fieldSplinesReady) {
        interpolatedDensity = interpolator.interpolateDensity(time);
        interpolatedTemperature = interpolator.interpolateTemperature(time);
    }
    
    const colorBy = document.getElementById('color-by').value;
    let colorField;

    switch(colorBy) {
        case 'density': 
            colorField = interpolatedDensity || snapshot.density; 
            break;
        case 'temperature': 
            colorField = interpolatedTemperature || snapshot.temperature; 
            break;
        case 'internal_energy': colorField = snapshot.internal_energy; break;
        case 'velocity_magnitude': colorField = snapshot.velocity_magnitude; break;
        case 'pressure': colorField = snapshot.pressure; break;
        case 'entropy': colorField = snapshot.entropy; break;
        default: 
            colorField = interpolatedDensity || snapshot.density;
    }
    
    // Use interpolated density for size scaling if available
    const densityForSize = interpolatedDensity || snapshot.density;
    
    // Update visualizer with interpolated positions
    if (activeVisualizer === visualizer3D) {
        if (!visualizer3D.instancedMesh) {
            visualizer3D.loadParticles(positions, colorField, densityForSize, colorBy);
        } else {
            visualizer3D.updateParticles(positions, colorField, densityForSize);
        }
    } else {
        if (!visualizer.particleSystem) {
            visualizer.loadParticles(positions, colorField, densityForSize, colorBy);
        } else {
            visualizer.updateParticles(positions, colorField, densityForSize);
        }
    }
    
    // Update time display
    document.getElementById('time-display').textContent = time.toFixed(2);
    
    // Update snapshot index (nearest)
    document.getElementById('snapshot-index').textContent = snapIdx;
    document.getElementById('snapshot-slider').value = snapIdx;
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

    lastAnimationTime = performance.now();
    
    // Use requestAnimationFrame for smooth playback
    animationLoop();
}

function animationLoop() {
    if (!isPlaying) return;
    
    const currentTime = performance.now();
    const deltaTime = (currentTime - lastAnimationTime) / 1000; // seconds
    lastAnimationTime = currentTime;
    
    // Handle streaming mode animation
    if (streamingMode && streamingLoader) {
        animationLoopStreaming();  // No longer needs deltaTime parameter - calculates its own
        return;
    }
    
    // Get time range from data
    const firstSnapshot = dataLoader.getSnapshotByIndex(0);
    const lastSnapshot = dataLoader.getSnapshotByIndex(dataLoader.getSnapshotCount() - 1);
    
    if (!firstSnapshot || !lastSnapshot) {
        pauseAnimation();
        return;
    }
    
    const minTime = firstSnapshot.time;
    const maxTime = lastSnapshot.time;
    
    // Advance playback time
    currentPlaybackTime += deltaTime * playbackSpeed;
    
    // Wrap or clamp
    if (currentPlaybackTime > maxTime) {
        currentPlaybackTime = minTime;
    }
    
    // Check if interpolation is available (splines ready OR linear fallback)
    const canInterpolate = interpolator && (interpolator.isReady || interpolator.useLinearFallback);
    
    if (interpolationEnabled && canInterpolate) {
        displayInterpolatedFrame(currentPlaybackTime);
    } else {
        // Step through snapshots based on time
        const idx = findSnapshotIndexForTime(currentPlaybackTime);
        if (idx !== dataLoader.currentIndex) {
            dataLoader.setCurrentIndex(idx);
            displaySnapshot(dataLoader.getCurrentSnapshot());
            document.getElementById('snapshot-slider').value = idx;
        }
    }
    
    // Limit framerate
    const maxFps = parseInt(document.getElementById('fps-slider').value);
    const frameTime = 1000 / maxFps;
    
    setTimeout(() => {
        animationFrameId = requestAnimationFrame(animationLoop);
    }, Math.max(0, frameTime - deltaTime * 1000));
}

function findSnapshotIndexForTime(time) {
    // Binary search for closest snapshot
    const count = dataLoader.getSnapshotCount();
    let low = 0, high = count - 1;
    
    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        const snapshot = dataLoader.getSnapshotByIndex(mid);
        
        if (snapshot.time < time) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    
    return Math.max(0, Math.min(count - 1, low));
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

    // Handle streaming mode
    if (streamingMode && streamingLoader) {
        const newIndex = Math.max(0, streamingCurrentIndex - 1);
        if (newIndex !== streamingCurrentIndex) {
            streamingCurrentIndex = newIndex;
            streamingLoader.getSnapshot(newIndex).then(snapshot => {
                if (snapshot) {
                    displaySnapshotFromStreaming(snapshot);
                    currentPlaybackTime = snapshot.time;
                    document.getElementById('snapshot-slider').value = newIndex;
                }
            });
        }
        return;
    }

    const snapshot = dataLoader.previousSnapshot();
    if (snapshot) {
        displaySnapshot(snapshot);
        currentPlaybackTime = snapshot.time;
        document.getElementById('snapshot-slider').value = dataLoader.currentIndex;
    }
}

function stepForward() {
    pauseAnimation();

    // Handle streaming mode
    if (streamingMode && streamingLoader) {
        const count = streamingLoader.getSnapshotCount();
        const newIndex = Math.min(count - 1, streamingCurrentIndex + 1);
        if (newIndex !== streamingCurrentIndex) {
            streamingCurrentIndex = newIndex;
            streamingLoader.getSnapshot(newIndex).then(snapshot => {
                if (snapshot) {
                    displaySnapshotFromStreaming(snapshot);
                    currentPlaybackTime = snapshot.time;
                    document.getElementById('snapshot-slider').value = newIndex;
                }
            });
        }
        return;
    }

    const snapshot = dataLoader.nextSnapshot();
    if (snapshot) {
        displaySnapshot(snapshot);
        currentPlaybackTime = snapshot.time;
        document.getElementById('snapshot-slider').value = dataLoader.currentIndex;
    }
}

function onSnapshotSliderChange(event) {
    pauseAnimation();

    const index = parseInt(event.target.value);
    
    // Handle streaming mode
    if (streamingMode && streamingLoader) {
        streamingCurrentIndex = index;
        streamingLoader.getSnapshot(index).then(snapshot => {
            if (snapshot) {
                displaySnapshotFromStreaming(snapshot);
                currentPlaybackTime = snapshot.time;
            }
        }).catch(err => {
            console.error('Failed to load snapshot:', err);
        });
        return;
    }
    
    dataLoader.setCurrentIndex(index);

    const snapshot = dataLoader.getCurrentSnapshot();
    displaySnapshot(snapshot);
    currentPlaybackTime = snapshot.time;
}

function onFPSChange(event) {
    document.getElementById('fps-value').textContent = event.target.value;
}

function onInterpolationChange(event) {
    interpolationEnabled = event.target.checked;
    
    // Build splines if enabled and data is loaded
    if (interpolationEnabled && dataLoader.isLoaded && interpolator) {
        setLoadingMessage('Building interpolation...');
        interpolator.buildSplines(dataLoader.snapshots).then((success) => {
            hideLoadingMessage();
            if (interpolator.useLinearFallback) {
                console.log('Using linear interpolation (dataset too large for splines)');
            } else if (success) {
                console.log('Spline interpolation ready');
            } else {
                console.log('Interpolation not available');
            }
        });
    }
}

function onPlaybackSpeedChange(event) {
    const logValue = parseFloat(event.target.value);
    playbackSpeed = Math.pow(10, logValue);
    document.getElementById('playback-speed-value').textContent = playbackSpeed.toFixed(4) + 'x';
}

// ============================================================================
// Renderer Mode
// ============================================================================

function onRendererModeChange(event) {
    setActiveRenderer(event.target.value);
}

// ============================================================================
// Visualization Control
// ============================================================================

function onColorByChange() {
    const snapshot = dataLoader.getCurrentSnapshot();
    if (snapshot) {
        displaySnapshot(snapshot);
    }
}

function onColormapChange(event) {
    const colormap = event.target.value;
    if (activeVisualizer === visualizer3D) {
        visualizer3D.setColormap(colormap);
    } else {
        visualizer.setColormap(colormap);
    }
}

function onPointSizeChange(event) {
    const size = parseFloat(event.target.value);
    document.getElementById('point-size-value').textContent = size.toFixed(4);
    
    if (activeVisualizer === visualizer3D) {
        visualizer3D.setPointSize(size);
    } else {
        visualizer.setPointSize(size);
    }
}

function onMaxSizeMultChange(event) {
    const mult = parseFloat(event.target.value);
    document.getElementById('max-size-mult-value').textContent = mult.toFixed(1) + 'x';
    
    if (visualizer3D) {
        visualizer3D.setMaxSizeMultiplier(mult);
    }
}

function onPointShapeChange(event) {
    const shape = event.target.value;
    visualizer.setPointShape(shape);
}

function onPointOpacityChange(event) {
    const opacity = parseFloat(event.target.value);
    document.getElementById('point-opacity-value').textContent = opacity.toFixed(2);
    
    if (activeVisualizer === visualizer3D) {
        visualizer3D.setPointOpacity(opacity);
    } else {
        visualizer.setPointOpacity(opacity);
    }
}

function onOpacityScaleChange(event) {
    const logValue = parseFloat(event.target.value);
    const scale = Math.pow(10, logValue);
    document.getElementById('opacity-scale-value').textContent = scale.toFixed(3);
    
    if (visualizer3D) {
        visualizer3D.setOpacityScale(scale);
    }
}

function onSizeByDensityChange(event) {
    const enabled = event.target.checked;
    
    if (activeVisualizer === visualizer3D) {
        visualizer3D.setSizeByDensity(enabled);
    } else {
        visualizer.setSizeByDensity(enabled);
    }
}

function onLogScaleChange(event) {
    const enabled = event.target.checked;
    
    if (activeVisualizer === visualizer3D) {
        visualizer3D.setLogScale(enabled);
    } else {
        visualizer.setLogScale(enabled);
    }
}

function onShowBHChange(event) {
    const show = event.target.checked;
    
    if (activeVisualizer === visualizer3D) {
        visualizer3D.setShowBlackHole(show);
    } else {
        visualizer.setShowBlackHole(show);
    }
}

function onShowAxesChange(event) {
    const show = event.target.checked;
    
    if (activeVisualizer === visualizer3D) {
        visualizer3D.setShowAxes(show);
    } else {
        visualizer.setShowAxes(show);
    }
}

function onShowSkyboxChange(event) {
    const show = event.target.checked;
    
    if (visualizer3D) {
        visualizer3D.setShowSkybox(show);
    }
}

// ============================================================================
// Bloom Control
// ============================================================================

function onBloomChange(event) {
    if (visualizer3D) {
        visualizer3D.setBloomEnabled(event.target.checked);
    }
}

function onBloomStrengthChange(event) {
    const strength = parseFloat(event.target.value);
    document.getElementById('bloom-strength-value').textContent = strength.toFixed(1);
    
    if (visualizer3D) {
        visualizer3D.setBloomStrength(strength);
    }
}

function onBloomRadiusChange(event) {
    const radius = parseFloat(event.target.value);
    document.getElementById('bloom-radius-value').textContent = radius.toFixed(2);
    
    if (visualizer3D) {
        visualizer3D.setBloomRadius(radius);
    }
}

function onBloomThresholdChange(event) {
    const threshold = parseFloat(event.target.value);
    document.getElementById('bloom-threshold-value').textContent = threshold.toFixed(2);
    
    if (visualizer3D) {
        visualizer3D.setBloomThreshold(threshold);
    }
}

// ============================================================================
// Camera Control
// ============================================================================

function onMoveSpeedChange(event) {
    const speed = parseFloat(event.target.value);
    document.getElementById('move-speed-value').textContent = speed.toFixed(1);
    
    if (visualizer3D) {
        visualizer3D.setMoveSpeed(speed);
    }
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

function updateProgress(percent) {
    const progress = document.getElementById('load-progress');
    const label = document.getElementById('load-progress-text');
    if (!progress || !label) return;
    const clamped = Math.max(0, Math.min(100, percent));
    progress.value = clamped;
    label.textContent = `${clamped}%`;
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
// Header Toggle
// ============================================================================

function toggleHeader() {
    const header = document.querySelector('header');
    const showBtn = document.getElementById('show-header-btn');
    
    if (header.classList.contains('collapsed')) {
        header.classList.remove('collapsed');
        showBtn.style.display = 'none';
    } else {
        header.classList.add('collapsed');
        showBtn.style.display = 'block';
    }
}

// ============================================================================
// Streaming Mode Management
// ============================================================================

function onStreamingModeChange(event) {
    streamingMode = event.target.checked;
    
    // Show/hide RAM cache slider and stats
    const memorySettings = document.getElementById('memory-settings');
    const cacheStats = document.getElementById('cache-stats');
    const splineOptions = document.getElementById('spline-options');
    
    if (streamingMode) {
        if (memorySettings) memorySettings.style.display = '';
        if (cacheStats) cacheStats.style.display = '';
        if (splineOptions) splineOptions.style.display = '';
        console.log('Streaming mode enabled - data will be loaded on-demand with disk cache');
    } else {
        if (memorySettings) memorySettings.style.display = 'none';
        if (cacheStats) cacheStats.style.display = 'none';
        if (splineOptions) splineOptions.style.display = 'none';
        console.log('Streaming mode disabled - all data loaded into memory');
    }
}

function onRamCacheChange(event) {
    ramCacheSizeGB = parseInt(event.target.value);
    document.getElementById('ram-cache-value').textContent = ramCacheSizeGB + ' GB';
    
    // Update streaming loader if active
    if (streamingLoader) {
        streamingLoader.setMaxRamCacheSize(ramCacheSizeGB * 1024 * 1024 * 1024);
    }
}

async function clearDiskCache() {
    if (streamingLoader) {
        await streamingLoader.clearDiskCache();
        updateCacheStats();
        console.log('Disk cache cleared');
    } else {
        // Clear IndexedDB directly if no loader
        try {
            const dbRequest = indexedDB.deleteDatabase('tde-sph-streaming-cache');
            dbRequest.onsuccess = () => {
                console.log('Disk cache cleared');
                alert('Disk cache cleared successfully');
            };
            dbRequest.onerror = () => {
                console.error('Failed to clear disk cache');
            };
        } catch (e) {
            console.error('Error clearing cache:', e);
        }
    }
}

function updateCacheStats() {
    const statsEl = document.getElementById('cache-stats');
    if (!statsEl) return;
    
    if (streamingLoader) {
        const stats = streamingLoader.getCacheStats();
        statsEl.innerHTML = `
            <span class="cache-label">RAM:</span> <span class="cache-value">${formatBytes(stats.ramUsed)}</span> / ${formatBytes(stats.ramLimit)}<br>
            <span class="cache-label">Disk:</span> <span class="cache-value">${stats.diskEntries} snapshots</span><br>
            <span class="cache-label">Loaded:</span> <span class="cache-value">${stats.ramEntries}</span> in memory
        `;
        
        // Update spline status
        const splineStatus = document.getElementById('spline-status');
        if (splineStatus) {
            if (streamingLoader.hasPrecomputedSplines) {
                splineStatus.innerHTML = '<span style="color:#4CAF50;">✓ Splines cached</span>';
            } else {
                splineStatus.innerHTML = '<span style="color:#888;">No spline cache</span>';
            }
        }
    } else {
        statsEl.innerHTML = '<span class="cache-label">No streaming loader active</span>';
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Precompute spline coefficients and cache to disk.
 */
async function precomputeSplines() {
    if (!streamingLoader) {
        alert('Please load data in streaming mode first.');
        return;
    }
    
    // Estimate size and warn if > 100GB
    const estimatedSize = streamingLoader.estimateSplineDataSize();
    const estimatedGB = estimatedSize / (1024 * 1024 * 1024);
    
    if (estimatedGB > 100) {
        const confirmed = confirm(
            `⚠️ Warning: Spline precomputation will generate approximately ${formatBytes(estimatedSize)} of cached data.\n\n` +
            `This will be stored in your browser's IndexedDB storage.\n\n` +
            `Make sure you have sufficient disk space. Continue?`
        );
        if (!confirmed) {
            return;
        }
    } else {
        const confirmed = confirm(
            `Spline precomputation will generate approximately ${formatBytes(estimatedSize)} of cached data.\n\n` +
            `This enables smooth cubic spline interpolation during playback.\n\n` +
            `Continue?`
        );
        if (!confirmed) {
            return;
        }
    }
    
    // Show progress UI
    const progressDiv = document.getElementById('spline-progress');
    const progressBar = document.getElementById('spline-progress-bar');
    const progressText = document.getElementById('spline-progress-text');
    const precomputeBtn = document.getElementById('precompute-splines-btn');
    
    progressDiv.style.display = 'block';
    precomputeBtn.disabled = true;
    
    try {
        const success = await streamingLoader.precomputeSplines(
            (progress, message) => {
                progressBar.value = progress;
                progressText.textContent = `${progress.toFixed(0)}% - ${message}`;
            }
        );
        
        if (success) {
            alert('Spline precomputation complete! Smooth interpolation is now available.');
        }
    } catch (error) {
        console.error('Spline precomputation failed:', error);
        alert(`Spline precomputation failed: ${error.message}`);
    } finally {
        progressDiv.style.display = 'none';
        precomputeBtn.disabled = false;
        updateCacheStats();
    }
}

// ============================================================================
// Streaming Data Loading
// ============================================================================

async function loadFilesStreaming(files) {
    setLoadingMessage('Initializing streaming loader...');
    
    // Create streaming loader with current RAM cache setting
    if (typeof StreamingDataLoader !== 'undefined') {
        streamingLoader = new StreamingDataLoader({
            maxRamCacheSize: ramCacheSizeGB * 1024 * 1024 * 1024,
            maxDiskCacheSize: 200 * 1024 * 1024 * 1024, // 200GB
            preloadAhead: 3,
            preloadBehind: 1
        });
        
        try {
            setLoadingMessage('Scanning files (metadata only)...');
            await streamingLoader.loadFilesMetadataOnly(files, (progress) => {
                updateProgress(progress * 100);
            });
            
            setLoadingMessage('Streaming loader ready');
            
            // Check for existing spline cache
            const hasSplines = await streamingLoader.hasSplineCache();
            if (hasSplines) {
                console.log('Found existing spline cache - smooth interpolation available');
            }
            
            // Get first snapshot to display
            const count = streamingLoader.getSnapshotCount();
            if (count > 0) {
                // Load first snapshot
                const snapshot = await streamingLoader.getSnapshot(0);
                
                // Set up visualizer
                if (visualizer3D) {
                    visualizer3D.setDensityBaseline(streamingLoader.getDensity90thPercentile());
                    const bhParams = streamingLoader.getBHParams();
                    if (bhParams && bhParams.mass) {
                        const r_h = streamingLoader.getEventHorizonRadius();
                        visualizer3D.setBHRadius(r_h);
                        updateBHStats(bhParams, r_h);
                    }
                }
                
                // Display first snapshot
                displaySnapshotFromStreaming(snapshot);
                currentPlaybackTime = snapshot.time;
                
                // Update UI
                document.getElementById('snapshot-count').textContent = count;
                document.getElementById('snapshot-slider').max = count - 1;
                document.getElementById('snapshot-slider').disabled = false;
                
                enableAnimationControls();
                hideLoadingMessage();
                
                // Start cache stats update interval
                setInterval(updateCacheStats, 1000);
                updateCacheStats();
                
                console.log(`Streaming loader ready: ${count} snapshots available`);
            }
        } catch (error) {
            showError(`Streaming load failed: ${error.message}`);
            console.error('Streaming load error:', error);
        }
    } else {
        showError('StreamingDataLoader not available');
    }
}

function displaySnapshotFromStreaming(snapshot) {
    if (!snapshot) {
        console.warn('No snapshot to display');
        return;
    }

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

    // Load particles into active visualizer
    if (activeVisualizer === visualizer3D) {
        if (!visualizer3D.particleSystem) {
            visualizer3D.loadParticles(snapshot.positions, colorField, snapshot.density, colorBy);
        } else {
            visualizer3D.updateParticles(snapshot.positions, colorField, snapshot.density);
        }
    } else {
        if (!visualizer.particleSystem) {
            visualizer.loadParticles(snapshot.positions, colorField, snapshot.density, colorBy);
        } else {
            visualizer.updateParticles(snapshot.positions, colorField, snapshot.density);
        }
    }

    // Update time display
    document.getElementById('time-display').textContent = snapshot.time.toFixed(2);

    // Update statistics (compute on the fly for streaming)
    const stats = computeStreamingStats(snapshot);
    updateStatistics(stats);

    // Update snapshot index display
    document.getElementById('snapshot-index').textContent = snapshot.index || 0;
}

function computeStreamingStats(snapshot) {
    const n = snapshot.positions.length / 3;
    
    // Quick stats computation
    let rhoMin = Infinity, rhoMax = -Infinity;
    let tempMin = Infinity, tempMax = -Infinity;
    let totalMass = 0;
    
    for (let i = 0; i < n; i++) {
        const rho = snapshot.density[i];
        const temp = snapshot.temperature ? snapshot.temperature[i] : 0;
        const mass = snapshot.mass ? snapshot.mass[i] : 1;
        
        if (rho < rhoMin) rhoMin = rho;
        if (rho > rhoMax) rhoMax = rho;
        if (temp < tempMin) tempMin = temp;
        if (temp > tempMax) tempMax = temp;
        totalMass += mass;
    }
    
    return {
        n_particles: n,
        total_mass: totalMass,
        total_energy: 0, // Skip for performance
        rho_min: rhoMin,
        rho_max: rhoMax,
        temp_min: tempMin,
        temp_max: tempMax
    };
}

// Streaming mode animation loop
let streamingCurrentIndex = 0;
let streamingLoadPending = false;  // Prevent multiple simultaneous loads
let streamingCachedSnapshots = new Map();  // Cache for interpolation: index -> snapshot
let streamingLastDisplayedTime = 0;
let streamingSplineLoadPending = false;

function animationLoopStreaming() {
    if (!isPlaying || !streamingLoader) return;
    
    const count = streamingLoader.getSnapshotCount();
    if (count === 0) {
        pauseAnimation();
        return;
    }
    
    // Calculate delta time properly each frame
    const currentTime = performance.now();
    const deltaTime = (currentTime - lastAnimationTime) / 1000;
    lastAnimationTime = currentTime;
    
    // Advance time
    const timeRange = streamingLoader.getTimeRange();
    currentPlaybackTime += deltaTime * playbackSpeed;
    
    // Wrap around
    if (currentPlaybackTime > timeRange.max) {
        currentPlaybackTime = timeRange.min;
        streamingCachedSnapshots.clear();  // Clear cache on wrap
    }
    
    // Use precomputed splines if available and interpolation is enabled
    if (interpolationEnabled && streamingLoader.hasPrecomputedSplines && !streamingSplineLoadPending) {
        streamingSplineLoadPending = true;
        
        // Load interpolated positions from spline cache (non-blocking)
        streamingLoader.getSplineInterpolatedPositions(currentPlaybackTime).then(positions => {
            streamingSplineLoadPending = false;
            
            if (positions && isPlaying) {
                // Get density from nearest snapshot for coloring
                const nearestIdx = streamingLoader.findSnapshotIndexForTime(currentPlaybackTime);
                const nearestSnap = streamingCachedSnapshots.get(nearestIdx);
                
                if (nearestSnap) {
                    displaySplineInterpolatedFrame(positions, nearestSnap);
                } else {
                    // Need to load the nearest snapshot for color data
                    streamingLoader.getSnapshot(nearestIdx).then(snap => {
                        if (snap) {
                            streamingCachedSnapshots.set(nearestIdx, snap);
                            displaySplineInterpolatedFrame(positions, snap);
                        }
                    });
                }
            }
            
            // Update time display
            document.getElementById('time-display').textContent = currentPlaybackTime.toFixed(2);
        }).catch(err => {
            streamingSplineLoadPending = false;
            console.warn('Spline interpolation failed:', err);
        });
        
        // Continue animation
        scheduleNextStreamingFrame(currentTime);
        return;
    }
    
    // Fall back to linear interpolation
    const times = streamingLoader.getSnapshotTimes();
    let lowerIdx = 0;
    let upperIdx = 0;
    
    for (let i = 0; i < times.length - 1; i++) {
        if (times[i] <= currentPlaybackTime && times[i + 1] >= currentPlaybackTime) {
            lowerIdx = i;
            upperIdx = i + 1;
            break;
        }
    }
    
    // Handle edge cases
    if (currentPlaybackTime <= times[0]) {
        lowerIdx = 0;
        upperIdx = 0;
    } else if (currentPlaybackTime >= times[times.length - 1]) {
        lowerIdx = times.length - 1;
        upperIdx = times.length - 1;
    }
    
    // Load snapshots for interpolation if needed
    const needsLower = !streamingCachedSnapshots.has(lowerIdx);
    const needsUpper = !streamingCachedSnapshots.has(upperIdx) && upperIdx !== lowerIdx;
    
    if ((needsLower || needsUpper) && !streamingLoadPending) {
        streamingLoadPending = true;
        
        const loadPromises = [];
        if (needsLower) {
            loadPromises.push(
                streamingLoader.getSnapshot(lowerIdx).then(s => {
                    if (s) streamingCachedSnapshots.set(lowerIdx, s);
                })
            );
        }
        if (needsUpper) {
            loadPromises.push(
                streamingLoader.getSnapshot(upperIdx).then(s => {
                    if (s) streamingCachedSnapshots.set(upperIdx, s);
                })
            );
        }
        
        Promise.all(loadPromises).then(() => {
            streamingLoadPending = false;
            // Clean up old cache entries (keep only nearby snapshots)
            for (const [idx] of streamingCachedSnapshots) {
                if (Math.abs(idx - lowerIdx) > 3) {
                    streamingCachedSnapshots.delete(idx);
                }
            }
        }).catch(err => {
            streamingLoadPending = false;
            console.warn('Failed to load snapshots for interpolation:', err);
        });
    }
    
    // Perform interpolation if we have both snapshots
    const lowerSnap = streamingCachedSnapshots.get(lowerIdx);
    const upperSnap = streamingCachedSnapshots.get(upperIdx);
    
    if (lowerSnap && (upperIdx === lowerIdx || upperSnap)) {
        if (interpolationEnabled && upperSnap && lowerIdx !== upperIdx) {
            // Linear interpolation between two snapshots
            const t0 = times[lowerIdx];
            const t1 = times[upperIdx];
            const alpha = (t1 > t0) ? (currentPlaybackTime - t0) / (t1 - t0) : 0;
            
            displayInterpolatedStreamingFrame(lowerSnap, upperSnap, alpha);
        } else {
            // Just display the lower snapshot
            if (streamingCurrentIndex !== lowerIdx) {
                streamingCurrentIndex = lowerIdx;
                displaySnapshotFromStreaming(lowerSnap);
                document.getElementById('snapshot-slider').value = lowerIdx;
            }
        }
        
        // Update time display
        document.getElementById('time-display').textContent = currentPlaybackTime.toFixed(2);
    }
    
    // Continue animation
    scheduleNextStreamingFrame(currentTime);
}

/**
 * Schedule the next streaming animation frame with FPS limiting.
 */
function scheduleNextStreamingFrame(startTime) {
    if (!isPlaying) return;
    
    const maxFps = parseInt(document.getElementById('fps-slider').value);
    const frameTime = 1000 / maxFps;
    
    const elapsed = performance.now() - startTime;
    const delay = Math.max(0, frameTime - elapsed);
    
    setTimeout(() => {
        if (isPlaying) {
            animationFrameId = requestAnimationFrame(animationLoopStreaming);
        }
    }, delay);
}

/**
 * Display frame using precomputed spline interpolated positions.
 */
function displaySplineInterpolatedFrame(positions, referenceSnapshot) {
    const n = positions.length / 3;
    
    // Get color field based on current setting
    const colorBy = document.getElementById('color-by').value;
    let colorField;
    
    switch(colorBy) {
        case 'density':
            colorField = referenceSnapshot.density;
            break;
        case 'temperature':
            colorField = referenceSnapshot.temperature;
            break;
        default:
            colorField = referenceSnapshot.density;
    }
    
    // Update visualizer
    if (activeVisualizer === visualizer3D) {
        if (!visualizer3D.particleSystem) {
            visualizer3D.loadParticles(positions, colorField, referenceSnapshot.density, colorBy);
        } else {
            visualizer3D.updateParticles(positions, colorField, referenceSnapshot.density);
        }
    } else {
        if (!visualizer.particleSystem) {
            visualizer.loadParticles(positions, colorField, referenceSnapshot.density, colorBy);
        } else {
            visualizer.updateParticles(positions, colorField, referenceSnapshot.density);
        }
    }
}

/**
 * Display linearly interpolated frame between two snapshots.
 */
function displayInterpolatedStreamingFrame(snap0, snap1, alpha) {
    const n = snap0.positions.length / 3;
    
    // Interpolate positions
    const positions = new Float32Array(snap0.positions.length);
    for (let i = 0; i < positions.length; i++) {
        positions[i] = snap0.positions[i] * (1 - alpha) + snap1.positions[i] * alpha;
    }
    
    // Interpolate density for coloring
    const density = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        density[i] = snap0.density[i] * (1 - alpha) + snap1.density[i] * alpha;
    }
    
    // Get color field based on current setting
    const colorBy = document.getElementById('color-by').value;
    let colorField;
    
    switch(colorBy) {
        case 'density':
            colorField = density;
            break;
        case 'temperature':
            if (snap0.temperature && snap1.temperature) {
                colorField = new Float32Array(n);
                for (let i = 0; i < n; i++) {
                    colorField[i] = snap0.temperature[i] * (1 - alpha) + snap1.temperature[i] * alpha;
                }
            } else {
                colorField = density;
            }
            break;
        default:
            colorField = density;
    }
    
    // Update visualizer
    if (activeVisualizer === visualizer3D) {
        if (!visualizer3D.particleSystem) {
            visualizer3D.loadParticles(positions, colorField, density, colorBy);
        } else {
            visualizer3D.updateParticles(positions, colorField, density);
        }
    } else {
        if (!visualizer.particleSystem) {
            visualizer.loadParticles(positions, colorField, density, colorBy);
        } else {
            visualizer.updateParticles(positions, colorField, density);
        }
    }
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

document.addEventListener('keydown', (event) => {
    // Ignore if focused on input
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') {
        return;
    }
    
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

    // Ctrl+R: Reset camera
    if (event.code === 'KeyR' && event.ctrlKey) {
        event.preventDefault();
        activeVisualizer.resetCamera();
    }

    // Ctrl+S: Screenshot
    if (event.code === 'KeyS' && event.ctrlKey) {
        event.preventDefault();
        activeVisualizer.screenshot();
    }
});

console.log('TDE-SPH Web Visualizer ready!');
console.log('Keyboard shortcuts:');
console.log('  Space: Play/Pause');
console.log('  ← →: Step backward/forward');
console.log('  W/S: Forward/Backward');
console.log('  A/D: Left/Right');
console.log('  Q/E: Rotate Left/Right');
console.log('  Shift: Move Up');
console.log('  Ctrl: Move Down');
console.log('  Ctrl+S: Screenshot');
console.log('  Ctrl+R: Reset Camera');
