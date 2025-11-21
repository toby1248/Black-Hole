/**
 * TDE-SPH Three.js Visualizer
 *
 * WebGL-based 3D particle visualization using Three.js.
 * Renders millions of SPH particles with GPU acceleration.
 *
 * Features:
 * - Point cloud rendering with custom vertex colors
 * - Multiple colormaps (viridis, plasma, inferno, etc.)
 * - Logarithmic and linear color scaling
 * - Orbit controls for camera manipulation
 * - Black hole visualization at origin
 * - Axes helper
 * - Efficient buffer geometry updates
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-11-18
 */

class Visualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);

        // Check for WebGL support
        if (!this.checkWebGLSupport()) {
            this.showWebGLError();
            return;
        }

        // Scene components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Visualization objects
        this.particleSystem = null;
        this.blackHole = null;
        this.axesHelper = null;

        // Data
        this.positions = null;  // Float32Array
        this.colorField = null; // Float32Array (density, temperature, etc.)
        this.particleCount = 0;

        // Settings
        this.pointSize = 2.0;
        this.colorBy = 'density';
        this.colormap = 'viridis';
        this.logScale = true;
        this.showBlackHole = true;
        this.showAxes = true;

        // Performance
        this.lastTime = 0;
        this.frameCount = 0;
        this.fps = 0;

        // Initialize
        this.init();
    }

    checkWebGLSupport() {
        /**
         * Check if WebGL is available in the browser.
         * Returns: boolean - true if WebGL is supported
         */
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            return !!(gl && gl instanceof WebGLRenderingContext);
        } catch (e) {
            return false;
        }
    }

    showWebGLError() {
        /**
         * Display WebGL not supported message.
         */
        const container = this.canvas.parentElement;
        const errorDiv = document.createElement('div');
        errorDiv.className = 'webgl-error';
        errorDiv.innerHTML = `
            <h2>WebGL Not Supported</h2>
            <p>Your browser or device does not support WebGL, which is required for 3D visualization.</p>
            <p>Please try:</p>
            <ul>
                <li>Updating your browser to the latest version</li>
                <li>Enabling hardware acceleration in browser settings</li>
                <li>Using a different browser (Chrome, Firefox, Edge recommended)</li>
            </ul>
        `;
        errorDiv.style.cssText = `
            padding: 40px;
            text-align: center;
            background: #2a2a2a;
            color: #fff;
            border-radius: 8px;
            max-width: 500px;
            margin: 50px auto;
        `;
        container.appendChild(errorDiv);
        this.canvas.style.display = 'none';
        console.error('WebGL is not supported in this browser');
    }

    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        // Create camera
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        this.camera.position.set(20, 20, 20);
        this.camera.lookAt(0, 0, 0);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enablePan = true;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 500;

        // Lighting (for black hole sphere)
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);

        // Black hole at origin
        this.createBlackHole();

        // Axes helper
        this.axesHelper = new THREE.AxesHelper(10);
        this.scene.add(this.axesHelper);

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start animation loop
        this.animate();
    }

    createBlackHole() {
        const geometry = new THREE.SphereGeometry(1.0, 32, 32);
        const material = new THREE.MeshStandardMaterial({
            color: 0x000000,
            emissive: 0x330033,
            roughness: 0.2,
            metalness: 0.8
        });

        this.blackHole = new THREE.Mesh(geometry, material);
        this.scene.add(this.blackHole);
    }

    loadParticles(positions, colorField, colorBy = 'density') {
        /**
         * Load particle data and create point cloud.
         *
         * Parameters:
         *   positions: Float32Array of shape (N, 3) - particle positions
         *   colorField: Float32Array of length N - quantity to color by
         *   colorBy: string - name of the field (for display)
         */

        // Remove old particle system
        if (this.particleSystem) {
            this.scene.remove(this.particleSystem);
            this.particleSystem.geometry.dispose();
            this.particleSystem.material.dispose();
        }

        // Store data
        this.positions = positions;
        this.colorField = colorField;
        this.particleCount = positions.length / 3;
        this.colorBy = colorBy;

        // Create geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Compute colors
        const colors = this.computeColors(colorField);
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create material
        const material = new THREE.PointsMaterial({
            size: this.pointSize,
            vertexColors: true,
            transparent: false,
            sizeAttenuation: true
        });

        // Create point cloud
        this.particleSystem = new THREE.Points(geometry, material);
        this.scene.add(this.particleSystem);

        // Auto-center camera on data
        this.fitCameraToParticles();
    }

    updateParticles(positions, colorField) {
        /**
         * Update existing particles with new data (efficient).
         */
        if (!this.particleSystem) {
            console.warn('No particle system loaded');
            return;
        }

        this.positions = positions;
        this.colorField = colorField;

        // Update positions
        this.particleSystem.geometry.attributes.position.array = positions;
        this.particleSystem.geometry.attributes.position.needsUpdate = true;

        // Update colors
        const colors = this.computeColors(colorField);
        this.particleSystem.geometry.attributes.color.array = colors;
        this.particleSystem.geometry.attributes.color.needsUpdate = true;
    }

    computeColors(field) {
        /**
         * Compute RGB colors from scalar field using current colormap.
         *
         * Returns: Float32Array of shape (N, 3) with RGB values [0, 1]
         */
        const N = field.length;
        const colors = new Float32Array(N * 3);

        // Compute field range
        let minVal = field[0];
        let maxVal = field[0];

        for (let i = 0; i < N; i++) {
            const val = field[i];
            if (!isNaN(val) && isFinite(val)) {
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
        }

        // Apply logarithmic scaling if enabled
        if (this.logScale && minVal > 0) {
            minVal = Math.log10(minVal);
            maxVal = Math.log10(maxVal);
        }

        // Normalize and apply colormap
        for (let i = 0; i < N; i++) {
            let val = field[i];

            // Handle invalid values
            if (isNaN(val) || !isFinite(val)) {
                colors[i*3] = 0.5;
                colors[i*3+1] = 0.5;
                colors[i*3+2] = 0.5;
                continue;
            }

            // Apply log scale
            if (this.logScale && val > 0) {
                val = Math.log10(val);
            }

            // Normalize to [0, 1]
            const normalized = (val - minVal) / (maxVal - minVal);
            const clamped = Math.max(0, Math.min(1, normalized));

            // Apply colormap
            const rgb = this.applyColormap(clamped, this.colormap);
            colors[i*3] = rgb[0];
            colors[i*3+1] = rgb[1];
            colors[i*3+2] = rgb[2];
        }

        return colors;
    }

    applyColormap(value, colormapName) {
        /**
         * Apply colormap to normalized value [0, 1].
         * Returns [r, g, b] in range [0, 1].
         *
         * Colormaps: viridis, plasma, inferno, hot, cool, rainbow
         */
        const t = value;

        switch(colormapName) {
            case 'viridis':
                // Analytical approximation of viridis
                return [
                    0.267 * (1-t) + 0.993 * t,
                    0.005 * (1-t) + 0.906 * t,
                    0.329 * (1-t) + 0.144 * t
                ];

            case 'plasma':
                // Analytical approximation of plasma
                return [
                    0.050 + 0.900 * Math.pow(t, 0.5),
                    0.030 + 0.700 * Math.pow(t, 1.5),
                    0.550 - 0.500 * t
                ];

            case 'inferno':
                // Analytical approximation of inferno
                return [
                    0.000 + 1.000 * Math.pow(t, 0.8),
                    0.000 + 0.700 * Math.pow(t, 2.0),
                    0.000 + 0.400 * Math.pow(t, 3.0)
                ];

            case 'hot':
                // Hot (black → red → yellow → white)
                if (t < 0.33) {
                    return [3.0 * t, 0, 0];
                } else if (t < 0.66) {
                    return [1.0, 3.0 * (t - 0.33), 0];
                } else {
                    return [1.0, 1.0, 3.0 * (t - 0.66)];
                }

            case 'cool':
                // Cool (cyan → blue → magenta)
                return [
                    t,
                    1.0 - t,
                    1.0
                ];

            case 'rainbow':
                // Rainbow (HSV with varying hue)
                const hue = (1.0 - t) * 240 / 360;  // Blue to red
                return this.hsvToRgb(hue, 1.0, 1.0);

            default:
                // Fallback: grayscale
                return [t, t, t];
        }
    }

    hsvToRgb(h, s, v) {
        /**
         * Convert HSV to RGB.
         * h, s, v in [0, 1]
         * Returns [r, g, b] in [0, 1]
         */
        const i = Math.floor(h * 6);
        const f = h * 6 - i;
        const p = v * (1 - s);
        const q = v * (1 - f * s);
        const t_val = v * (1 - (1 - f) * s);

        switch(i % 6) {
            case 0: return [v, t_val, p];
            case 1: return [q, v, p];
            case 2: return [p, v, t_val];
            case 3: return [p, q, v];
            case 4: return [t_val, p, v];
            case 5: return [v, p, q];
            default: return [0, 0, 0];
        }
    }

    fitCameraToParticles() {
        /**
         * Automatically position camera to view all particles.
         */
        if (!this.positions || this.positions.length === 0) return;

        // Compute bounding box
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = 0; i < this.positions.length; i += 3) {
            const x = this.positions[i];
            const y = this.positions[i+1];
            const z = this.positions[i+2];

            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }

        // Compute center and size
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;

        const sizeX = maxX - minX;
        const sizeY = maxY - minY;
        const sizeZ = maxZ - minZ;
        const maxSize = Math.max(sizeX, sizeY, sizeZ);

        // Position camera
        const distance = maxSize * 1.5;
        this.camera.position.set(
            centerX + distance * 0.7,
            centerY + distance * 0.7,
            centerZ + distance * 0.7
        );
        this.camera.lookAt(centerX, centerY, centerZ);
        this.controls.target.set(centerX, centerY, centerZ);
    }

    setPointSize(size) {
        this.pointSize = size;
        if (this.particleSystem) {
            this.particleSystem.material.size = size;
        }
    }

    setColormap(colormapName) {
        this.colormap = colormapName;
        if (this.colorField) {
            // Recompute colors
            const colors = this.computeColors(this.colorField);
            if (this.particleSystem) {
                this.particleSystem.geometry.attributes.color.array = colors;
                this.particleSystem.geometry.attributes.color.needsUpdate = true;
            }
        }
    }

    setLogScale(enabled) {
        this.logScale = enabled;
        if (this.colorField) {
            // Recompute colors
            const colors = this.computeColors(this.colorField);
            if (this.particleSystem) {
                this.particleSystem.geometry.attributes.color.array = colors;
                this.particleSystem.geometry.attributes.color.needsUpdate = true;
            }
        }
    }

    setShowBlackHole(show) {
        this.showBlackHole = show;
        if (this.blackHole) {
            this.blackHole.visible = show;
        }
    }

    setShowAxes(show) {
        this.showAxes = show;
        if (this.axesHelper) {
            this.axesHelper.visible = show;
        }
    }

    resetCamera() {
        this.camera.position.set(20, 20, 20);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    setTopView() {
        if (this.positions && this.positions.length > 0) {
            // Compute center
            let centerX = 0, centerY = 0, centerZ = 0;
            const N = this.positions.length / 3;
            for (let i = 0; i < this.positions.length; i += 3) {
                centerX += this.positions[i] / N;
                centerY += this.positions[i+1] / N;
                centerZ += this.positions[i+2] / N;
            }

            const distance = 30;
            this.camera.position.set(centerX, centerY, centerZ + distance);
            this.camera.lookAt(centerX, centerY, centerZ);
            this.controls.target.set(centerX, centerY, centerZ);
            this.controls.update();
        }
    }

    setSideView() {
        if (this.positions && this.positions.length > 0) {
            let centerX = 0, centerY = 0, centerZ = 0;
            const N = this.positions.length / 3;
            for (let i = 0; i < this.positions.length; i += 3) {
                centerX += this.positions[i] / N;
                centerY += this.positions[i+1] / N;
                centerZ += this.positions[i+2] / N;
            }

            const distance = 30;
            this.camera.position.set(centerX + distance, centerY, centerZ);
            this.camera.lookAt(centerX, centerY, centerZ);
            this.controls.target.set(centerX, centerY, centerZ);
            this.controls.update();
        }
    }

    screenshot() {
        /**
         * Capture current view as PNG image.
         */
        this.renderer.render(this.scene, this.camera);
        const dataURL = this.renderer.domElement.toDataURL('image/png');

        // Trigger download
        const link = document.createElement('a');
        link.download = `tde_sph_screenshot_${Date.now()}.png`;
        link.href = dataURL;
        link.click();
    }

    onWindowResize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Update controls
        this.controls.update();

        // Render scene
        this.renderer.render(this.scene, this.camera);

        // Calculate FPS
        const currentTime = performance.now();
        this.frameCount++;

        if (currentTime - this.lastTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = currentTime;

            // Update FPS display
            const fpsElement = document.getElementById('fps-counter-value');
            if (fpsElement) {
                fpsElement.textContent = this.fps;
            }
        }
    }
}
