/**
 * TDE-SPH Three.js Visualizer
 *
 * WebGL-based 3D particle visualization using Three.js.
 * Supports configurable point shapes, transparency, and density-based sizing.
 */

class Visualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);

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
        this.positions = null;      // Float32Array
        this.colorField = null;     // Float32Array (density, temperature, etc.)
        this.densityField = null;   // Float32Array (used for size scaling)
        this.particleCount = 0;

        // Settings
        this.pointSize = 2.0;
        this.pointShape = 'square'; // square | circle | smooth
        this.pointOpacity = 1.0;
        this.sizeByDensity = false;
        this.densityScale = 1.0;
        this.colorBy = 'density';
        this.colormap = 'viridis';
        this.logScale = true;
        this.showBlackHole = true;
        this.showAxes = true;

        // Performance
        this.lastTime = 0;
        this.frameCount = 0;
        this.fps = 0;

        // Animation state
        this.isActive = true;
        this.animationId = null;

        this.init();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        // Camera
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        this.camera.position.set(20, 20, 20);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enablePan = true;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 500;

        // Lights
        this.scene.add(new THREE.AmbientLight(0x404040, 0.5));
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);

        // Objects
        this.createBlackHole();
        this.axesHelper = new THREE.AxesHelper(10);
        this.scene.add(this.axesHelper);

        // Resize handling
        window.addEventListener('resize', () => this.onWindowResize());

        // Start loop
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

    loadParticles(positions, colorField, densityField, colorBy = 'density') {
        // Dispose old system
        if (this.particleSystem) {
            this.scene.remove(this.particleSystem);
            this.particleSystem.geometry.dispose();
            this.particleSystem.material.dispose();
        }

        // Store data
        this.positions = positions;
        this.colorField = colorField;
        this.densityField = densityField;
        this.particleCount = positions.length / 3;
        this.colorBy = colorBy;

        // Geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const colors = this.computeColors(colorField);
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const sizeAttribute = this.buildSizeAttribute(densityField);
        geometry.setAttribute('size', new THREE.BufferAttribute(sizeAttribute, 1));

        // Material + points
        const material = this.buildParticleMaterial();
        this.particleSystem = new THREE.Points(geometry, material);
        this.scene.add(this.particleSystem);

        // Center view
        this.fitCameraToParticles();
    }

    updateParticles(positions, colorField, densityField) {
        if (!this.particleSystem) {
            console.warn('No particle system loaded');
            return;
        }

        this.positions = positions;
        this.colorField = colorField;
        this.densityField = densityField;
        this.particleCount = positions.length / 3;

        // Positions
        this.particleSystem.geometry.attributes.position.array = positions;
        this.particleSystem.geometry.attributes.position.needsUpdate = true;

        // Colors
        const colors = this.computeColors(colorField);
        this.particleSystem.geometry.attributes.color.array = colors;
        this.particleSystem.geometry.attributes.color.needsUpdate = true;

        // Sizes
        const sizeAttr = this.buildSizeAttribute(densityField);
        this.particleSystem.geometry.attributes.size.array = sizeAttr;
        this.particleSystem.geometry.attributes.size.needsUpdate = true;

        this.updateMaterialUniforms();
    }

    computeColors(field) {
        const N = field.length;
        const colors = new Float32Array(N * 3);

        let minVal = field[0];
        let maxVal = field[0];
        for (let i = 0; i < N; i++) {
            const val = field[i];
            if (!isNaN(val) && isFinite(val)) {
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
        }

        if (this.logScale && minVal > 0) {
            minVal = Math.log10(minVal);
            maxVal = Math.log10(maxVal);
        }
        const range = (maxVal - minVal) !== 0 ? (maxVal - minVal) : 1e-6;

        for (let i = 0; i < N; i++) {
            let val = field[i];
            if (isNaN(val) || !isFinite(val)) {
                colors[i * 3] = 0.5;
                colors[i * 3 + 1] = 0.5;
                colors[i * 3 + 2] = 0.5;
                continue;
            }

            if (this.logScale && val > 0) {
                val = Math.log10(val);
            }

            const normalized = (val - minVal) / range;
            const clamped = Math.max(0, Math.min(1, normalized));
            const rgb = this.applyColormap(clamped, this.colormap);
            colors[i * 3] = rgb[0];
            colors[i * 3 + 1] = rgb[1];
            colors[i * 3 + 2] = rgb[2];
        }

        return colors;
    }

    applyColormap(value, colormapName) {
        const t = value;
        switch (colormapName) {
            case 'viridis':
                return [
                    0.267 * (1 - t) + 0.993 * t,
                    0.004 * (1 - t) + 0.906 * t,
                    0.329 * (1 - t) + 0.143 * t
                ];
            case 'plasma':
                return [
                    0.050 * (1 - t) + 0.940 * t,
                    0.030 * (1 - t) + 0.510 * t,
                    0.527 * (1 - t) + 0.150 * t
                ];
            case 'inferno':
                return [
                    0.100 * (1 - t) + 0.988 * t,
                    0.041 * (1 - t) + 0.645 * t,
                    0.200 * (1 - t) + 0.010 * t
                ];
            case 'hot':
                return [
                    Math.min(1, t * 3),
                    Math.min(1, Math.max(0, (t - 1 / 3) * 3)),
                    Math.min(1, Math.max(0, (t - 2 / 3) * 3))
                ];
            case 'cool':
                return [t, 1 - t, 1];
            case 'rainbow':
                return this.hsvToRgb(0.7 * (1 - t), 1, 1);
            default:
                return [t, t, t];
        }
    }

    hsvToRgb(h, s, v) {
        const i = Math.floor(h * 6);
        const f = h * 6 - i;
        const p = v * (1 - s);
        const q = v * (1 - f * s);
        const t = v * (1 - (1 - f) * s);
        switch (i % 6) {
            case 0: return [v, t, p];
            case 1: return [q, v, p];
            case 2: return [p, v, t];
            case 3: return [p, q, v];
            case 4: return [t, p, v];
            case 5: return [v, p, q];
            default: return [0, 0, 0];
        }
    }

    buildSizeAttribute(densityField) {
        const sizeAttr = new Float32Array(this.particleCount);
        if (!densityField || densityField.length === 0) {
            sizeAttr.fill(1.0);
            return sizeAttr;
        }

        // Compute median density
        const n = Math.min(densityField.length, this.particleCount);
        const copy = Array.from(densityField.slice(0, n)).map(v => Math.max(v, 1e-20)).sort((a, b) => a - b);
        const mid = Math.floor(copy.length / 2);
        const median = copy.length % 2 === 0 ? 0.5 * (copy[mid - 1] + copy[mid]) : copy[mid];
        const medianSafe = Math.max(median, 1e-20);

        for (let i = 0; i < n; i++) {
            const rho = Math.max(densityField[i], 1e-20);
            if (rho < medianSafe) {
                let scale = Math.pow(medianSafe / rho, 1 / 3); // inverse density
                scale = Math.min(scale, 5.0);
                sizeAttr[i] = scale;
            } else {
                sizeAttr[i] = 1.0;
            }
        }
        for (let i = n; i < this.particleCount; i++) {
            sizeAttr[i] = 1.0;
        }
        return sizeAttr;
    }

    buildParticleMaterial() {
        const shapeMode = this.pointShape === 'circle' ? 1 : (this.pointShape === 'smooth' ? 2 : 0);
        const uniforms = {
            uSize: { value: this.pointSize },
            uOpacity: { value: this.pointOpacity },
            uShapeMode: { value: shapeMode },
            uUseDensity: { value: this.sizeByDensity },
            uDensityScale: { value: this.densityScale },
            uPixelRatio: { value: this.renderer ? this.renderer.getPixelRatio() : 1.0 },
        };

        return new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexColors: true,
            transparent: true,
            depthWrite: false,
            blending: THREE.NormalBlending,
            vertexShader: `
                uniform float uSize;
                uniform bool uUseDensity;
                uniform float uDensityScale;
                uniform float uPixelRatio;
                attribute float size;
                varying vec3 vColor;
                varying float vSizeScale;
                void main() {
                    vColor = color;
                    vSizeScale = uUseDensity ? size * uDensityScale : 1.0;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    float pointSize = uSize * vSizeScale * uPixelRatio;
                    pointSize *= 300.0 / -mvPosition.z;
                    gl_PointSize = pointSize;
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform float uOpacity;
                uniform int uShapeMode;
                varying vec3 vColor;
                varying float vSizeScale;
                void main() {
                    vec2 c = gl_PointCoord - vec2(0.5);
                    float dist = length(c);
                    float alpha = uOpacity;

                    if (uShapeMode == 1 && dist > 0.5) {
                        discard;
                    }
                    if (uShapeMode == 2) {
                        float edge = smoothstep(0.01, 0.6, dist);
                        alpha *= (1.0 - edge);
                        if (alpha <= 0.001) discard;
                    }

                    if (vSizeScale > 1.0) {
                        alpha /= vSizeScale; // larger (low-density) particles become more transparent
                    }

                    gl_FragColor = vec4(vColor, alpha);
                }
            `
        });
    }

    fitCameraToParticles() {
        if (!this.positions || this.positions.length === 0) return;

        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = 0; i < this.positions.length; i += 3) {
            const x = this.positions[i];
            const y = this.positions[i + 1];
            const z = this.positions[i + 2];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        const maxSize = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

        const distance = Math.max(5, maxSize * 1.5);
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
        this.updateMaterialUniforms();
    }

    setColormap(colormapName) {
        this.colormap = colormapName;
        if (this.colorField && this.particleSystem) {
            const colors = this.computeColors(this.colorField);
            this.particleSystem.geometry.attributes.color.array = colors;
            this.particleSystem.geometry.attributes.color.needsUpdate = true;
        }
    }

    setLogScale(enabled) {
        this.logScale = enabled;
        if (this.colorField && this.particleSystem) {
            const colors = this.computeColors(this.colorField);
            this.particleSystem.geometry.attributes.color.array = colors;
            this.particleSystem.geometry.attributes.color.needsUpdate = true;
        }
    }

    setShowBlackHole(show) {
        this.showBlackHole = show;
        if (this.blackHole) this.blackHole.visible = show;
    }

    setShowAxes(show) {
        this.showAxes = show;
        if (this.axesHelper) this.axesHelper.visible = show;
    }

    setPointShape(shape) {
        this.pointShape = shape;
        this.rebuildMaterial();
    }

    setPointOpacity(opacity) {
        this.pointOpacity = opacity;
        this.updateMaterialUniforms();
    }

    setSizeByDensity(enabled) {
        this.sizeByDensity = enabled;
        this.updateMaterialUniforms();
    }

    setDensityScale(scale) {
        this.densityScale = scale;
        this.updateMaterialUniforms();
    }

    rebuildMaterial() {
        if (!this.particleSystem) return;
        const geometry = this.particleSystem.geometry;
        this.scene.remove(this.particleSystem);
        this.particleSystem.material.dispose();
        const material = this.buildParticleMaterial();
        this.particleSystem = new THREE.Points(geometry, material);
        this.scene.add(this.particleSystem);
    }

    updateMaterialUniforms() {
        if (!this.particleSystem || !this.particleSystem.material.uniforms) return;
        const uniforms = this.particleSystem.material.uniforms;
        uniforms.uSize.value = this.pointSize;
        uniforms.uOpacity.value = this.pointOpacity;
        uniforms.uShapeMode.value = this.pointShape === 'circle' ? 1 : (this.pointShape === 'smooth' ? 2 : 0);
        uniforms.uUseDensity.value = this.sizeByDensity;
        uniforms.uDensityScale.value = this.densityScale;
        uniforms.uPixelRatio.value = this.renderer ? this.renderer.getPixelRatio() : 1.0;
    }

    resetCamera() {
        this.camera.position.set(20, 20, 20);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    setTopView() {
        if (!this.positions || this.positions.length === 0) return;
        let cx = 0, cy = 0, cz = 0;
        const N = this.positions.length / 3;
        for (let i = 0; i < this.positions.length; i += 3) {
            cx += this.positions[i] / N;
            cy += this.positions[i + 1] / N;
            cz += this.positions[i + 2] / N;
        }
        const distance = 30;
        this.camera.position.set(cx, cy, cz + distance);
        this.camera.lookAt(cx, cy, cz);
        this.controls.target.set(cx, cy, cz);
        this.controls.update();
    }

    setSideView() {
        if (!this.positions || this.positions.length === 0) return;
        let cx = 0, cy = 0, cz = 0;
        const N = this.positions.length / 3;
        for (let i = 0; i < this.positions.length; i += 3) {
            cx += this.positions[i] / N;
            cy += this.positions[i + 1] / N;
            cz += this.positions[i + 2] / N;
        }
        const distance = 30;
        this.camera.position.set(cx + distance, cy, cz);
        this.camera.lookAt(cx, cy, cz);
        this.controls.target.set(cx, cy, cz);
        this.controls.update();
    }

    screenshot() {
        this.renderer.render(this.scene, this.camera);
        const dataURL = this.renderer.domElement.toDataURL('image/png');
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
        this.updateMaterialUniforms();
    }

    animate() {
        if (!this.isActive) return;
        
        this.animationId = requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);

        const currentTime = performance.now();
        this.frameCount++;
        if (currentTime - this.lastTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = currentTime;
            const fpsElement = document.getElementById('fps-counter-value');
            if (fpsElement) {
                fpsElement.textContent = this.fps;
            }
        }
    }
    
    pauseAnimation() {
        this.isActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    resumeAnimation() {
        if (!this.isActive) {
            this.isActive = true;
            this.animate();
        }
    }
    
    /**
     * Dispose all resources and free WebGL context.
     */
    dispose() {
        this.pauseAnimation();
        
        if (this.particleSystem) {
            this.scene.remove(this.particleSystem);
            this.particleSystem.geometry.dispose();
            this.particleSystem.material.dispose();
        }
        
        if (this.axesHelper) {
            this.scene.remove(this.axesHelper);
        }
        
        if (this.blackHole) {
            this.scene.remove(this.blackHole);
            this.blackHole.geometry.dispose();
            this.blackHole.material.dispose();
        }
        
        this.renderer.dispose();
        
        // Remove event listener (need to store reference for proper removal)
        // Note: This may not fully remove if anonymous function was used
    }
}
