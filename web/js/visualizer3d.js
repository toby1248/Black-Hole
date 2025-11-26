/**
 * TDE-SPH 3D Instanced Mesh Visualizer
 *
 * High-performance WebGL-based 3D particle visualization using Three.js InstancedMesh.
 * Optimized for 1M+ particles with:
 * - InstancedMesh for efficient GPU instancing
 * - Emissive bloom post-processing
 * - Blackbody colormap for realistic thermal visualization
 * - Density-based sizing with 10th percentile baseline
 * - Skybox background
 * - WASDQE camera controls
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-01-25
 */

class Visualizer3D {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);

        // Scene components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Post-processing
        this.composer = null;
        this.bloomPass = null;
        this.renderPass = null;
        this.bloomEnabled = true;
        this.bloomStrength = 1.0;
        this.bloomRadius = 0.1;
        this.bloomThreshold = 0.0;

        // Visualization objects
        this.instancedMesh = null;
        this.blackHole = null;
        this.axesHelper = null;
        this.skybox = null;

        // Data
        this.positions = null;
        this.colorField = null;
        this.densityField = null;
        this.particleCount = 0;
        
        // Density baseline for sizing (90th percentile clamp threshold)
        this.density90thPercentile = 1e-10;

        // Settings
        this.pointSize = 0.01;  // Default: 0.01 (slider min is 0.001)
        this.maxSizeMultiplier = 5.0;
        this.pointOpacity = 1.0;
        this.opacityScale = 1.0;  // Logarithmic opacity slider value
        this.sizeByDensity = true;
        this.colorBy = 'density';
        this.colormap = 'blackbody';
        this.logScale = true;
        this.showBlackHole = true;
        this.showAxes = true;
        this.showSkybox = true;
        
        // BH sizing from metadata
        this.bhRadius = 1.0;

        // Camera movement
        this.moveSpeed = 0.2;
        this.keysPressed = {};

        // Performance
        this.lastTime = 0;
        this.frameCount = 0;
        this.fps = 0;

        // Animation state
        this.isActive = true;
        this.animationId = null;

        // Dummy matrix for instance updates
        this.dummyMatrix = new THREE.Matrix4();
        this.dummyColor = new THREE.Color();
        this.dummyPosition = new THREE.Vector3();
        this.dummyScale = new THREE.Vector3();
        this.dummyQuaternion = new THREE.Quaternion();
        
        // THREE.js Lut for colormaps (initialized in init())
        this.lut = null;

        this.init();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Initialize the Lut colormap with high resolution
        this.initLut();

        // Ensure canvas has valid dimensions
        const width = Math.max(1, this.canvas.clientWidth || 800);
        const height = Math.max(1, this.canvas.clientHeight || 600);
        console.log(`Initializing visualizer3d at ${width}x${height}`);

        // Camera
        const aspect = width / height;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.01, 10000);
        this.camera.position.set(20, 20, 20);
        this.camera.lookAt(0, 0, 0);
        // Enable camera to see all layers (0 and 1) for selective bloom
        this.camera.layers.enableAll();

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap at 2x for performance
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        // Controls with reduced pan sensitivity
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enablePan = true;
        this.controls.panSpeed = 0.25;  // Reduced from default 1.0
        this.controls.rotateSpeed = 0.5;
        this.controls.minDistance = 0.1;
        this.controls.maxDistance = 5000;

        // Ambient light only (no specular)
        this.scene.add(new THREE.AmbientLight(0xffffff, 1.0));

        // Objects
        this.createBlackHole();
        this.axesHelper = new THREE.AxesHelper(10);
        this.scene.add(this.axesHelper);

        // Setup post-processing (bloom)
        this.setupPostProcessing();
        
        // Setup skybox (will load asynchronously)
        this.loadSkybox();

        // Event handling
        window.addEventListener('resize', () => this.onWindowResize());
        window.addEventListener('keydown', (e) => this.onKeyDown(e));
        window.addEventListener('keyup', (e) => this.onKeyUp(e));

        // Start loop
        this.animate();
    }

    setupPostProcessing() {
        // Check if EffectComposer is available
        if (typeof THREE.EffectComposer === 'undefined') {
            console.warn('EffectComposer not available, bloom disabled');
            this.bloomEnabled = false;
            return;
        }
        
        // Ensure canvas has valid dimensions before creating render targets
        const width = Math.max(1, this.canvas.clientWidth);
        const height = Math.max(1, this.canvas.clientHeight);
        
        if (width <= 1 || height <= 1) {
            console.warn('Canvas has zero size, deferring post-processing setup');
            this.bloomEnabled = false;
            // Retry after a short delay
            setTimeout(() => this.setupPostProcessing(), 100);
            return;
        }
        
        console.log(`Setting up selective bloom post-processing at ${width}x${height}`);

        // LAYER SETUP for selective bloom:
        // Layer 0 = default (scene objects without bloom - skybox, black hole)
        // Layer 1 = bloom objects only (particles)
        this.BLOOM_LAYER = 1;
        this.bloomLayer = new THREE.Layers();
        this.bloomLayer.set(this.BLOOM_LAYER);
        
        // Materials to darken for bloom pass (non-bloom objects)
        this.darkMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
        this.materials = {};

        // Create render targets
        const renderTargetParams = {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat,
            colorSpace: THREE.SRGBColorSpace
        };
        
        // Main composer for bloom layer
        const bloomRenderTarget = new THREE.WebGLRenderTarget(width, height, renderTargetParams);
        this.bloomComposer = new THREE.EffectComposer(this.renderer, bloomRenderTarget);
        this.bloomComposer.renderToScreen = false;
        
        const renderScene = new THREE.RenderPass(this.scene, this.camera);
        this.bloomComposer.addPass(renderScene);
        
        // Bloom pass (only applied to bloom layer)
        if (typeof THREE.UnrealBloomPass !== 'undefined') {
            const resolution = new THREE.Vector2(width, height);
            this.bloomPass = new THREE.UnrealBloomPass(
                resolution,
                this.bloomStrength * 1.5,  // Increase bloom for more glow
                this.bloomRadius * 1.2,
                this.bloomThreshold * 0.5   // Lower threshold for more bloom
            );
            this.bloomComposer.addPass(this.bloomPass);
        }
        
        // Final composer that combines scene + bloomed particles
        const finalRenderTarget = new THREE.WebGLRenderTarget(width, height, renderTargetParams);
        this.finalComposer = new THREE.EffectComposer(this.renderer, finalRenderTarget);
        this.finalComposer.setSize(width, height);
        
        const finalRenderPass = new THREE.RenderPass(this.scene, this.camera);
        this.finalComposer.addPass(finalRenderPass);
        
        // Shader to blend the bloom texture with the final render
        if (typeof THREE.ShaderPass !== 'undefined') {
            const bloomBlendShader = {
                uniforms: {
                    baseTexture: { value: null },
                    bloomTexture: { value: null }  // Set after pass creation to avoid clone warning
                },
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler2D baseTexture;
                    uniform sampler2D bloomTexture;
                    varying vec2 vUv;
                    void main() {
                        vec4 base = texture2D(baseTexture, vUv);
                        vec4 bloom = texture2D(bloomTexture, vUv);
                        // Additive blending - bloom adds to base
                        gl_FragColor = base + bloom;
                    }
                `
            };
            
            this.bloomBlendPass = new THREE.ShaderPass(bloomBlendShader, 'baseTexture');
            this.bloomBlendPass.needsSwap = true;
            // Set bloom texture reference AFTER pass creation to avoid cloning the render target texture
            this.bloomBlendPass.uniforms.bloomTexture.value = this.bloomComposer.renderTarget2.texture;
            this.finalComposer.addPass(this.bloomBlendPass);
        }
        
        this.bloomEnabled = true;
        console.log('Selective bloom configured - particles only');
    }
    
    // Helper: darken non-bloom materials for bloom pass
    darkenNonBloomObjects() {
        this.scene.traverse((obj) => {
            if (obj.isMesh && !this.bloomLayer.test(obj.layers)) {
                this.materials[obj.uuid] = obj.material;
                obj.material = this.darkMaterial;
            }
        });
    }
    
    // Helper: restore original materials after bloom pass
    restoreMaterials() {
        this.scene.traverse((obj) => {
            if (this.materials[obj.uuid]) {
                obj.material = this.materials[obj.uuid];
                delete this.materials[obj.uuid];
            }
        });
    }
    
    loadSkybox() {
        // Check if we're running from file:// protocol (CORS won't work)
        if (window.location.protocol === 'file:') {
            console.log('Running from file:// protocol - using procedural skybox (no CORS)');
            this.createProceduralSkybox();
            return;
        }
        
        // Try to load skybox from assets - try multiple formats and paths
        const basePaths = [
            '../assets/skybox/',
            './assets/skybox/',
            'assets/skybox/'
        ];
        
        // Only try image-based skybox loading when running on http(s)
        const tryLoadImageSkybox = (pathIndex) => {
            if (pathIndex >= basePaths.length) {
                console.log('Could not load skybox textures, using procedural skybox');
                this.createProceduralSkybox();
                return;
            }
            
            const basePath = basePaths[pathIndex];
            const loader = new THREE.CubeTextureLoader();
            loader.setPath(basePath);
            
            // Try PNG first
            loader.load(
                ['px.png', 'nx.png', 'py.png', 'ny.png', 'pz.png', 'nz.png'],
                (texture) => {
                    texture.colorSpace = THREE.SRGBColorSpace;
                    this.skybox = texture;
                    this.scene.background = texture;
                    console.log('Skybox loaded (PNG) from', basePath);
                },
                undefined,
                () => {
                    // Try JPG
                    loader.load(
                        ['px.jpg', 'nx.jpg', 'py.jpg', 'ny.jpg', 'pz.jpg', 'nz.jpg'],
                        (texture) => {
                            texture.colorSpace = THREE.SRGBColorSpace;
                            this.skybox = texture;
                            this.scene.background = texture;
                            console.log('Skybox loaded (JPG) from', basePath);
                        },
                        undefined,
                        () => {
                            tryLoadImageSkybox(pathIndex + 1);
                        }
                    );
                }
            );
        };
        
        // Start trying to load image-based skybox
        tryLoadImageSkybox(0);
    }
    
    /**
     * Create a procedural starfield skybox when no texture is available.
     */
    createProceduralSkybox() {
        // Create a gradient sphere for background
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Create gradient
        const gradient = ctx.createRadialGradient(256, 256, 0, 256, 256, 256);
        gradient.addColorStop(0, '#1a0a2e');  // Dark purple center
        gradient.addColorStop(0.5, '#0d0f1a'); // Dark blue
        gradient.addColorStop(1, '#000000');   // Black edge
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 512, 512);
        
        // Add stars
        ctx.fillStyle = '#ffffff';
        for (let i = 0; i < 500; i++) {
            const x = Math.random() * 512;
            const y = Math.random() * 512;
            const size = Math.random() * 1.5 + 0.5;
            const alpha = Math.random() * 0.8 + 0.2;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Create texture from canvas
        const texture = new THREE.CanvasTexture(canvas);
        texture.colorSpace = THREE.SRGBColorSpace;
        
        // Create sphere geometry for skybox
        const geometry = new THREE.SphereGeometry(1000, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            side: THREE.BackSide,
            depthWrite: false
        });
        
        const skyboxMesh = new THREE.Mesh(geometry, material);
        this.scene.add(skyboxMesh);
        
        // Store reference for toggling
        this.skyboxMesh = skyboxMesh;
        console.log('Procedural skybox created');
    }

    createBlackHole(radius = null) {
        if (this.blackHole) {
            this.scene.remove(this.blackHole);
            this.blackHole.geometry.dispose();
            this.blackHole.material.dispose();
        }
        
        const r = radius || this.bhRadius;
        const geometry = new THREE.SphereGeometry(r, 64, 64);
        const material = new THREE.MeshBasicMaterial({
            color: 0x000000,
            side: THREE.FrontSide
        });
        
        this.blackHole = new THREE.Mesh(geometry, material);
        this.blackHole.visible = this.showBlackHole;
        this.scene.add(this.blackHole);
        
        // Add accretion disc glow ring
        //const ringGeometry = new THREE.RingGeometry(r * 1.01, r * 1.5, 64);
        //const ringMaterial = new THREE.MeshBasicMaterial({
        //    color: 0xff4400,
        //    transparent: true,
        //    opacity: 0.3,
        //    side: THREE.DoubleSide
        //});
        //const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        //this.blackHole.add(ring);
    }
    
    setBHRadius(radius) {
        this.bhRadius = radius;
        this.createBlackHole(radius);
    }

    createParticleSystem(count) {
        // Dispose old particle system
        if (this.particleSystem) {
            this.scene.remove(this.particleSystem);
            this.particleSystem.geometry.dispose();
            this.particleSystem.material.dispose();
        }
        
        // Also dispose old instanced mesh if it exists
        if (this.instancedMesh) {
            this.scene.remove(this.instancedMesh);
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
            this.instancedMesh = null;
        }

        // Create geometry with positions, colors, sizes, and per-particle opacity
        const geometry = new THREE.BufferGeometry();
        
        // Placeholder attributes (will be updated with real data)
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);
        const sizes = new Float32Array(count);
        const particleOpacities = new Float32Array(count);  // Per-particle opacity
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        // Use 'customColor' to avoid conflict with Three.js built-in 'color'
        geometry.setAttribute('customColor', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('particleOpacity', new THREE.BufferAttribute(particleOpacities, 1));

        // Custom shader material for soft glowing plasma particles
        const material = new THREE.ShaderMaterial({
            uniforms: {
                pointSize: { value: this.pointSize * 100 },  // Base size multiplier
                opacity: { value: this.pointOpacity },
                opacityScale: { value: this.opacityScale },  // Logarithmic opacity multiplier
                glowFalloff: { value: 2.5 },  // Controls how quickly glow fades (higher = softer)
            },
            vertexShader: `
                attribute float size;
                attribute vec3 customColor;
                attribute float particleOpacity;
                varying vec3 vColor;
                varying float vParticleOpacity;
                uniform float pointSize;
                
                void main() {
                    vColor = customColor;
                    vParticleOpacity = particleOpacity;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    
                    // Size attenuation based on distance
                    float sizeAttenuation = 300.0 / length(mvPosition.xyz);
                    gl_PointSize = size * pointSize * sizeAttenuation;
                    gl_PointSize = clamp(gl_PointSize, 1.0, 500.0);
                    
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                varying float vParticleOpacity;
                uniform float opacity;
                uniform float opacityScale;
                uniform float glowFalloff;
                
                void main() {
                    // Distance from center of point (0 at center, 1 at edge)
                    vec2 center = gl_PointCoord - vec2(0.5);
                    float dist = length(center) * 2.0;
                    
                    // Soft gaussian-like falloff - NO hard core
                    // This creates a pure diffuse glow with no visible center
                    float glow = exp(-dist * dist * glowFalloff);
                    
                    // Discard pixels outside the glow
                    if (glow < 0.01) discard;
                    
                    // Color with glow intensity - emissive style
                    vec3 emissiveColor = vColor * glow * 2.0;  // Boost emission
                    
                    // Final alpha: glow * global opacity * opacity scale * per-particle opacity
                    float alpha = glow * opacity * opacityScale * vParticleOpacity;
                    
                    gl_FragColor = vec4(emissiveColor, alpha);
                }
            `,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        this.particleSystem = new THREE.Points(geometry, material);
        this.particleSystem.frustumCulled = false;
        
        // Put particles on bloom layer (layer 1) for selective bloom
        this.particleSystem.layers.set(1);
        
        this.scene.add(this.particleSystem);
        console.log(`Created plasma particle system with ${count} particles`);
        
        return this.particleSystem;
    }

    // Keep createInstancedMesh as fallback but we won't use it
    createInstancedMesh(count) {
        // Redirect to new particle system
        return this.createParticleSystem(count);
    }

    loadParticles(positions, colorField, densityField, colorBy = 'density') {
        this.positions = positions;
        this.colorField = colorField;
        this.densityField = densityField;
        this.particleCount = positions.length / 3;
        this.colorBy = colorBy;

        // Create or resize particle system
        if (!this.particleSystem || this.particleSystem.geometry.getAttribute('position').count !== this.particleCount) {
            this.createParticleSystem(this.particleCount);
        }

        this.updateParticleData();
        
        // Only fit camera on first load, not on every call
        if (!this._cameraInitialized) {
            this.fitCameraToParticles();
            this._cameraInitialized = true;
        }
    }

    updateParticles(positions, colorField, densityField) {
        if (!positions || positions.length === 0) {
            console.warn('updateParticles called with empty positions');
            return;
        }
        
        const numParticles = Math.floor(positions.length / 3);

        this.positions = positions;
        this.colorField = colorField;
        this.densityField = densityField;

        // Handle particle count changes - use new particle system
        const newCount = numParticles;
        if (newCount !== this.particleCount || !this.particleSystem) {
            console.log(`Creating new particle system for ${newCount} particles`);
            this.particleCount = newCount;
            this.createParticleSystem(this.particleCount);
        }

        this.updateParticleData();
        
        // Auto-frame camera to see particles on first load of new file
        if (!this._cameraInitialized) {
            this.fitCameraToParticles();
            this._cameraInitialized = true;
        }
    }

    updateParticleData() {
        if (!this.particleSystem) {
            console.warn('updateParticleData: No particle system');
            return;
        }
        
        if (!this.positions || this.positions.length === 0) {
            console.warn('updateParticleData: No positions data');
            return;
        }

        const colors = this.computeColors(this.colorField);
        const { sizes, opacities } = this.computeSizesAndOpacities(this.densityField);
        
        // Get geometry attributes - use direct array access for performance
        const geometry = this.particleSystem.geometry;
        const positionAttr = geometry.getAttribute('position');
        const colorAttr = geometry.getAttribute('customColor');
        const sizeAttr = geometry.getAttribute('size');
        const opacityAttr = geometry.getAttribute('particleOpacity');
        
        // Direct buffer array copy - MUCH faster than setXYZ for 1M+ particles
        // This avoids function call overhead per particle
        const posArray = positionAttr.array;
        const colorArray = colorAttr.array;
        const sizeArray = sizeAttr.array;
        const opacityArray = opacityAttr.array;
        
        // Copy positions directly (same layout: x,y,z,x,y,z,...)
        posArray.set(this.positions);
        
        // Copy colors directly (same layout: r,g,b,r,g,b,...)
        colorArray.set(colors);
        
        // Copy sizes and opacities directly
        sizeArray.set(sizes);
        opacityArray.set(opacities);
        
        // Mark attributes as needing update
        positionAttr.needsUpdate = true;
        colorAttr.needsUpdate = true;
        sizeAttr.needsUpdate = true;
        opacityAttr.needsUpdate = true;
    }
    
    // Legacy method name redirect
    updateInstanceData() {
        this.updateParticleData();
    }
    
    /**
     * Initialize the THREE.Lut colormap system with custom blackbody colormap.
     * Blackbody colors sampled from physical temperature scale (3000K-9000K).
     */
    initLut() {
        if (typeof THREE.Lut === 'undefined') {
            console.warn('THREE.Lut not available, using fallback colormap');
            this.lut = null;
            return;
        }
        
        // Add custom blackbody colormap if not already defined
        // Sampled from physical blackbody radiation (3000K-9000K)
        // Colors: deep red -> orange -> yellow -> white -> pale blue
        if (!THREE.Lut.prototype.hasOwnProperty('blackbody')) {
            // 32 evenly spaced samples from the blackbody spectrum image
            const blackbodyColors = [
                [0.00, 0xFF1C00],  // 3000K - deep red-orange
                [0.03, 0xFF2A00],  // 3200K
                [0.06, 0xFF3800],  // 3400K
                [0.10, 0xFF4700],  // 3600K
                [0.13, 0xFF5500],  // 3800K - orange
                [0.16, 0xFF6300],  // 4000K
                [0.19, 0xFF7100],  // 4200K
                [0.23, 0xFF7F00],  // 4400K
                [0.26, 0xFF8C12],  // 4600K - warm orange
                [0.29, 0xFF9A29],  // 4800K
                [0.32, 0xFFA740],  // 5000K
                [0.35, 0xFFB457],  // 5200K - yellow-orange
                [0.39, 0xFFC16E],  // 5400K
                [0.42, 0xFFCD85],  // 5600K
                [0.45, 0xFFD99C],  // 5800K - pale yellow
                [0.48, 0xFFE4B3],  // 6000K
                [0.52, 0xFFEFCA],  // 6200K
                [0.55, 0xFFF9E0],  // 6400K - warm white
                [0.58, 0xFFFFF4],  // 6600K - pure white
                [0.61, 0xF5F7FF],  // 6800K
                [0.65, 0xECF0FF],  // 7000K - cool white
                [0.68, 0xE3E9FF],  // 7200K
                [0.71, 0xDAE2FF],  // 7400K
                [0.74, 0xD1DBFF],  // 7600K - pale blue-white
                [0.77, 0xC9D4FF],  // 7800K
                [0.81, 0xC1CDFF],  // 8000K
                [0.84, 0xB9C7FF],  // 8200K
                [0.87, 0xB2C0FF],  // 8400K - light blue
                [0.90, 0xABBAFF],  // 8600K
                [0.94, 0xA4B4FF],  // 8800K
                [0.97, 0x9EAEFF],  // 9000K
                [1.00, 0x99A9FF],  // 9200K - pale blue
            ];
            
            // Register the custom colormap
            THREE.Lut.prototype.addColorMap('blackbody', blackbodyColors);
            console.log('Custom blackbody colormap registered with THREE.Lut');
        }
        
        // Create Lut with the selected colormap
        try {
            this.lut = new THREE.Lut(this.colormap, 512);
            this.lut.setMin(0);
            this.lut.setMax(1);
            console.log('THREE.Lut colormap initialized with:', this.colormap);
        } catch (e) {
            console.warn('Failed to create Lut with colormap:', this.colormap, '- falling back to rainbow');
            this.lut = new THREE.Lut('rainbow', 512);
            this.lut.setMin(0);
            this.lut.setMax(1);
        }
    }

    computeColors(field) {
        const N = field.length;
        const colors = new Float32Array(N * 3);

        // Find min/max in single pass
        let minVal = Infinity;
        let maxVal = -Infinity;
        
        for (let i = 0; i < N; i++) {
            const val = field[i];
            if (val > 0 && val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }

        if (minVal === Infinity) minVal = 1;
        if (maxVal === -Infinity) maxVal = 10;

        // Pre-compute log values if needed
        const useLog = this.logScale && minVal > 0;
        const logMin = useLog ? Math.log10(minVal) : minVal;
        const logMax = useLog ? Math.log10(maxVal) : maxVal;
        const range = (logMax - logMin) || 1e-6;
        const invRange = 1.0 / range;
        
        // Pre-build a lookup table for colormap (256 entries for fast lookup)
        const lutSize = 256;
        const lutColors = new Float32Array(lutSize * 3);
        
        if (this.lut) {
            const tempColor = new THREE.Color();
            for (let i = 0; i < lutSize; i++) {
                const t = i / (lutSize - 1);
                const lutColor = this.lut.getColor(t);
                if (lutColor) {
                    tempColor.copy(lutColor).convertSRGBToLinear();
                    lutColors[i * 3] = tempColor.r;
                    lutColors[i * 3 + 1] = tempColor.g;
                    lutColors[i * 3 + 2] = tempColor.b;
                } else {
                    lutColors[i * 3] = t;
                    lutColors[i * 3 + 1] = t;
                    lutColors[i * 3 + 2] = t;
                }
            }
        } else {
            // Grayscale fallback
            for (let i = 0; i < lutSize; i++) {
                const t = i / (lutSize - 1);
                lutColors[i * 3] = t;
                lutColors[i * 3 + 1] = t;
                lutColors[i * 3 + 2] = t;
            }
        }

        // Apply colormap using lookup table (much faster than calling Lut per particle)
        for (let i = 0; i < N; i++) {
            const val = field[i];
            
            if (!(val > 0)) {
                // Invalid value - gray
                colors[i * 3] = 0.2;
                colors[i * 3 + 1] = 0.2;
                colors[i * 3 + 2] = 0.2;
                continue;
            }

            const normalizedVal = useLog ? Math.log10(val) : val;
            const normalized = (normalizedVal - logMin) * invRange;
            
            // Clamp and convert to LUT index
            const lutIdx = Math.max(0, Math.min(lutSize - 1, (normalized * (lutSize - 1)) | 0));
            
            colors[i * 3] = lutColors[lutIdx * 3];
            colors[i * 3 + 1] = lutColors[lutIdx * 3 + 1];
            colors[i * 3 + 2] = lutColors[lutIdx * 3 + 2];
        }

        return colors;
    }

    /**
     * Compute particle sizes and opacities based on density.
     * Uses 90th percentile as the baseline for size scaling.
     * Size scales with density^(-1/3) for volume conservation.
     * Opacity scales with size^(-2) for visibility balance (larger particles are dimmer).
     * @returns {{sizes: Float32Array, opacities: Float32Array}}
     */
    computeSizesAndOpacities(densityField) {
        const sizes = new Float32Array(this.particleCount);
        const opacities = new Float32Array(this.particleCount);
        
        if (!densityField || densityField.length === 0 || !this.sizeByDensity) {
            sizes.fill(1.0);
            opacities.fill(1.0);
            return { sizes, opacities };
        }

        const baseline = Math.max(this.density90thPercentile, 1e-30);
        
        for (let i = 0; i < this.particleCount; i++) {
            const rho = Math.max(densityField[i], 1e-30);
            
            // Size scales inversely with density (volume conservation)
            // Higher density = smaller size, 90th percentile is the clamp threshold
            let scale = Math.pow(baseline / rho, 1 / 3);
            scale = Math.max(1.0, Math.min(scale, this.maxSizeMultiplier));
            
            sizes[i] = scale;
            
            // Opacity scales with size^(-2) for visibility balance
            // Larger particles (lower density) are made dimmer
            // This prevents large diffuse particles from dominating the view
            // Clamp opacity to [0.1, 1.0] to keep particles visible
            const opacityFromSize = 1.0 / (scale * scale);
            opacities[i] = Math.max(0.1, Math.min(1.0, opacityFromSize));
        }
        
        return { sizes, opacities };
    }
    
    // Legacy method for compatibility
    computeSizes(densityField) {
        return this.computeSizesAndOpacities(densityField).sizes;
    }
    
    setDensityBaseline(percentile90) {
        this.density90thPercentile = percentile90;
    }

    fitCameraToParticles() {
        if (!this.positions || this.positions.length === 0) {
            console.warn('fitCameraToParticles: No positions');
            return;
        }

        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = 0; i < this.positions.length; i += 3) {
            const x = this.positions[i];
            const y = this.positions[i + 1];
            const z = this.positions[i + 2];
            if (!isNaN(x) && isFinite(x)) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
            }
            if (!isNaN(y) && isFinite(y)) {
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
            if (!isNaN(z) && isFinite(z)) {
                if (z < minZ) minZ = z;
                if (z > maxZ) maxZ = z;
            }
        }

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

        // Ensure minimum extent for very compact datasets
        const effectiveExtent = Math.max(extent, 1.0);
        const distance = Math.max(5, effectiveExtent * 2.0);
        
        this.camera.position.set(
            centerX + distance * 0.5,
            centerY + distance * 0.5,
            centerZ + distance
        );
        this.camera.lookAt(centerX, centerY, centerZ);
        this.controls.target.set(centerX, centerY, centerZ);
        
        // Update near/far planes based on data extent
        this.camera.near = Math.max(0.001, distance * 0.001);
        this.camera.far = Math.max(10000, distance * 20);
        this.camera.updateProjectionMatrix();
        
        this.controls.update();
        
        console.log(`Camera fitted: center=(${centerX.toFixed(2)}, ${centerY.toFixed(2)}, ${centerZ.toFixed(2)}), extent=${extent.toFixed(2)}, distance=${distance.toFixed(2)}, near=${this.camera.near.toFixed(4)}, far=${this.camera.far.toFixed(0)}`);
    }

    // Settings methods
    setPointSize(size) {
        this.pointSize = size;
        // Update shader uniform
        if (this.particleSystem && this.particleSystem.material.uniforms) {
            this.particleSystem.material.uniforms.pointSize.value = size * 100;
        }
    }
    
    setMaxSizeMultiplier(mult) {
        this.maxSizeMultiplier = mult;
        if (this.particleSystem && this.densityField) {
            this.updateParticleData();
        }
    }

    setColormap(colormapName) {
        this.colormap = colormapName;
        
        // Update the Lut colormap
        if (this.lut && typeof THREE.Lut !== 'undefined') {
            this.lut.setColorMap(colormapName);
        }
        
        if (this.colorField && this.particleSystem) {
            this.updateParticleData();
        }
    }

    setLogScale(enabled) {
        this.logScale = enabled;
        if (this.colorField && this.particleSystem) {
            this.updateParticleData();
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
    
    setShowSkybox(show) {
        this.showSkybox = show;
        if (show) {
            if (this.skybox) {
                this.scene.background = this.skybox;
            }
            if (this.skyboxMesh) {
                this.skyboxMesh.visible = true;
            }
        } else {
            this.scene.background = new THREE.Color(0x000000);
            if (this.skyboxMesh) {
                this.skyboxMesh.visible = false;
            }
        }
    }

    setPointOpacity(opacity) {
        this.pointOpacity = opacity;
        // Update shader uniform
        if (this.particleSystem && this.particleSystem.material.uniforms) {
            this.particleSystem.material.uniforms.opacity.value = opacity;
        }
    }
    
    /**
     * Set glow falloff - higher values make glow softer/more diffuse.
     */
    setGlowFalloff(falloff) {
        if (this.particleSystem && this.particleSystem.material.uniforms) {
            this.particleSystem.material.uniforms.glowFalloff.value = falloff;
        }
    }
    
    /**
     * Set opacity scale (logarithmic, 0.001 to 10).
     * This is applied as a multiplier to per-particle opacity.
     */
    setOpacityScale(scale) {
        this.opacityScale = scale;
        // Update shader uniform
        if (this.particleSystem && this.particleSystem.material.uniforms) {
            this.particleSystem.material.uniforms.opacityScale.value = scale;
        }
    }

    setSizeByDensity(enabled) {
        this.sizeByDensity = enabled;
        if (this.particleSystem && this.densityField) {
            this.updateParticleData();
        }
    }
    
    // Bloom controls
    setBloomEnabled(enabled) {
        this.bloomEnabled = enabled;
    }
    
    setBloomStrength(strength) {
        this.bloomStrength = strength;
        if (this.bloomPass) {
            this.bloomPass.strength = strength;
        }
    }
    
    setBloomRadius(radius) {
        this.bloomRadius = radius;
        if (this.bloomPass) {
            this.bloomPass.radius = radius;
        }
    }
    
    setBloomThreshold(threshold) {
        this.bloomThreshold = threshold;
        if (this.bloomPass) {
            this.bloomPass.threshold = threshold;
        }
    }

    // Camera controls
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
        // Render first
        if (this.bloomEnabled && this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
        
        const dataURL = this.renderer.domElement.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `tde_sph_screenshot_${Date.now()}.png`;
        link.href = dataURL;
        link.click();
    }

    // Keyboard controls (WASDQE + Shift/Ctrl)
    onKeyDown(event) {
        this.keysPressed[event.code] = true;
        
        // Prevent default for Ctrl key to avoid browser shortcuts
        // (But allow Ctrl+S for screenshot which is handled in app.js)
        if ((event.code === 'ControlLeft' || event.code === 'ControlRight') && 
            !event.key.toLowerCase().includes('s')) {
            // Don't prevent default - let app.js handle Ctrl+S
        }
    }

    onKeyUp(event) {
        this.keysPressed[event.code] = false;
    }

    updateCameraMovement() {
        const forward = new THREE.Vector3();
        const right = new THREE.Vector3();
        const up = new THREE.Vector3(0, 1, 0);
        
        this.camera.getWorldDirection(forward);
        right.crossVectors(forward, up).normalize();

        const speed = this.moveSpeed;
        const rotateSpeed = 0.02; // Radians per frame for rotation

        // WASD movement (W/S = forward/backward, A/D = left/right)
        if (this.keysPressed['KeyW']) {
            this.camera.position.addScaledVector(forward, speed);
            this.controls.target.addScaledVector(forward, speed);
        }
        if (this.keysPressed['KeyS']) {
            this.camera.position.addScaledVector(forward, -speed);
            this.controls.target.addScaledVector(forward, -speed);
        }
        if (this.keysPressed['KeyA']) {
            this.camera.position.addScaledVector(right, -speed);
            this.controls.target.addScaledVector(right, -speed);
        }
        if (this.keysPressed['KeyD']) {
            this.camera.position.addScaledVector(right, speed);
            this.controls.target.addScaledVector(right, speed);
        }
        
        // Q/E for left/right rotation (yaw)
        if (this.keysPressed['KeyQ']) {
            // Rotate camera around its target (orbit left)
            const offset = this.camera.position.clone().sub(this.controls.target);
            offset.applyAxisAngle(up, rotateSpeed);
            this.camera.position.copy(this.controls.target).add(offset);
            this.camera.lookAt(this.controls.target);
        }
        if (this.keysPressed['KeyE']) {
            // Rotate camera around its target (orbit right)
            const offset = this.camera.position.clone().sub(this.controls.target);
            offset.applyAxisAngle(up, -rotateSpeed);
            this.camera.position.copy(this.controls.target).add(offset);
            this.camera.lookAt(this.controls.target);
        }
        
        // Shift/Ctrl for up/down movement
        if (this.keysPressed['ShiftLeft'] || this.keysPressed['ShiftRight']) {
            this.camera.position.addScaledVector(up, speed);
            this.controls.target.addScaledVector(up, speed);
        }
        if (this.keysPressed['ControlLeft'] || this.keysPressed['ControlRight']) {
            this.camera.position.addScaledVector(up, -speed);
            this.controls.target.addScaledVector(up, -speed);
        }
    }
    
    setMoveSpeed(speed) {
        this.moveSpeed = speed;
    }

    onWindowResize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
        
        // Update composers for selective bloom
        if (this.bloomComposer) {
            this.bloomComposer.setSize(width, height);
        }
        
        if (this.finalComposer) {
            this.finalComposer.setSize(width, height);
        }
        
        if (this.bloomPass) {
            this.bloomPass.resolution.set(width, height);
        }
    }

    animate() {
        if (!this.isActive) return;
        
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // Update keyboard-based camera movement
        this.updateCameraMovement();
        
        // Update orbit controls
        this.controls.update();
        
        // Render with selective bloom if enabled
        if (this.bloomEnabled && this.bloomComposer && this.finalComposer && this.bloomBlendPass) {
            try {
                // Step 1: Render bloom pass with non-bloom objects darkened
                this.darkenNonBloomObjects();
                this.bloomComposer.render();
                this.restoreMaterials();
                
                // Step 2: Render final scene with bloom composited
                this.bloomBlendPass.uniforms.bloomTexture.value = this.bloomComposer.renderTarget2.texture;
                this.finalComposer.render();
            } catch (e) {
                // Fallback to direct rendering if composer fails
                console.warn('Selective bloom render failed, falling back to direct render:', e);
                this.bloomEnabled = false;
                this.renderer.render(this.scene, this.camera);
            }
        } else {
            this.renderer.render(this.scene, this.camera);
        }

        // FPS counter
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
     * Dispose all resources.
     */
    dispose() {
        // Dispose particle system
        if (this.particleSystem) {
            this.scene.remove(this.particleSystem);
            this.particleSystem.geometry.dispose();
            this.particleSystem.material.dispose();
        }
        
        // Dispose legacy instanced mesh if it exists
        if (this.instancedMesh) {
            this.scene.remove(this.instancedMesh);
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
        }
        
        if (this.blackHole) {
            this.scene.remove(this.blackHole);
            this.blackHole.geometry.dispose();
            this.blackHole.material.dispose();
        }
        
        if (this.axesHelper) {
            this.scene.remove(this.axesHelper);
        }
        
        // Dispose selective bloom composers
        if (this.bloomComposer) {
            this.bloomComposer.dispose();
        }
        
        if (this.finalComposer) {
            this.finalComposer.dispose();
        }
        
        // Dispose dark material
        if (this.darkMaterial) {
            this.darkMaterial.dispose();
        }
        
        this.renderer.dispose();
        
        window.removeEventListener('resize', this.onWindowResize);
        window.removeEventListener('keydown', this.onKeyDown);
        window.removeEventListener('keyup', this.onKeyUp);
    }
}
