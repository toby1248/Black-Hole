/**
 * TDE-SPH Data Loader
 *
 * Handles loading and parsing of simulation data from various sources:
 * - Local HDF5 files (via h5wasm WebAssembly library)
 * - Local JSON files
 * - Simulation server (WebSocket or HTTP)
 * - Demo data (procedurally generated)
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-11-25
 */

class DataLoader {
    constructor() {
        this.snapshots = [];  // Array of snapshot objects
        this.currentIndex = 0;
        this.isLoaded = false;

        // Server connection
        this.serverURL = null;
        this.websocket = null;
        this.isConnected = false;

        // HDF5 wasm loader
        this.h5wasm = null;
        this.h5wasmReady = null;
        
        // Black hole metadata from HDF5
        this.bhMass = null;
        this.bhSpin = null;
        this.metricType = null;
        
        // Density statistics for scaling (90th percentile for clamp threshold)
        this.density90thPercentile = null;
    }

    loadDemoData() {
        /**
         * Generate demo SPH data for testing (Keplerian disc).
         */
        console.log('Generating demo data...');

        const numSnapshots = 50;
        const particlesPerSnapshot = 10000;

        this.snapshots = [];

        for (let snap = 0; snap < numSnapshots; snap++) {
            const time = snap * 2.0;  // Time spacing

            // Generate Keplerian disc
            const positions = new Float32Array(particlesPerSnapshot * 3);
            const density = new Float32Array(particlesPerSnapshot);
            const temperature = new Float32Array(particlesPerSnapshot);
            const velocityMag = new Float32Array(particlesPerSnapshot);

            for (let i = 0; i < particlesPerSnapshot; i++) {
                // Disc parameters
                const r = 5.0 + Math.random() * 15.0;  // Radius [5, 20]
                const phi = Math.random() * 2 * Math.PI;  // Azimuthal angle
                const z = (Math.random() - 0.5) * 0.5 * Math.exp(-r / 10.0);  // Scale height

                // Add time-dependent precession
                const precessionAngle = time * 0.05;
                const phiRotated = phi + precessionAngle;

                // Positions
                positions[i*3] = r * Math.cos(phiRotated);
                positions[i*3+1] = r * Math.sin(phiRotated);
                positions[i*3+2] = z;

                // Keplerian velocity
                const v_kepler = 1.0 / Math.sqrt(r);

                // Density (decreases with radius)
                density[i] = 10.0 * Math.exp(-r / 10.0) * (1.0 + 0.5 * Math.random());

                // Temperature (correlated with density)
                temperature[i] = 1000.0 + 5000.0 * density[i] / 10.0;

                // Velocity magnitude
                velocityMag[i] = v_kepler;
            }

            // Add some bound debris streams (eccentric orbits)
            const streamParticles = 2000;
            const streamOffset = particlesPerSnapshot;

            const streamPositions = new Float32Array(streamParticles * 3);
            const streamDensity = new Float32Array(streamParticles);
            const streamTemperature = new Float32Array(streamParticles);
            const streamVelocity = new Float32Array(streamParticles);

            for (let i = 0; i < streamParticles; i++) {
                // Eccentric orbit
                const phase = (i / streamParticles) * 2 * Math.PI;
                const ecc = 0.8;
                const a = 25.0;  // Semi-major axis

                const r = a * (1 - ecc**2) / (1 + ecc * Math.cos(phase));
                const theta = phase + time * 0.1;  // Orbiting

                streamPositions[i*3] = r * Math.cos(theta);
                streamPositions[i*3+1] = r * Math.sin(theta);
                streamPositions[i*3+2] = (Math.random() - 0.5) * 2.0;

                streamDensity[i] = 2.0 * Math.exp(-r / 30.0);
                streamTemperature[i] = 500.0 + 2000.0 * Math.random();
                streamVelocity[i] = 0.5;
            }

            // Combine disc + stream
            const totalParticles = particlesPerSnapshot + streamParticles;
            const combinedPositions = new Float32Array(totalParticles * 3);
            const combinedDensity = new Float32Array(totalParticles);
            const combinedTemp = new Float32Array(totalParticles);
            const combinedVel = new Float32Array(totalParticles);

            combinedPositions.set(positions);
            combinedPositions.set(streamPositions, particlesPerSnapshot * 3);

            combinedDensity.set(density);
            combinedDensity.set(streamDensity, particlesPerSnapshot);

            combinedTemp.set(temperature);
            combinedTemp.set(streamTemperature, particlesPerSnapshot);

            combinedVel.set(velocityMag);
            combinedVel.set(streamVelocity, particlesPerSnapshot);

            // Create derived fields using for loops (avoid .map() stack overflow on large arrays)
            const internalEnergy = new Float32Array(totalParticles);
            const pressure = new Float32Array(totalParticles);
            const entropy = new Float32Array(totalParticles);
            for (let i = 0; i < totalParticles; i++) {
                internalEnergy[i] = combinedTemp[i] / 10000.0;
                pressure[i] = combinedDensity[i] * combinedTemp[i] / 10000.0;
                const rho = Math.max(combinedDensity[i], 1e-30);
                entropy[i] = Math.log(combinedTemp[i] / Math.pow(rho, 0.4));
            }
            
            // Create snapshot object
            const snapshot = {
                time: time,
                step: snap,
                n_particles: totalParticles,
                positions: combinedPositions,
                density: combinedDensity,
                temperature: combinedTemp,
                internal_energy: internalEnergy,
                velocity_magnitude: combinedVel,
                pressure: pressure,
                entropy: entropy
            };

            this.snapshots.push(snapshot);
        }

        this.currentIndex = 0;
        this.isLoaded = true;
        
        // Compute density statistics
        this.computeDensityPercentile();
        
        // Set demo BH parameters
        this.bhMass = 1e6;
        this.bhSpin = 0.0;
        this.metricType = 'schwarzschild';

        console.log(`Demo data generated: ${numSnapshots} snapshots, ${this.snapshots[0].n_particles} particles each`);

        return this.snapshots;
    }

    loadJSONFile(file) {
        /**
         * Load a single JSON snapshot file.
         *
         * Expected format:
         * {
         *   "time": 10.5,
         *   "step": 100,
         *   "n_particles": 100000,
         *   "positions": [[x1,y1,z1], [x2,y2,z2], ...],
         *   "density": [rho1, rho2, ...],
         *   "temperature": [T1, T2, ...],
         *   ...
         * }
         */
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                try {
                    const data = JSON.parse(event.target.result);

                    // Manual flatten for positions (avoids stack overflow on large arrays)
                    const flattenPositions = (arr) => {
                        const result = new Float32Array(arr.length * 3);
                        for (let i = 0; i < arr.length; i++) {
                            result[i * 3] = arr[i][0];
                            result[i * 3 + 1] = arr[i][1];
                            result[i * 3 + 2] = arr[i][2];
                        }
                        return result;
                    };
                    
                    const positions = flattenPositions(data.positions);

                    // Helper function to create filled array without .map() (avoids stack overflow on large arrays)
                    const createFilledArray = (length, value) => {
                        const arr = new Float32Array(length);
                        arr.fill(value);
                        return arr;
                    };
                    
                    const n = data.n_particles;
                    
                    const snapshot = {
                        time: data.time,
                        step: data.step,
                        n_particles: n,
                        positions: positions,
                        density: new Float32Array(data.density),
                        temperature: new Float32Array(data.temperature || createFilledArray(n, 1000)),
                        internal_energy: new Float32Array(data.internal_energy || createFilledArray(n, 1.0)),
                        velocity_magnitude: new Float32Array(data.velocity_magnitude || createFilledArray(n, 0.1)),
                        pressure: new Float32Array(data.pressure || data.density),
                        entropy: new Float32Array(data.entropy || createFilledArray(n, 1.0))
                    };

                    resolve(snapshot);

                } catch(error) {
                    reject(new Error(`Failed to parse JSON: ${error.message}`));
                }
            };

            reader.onerror = () => reject(new Error('File reading failed'));
            reader.readAsText(file);
        });
    }

    async loadMultipleJSONFiles(files) {
        /**
         * Load multiple JSON snapshot files.
         *
         * Parameters:
         *   files: FileList from input element
         */
        return this.loadMultipleFiles(files);
    }

    connectToServer(url) {
        /**
         * Connect to TDE-SPH simulation server via WebSocket.
         *
         * Server should send snapshot data in JSON format.
         */
        return new Promise((resolve, reject) => {
            this.serverURL = url;

            // Try WebSocket connection
            const wsURL = url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws';

            try {
                this.websocket = new WebSocket(wsURL);

                this.websocket.onopen = () => {
                    console.log(`Connected to server: ${wsURL}`);
                    this.isConnected = true;
                    resolve();
                };

                this.websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);

                        if (data.type === 'snapshot') {
                            // Convert and add to snapshots
                            const snapshot = this.parseServerSnapshot(data.snapshot);
                            this.snapshots.push(snapshot);
                            this.isLoaded = true;

                            console.log(`Received snapshot at t=${snapshot.time}`);
                        }

                    } catch(error) {
                        console.error('Error parsing server message:', error);
                    }
                };

                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnected = false;
                    reject(error);
                };

                this.websocket.onclose = () => {
                    console.log('WebSocket connection closed');
                    this.isConnected = false;
                };

            } catch(error) {
                reject(error);
            }
        });
    }

    parseServerSnapshot(data) {
        /**
         * Parse snapshot data from server.
         */
        // Manual flatten for positions (avoids stack overflow on large arrays)
        const flattenPositions = (arr) => {
            const result = new Float32Array(arr.length * 3);
            for (let i = 0; i < arr.length; i++) {
                result[i * 3] = arr[i][0];
                result[i * 3 + 1] = arr[i][1];
                result[i * 3 + 2] = arr[i][2];
            }
            return result;
        };
        
        const positions = flattenPositions(data.positions);

        return {
            time: data.time,
            step: data.step,
            n_particles: data.n_particles,
            positions: positions,
            density: new Float32Array(data.density),
            temperature: new Float32Array(data.temperature),
            internal_energy: new Float32Array(data.internal_energy),
            velocity_magnitude: new Float32Array(data.velocity_magnitude),
            pressure: new Float32Array(data.pressure),
            entropy: new Float32Array(data.entropy)
        };
    }

    disconnectFromServer() {
        if (this.websocket) {
            this.websocket.close();
            this.isConnected = false;
        }
    }

    getCurrentSnapshot() {
        if (!this.isLoaded || this.snapshots.length === 0) {
            return null;
        }

        return this.snapshots[this.currentIndex];
    }

    getSnapshotByIndex(index) {
        if (!this.isLoaded || index < 0 || index >= this.snapshots.length) {
            return null;
        }

        return this.snapshots[index];
    }

    getSnapshotCount() {
        return this.snapshots.length;
    }

    setCurrentIndex(index) {
        if (index >= 0 && index < this.snapshots.length) {
            this.currentIndex = index;
            return true;
        }
        return false;
    }

    nextSnapshot() {
        if (this.currentIndex < this.snapshots.length - 1) {
            this.currentIndex++;
            return this.getCurrentSnapshot();
        }
        return null;
    }

    previousSnapshot() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            return this.getCurrentSnapshot();
        }
        return null;
    }

    computeStatistics(snapshot) {
        /**
         * Compute basic statistics for a snapshot.
         * Uses iterative loops instead of spread/reduce to avoid stack overflow on large arrays.
         */
        const stats = {
            n_particles: snapshot.n_particles,
            time: snapshot.time,
            step: snapshot.step
        };

        // Total mass (assuming uniform particle masses)
        stats.total_mass = snapshot.n_particles * 0.00001;  // Placeholder

        // Density statistics (iterative to avoid stack overflow)
        let rhoMin = Infinity, rhoMax = -Infinity, rhoSum = 0;
        const density = snapshot.density;
        for (let i = 0; i < density.length; i++) {
            const val = density[i];
            if (val < rhoMin) rhoMin = val;
            if (val > rhoMax) rhoMax = val;
            rhoSum += val;
        }
        stats.rho_min = rhoMin;
        stats.rho_max = rhoMax;
        stats.rho_mean = rhoSum / density.length;

        // Temperature statistics (iterative to avoid stack overflow)
        let tempMin = Infinity, tempMax = -Infinity, tempSum = 0;
        const temperature = snapshot.temperature;
        for (let i = 0; i < temperature.length; i++) {
            const val = temperature[i];
            if (val < tempMin) tempMin = val;
            if (val > tempMax) tempMax = val;
            tempSum += val;
        }
        stats.temp_min = tempMin;
        stats.temp_max = tempMax;
        stats.temp_mean = tempSum / temperature.length;

        // Total energy (approximate) - iterative to avoid stack overflow
        let kineticSum = 0, internalSum = 0;
        const velocityMag = snapshot.velocity_magnitude;
        const internalEnergy = snapshot.internal_energy;
        for (let i = 0; i < velocityMag.length; i++) {
            kineticSum += 0.5 * velocityMag[i] * velocityMag[i];
        }
        for (let i = 0; i < internalEnergy.length; i++) {
            internalSum += internalEnergy[i];
        }
        stats.total_energy = (kineticSum + internalSum) * 0.00001;

        return stats;
    }
    
    /**
     * Compute the 10th percentile of density across selected snapshots.
     * Uses up to 100 snapshots evenly spaced throughout the dataset.
     */
    computeDensityPercentile() {
        if (this.snapshots.length === 0) {
            this.density90thPercentile = 1e-10;
            return;
        }
        
        // Select up to 100 snapshots evenly spaced
        const numSamples = Math.min(100, this.snapshots.length);
        const step = this.snapshots.length / numSamples;
        
        let allDensities = [];
        
        for (let i = 0; i < numSamples; i++) {
            const snapIdx = Math.floor(i * step);
            const snapshot = this.snapshots[snapIdx];
            
            // Sample up to 1000 particles per snapshot for efficiency
            const sampleSize = Math.min(1000, snapshot.density.length);
            const particleStep = snapshot.density.length / sampleSize;
            
            for (let j = 0; j < sampleSize; j++) {
                const pIdx = Math.floor(j * particleStep);
                allDensities.push(snapshot.density[pIdx]);
            }
        }
        
        // Sort and find 90th percentile (for clamping high-density particles)
        allDensities.sort((a, b) => a - b);
        const idx90 = Math.floor(allDensities.length * 0.9);
        this.density90thPercentile = allDensities[idx90];
        
        console.log(`Density 90th percentile: ${this.density90thPercentile.toExponential(2)}`);
    }
    
    /**
     * Get the computed 90th percentile density value (for clamping).
     */
    getDensity90thPercentile() {
        return this.density90thPercentile;
    }
    
    /**
     * Get black hole parameters from loaded data.
     */
    getBHParams() {
        return {
            mass: this.bhMass,
            spin: this.bhSpin,
            metricType: this.metricType
        };
    }
    
    /**
     * Compute event horizon radius from BH mass.
     * r_s = 2 * G * M / c^2
     * In code units where G=c=1, r_s = 2*M
     * For Kerr: r+ = M + sqrt(M^2 - a^2)
     */
    getEventHorizonRadius() {
        if (!this.bhMass) return 1.0;
        
        const M = this.bhMass;
        const a = this.bhSpin || 0;
        
        if (this.metricType === 'kerr' && Math.abs(a) > 0) {
            // Kerr metric: r+ = M + sqrt(M^2 - a^2)
            const aMax = Math.min(Math.abs(a), M); // Ensure a <= M
            return M + Math.sqrt(M * M - aMax * aMax);
        } else {
            // Schwarzschild: r_s = 2M
            return 2 * M;
        }
    }

    exportToJSON(filename = 'tde_sph_snapshots.json') {
        /**
         * Export all loaded snapshots to JSON file.
         */
        const dataToExport = this.snapshots.map(snapshot => ({
            time: snapshot.time,
            step: snapshot.step,
            n_particles: snapshot.n_particles,
            positions: Array.from(snapshot.positions),
            density: Array.from(snapshot.density),
            temperature: Array.from(snapshot.temperature),
            internal_energy: Array.from(snapshot.internal_energy),
            velocity_magnitude: Array.from(snapshot.velocity_magnitude),
            pressure: Array.from(snapshot.pressure),
            entropy: Array.from(snapshot.entropy)
        }));

        const json = JSON.stringify(dataToExport, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();

        URL.revokeObjectURL(url);
    }

    async loadMultipleFiles(files, progressCallback = null) {
        /**
         * Load multiple files (JSON or HDF5). HDF5 files are converted client-side.
         */
        this.snapshots = [];
        let processed = 0;
        const errors = [];

        for (const file of files) {
            try {
                const ext = file.name.toLowerCase();
                let snapshot;
                if (ext.endsWith('.json')) {
                    snapshot = await this.loadJSONFile(file);
                } else if (ext.endsWith('.h5') || ext.endsWith('.hdf5')) {
                    snapshot = await this.loadHDF5File(file);
                } else {
                    console.warn(`Skipping unsupported file: ${file.name}`);
                    continue;
                }

                this.snapshots.push(snapshot);
            } catch(error) {
                console.error(`Error loading ${file.name}:`, error);
                errors.push(`${file.name}: ${error.message}`);
            }

            processed += 1;
            if (progressCallback) {
                const pct = Math.round((processed / files.length) * 100);
                progressCallback(pct);
            }
        }

        // Sort by time
        this.snapshots.sort((a, b) => a.time - b.time);

        this.currentIndex = 0;
        this.isLoaded = this.snapshots.length > 0;
        
        // Compute density statistics after loading
        if (this.isLoaded) {
            this.computeDensityPercentile();
        }

        if (!this.isLoaded && errors.length) {
            throw new Error(`Failed to load snapshots: ${errors.join('; ')}`);
        }

        console.log(`Loaded ${this.snapshots.length} snapshots`);

        return this.snapshots;
    }

    async loadHDF5File(file) {
        /**
         * Load a single HDF5 snapshot using h5wasm and convert to web schema.
         */
        await this.ensureH5Wasm();
        if (!this.h5wasm) {
            throw new Error('HDF5 support unavailable (h5wasm not initialized)');
        }
        
        const arrayBuffer = await file.arrayBuffer();
        
        // Write to virtual filesystem and open
        const filename = file.name;
        this.h5wasm.FS.writeFile(filename, new Uint8Array(arrayBuffer));
        const f = new this.h5wasm.File(filename, 'r');

        const getDataset = (path) => {
            try {
                // Try various path formats
                const paths = [
                    path,
                    path.startsWith('/') ? path.slice(1) : '/' + path,
                    path.replace('/particles/', 'particles/')
                ];
                
                for (const p of paths) {
                    try {
                        const dset = f.get(p);
                        if (dset && dset.dtype) {
                            const data = dset.value;
                            // Convert to Float32Array without spread/apply (avoids stack overflow on large arrays)
                            if (data instanceof Float32Array) {
                                return data;
                            } else if (data instanceof Float64Array) {
                                // Manual copy from Float64Array to Float32Array
                                const result = new Float32Array(data.length);
                                for (let i = 0; i < data.length; i++) {
                                    result[i] = data[i];
                                }
                                return result;
                            } else if (ArrayBuffer.isView(data)) {
                                // Other typed arrays - manual copy
                                const result = new Float32Array(data.length);
                                for (let i = 0; i < data.length; i++) {
                                    result[i] = data[i];
                                }
                                return result;
                            } else if (Array.isArray(data)) {
                                // Manual flatten for nested arrays
                                if (data.length > 0 && Array.isArray(data[0])) {
                                    const innerLen = data[0].length;
                                    const result = new Float32Array(data.length * innerLen);
                                    for (let i = 0; i < data.length; i++) {
                                        for (let j = 0; j < innerLen; j++) {
                                            result[i * innerLen + j] = data[i][j];
                                        }
                                    }
                                    return result;
                                } else {
                                    // Manual copy from regular array
                                    const result = new Float32Array(data.length);
                                    for (let i = 0; i < data.length; i++) {
                                        result[i] = data[i];
                                    }
                                    return result;
                                }
                            } else if (typeof data === 'number') {
                                return new Float32Array([data]);
                            } else {
                                // Unknown type - try manual iteration
                                const len = data.length || 0;
                                const result = new Float32Array(len);
                                for (let i = 0; i < len; i++) {
                                    result[i] = data[i];
                                }
                                return result;
                            }
                        }
                    } catch(e) {
                        // Try next path
                    }
                }
                return null;
            } catch (e) {
                return null;
            }
        };
        
        const getAttr = (obj, attrName) => {
            try {
                if (obj && obj.attrs) {
                    const attr = obj.attrs.get(attrName);
                    if (attr !== undefined && attr !== null) {
                        // Handle different attribute types
                        if (typeof attr === 'object' && attr.value !== undefined) {
                            return attr.value;
                        }
                        return attr;
                    }
                }
            } catch(e) {
                // Attribute not found
            }
            return null;
        };

        // Load particle data
        const positions = getDataset('/particles/positions') || getDataset('particles/positions');
        if (!positions) {
            f.close();
            this.h5wasm.FS.unlink(filename);
            throw new Error("positions dataset missing in HDF5 file");
        }

        const density = getDataset('/particles/density') || getDataset('particles/density');
        if (!density) {
            f.close();
            this.h5wasm.FS.unlink(filename);
            throw new Error("density dataset missing in HDF5 file");
        }

        const internalEnergy = getDataset('/particles/internal_energy') || getDataset('particles/internal_energy');
        if (!internalEnergy) {
            f.close();
            this.h5wasm.FS.unlink(filename);
            throw new Error("internal_energy dataset missing in HDF5 file");
        }

        const velocities = getDataset('/particles/velocities') || getDataset('particles/velocities');
        let velocityMag = getDataset('/particles/velocity_magnitude') || getDataset('particles/velocity_magnitude');
        if (!velocityMag && velocities) {
            velocityMag = new Float32Array(velocities.length / 3);
            for (let i = 0; i < velocityMag.length; i++) {
                const vx = velocities[i*3];
                const vy = velocities[i*3+1];
                const vz = velocities[i*3+2];
                velocityMag[i] = Math.sqrt(vx*vx + vy*vy + vz*vz);
            }
        }

        let pressure = getDataset('/particles/pressure') || getDataset('particles/pressure');
        const gamma = 5.0 / 3.0;
        if (!pressure && density && internalEnergy) {
            pressure = new Float32Array(density.length);
            for (let i = 0; i < density.length; i++) {
                pressure[i] = (gamma - 1.0) * density[i] * internalEnergy[i];
            }
        }

        let temperature = getDataset('/particles/temperature') || getDataset('particles/temperature');
        if (!temperature) {
            // Manual copy instead of .slice() to avoid stack overflow on large arrays
            temperature = new Float32Array(internalEnergy.length);
            for (let i = 0; i < internalEnergy.length; i++) {
                temperature[i] = internalEnergy[i];
            }
        }

        let entropy = getDataset('/particles/entropy') || getDataset('particles/entropy');
        if (!entropy && pressure && density) {
            entropy = new Float32Array(density.length);
            for (let i = 0; i < density.length; i++) {
                const rho = Math.max(density[i], 1e-30);
                entropy[i] = pressure[i] / Math.pow(rho, gamma);
            }
        }

        // Get metadata
        const root = f.get('/');
        let timeVal = 0.0;
        let stepVal = 0;
        
        // Try root attributes first
        timeVal = getAttr(root, 'time');
        stepVal = getAttr(root, 'step') || getAttr(root, 'iteration') || getAttr(root, 'timestep');
        
        // Try metadata group
        let metaGroup = null;
        try {
            metaGroup = f.get('/metadata') || f.get('metadata');
        } catch(e) {
            // No metadata group
        }
        
        if (metaGroup) {
            if (timeVal === null || timeVal === undefined) {
                timeVal = getAttr(metaGroup, 'simulation_time') || getAttr(metaGroup, 'time');
            }
            if (stepVal === null || stepVal === undefined) {
                stepVal = getAttr(metaGroup, 'step');
            }
            
            // Get BH parameters
            const bhMass = getAttr(metaGroup, 'bh_mass');
            const bhSpin = getAttr(metaGroup, 'bh_spin');
            const metricType = getAttr(metaGroup, 'metric_type');
            
            if (bhMass !== null && this.bhMass === null) {
                this.bhMass = parseFloat(bhMass);
                console.log(`Loaded BH mass from HDF5: ${this.bhMass}`);
            }
            if (bhSpin !== null && this.bhSpin === null) {
                this.bhSpin = parseFloat(bhSpin);
            }
            if (metricType !== null && this.metricType === null) {
                this.metricType = metricType;
            }
        }
        
        // Parse time value
        if (timeVal !== null && timeVal !== undefined) {
            timeVal = parseFloat(timeVal);
        } else {
            // Try to extract from filename
            const match = file.name.match(/snapshot_(\d+)/);
            if (match) {
                timeVal = parseFloat(match[1]) * 0.1;  // Estimate
            } else {
                timeVal = 0.0;
            }
        }

        const snapshot = {
            time: timeVal,
            step: stepVal ? parseInt(stepVal) : 0,
            n_particles: positions.length / 3,
            positions: positions,
            density: density,
            temperature: temperature,
            internal_energy: internalEnergy,
            velocity_magnitude: velocityMag || new Float32Array(density.length),
            pressure: pressure || new Float32Array(density.length),
            entropy: entropy || new Float32Array(density.length)
        };

        f.close();
        
        // Clean up virtual filesystem
        try {
            this.h5wasm.FS.unlink(filename);
        } catch(e) {
            // Ignore cleanup errors
        }
        
        return snapshot;
    }

    async ensureH5Wasm() {
        if (this.h5wasm) {
            return;
        }
        
        if (this.h5wasmReady) {
            await this.h5wasmReady;
            return;
        }

        // Lazy-load h5wasm from CDN using ESM
        this.h5wasmReady = new Promise(async (resolve, reject) => {
            try {
                // Use dynamic import for ESM module
                const h5wasmModule = await import('https://cdn.jsdelivr.net/npm/h5wasm@0.7.4/dist/esm/hdf5_hl.js');
                
                // Wait for WASM to be ready
                await h5wasmModule.ready;
                
                this.h5wasm = h5wasmModule;
                console.log('h5wasm loaded successfully');
                resolve();
            } catch (error) {
                console.error('Failed to load h5wasm:', error);
                
                // Fallback: Try script tag approach for non-ESM environments
                try {
                    await this.loadH5WasmScript();
                    resolve();
                } catch (e2) {
                    reject(new Error(`h5wasm init failed: ${error.message}`));
                }
            }
        });

        await this.h5wasmReady;
    }
    
    loadH5WasmScript() {
        return new Promise((resolve, reject) => {
            // Check if already loaded globally
            if (window.h5wasm) {
                this.h5wasm = window.h5wasm;
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.type = 'module';
            script.textContent = `
                import * as h5wasm from 'https://cdn.jsdelivr.net/npm/h5wasm@0.7.4/dist/esm/hdf5_hl.js';
                await h5wasm.ready;
                window.h5wasm = h5wasm;
                window.dispatchEvent(new Event('h5wasm-loaded'));
            `;
            
            window.addEventListener('h5wasm-loaded', () => {
                this.h5wasm = window.h5wasm;
                console.log('h5wasm loaded via script');
                resolve();
            }, { once: true });
            
            script.onerror = () => reject(new Error('Failed to load h5wasm script'));
            document.head.appendChild(script);
            
            // Timeout fallback
            setTimeout(() => {
                if (!this.h5wasm) {
                    reject(new Error('h5wasm loading timeout'));
                }
            }, 10000);
        });
    }
}
