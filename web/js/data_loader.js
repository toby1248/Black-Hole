/**
 * TDE-SPH Data Loader
 *
 * Handles loading and parsing of simulation data from various sources:
 * - Local HDF5 files (via conversion to JSON)
 * - Simulation server (WebSocket or HTTP)
 * - Demo data (procedurally generated)
 *
 * Note: Browser-based HDF5 reading requires h5wasm library (not included by default).
 * For production use, consider serving pre-converted JSON snapshots.
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-11-18
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

            // Create snapshot object
            const snapshot = {
                time: time,
                step: snap,
                n_particles: totalParticles,
                positions: combinedPositions,
                density: combinedDensity,
                temperature: combinedTemp,
                internal_energy: combinedTemp.map(T => T / 10000.0),  // Approx
                velocity_magnitude: combinedVel,
                pressure: combinedDensity.map((rho, idx) => rho * combinedTemp[idx] / 10000.0),
                entropy: combinedTemp.map((T, idx) => Math.log(T / (combinedDensity[idx]**0.4)))
            };

            this.snapshots.push(snapshot);
        }

        this.currentIndex = 0;
        this.isLoaded = true;

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

                    // Convert nested arrays to Float32Array
                    const positions = new Float32Array(data.positions.flat());

                    const snapshot = {
                        time: data.time,
                        step: data.step,
                        n_particles: data.n_particles,
                        positions: positions,
                        density: new Float32Array(data.density),
                        temperature: new Float32Array(data.temperature || data.density.map(()=>1000)),
                        internal_energy: new Float32Array(data.internal_energy || data.density.map(()=>1.0)),
                        velocity_magnitude: new Float32Array(data.velocity_magnitude || data.density.map(()=>0.1)),
                        pressure: new Float32Array(data.pressure || data.density),
                        entropy: new Float32Array(data.entropy || data.density.map(()=>1.0))
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
        this.snapshots = [];

        for (const file of files) {
            try {
                const snapshot = await this.loadJSONFile(file);
                this.snapshots.push(snapshot);
            } catch(error) {
                console.error(`Error loading ${file.name}:`, error);
            }
        }

        // Sort by time
        this.snapshots.sort((a, b) => a.time - b.time);

        this.currentIndex = 0;
        this.isLoaded = true;

        console.log(`Loaded ${this.snapshots.length} snapshots`);

        return this.snapshots;
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
        const positions = new Float32Array(data.positions.flat());

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
         */
        const stats = {
            n_particles: snapshot.n_particles,
            time: snapshot.time,
            step: snapshot.step
        };

        // Total mass (assuming uniform particle masses)
        stats.total_mass = snapshot.n_particles * 0.00001;  // Placeholder

        // Density statistics
        stats.rho_min = Math.min(...snapshot.density);
        stats.rho_max = Math.max(...snapshot.density);
        stats.rho_mean = snapshot.density.reduce((a,b) => a+b, 0) / snapshot.density.length;

        // Temperature statistics
        stats.temp_min = Math.min(...snapshot.temperature);
        stats.temp_max = Math.max(...snapshot.temperature);
        stats.temp_mean = snapshot.temperature.reduce((a,b) => a+b, 0) / snapshot.temperature.length;

        // Total energy (approximate)
        const kinetic = snapshot.velocity_magnitude.reduce((sum, v) => sum + 0.5 * v**2, 0) * 0.00001;
        const internal = snapshot.internal_energy.reduce((a,b) => a+b, 0) * 0.00001;
        stats.total_energy = kinetic + internal;

        return stats;
    }

    exportToJSON(filename = 'tde_sph_snapshots.json') {
        /**
         * Export all loaded snapshots to JSON file.
         *
         * JSON Format:
         * [
         *   {
         *     "time": 0.0,           // Simulation time (code units)
         *     "step": 0,             // Simulation step number
         *     "n_particles": 12000,  // Number of particles
         *     "positions": [...],    // Flat array [x1,y1,z1,x2,y2,z2,...] (3*n_particles)
         *     "density": [...],      // Array of density values (n_particles)
         *     "temperature": [...],  // Array of temperature values (n_particles)
         *     "internal_energy": [...], // Array of internal energy (n_particles)
         *     "velocity_magnitude": [...], // Array of velocity magnitudes (n_particles)
         *     "pressure": [...],     // Array of pressure values (n_particles)
         *     "entropy": [...]       // Array of entropy values (n_particles)
         *   },
         *   ...
         * ]
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
}
