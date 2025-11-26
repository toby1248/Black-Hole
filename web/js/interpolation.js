/**
 * TDE-SPH Cubic Spline Interpolation Module
 *
 * Provides smooth continuous playback of particle positions, density, and 
 * temperature using cubic spline interpolation, with Web Worker support 
 * for multithreaded computation.
 *
 * For large datasets (>100K particles), falls back to linear interpolation
 * to avoid memory issues.
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-11-25
 */

class SplineInterpolator {
    constructor() {
        // Spline coefficients per particle [a, b, c, d] for each segment
        this.splineCoeffsX = null;
        this.splineCoeffsY = null;
        this.splineCoeffsZ = null;
        // Spline coefficients for density and temperature (for smooth color/size)
        this.splineCoeffsDensity = null;
        this.splineCoeffsTemperature = null;
        this.times = null;
        this.numParticles = 0;
        this.numSegments = 0;
        this.isReady = false;
        this.fieldSplinesReady = false;  // Whether density/temp splines are ready
        
        // Linear fallback for large datasets
        this.useLinearFallback = false;
        this.snapshots = null;  // Reference to snapshots for linear interpolation
        
        // Web Worker for multithreaded computation
        this.worker = null;
        this.workerReady = false;
        this.pendingCallbacks = new Map();
        this.callbackId = 0;
        
        // Try to initialize worker
        this.initWorker();
    }
    
    initWorker() {
        try {
            // Create inline worker from blob
            const workerCode = this.getWorkerCode();
            const blob = new Blob([workerCode], { type: 'application/javascript' });
            const workerUrl = URL.createObjectURL(blob);
            this.worker = new Worker(workerUrl);
            
            this.worker.onmessage = (e) => this.handleWorkerMessage(e);
            this.worker.onerror = (e) => {
                console.warn('Spline worker error:', e.message);
                this.workerReady = false;
                // Reject any pending callbacks so they can fall back to main thread
                for (const [id, callback] of this.pendingCallbacks) {
                    // Signal error by passing null
                    callback(null);
                }
                this.pendingCallbacks.clear();
            };
            
            this.workerReady = true;
            console.log('Spline interpolation worker initialized');
        } catch (error) {
            console.warn('Could not initialize spline worker, using main thread:', error);
            this.workerReady = false;
        }
    }
    
    getWorkerCode() {
        return `
// Cubic Spline Interpolation Web Worker
// Computes spline coefficients and evaluates interpolated positions
// Uses transferable ArrayBuffers for efficient memory transfer

self.onmessage = function(e) {
    const { type, id, data } = e.data;
    
    switch(type) {
        case 'computeSplines':
            const coeffs = computeSplineCoefficients(data.times, data.positionsX, data.positionsY, data.positionsZ, data.numParticles);
            // Transfer ArrayBuffers back instead of cloning (much more memory efficient)
            self.postMessage(
                { type: 'splinesComputed', id, coeffs },
                [coeffs.coeffsX.buffer, coeffs.coeffsY.buffer, coeffs.coeffsZ.buffer]
            );
            break;
            
        case 'interpolate':
            const positions = interpolatePositions(data.t, data.coeffsX, data.coeffsY, data.coeffsZ, data.times, data.numParticles);
            self.postMessage({ type: 'interpolated', id, positions });
            break;
            
        case 'interpolateBatch':
            const batchResult = interpolateBatch(data.startIdx, data.endIdx, data.t, data.coeffsX, data.coeffsY, data.coeffsZ, data.times);
            self.postMessage({ type: 'batchInterpolated', id, result: batchResult });
            break;
    }
};

function computeSplineCoefficients(times, positionsX, positionsY, positionsZ, numParticles) {
    const n = times.length;
    if (n < 2) return null;
    
    const numSegments = n - 1;
    
    // Allocate coefficient arrays [numParticles][numSegments][4]
    // Flatten for transfer: [particle][segment][coeff]
    const coeffsX = new Float32Array(numParticles * numSegments * 4);
    const coeffsY = new Float32Array(numParticles * numSegments * 4);
    const coeffsZ = new Float32Array(numParticles * numSegments * 4);
    
    // Compute spline for each particle
    for (let p = 0; p < numParticles; p++) {
        // Extract this particle's positions at each time
        const yX = new Float32Array(n);
        const yY = new Float32Array(n);
        const yZ = new Float32Array(n);
        
        for (let t = 0; t < n; t++) {
            yX[t] = positionsX[t * numParticles + p];
            yY[t] = positionsY[t * numParticles + p];
            yZ[t] = positionsZ[t * numParticles + p];
        }
        
        // Compute natural cubic spline coefficients
        const cX = naturalCubicSpline(times, yX);
        const cY = naturalCubicSpline(times, yY);
        const cZ = naturalCubicSpline(times, yZ);
        
        // Store coefficients
        const baseIdx = p * numSegments * 4;
        for (let s = 0; s < numSegments; s++) {
            const segIdx = baseIdx + s * 4;
            coeffsX[segIdx] = cX.a[s];
            coeffsX[segIdx + 1] = cX.b[s];
            coeffsX[segIdx + 2] = cX.c[s];
            coeffsX[segIdx + 3] = cX.d[s];
            
            coeffsY[segIdx] = cY.a[s];
            coeffsY[segIdx + 1] = cY.b[s];
            coeffsY[segIdx + 2] = cY.c[s];
            coeffsY[segIdx + 3] = cY.d[s];
            
            coeffsZ[segIdx] = cZ.a[s];
            coeffsZ[segIdx + 1] = cZ.b[s];
            coeffsZ[segIdx + 2] = cZ.c[s];
            coeffsZ[segIdx + 3] = cZ.d[s];
        }
    }
    
    return {
        coeffsX: coeffsX,
        coeffsY: coeffsY,
        coeffsZ: coeffsZ,
        numSegments: numSegments
    };
}

function naturalCubicSpline(x, y) {
    // Natural cubic spline interpolation
    // Returns coefficients a, b, c, d for each segment
    // S_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3
    
    const n = x.length - 1; // number of segments
    
    const a = new Float32Array(n);
    const b = new Float32Array(n);
    const c = new Float32Array(n + 1);
    const d = new Float32Array(n);
    
    const h = new Float32Array(n);
    const alpha = new Float32Array(n);
    const l = new Float32Array(n + 1);
    const mu = new Float32Array(n);
    const z = new Float32Array(n + 1);
    
    // Initialize a with y values
    for (let i = 0; i < n; i++) {
        a[i] = y[i];
    }
    
    // Calculate h (intervals)
    for (let i = 0; i < n; i++) {
        h[i] = x[i + 1] - x[i];
    }
    
    // Calculate alpha
    for (let i = 1; i < n; i++) {
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1]);
    }
    
    // Solve tridiagonal system for c
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;
    
    for (let i = 1; i < n; i++) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    
    l[n] = 1;
    z[n] = 0;
    c[n] = 0;
    
    // Back substitution
    for (let j = n - 1; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }
    
    return { a, b, c: c.slice(0, n), d };
}

function interpolatePositions(t, coeffsX, coeffsY, coeffsZ, times, numParticles) {
    const n = times.length;
    const numSegments = n - 1;
    
    // Find the segment containing t
    let segmentIdx = 0;
    for (let i = 0; i < numSegments; i++) {
        if (t >= times[i] && t <= times[i + 1]) {
            segmentIdx = i;
            break;
        }
        if (t > times[i + 1]) {
            segmentIdx = i;
        }
    }
    
    // Clamp to valid range
    segmentIdx = Math.max(0, Math.min(segmentIdx, numSegments - 1));
    const tLocal = t - times[segmentIdx];
    
    // Interpolate all particles
    const positions = new Float32Array(numParticles * 3);
    
    for (let p = 0; p < numParticles; p++) {
        const baseIdx = p * numSegments * 4 + segmentIdx * 4;
        
        // S(t) = a + b*t + c*t^2 + d*t^3
        const t2 = tLocal * tLocal;
        const t3 = t2 * tLocal;
        
        positions[p * 3] = coeffsX[baseIdx] + coeffsX[baseIdx + 1] * tLocal + 
                           coeffsX[baseIdx + 2] * t2 + coeffsX[baseIdx + 3] * t3;
        positions[p * 3 + 1] = coeffsY[baseIdx] + coeffsY[baseIdx + 1] * tLocal + 
                               coeffsY[baseIdx + 2] * t2 + coeffsY[baseIdx + 3] * t3;
        positions[p * 3 + 2] = coeffsZ[baseIdx] + coeffsZ[baseIdx + 1] * tLocal + 
                               coeffsZ[baseIdx + 2] * t2 + coeffsZ[baseIdx + 3] * t3;
    }
    
    return positions;
}

function interpolateBatch(startIdx, endIdx, t, coeffsX, coeffsY, coeffsZ, times) {
    const numSegments = times.length - 1;
    
    // Find segment
    let segmentIdx = 0;
    for (let i = 0; i < numSegments; i++) {
        if (t >= times[i] && t <= times[i + 1]) {
            segmentIdx = i;
            break;
        }
        if (t > times[i + 1]) segmentIdx = i;
    }
    segmentIdx = Math.max(0, Math.min(segmentIdx, numSegments - 1));
    
    const tLocal = t - times[segmentIdx];
    const t2 = tLocal * tLocal;
    const t3 = t2 * tLocal;
    
    const batchSize = endIdx - startIdx;
    const positions = new Float32Array(batchSize * 3);
    
    for (let i = 0; i < batchSize; i++) {
        const p = startIdx + i;
        const baseIdx = p * numSegments * 4 + segmentIdx * 4;
        
        positions[i * 3] = coeffsX[baseIdx] + coeffsX[baseIdx + 1] * tLocal + 
                           coeffsX[baseIdx + 2] * t2 + coeffsX[baseIdx + 3] * t3;
        positions[i * 3 + 1] = coeffsY[baseIdx] + coeffsY[baseIdx + 1] * tLocal + 
                               coeffsY[baseIdx + 2] * t2 + coeffsY[baseIdx + 3] * t3;
        positions[i * 3 + 2] = coeffsZ[baseIdx] + coeffsZ[baseIdx + 1] * tLocal + 
                               coeffsZ[baseIdx + 2] * t2 + coeffsZ[baseIdx + 3] * t3;
    }
    
    return { startIdx, positions };
}
`;
    }
    
    handleWorkerMessage(e) {
        const { type, id, coeffs, positions, result } = e.data;
        
        if (this.pendingCallbacks.has(id)) {
            const callback = this.pendingCallbacks.get(id);
            this.pendingCallbacks.delete(id);
            
            switch (type) {
                case 'splinesComputed':
                    callback(coeffs);
                    break;
                case 'interpolated':
                    callback(positions);
                    break;
                case 'batchInterpolated':
                    callback(result);
                    break;
            }
        }
    }
    
    /**
     * Build spline coefficients from snapshot data.
     * @param {Array} snapshots - Array of snapshot objects with positions
     * @returns {Promise} Resolves when splines are ready
     */
    async buildSplines(snapshots) {
        if (snapshots.length < 2) {
            console.warn('Need at least 2 snapshots for interpolation');
            return false;
        }
        
        this.numParticles = snapshots[0].n_particles;
        
        // Check if particle count is too high for spline interpolation
        // Splines require numParticles × numSegments × 4 coefficients × 3 axes × 4 bytes
        // For 1M particles × 100 snapshots = 4.8GB - too much for browsers
        const maxParticlesForSplines = 100000; // 100K is reasonable
        if (this.numParticles > maxParticlesForSplines) {
            console.warn(`Particle count (${this.numParticles.toLocaleString()}) exceeds spline limit (${maxParticlesForSplines.toLocaleString()}). Spline interpolation disabled - using linear interpolation.`);
            this.isReady = false;
            this.useLinearFallback = true;
            // Store times for linear interpolation fallback
            this.times = new Float32Array(snapshots.length);
            for (let t = 0; t < snapshots.length; t++) {
                this.times[t] = snapshots[t].time;
            }
            this.snapshots = snapshots; // Store reference for linear interpolation
            return false;
        }
        
        console.log(`Building cubic splines for ${snapshots.length} snapshots...`);
        const startTime = performance.now();
        
        // Extract times and positions
        this.times = new Float32Array(snapshots.length);
        
        const positionsX = new Float32Array(snapshots.length * this.numParticles);
        const positionsY = new Float32Array(snapshots.length * this.numParticles);
        const positionsZ = new Float32Array(snapshots.length * this.numParticles);
        
        for (let t = 0; t < snapshots.length; t++) {
            this.times[t] = snapshots[t].time;
            const positions = snapshots[t].positions;
            
            for (let p = 0; p < this.numParticles; p++) {
                positionsX[t * this.numParticles + p] = positions[p * 3];
                positionsY[t * this.numParticles + p] = positions[p * 3 + 1];
                positionsZ[t * this.numParticles + p] = positions[p * 3 + 2];
            }
        }
        
        // Check if data is too large for worker (>500MB total)
        const estimatedSize = this.numParticles * snapshots.length * 4 * 3; // bytes
        const useWorker = this.workerReady && this.worker && estimatedSize < 500 * 1024 * 1024;
        
        if (useWorker) {
            return new Promise((resolve, reject) => {
                const id = this.callbackId++;
                
                // Set timeout for worker - fall back to main thread if it takes too long or fails
                const timeout = setTimeout(() => {
                    console.warn('Spline worker timeout, falling back to main thread');
                    this.pendingCallbacks.delete(id);
                    this.computeSplinesMainThread(positionsX, positionsY, positionsZ).then(resolve);
                }, 30000);
                
                this.pendingCallbacks.set(id, (coeffs) => {
                    clearTimeout(timeout);
                    this.splineCoeffsX = coeffs.coeffsX;
                    this.splineCoeffsY = coeffs.coeffsY;
                    this.splineCoeffsZ = coeffs.coeffsZ;
                    this.numSegments = coeffs.numSegments;
                    this.isReady = true;
                    
                    const elapsed = performance.now() - startTime;
                    console.log(`Spline computation complete in ${elapsed.toFixed(0)}ms (worker)`);
                    resolve(true);
                });
                
                try {
                    // Use transferable objects for efficient memory transfer to worker
                    this.worker.postMessage({
                        type: 'computeSplines',
                        id,
                        data: {
                            times: this.times,
                            positionsX,
                            positionsY,
                            positionsZ,
                            numParticles: this.numParticles
                        }
                    }, [positionsX.buffer, positionsY.buffer, positionsZ.buffer]);
                } catch (e) {
                    clearTimeout(timeout);
                    console.warn('Worker postMessage failed, falling back to main thread:', e.message);
                    this.pendingCallbacks.delete(id);
                    // Re-extract positions since buffers were transferred
                    const posX2 = new Float32Array(snapshots.length * this.numParticles);
                    const posY2 = new Float32Array(snapshots.length * this.numParticles);
                    const posZ2 = new Float32Array(snapshots.length * this.numParticles);
                    for (let t = 0; t < snapshots.length; t++) {
                        const positions = snapshots[t].positions;
                        for (let p = 0; p < this.numParticles; p++) {
                            posX2[t * this.numParticles + p] = positions[p * 3];
                            posY2[t * this.numParticles + p] = positions[p * 3 + 1];
                            posZ2[t * this.numParticles + p] = positions[p * 3 + 2];
                        }
                    }
                    this.computeSplinesMainThread(posX2, posY2, posZ2).then(resolve);
                }
            });
        } else {
            // Fallback to main thread computation
            return this.computeSplinesMainThread(positionsX, positionsY, positionsZ);
        }
    }
    
    computeSplinesMainThread(positionsX, positionsY, positionsZ) {
        // Natural cubic spline computation on main thread (slower)
        const n = this.times.length;
        this.numSegments = n - 1;
        
        this.splineCoeffsX = new Float32Array(this.numParticles * this.numSegments * 4);
        this.splineCoeffsY = new Float32Array(this.numParticles * this.numSegments * 4);
        this.splineCoeffsZ = new Float32Array(this.numParticles * this.numSegments * 4);
        
        for (let p = 0; p < this.numParticles; p++) {
            const yX = new Float32Array(n);
            const yY = new Float32Array(n);
            const yZ = new Float32Array(n);
            
            for (let t = 0; t < n; t++) {
                yX[t] = positionsX[t * this.numParticles + p];
                yY[t] = positionsY[t * this.numParticles + p];
                yZ[t] = positionsZ[t * this.numParticles + p];
            }
            
            const cX = this.naturalCubicSpline(this.times, yX);
            const cY = this.naturalCubicSpline(this.times, yY);
            const cZ = this.naturalCubicSpline(this.times, yZ);
            
            const baseIdx = p * this.numSegments * 4;
            for (let s = 0; s < this.numSegments; s++) {
                const segIdx = baseIdx + s * 4;
                this.splineCoeffsX[segIdx] = cX.a[s];
                this.splineCoeffsX[segIdx + 1] = cX.b[s];
                this.splineCoeffsX[segIdx + 2] = cX.c[s];
                this.splineCoeffsX[segIdx + 3] = cX.d[s];
                
                this.splineCoeffsY[segIdx] = cY.a[s];
                this.splineCoeffsY[segIdx + 1] = cY.b[s];
                this.splineCoeffsY[segIdx + 2] = cY.c[s];
                this.splineCoeffsY[segIdx + 3] = cY.d[s];
                
                this.splineCoeffsZ[segIdx] = cZ.a[s];
                this.splineCoeffsZ[segIdx + 1] = cZ.b[s];
                this.splineCoeffsZ[segIdx + 2] = cZ.c[s];
                this.splineCoeffsZ[segIdx + 3] = cZ.d[s];
            }
        }
        
        this.isReady = true;
        return Promise.resolve(true);
    }
    
    naturalCubicSpline(x, y) {
        const n = x.length - 1;
        
        const a = new Float32Array(n);
        const b = new Float32Array(n);
        const c = new Float32Array(n + 1);
        const d = new Float32Array(n);
        
        const h = new Float32Array(n);
        const alpha = new Float32Array(n);
        const l = new Float32Array(n + 1);
        const mu = new Float32Array(n);
        const z = new Float32Array(n + 1);
        
        for (let i = 0; i < n; i++) {
            a[i] = y[i];
            h[i] = x[i + 1] - x[i];
        }
        
        for (let i = 1; i < n; i++) {
            alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1]);
        }
        
        l[0] = 1;
        mu[0] = 0;
        z[0] = 0;
        
        for (let i = 1; i < n; i++) {
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        
        l[n] = 1;
        z[n] = 0;
        c[n] = 0;
        
        for (let j = n - 1; j >= 0; j--) {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
            d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        }
        
        return { a, b, c: c.slice(0, n), d };
    }
    
    /**
     * Get interpolated positions at arbitrary time t
     * @param {number} t - Time value
     * @returns {Float32Array} Interpolated positions
     */
    interpolate(t) {
        // Use linear fallback for large datasets
        if (this.useLinearFallback) {
            return this.interpolateLinear(t);
        }
        
        if (!this.isReady) {
            console.warn('Splines not ready');
            return null;
        }
        
        // Clamp t to valid range
        const tMin = this.times[0];
        const tMax = this.times[this.times.length - 1];
        t = Math.max(tMin, Math.min(tMax, t));
        
        // Find segment
        let segmentIdx = 0;
        for (let i = 0; i < this.numSegments; i++) {
            if (t >= this.times[i] && t <= this.times[i + 1]) {
                segmentIdx = i;
                break;
            }
            if (t > this.times[i + 1]) {
                segmentIdx = i;
            }
        }
        
        const tLocal = t - this.times[segmentIdx];
        const t2 = tLocal * tLocal;
        const t3 = t2 * tLocal;
        
        const positions = new Float32Array(this.numParticles * 3);
        
        for (let p = 0; p < this.numParticles; p++) {
            const baseIdx = p * this.numSegments * 4 + segmentIdx * 4;
            
            positions[p * 3] = this.splineCoeffsX[baseIdx] + 
                               this.splineCoeffsX[baseIdx + 1] * tLocal + 
                               this.splineCoeffsX[baseIdx + 2] * t2 + 
                               this.splineCoeffsX[baseIdx + 3] * t3;
            positions[p * 3 + 1] = this.splineCoeffsY[baseIdx] + 
                                   this.splineCoeffsY[baseIdx + 1] * tLocal + 
                                   this.splineCoeffsY[baseIdx + 2] * t2 + 
                                   this.splineCoeffsY[baseIdx + 3] * t3;
            positions[p * 3 + 2] = this.splineCoeffsZ[baseIdx] + 
                                   this.splineCoeffsZ[baseIdx + 1] * tLocal + 
                                   this.splineCoeffsZ[baseIdx + 2] * t2 + 
                                   this.splineCoeffsZ[baseIdx + 3] * t3;
        }
        
        return positions;
    }
    
    /**
     * Linear interpolation fallback for large datasets.
     * Much lower memory usage than cubic splines.
     * @param {number} t - Time value
     * @returns {Float32Array} Interpolated positions
     */
    interpolateLinear(t) {
        if (!this.snapshots || this.snapshots.length === 0) {
            return null;
        }
        
        // Clamp t to valid range
        const tMin = this.times[0];
        const tMax = this.times[this.times.length - 1];
        t = Math.max(tMin, Math.min(tMax, t));
        
        // Find bounding snapshots
        let idx0 = 0;
        for (let i = 0; i < this.times.length - 1; i++) {
            if (t >= this.times[i] && t <= this.times[i + 1]) {
                idx0 = i;
                break;
            }
            if (t > this.times[i + 1]) {
                idx0 = i;
            }
        }
        const idx1 = Math.min(idx0 + 1, this.times.length - 1);
        
        // Edge case: same snapshot
        if (idx0 === idx1) {
            return new Float32Array(this.snapshots[idx0].positions);
        }
        
        // Compute interpolation factor
        const t0 = this.times[idx0];
        const t1 = this.times[idx1];
        const alpha = (t - t0) / (t1 - t0);
        
        const pos0 = this.snapshots[idx0].positions;
        const pos1 = this.snapshots[idx1].positions;
        const positions = new Float32Array(this.numParticles * 3);
        
        // Linear interpolation: pos = pos0 * (1-alpha) + pos1 * alpha
        const oneMinusAlpha = 1 - alpha;
        for (let i = 0; i < positions.length; i++) {
            positions[i] = pos0[i] * oneMinusAlpha + pos1[i] * alpha;
        }
        
        return positions;
    }
    
    /**
     * Find the closest snapshot index for a given time
     * @param {number} t - Time value
     * @returns {number} Index of closest snapshot
     */
    findClosestSnapshotIndex(t) {
        if (!this.times || this.times.length === 0) return 0;
        
        let closest = 0;
        let minDiff = Math.abs(this.times[0] - t);
        
        for (let i = 1; i < this.times.length; i++) {
            const diff = Math.abs(this.times[i] - t);
            if (diff < minDiff) {
                minDiff = diff;
                closest = i;
            }
        }
        
        return closest;
    }
    
    /**
     * Get time range
     * @returns {Object} { min, max } time values
     */
    getTimeRange() {
        if (!this.times || this.times.length === 0) {
            return { min: 0, max: 0 };
        }
        return {
            min: this.times[0],
            max: this.times[this.times.length - 1]
        };
    }
    
    /**
     * Build spline coefficients for density and temperature fields.
     * This enables smooth interpolation of colors and sizes during playback.
     * @param {Array} snapshots - Array of snapshot objects with density/temperature
     * @returns {Promise} Resolves when field splines are ready
     */
    async buildFieldSplines(snapshots) {
        if (snapshots.length < 2) {
            console.warn('Need at least 2 snapshots for field interpolation');
            return false;
        }
        
        console.log(`Building field splines (density/temperature) for ${snapshots.length} snapshots...`);
        const startTime = performance.now();
        
        const n = snapshots.length;
        const numParticles = snapshots[0].n_particles;
        
        // Extract density and temperature for all snapshots
        const densityData = new Float32Array(n * numParticles);
        const temperatureData = new Float32Array(n * numParticles);
        
        for (let t = 0; t < n; t++) {
            const snapshot = snapshots[t];
            for (let p = 0; p < numParticles; p++) {
                // Apply log transform for smoother interpolation of density
                const rho = Math.max(snapshot.density[p], 1e-30);
                densityData[t * numParticles + p] = Math.log10(rho);
                
                // Temperature (already linear scale)
                temperatureData[t * numParticles + p] = snapshot.temperature ? 
                    snapshot.temperature[p] : snapshot.internal_energy[p];
            }
        }
        
        // Compute splines on main thread (for now - can add worker later)
        this.splineCoeffsDensity = new Float32Array(numParticles * this.numSegments * 4);
        this.splineCoeffsTemperature = new Float32Array(numParticles * this.numSegments * 4);
        
        for (let p = 0; p < numParticles; p++) {
            const yDensity = new Float32Array(n);
            const yTemperature = new Float32Array(n);
            
            for (let t = 0; t < n; t++) {
                yDensity[t] = densityData[t * numParticles + p];
                yTemperature[t] = temperatureData[t * numParticles + p];
            }
            
            const cDensity = this.naturalCubicSpline(this.times, yDensity);
            const cTemperature = this.naturalCubicSpline(this.times, yTemperature);
            
            const baseIdx = p * this.numSegments * 4;
            for (let s = 0; s < this.numSegments; s++) {
                const segIdx = baseIdx + s * 4;
                
                this.splineCoeffsDensity[segIdx] = cDensity.a[s];
                this.splineCoeffsDensity[segIdx + 1] = cDensity.b[s];
                this.splineCoeffsDensity[segIdx + 2] = cDensity.c[s];
                this.splineCoeffsDensity[segIdx + 3] = cDensity.d[s];
                
                this.splineCoeffsTemperature[segIdx] = cTemperature.a[s];
                this.splineCoeffsTemperature[segIdx + 1] = cTemperature.b[s];
                this.splineCoeffsTemperature[segIdx + 2] = cTemperature.c[s];
                this.splineCoeffsTemperature[segIdx + 3] = cTemperature.d[s];
            }
        }
        
        this.fieldSplinesReady = true;
        const elapsed = performance.now() - startTime;
        console.log(`Field spline computation complete in ${elapsed.toFixed(0)}ms`);
        return true;
    }
    
    /**
     * Interpolate density field at arbitrary time t
     * @param {number} t - Time value
     * @returns {Float32Array} Interpolated density values
     */
    interpolateDensity(t) {
        if (!this.fieldSplinesReady || !this.splineCoeffsDensity) {
            return null;
        }
        
        // Clamp t to valid range
        const tMin = this.times[0];
        const tMax = this.times[this.times.length - 1];
        t = Math.max(tMin, Math.min(tMax, t));
        
        // Find segment
        let segmentIdx = 0;
        for (let i = 0; i < this.numSegments; i++) {
            if (t >= this.times[i] && t <= this.times[i + 1]) {
                segmentIdx = i;
                break;
            }
            if (t > this.times[i + 1]) segmentIdx = i;
        }
        
        const tLocal = t - this.times[segmentIdx];
        const t2 = tLocal * tLocal;
        const t3 = t2 * tLocal;
        
        const density = new Float32Array(this.numParticles);
        
        for (let p = 0; p < this.numParticles; p++) {
            const baseIdx = p * this.numSegments * 4 + segmentIdx * 4;
            
            // Interpolate in log space, then convert back
            const logRho = this.splineCoeffsDensity[baseIdx] + 
                          this.splineCoeffsDensity[baseIdx + 1] * tLocal + 
                          this.splineCoeffsDensity[baseIdx + 2] * t2 + 
                          this.splineCoeffsDensity[baseIdx + 3] * t3;
            
            density[p] = Math.pow(10, logRho);
        }
        
        return density;
    }
    
    /**
     * Interpolate temperature field at arbitrary time t
     * @param {number} t - Time value
     * @returns {Float32Array} Interpolated temperature values
     */
    interpolateTemperature(t) {
        if (!this.fieldSplinesReady || !this.splineCoeffsTemperature) {
            return null;
        }
        
        // Clamp t to valid range
        const tMin = this.times[0];
        const tMax = this.times[this.times.length - 1];
        t = Math.max(tMin, Math.min(tMax, t));
        
        // Find segment
        let segmentIdx = 0;
        for (let i = 0; i < this.numSegments; i++) {
            if (t >= this.times[i] && t <= this.times[i + 1]) {
                segmentIdx = i;
                break;
            }
            if (t > this.times[i + 1]) segmentIdx = i;
        }
        
        const tLocal = t - this.times[segmentIdx];
        const t2 = tLocal * tLocal;
        const t3 = t2 * tLocal;
        
        const temperature = new Float32Array(this.numParticles);
        
        for (let p = 0; p < this.numParticles; p++) {
            const baseIdx = p * this.numSegments * 4 + segmentIdx * 4;
            
            temperature[p] = this.splineCoeffsTemperature[baseIdx] + 
                            this.splineCoeffsTemperature[baseIdx + 1] * tLocal + 
                            this.splineCoeffsTemperature[baseIdx + 2] * t2 + 
                            this.splineCoeffsTemperature[baseIdx + 3] * t3;
        }
        
        return temperature;
    }
    
    /**
     * Clean up resources
     */
    dispose() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
        this.splineCoeffsX = null;
        this.splineCoeffsY = null;
        this.splineCoeffsZ = null;
        this.splineCoeffsDensity = null;
        this.splineCoeffsTemperature = null;
        this.times = null;
        this.isReady = false;
        this.fieldSplinesReady = false;
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.SplineInterpolator = SplineInterpolator;
}
