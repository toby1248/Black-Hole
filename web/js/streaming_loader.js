/**
 * TDE-SPH Streaming Data Loader
 *
 * Memory-efficient data loader that streams snapshots from disk on-demand
 * using an LRU cache instead of loading everything into RAM.
 *
 * Features:
 * - LRU cache with configurable memory limit (default 8GB, max 32GB)
 * - On-demand snapshot loading from File objects via Web Worker (non-blocking)
 * - IndexedDB cache for parsed snapshot data (up to 200GB)
 * - Preloading of nearby snapshots for smooth playback
 * - Precomputed spline coefficients for smooth interpolation
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-11-26
 */

class StreamingDataLoader {
    constructor(options = {}) {
        // Configuration
        this.maxMemoryGB = options.maxMemoryGB || 8;  // RAM cache limit in GB
        this.maxDiskCacheGB = options.maxDiskCacheGB || 200;  // IndexedDB cache limit
        this.preloadAhead = options.preloadAhead || 3;  // Snapshots to preload ahead
        this.preloadBehind = options.preloadBehind || 1;  // Snapshots to preload behind
        
        // File references (don't store data, just File objects)
        this.files = [];  // Array of { file: File, index: number, time: number }
        this.fileMap = new Map();  // filename -> file info
        
        // LRU cache for loaded snapshots
        this.cache = new Map();  // index -> { snapshot, lastAccess, size }
        this.cacheOrder = [];  // LRU order (most recent at end)
        this.currentCacheSize = 0;  // Current cache size in bytes
        this.maxCacheSize = this.maxMemoryGB * 1024 * 1024 * 1024;
        
        // IndexedDB for disk cache
        this.db = null;
        this.dbName = 'tde-sph-streaming-cache';
        this.dbVersion = 2;  // Bumped for spline store
        
        // Web Worker for non-blocking file loading
        this.worker = null;
        this.workerReady = false;
        this.pendingWorkerRequests = new Map();  // id -> { resolve, reject }
        this.workerRequestId = 0;
        
        // Spline cache
        this.splineDb = null;
        this.splineDbName = 'tde-sph-spline-cache';
        this.hasPrecomputedSplines = false;
        this.splineComputeProgress = 0;
        
        // Metadata
        this.snapshotCount = 0;
        this.currentIndex = 0;
        this.isLoaded = false;
        this.isStreaming = true;
        
        // HDF5 wasm loader reference (fallback if worker fails)
        this.h5wasm = null;
        this.h5wasmReady = null;
        
        // Black hole metadata
        this.bhMass = null;
        this.bhSpin = null;
        this.metricType = null;
        
        // Density statistics
        this.density90thPercentile = null;
        
        // Preloading state
        this.preloadQueue = [];
        this.isPreloading = false;
        
        // Statistics
        this.cacheHits = 0;
        this.cacheMisses = 0;
        this.diskCacheHits = 0;
        
        // Initialize worker
        this.initWorker();
    }
    
    /**
     * Initialize the Web Worker for non-blocking file loading.
     */
    initWorker() {
        try {
            // Try to create worker - may fail due to CORS or file:// restrictions
            const workerUrl = new URL('snapshot_worker.js', import.meta.url || window.location.href).href;
            this.worker = new Worker(workerUrl, { type: 'module' });
            
            this.worker.onmessage = (e) => {
                const { type, id, snapshot, coefficients, error } = e.data;
                const pending = this.pendingWorkerRequests.get(id);
                
                if (!pending) return;
                
                this.pendingWorkerRequests.delete(id);
                
                if (type === 'error') {
                    pending.reject(new Error(error));
                } else if (type === 'snapshotLoaded') {
                    pending.resolve(snapshot);
                } else if (type === 'splineCoefficients') {
                    pending.resolve(coefficients);
                }
            };
            
            this.worker.onerror = (e) => {
                console.warn('Worker error:', e);
                this.workerReady = false;
            };
            
            this.workerReady = true;
            console.log('Snapshot worker initialized for non-blocking loading');
        } catch (e) {
            console.warn('Failed to create worker, falling back to main thread:', e);
            this.workerReady = false;
        }
    }
    
    /**
     * Initialize the streaming loader with file list.
     * Only stores file references and extracts metadata.
     */
    async initialize(files, progressCallback = null) {
        console.log(`Initializing streaming loader with ${files.length} files...`);
        
        this.files = [];
        this.fileMap.clear();
        this.cache.clear();
        this.cacheOrder = [];
        this.currentCacheSize = 0;
        
        // Initialize IndexedDB
        await this.initIndexedDB();
        
        // Ensure h5wasm is ready
        await this.ensureH5Wasm();
        
        // Extract metadata from each file (time, particle count) without loading full data
        let processed = 0;
        const fileInfos = [];
        
        for (const file of files) {
            try {
                const ext = file.name.toLowerCase();
                if (!ext.endsWith('.h5') && !ext.endsWith('.hdf5') && !ext.endsWith('.json')) {
                    continue;
                }
                
                // Extract just the metadata (time, step, n_particles)
                const metadata = await this.extractMetadata(file);
                
                fileInfos.push({
                    file: file,
                    name: file.name,
                    time: metadata.time,
                    step: metadata.step,
                    n_particles: metadata.n_particles,
                    size: file.size
                });
                
            } catch (error) {
                console.warn(`Error extracting metadata from ${file.name}:`, error);
            }
            
            processed++;
            if (progressCallback) {
                progressCallback(Math.round((processed / files.length) * 50));  // First 50%
            }
        }
        
        // Sort by time
        fileInfos.sort((a, b) => a.time - b.time);
        
        // Assign indices and store
        for (let i = 0; i < fileInfos.length; i++) {
            fileInfos[i].index = i;
            this.files.push(fileInfos[i]);
            this.fileMap.set(fileInfos[i].name, fileInfos[i]);
        }
        
        this.snapshotCount = this.files.length;
        this.currentIndex = 0;
        this.isLoaded = this.snapshotCount > 0;
        
        // Load first snapshot to get density percentile
        if (this.isLoaded) {
            const firstSnapshot = await this.getSnapshot(0);
            if (firstSnapshot) {
                this.computeDensityPercentile(firstSnapshot);
                
                // Extract BH parameters
                this.extractBHParams(firstSnapshot);
            }
            
            if (progressCallback) {
                progressCallback(100);
            }
        }
        
        console.log(`Streaming loader initialized: ${this.snapshotCount} snapshots`);
        console.log(`Memory limit: ${this.maxMemoryGB}GB, Disk cache: ${this.maxDiskCacheGB}GB`);
        
        return this.snapshotCount;
    }
    
    /**
     * Extract metadata from a file without loading full particle data.
     */
    async extractMetadata(file) {
        const ext = file.name.toLowerCase();
        
        if (ext.endsWith('.json')) {
            // For JSON, we need to parse the whole file unfortunately
            const text = await file.text();
            const data = JSON.parse(text);
            return {
                time: data.time || 0,
                step: data.step || 0,
                n_particles: data.n_particles || (data.positions ? data.positions.length : 0)
            };
        }
        
        // HDF5: Read only attributes/metadata
        const arrayBuffer = await file.arrayBuffer();
        const filename = `meta_${file.name}`;
        
        this.h5wasm.FS.writeFile(filename, new Uint8Array(arrayBuffer));
        const f = new this.h5wasm.File(filename, 'r');
        
        let time = 0;
        let step = 0;
        let n_particles = 0;
        
        try {
            // Try to get time from root attributes
            const root = f.get('/');
            
            const getAttr = (obj, name) => {
                try {
                    if (obj.attrs && obj.attrs[name]) {
                        const val = obj.attrs[name].value;
                        return Array.isArray(val) ? val[0] : val;
                    }
                } catch (e) {}
                return null;
            };
            
            time = getAttr(root, 'time') || 0;
            step = getAttr(root, 'step') || getAttr(root, 'iteration') || 0;
            
            // Try metadata group
            try {
                const meta = f.get('/metadata');
                if (meta) {
                    time = getAttr(meta, 'time') || time;
                    step = getAttr(meta, 'step') || step;
                    
                    // BH parameters
                    this.bhMass = this.bhMass || getAttr(meta, 'bh_mass');
                    this.bhSpin = this.bhSpin || getAttr(meta, 'bh_spin');
                    this.metricType = this.metricType || getAttr(meta, 'metric_type');
                }
            } catch (e) {}
            
            // Get particle count from positions dataset shape
            try {
                const posDset = f.get('/particles/positions') || f.get('particles/positions');
                if (posDset && posDset.shape) {
                    n_particles = posDset.shape[0];
                }
            } catch (e) {}
            
            // Parse time from filename as fallback
            if (time === 0) {
                const match = file.name.match(/snapshot_(\d+)/);
                if (match) {
                    step = parseInt(match[1]);
                    time = step * 0.1;  // Assume dt=0.1
                }
            }
            
        } finally {
            f.close();
            try {
                this.h5wasm.FS.unlink(filename);
            } catch (e) {}
        }
        
        return { time, step, n_particles };
    }
    
    /**
     * Get a snapshot by index, loading from cache, disk, or file.
     * Uses Web Worker for non-blocking file loading.
     */
    async getSnapshot(index) {
        if (index < 0 || index >= this.snapshotCount) {
            return null;
        }
        
        // Check RAM cache first
        if (this.cache.has(index)) {
            this.cacheHits++;
            this.updateCacheAccess(index);
            return this.cache.get(index).snapshot;
        }
        
        this.cacheMisses++;
        
        // Check IndexedDB cache
        const diskCached = await this.loadFromDiskCache(index);
        if (diskCached) {
            this.diskCacheHits++;
            this.addToCache(index, diskCached);
            return diskCached;
        }
        
        // Load from file using worker (non-blocking)
        const fileInfo = this.files[index];
        if (!fileInfo) {
            return null;
        }
        
        try {
            const snapshot = await this.loadSnapshotNonBlocking(fileInfo.file);
            snapshot.index = index;
            
            // Add to caches
            this.addToCache(index, snapshot);
            await this.saveToDiskCache(index, snapshot);
            
            // Trigger preloading of nearby snapshots
            this.preloadNearby(index);
            
            return snapshot;
        } catch (error) {
            console.error(`Error loading snapshot ${index}:`, error);
            return null;
        }
    }
    
    /**
     * Load snapshot using Web Worker (non-blocking).
     * Falls back to main thread if worker unavailable.
     */
    async loadSnapshotNonBlocking(file) {
        const ext = file.name.toLowerCase();
        const isJson = ext.endsWith('.json');
        
        if (this.workerReady && this.worker) {
            return new Promise(async (resolve, reject) => {
                const id = ++this.workerRequestId;
                this.pendingWorkerRequests.set(id, { resolve, reject });
                
                try {
                    if (isJson) {
                        const jsonText = await file.text();
                        this.worker.postMessage({
                            type: 'loadSnapshot',
                            id,
                            data: { isJson: true, jsonText, filename: file.name }
                        });
                    } else {
                        // Read file as ArrayBuffer and transfer to worker
                        const arrayBuffer = await file.arrayBuffer();
                        this.worker.postMessage({
                            type: 'loadSnapshot',
                            id,
                            data: { isJson: false, arrayBuffer, filename: file.name }
                        }, [arrayBuffer]);
                    }
                    
                    // Timeout after 30 seconds
                    setTimeout(() => {
                        if (this.pendingWorkerRequests.has(id)) {
                            this.pendingWorkerRequests.delete(id);
                            reject(new Error('Worker timeout'));
                        }
                    }, 30000);
                } catch (e) {
                    this.pendingWorkerRequests.delete(id);
                    reject(e);
                }
            });
        } else {
            // Fallback to main thread loading
            return this.loadFullSnapshot(file);
        }
    }
    
    /**
     * Load full snapshot data from a file.
     */
    async loadFullSnapshot(file) {
        await this.ensureH5Wasm();
        
        const ext = file.name.toLowerCase();
        
        if (ext.endsWith('.json')) {
            const text = await file.text();
            const data = JSON.parse(text);
            return this.parseJSONSnapshot(data);
        }
        
        // HDF5
        const arrayBuffer = await file.arrayBuffer();
        const filename = `load_${file.name}`;
        
        this.h5wasm.FS.writeFile(filename, new Uint8Array(arrayBuffer));
        const f = new this.h5wasm.File(filename, 'r');
        
        try {
            return this.parseHDF5Snapshot(f, file.name);
        } finally {
            f.close();
            try {
                this.h5wasm.FS.unlink(filename);
            } catch (e) {}
        }
    }
    
    /**
     * Parse HDF5 file into snapshot object.
     */
    parseHDF5Snapshot(f, filename) {
        const getDataset = (path) => {
            try {
                const paths = [path, path.startsWith('/') ? path.slice(1) : '/' + path];
                for (const p of paths) {
                    try {
                        const dset = f.get(p);
                        if (dset && dset.dtype) {
                            const data = dset.value;
                            if (data instanceof Float32Array) return data;
                            if (data instanceof Float64Array) {
                                const f32 = new Float32Array(data.length);
                                for (let i = 0; i < data.length; i++) f32[i] = data[i];
                                return f32;
                            }
                            if (Array.isArray(data)) return new Float32Array(data.flat());
                            return new Float32Array(data);
                        }
                    } catch (e) {}
                }
            } catch (e) {}
            return null;
        };
        
        const getAttr = (obj, name) => {
            try {
                if (obj.attrs && obj.attrs[name]) {
                    const val = obj.attrs[name].value;
                    return Array.isArray(val) ? val[0] : val;
                }
            } catch (e) {}
            return null;
        };
        
        const positions = getDataset('/particles/positions');
        const density = getDataset('/particles/density');
        const internalEnergy = getDataset('/particles/internal_energy');
        const velocities = getDataset('/particles/velocities');
        
        if (!positions || !density) {
            throw new Error('Required datasets missing in HDF5 file');
        }
        
        // Compute derived fields
        let velocityMag = getDataset('/particles/velocity_magnitude');
        if (!velocityMag && velocities) {
            velocityMag = new Float32Array(velocities.length / 3);
            for (let i = 0; i < velocityMag.length; i++) {
                const vx = velocities[i * 3], vy = velocities[i * 3 + 1], vz = velocities[i * 3 + 2];
                velocityMag[i] = Math.sqrt(vx * vx + vy * vy + vz * vz);
            }
        }
        
        const gamma = 5.0 / 3.0;
        let pressure = getDataset('/particles/pressure');
        if (!pressure && density && internalEnergy) {
            pressure = new Float32Array(density.length);
            for (let i = 0; i < density.length; i++) {
                pressure[i] = (gamma - 1) * density[i] * internalEnergy[i];
            }
        }
        
        let temperature = getDataset('/particles/temperature');
        if (!temperature && internalEnergy) {
            temperature = new Float32Array(internalEnergy.length);
            for (let i = 0; i < internalEnergy.length; i++) {
                temperature[i] = internalEnergy[i] * 1000;
            }
        }
        
        let entropy = getDataset('/particles/entropy');
        if (!entropy && pressure && density) {
            entropy = new Float32Array(density.length);
            for (let i = 0; i < density.length; i++) {
                const rho = Math.max(density[i], 1e-30);
                entropy[i] = pressure[i] / Math.pow(rho, gamma);
            }
        }
        
        // Get time/step
        const root = f.get('/');
        let time = getAttr(root, 'time') || 0;
        let step = getAttr(root, 'step') || 0;
        
        try {
            const meta = f.get('/metadata');
            if (meta) {
                time = getAttr(meta, 'time') || time;
                step = getAttr(meta, 'step') || step;
            }
        } catch (e) {}
        
        // Parse from filename as fallback
        if (time === 0) {
            const match = filename.match(/snapshot_(\d+)/);
            if (match) {
                step = parseInt(match[1]);
                time = step * 0.1;
            }
        }
        
        return {
            time: parseFloat(time),
            step: parseInt(step),
            n_particles: positions.length / 3,
            positions,
            density,
            temperature: temperature || new Float32Array(density.length),
            internal_energy: internalEnergy || new Float32Array(density.length),
            velocity_magnitude: velocityMag || new Float32Array(density.length),
            pressure: pressure || new Float32Array(density.length),
            entropy: entropy || new Float32Array(density.length)
        };
    }
    
    /**
     * Parse JSON data into snapshot object.
     */
    parseJSONSnapshot(data) {
        const flattenPositions = (arr) => {
            if (arr[0] && Array.isArray(arr[0])) {
                const result = new Float32Array(arr.length * 3);
                for (let i = 0; i < arr.length; i++) {
                    result[i * 3] = arr[i][0];
                    result[i * 3 + 1] = arr[i][1];
                    result[i * 3 + 2] = arr[i][2];
                }
                return result;
            }
            return new Float32Array(arr);
        };
        
        return {
            time: data.time || 0,
            step: data.step || 0,
            n_particles: data.n_particles || data.positions.length,
            positions: flattenPositions(data.positions),
            density: new Float32Array(data.density),
            temperature: new Float32Array(data.temperature || data.internal_energy),
            internal_energy: new Float32Array(data.internal_energy),
            velocity_magnitude: new Float32Array(data.velocity_magnitude || []),
            pressure: new Float32Array(data.pressure || []),
            entropy: new Float32Array(data.entropy || [])
        };
    }
    
    /**
     * Add snapshot to RAM cache with LRU eviction.
     */
    addToCache(index, snapshot) {
        // Estimate snapshot size in bytes
        const size = this.estimateSnapshotSize(snapshot);
        
        // Evict old entries if needed
        while (this.currentCacheSize + size > this.maxCacheSize && this.cacheOrder.length > 0) {
            const oldestIndex = this.cacheOrder.shift();
            const oldEntry = this.cache.get(oldestIndex);
            if (oldEntry) {
                this.currentCacheSize -= oldEntry.size;
                this.cache.delete(oldestIndex);
            }
        }
        
        // Add new entry
        this.cache.set(index, {
            snapshot,
            lastAccess: Date.now(),
            size
        });
        this.cacheOrder.push(index);
        this.currentCacheSize += size;
    }
    
    /**
     * Update LRU access order for cache entry.
     */
    updateCacheAccess(index) {
        const pos = this.cacheOrder.indexOf(index);
        if (pos !== -1) {
            this.cacheOrder.splice(pos, 1);
            this.cacheOrder.push(index);
        }
        const entry = this.cache.get(index);
        if (entry) {
            entry.lastAccess = Date.now();
        }
    }
    
    /**
     * Estimate memory size of a snapshot in bytes.
     */
    estimateSnapshotSize(snapshot) {
        let size = 0;
        if (snapshot.positions) size += snapshot.positions.byteLength;
        if (snapshot.density) size += snapshot.density.byteLength;
        if (snapshot.temperature) size += snapshot.temperature.byteLength;
        if (snapshot.internal_energy) size += snapshot.internal_energy.byteLength;
        if (snapshot.velocity_magnitude) size += snapshot.velocity_magnitude.byteLength;
        if (snapshot.pressure) size += snapshot.pressure.byteLength;
        if (snapshot.entropy) size += snapshot.entropy.byteLength;
        return size;
    }
    
    /**
     * Initialize IndexedDB for disk caching.
     */
    async initIndexedDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);
            
            request.onerror = () => {
                console.warn('IndexedDB not available, disk cache disabled');
                resolve();
            };
            
            request.onsuccess = () => {
                this.db = request.result;
                console.log('IndexedDB cache initialized');
                resolve();
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Create object store for snapshots
                if (!db.objectStoreNames.contains('snapshots')) {
                    db.createObjectStore('snapshots', { keyPath: 'index' });
                }
                
                // Create object store for metadata
                if (!db.objectStoreNames.contains('metadata')) {
                    db.createObjectStore('metadata', { keyPath: 'key' });
                }
            };
        });
    }
    
    /**
     * Load snapshot from IndexedDB cache.
     */
    async loadFromDiskCache(index) {
        if (!this.db) return null;
        
        return new Promise((resolve) => {
            try {
                const tx = this.db.transaction(['snapshots'], 'readonly');
                const store = tx.objectStore('snapshots');
                const request = store.get(index);
                
                request.onsuccess = () => {
                    if (request.result) {
                        // Reconstruct Float32Arrays from stored data
                        const data = request.result;
                        resolve({
                            time: data.time,
                            step: data.step,
                            n_particles: data.n_particles,
                            positions: new Float32Array(data.positions),
                            density: new Float32Array(data.density),
                            temperature: new Float32Array(data.temperature),
                            internal_energy: new Float32Array(data.internal_energy),
                            velocity_magnitude: new Float32Array(data.velocity_magnitude),
                            pressure: new Float32Array(data.pressure),
                            entropy: new Float32Array(data.entropy)
                        });
                    } else {
                        resolve(null);
                    }
                };
                
                request.onerror = () => resolve(null);
            } catch (e) {
                resolve(null);
            }
        });
    }
    
    /**
     * Save snapshot to IndexedDB cache.
     */
    async saveToDiskCache(index, snapshot) {
        if (!this.db) return;
        
        try {
            const tx = this.db.transaction(['snapshots'], 'readwrite');
            const store = tx.objectStore('snapshots');
            
            // Convert Float32Arrays to regular arrays for storage
            await store.put({
                index,
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
            });
        } catch (e) {
            console.warn('Failed to save to disk cache:', e);
        }
    }
    
    /**
     * Clear IndexedDB cache.
     */
    async clearDiskCache() {
        if (!this.db) return;
        
        try {
            const tx = this.db.transaction(['snapshots'], 'readwrite');
            const store = tx.objectStore('snapshots');
            await store.clear();
            console.log('Disk cache cleared');
        } catch (e) {
            console.warn('Failed to clear disk cache:', e);
        }
    }
    
    /**
     * Preload nearby snapshots in background.
     */
    preloadNearby(currentIndex) {
        if (this.isPreloading) return;
        
        const toPreload = [];
        
        // Add snapshots ahead
        for (let i = 1; i <= this.preloadAhead; i++) {
            const idx = currentIndex + i;
            if (idx < this.snapshotCount && !this.cache.has(idx)) {
                toPreload.push(idx);
            }
        }
        
        // Add snapshots behind
        for (let i = 1; i <= this.preloadBehind; i++) {
            const idx = currentIndex - i;
            if (idx >= 0 && !this.cache.has(idx)) {
                toPreload.push(idx);
            }
        }
        
        if (toPreload.length === 0) return;
        
        this.isPreloading = true;
        
        // Preload in background
        const preloadNext = async () => {
            while (toPreload.length > 0) {
                const idx = toPreload.shift();
                try {
                    await this.getSnapshot(idx);
                } catch (e) {
                    console.warn(`Failed to preload snapshot ${idx}:`, e);
                }
                
                // Small delay to avoid blocking
                await new Promise(r => setTimeout(r, 10));
            }
            this.isPreloading = false;
        };
        
        preloadNext();
    }
    
    /**
     * Compute 90th percentile density from a sample snapshot.
     */
    computeDensityPercentile(snapshot) {
        if (!snapshot || !snapshot.density) {
            this.density90thPercentile = 1e-10;
            return;
        }
        
        // Sample for efficiency
        const sampleSize = Math.min(10000, snapshot.density.length);
        const step = snapshot.density.length / sampleSize;
        const samples = [];
        
        for (let i = 0; i < sampleSize; i++) {
            const idx = Math.floor(i * step);
            const val = snapshot.density[idx];
            if (val > 0 && isFinite(val)) {
                samples.push(val);
            }
        }
        
        samples.sort((a, b) => a - b);
        const idx90 = Math.floor(samples.length * 0.9);
        this.density90thPercentile = samples[idx90] || 1e-10;
        
        console.log(`Density 90th percentile: ${this.density90thPercentile.toExponential(2)}`);
    }
    
    /**
     * Extract BH parameters from snapshot.
     */
    extractBHParams(snapshot) {
        // BH params are extracted during metadata reading
    }
    
    // ========================================================================
    // API compatibility with DataLoader
    // ========================================================================
    
    getCurrentSnapshot() {
        return this.getSnapshotSync(this.currentIndex);
    }
    
    getSnapshotByIndex(index) {
        return this.getSnapshotSync(index);
    }
    
    /**
     * Synchronous snapshot getter (returns cached or null).
     * Use getSnapshot() for async loading.
     */
    getSnapshotSync(index) {
        if (this.cache.has(index)) {
            this.updateCacheAccess(index);
            return this.cache.get(index).snapshot;
        }
        return null;
    }
    
    getSnapshotCount() {
        return this.snapshotCount;
    }
    
    setCurrentIndex(index) {
        if (index >= 0 && index < this.snapshotCount) {
            this.currentIndex = index;
            return true;
        }
        return false;
    }
    
    getDensity90thPercentile() {
        return this.density90thPercentile;
    }
    
    getBHParams() {
        return {
            mass: this.bhMass,
            spin: this.bhSpin,
            metricType: this.metricType
        };
    }
    
    getEventHorizonRadius() {
        if (!this.bhMass) return 1.0;
        const M = this.bhMass;
        const a = this.bhSpin || 0;
        if (this.metricType === 'kerr' && Math.abs(a) > 0) {
            const aMax = Math.min(Math.abs(a), M);
            return M + Math.sqrt(M * M - aMax * aMax);
        }
        return 2 * M;
    }
    
    /**
     * Get snapshot times for interpolation.
     */
    getSnapshotTimes() {
        return this.files.map(f => f.time);
    }
    
    /**
     * Get cache statistics.
     */
    getCacheStats() {
        return {
            cacheHits: this.cacheHits,
            cacheMisses: this.cacheMisses,
            diskCacheHits: this.diskCacheHits,
            cacheSize: this.currentCacheSize,
            maxCacheSize: this.maxCacheSize,
            cachedSnapshots: this.cache.size,
            totalSnapshots: this.snapshotCount,
            hitRate: this.cacheHits / (this.cacheHits + this.cacheMisses) || 0,
            // Aliases for app.js compatibility
            ramUsed: this.currentCacheSize,
            ramLimit: this.maxCacheSize,
            ramEntries: this.cache.size,
            diskEntries: this.diskCacheHits + this.cacheMisses  // Approximate disk entries
        };
    }
    
    /**
     * Set memory limit for RAM cache.
     */
    setMaxMemory(gb) {
        this.maxMemoryGB = gb;
        this.maxCacheSize = gb * 1024 * 1024 * 1024;
        console.log(`RAM cache limit set to ${gb}GB`);
        
        // Evict if over limit
        while (this.currentCacheSize > this.maxCacheSize && this.cacheOrder.length > 0) {
            const oldestIndex = this.cacheOrder.shift();
            const oldEntry = this.cache.get(oldestIndex);
            if (oldEntry) {
                this.currentCacheSize -= oldEntry.size;
                this.cache.delete(oldestIndex);
            }
        }
    }
    
    /**
     * Alternative method name for setting RAM cache size (in bytes).
     */
    setMaxRamCacheSize(bytes) {
        const gb = bytes / (1024 * 1024 * 1024);
        this.setMaxMemory(gb);
    }
    
    /**
     * Get the time range of all snapshots.
     */
    getTimeRange() {
        if (this.files.length === 0) {
            return { min: 0, max: 1 };
        }
        
        let minTime = Infinity;
        let maxTime = -Infinity;
        
        for (const fileInfo of this.files) {
            if (fileInfo.time < minTime) minTime = fileInfo.time;
            if (fileInfo.time > maxTime) maxTime = fileInfo.time;
        }
        
        return { min: minTime, max: maxTime };
    }
    
    /**
     * Find the snapshot index for a given time (binary search).
     */
    findSnapshotIndexForTime(time) {
        if (this.files.length === 0) return 0;
        
        // Binary search for closest snapshot
        let low = 0;
        let high = this.files.length - 1;
        
        while (low < high) {
            const mid = Math.floor((low + high) / 2);
            if (this.files[mid].time < time) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        // Return closest index
        if (low > 0) {
            const prevDiff = Math.abs(this.files[low - 1].time - time);
            const currDiff = Math.abs(this.files[low].time - time);
            if (prevDiff < currDiff) {
                return low - 1;
            }
        }
        
        return Math.max(0, Math.min(this.files.length - 1, low));
    }
    
    /**
     * Clear the IndexedDB disk cache.
     */
    async clearDiskCache() {
        if (this.db) {
            const tx = this.db.transaction('snapshots', 'readwrite');
            const store = tx.objectStore('snapshots');
            await new Promise((resolve, reject) => {
                const request = store.clear();
                request.onsuccess = resolve;
                request.onerror = reject;
            });
            console.log('Disk cache cleared');
        }
    }
    
    /**
     * Alias for loadFilesMetadataOnly for compatibility with app.js.
     */
    async loadFilesMetadataOnly(files, progressCallback = null) {
        return this.initialize(files, progressCallback);
    }
    
    // ========================================================================
    // Spline Precomputation and Caching
    // ========================================================================
    
    /**
     * Estimate the size of precomputed spline data.
     * Returns size in bytes.
     */
    estimateSplineDataSize() {
        if (this.files.length === 0) return 0;
        
        // Get particle count from first snapshot metadata
        const firstFile = this.files[0];
        const nParticles = firstFile.nParticles || 100000;  // Default estimate
        const nSnapshots = this.files.length;
        
        // Each particle needs:
        // - 3 dimensions (x, y, z)
        // - 4 coefficients per segment (a, b, c, d) as Float32 = 4 bytes each
        // - (nSnapshots - 1) segments
        // Plus times array
        
        const coeffsPerDimension = 4 * (nSnapshots - 1) * 4;  // 4 coeffs * segments * 4 bytes
        const timesSize = nSnapshots * 4;  // Float32
        const perParticleSize = 3 * coeffsPerDimension + timesSize;
        
        return nParticles * perParticleSize;
    }
    
    /**
     * Format bytes to human readable string.
     */
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    /**
     * Check if precomputed splines exist in IndexedDB.
     */
    async hasSplineCache() {
        try {
            await this.initSplineDb();
            const tx = this.splineDb.transaction('metadata', 'readonly');
            const store = tx.objectStore('metadata');
            
            return new Promise((resolve) => {
                const request = store.get('splineInfo');
                request.onsuccess = () => {
                    const info = request.result;
                    if (info && info.snapshotCount === this.snapshotCount) {
                        this.hasPrecomputedSplines = true;
                        resolve(true);
                    } else {
                        resolve(false);
                    }
                };
                request.onerror = () => resolve(false);
            });
        } catch (e) {
            return false;
        }
    }
    
    /**
     * Initialize spline IndexedDB.
     */
    async initSplineDb() {
        if (this.splineDb) return;
        
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.splineDbName, 1);
            
            request.onerror = () => reject(request.error);
            
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                
                if (!db.objectStoreNames.contains('coefficients')) {
                    db.createObjectStore('coefficients', { keyPath: 'key' });
                }
                if (!db.objectStoreNames.contains('metadata')) {
                    db.createObjectStore('metadata', { keyPath: 'id' });
                }
            };
            
            request.onsuccess = () => {
                this.splineDb = request.result;
                resolve();
            };
        });
    }
    
    /**
     * Precompute and cache spline coefficients for all particles.
     * Shows warning if estimated size > 100GB.
     * 
     * @param {Function} progressCallback - Called with (progress, message)
     * @param {Function} confirmCallback - Called with (estimatedSize) to confirm large operations
     * @returns {Promise<boolean>} - True if computation completed
     */
    async precomputeSplines(progressCallback = null, confirmCallback = null) {
        const estimatedSize = this.estimateSplineDataSize();
        const estimatedGB = estimatedSize / (1024 * 1024 * 1024);
        
        console.log(`Estimated spline data size: ${this.formatBytes(estimatedSize)}`);
        
        // Warn if > 100GB
        if (estimatedGB > 100) {
            if (confirmCallback) {
                const confirmed = await confirmCallback(estimatedSize);
                if (!confirmed) {
                    console.log('Spline precomputation cancelled by user');
                    return false;
                }
            } else {
                const msg = `Warning: Spline precomputation will generate approximately ${this.formatBytes(estimatedSize)} of data. Continue?`;
                if (!confirm(msg)) {
                    return false;
                }
            }
        }
        
        await this.initSplineDb();
        
        // Clear existing spline data
        const clearTx = this.splineDb.transaction('coefficients', 'readwrite');
        await new Promise((resolve, reject) => {
            const req = clearTx.objectStore('coefficients').clear();
            req.onsuccess = resolve;
            req.onerror = reject;
        });
        
        // Load all snapshots to compute splines
        const snapshots = [];
        
        if (progressCallback) progressCallback(0, 'Loading snapshots for spline computation...');
        
        for (let i = 0; i < this.snapshotCount; i++) {
            const snapshot = await this.getSnapshot(i);
            if (!snapshot) {
                console.error(`Failed to load snapshot ${i} for spline computation`);
                return false;
            }
            snapshots.push(snapshot);
            
            if (progressCallback) {
                const loadProgress = ((i + 1) / this.snapshotCount) * 30;  // 0-30%
                progressCallback(loadProgress, `Loading snapshot ${i + 1}/${this.snapshotCount}...`);
            }
        }
        
        const nParticles = snapshots[0].n_particles;
        const nSnapshots = snapshots.length;
        const times = snapshots.map(s => s.time);
        
        if (progressCallback) progressCallback(30, 'Computing spline coefficients...');
        
        // Compute splines in batches to avoid blocking
        const batchSize = 1000;  // Particles per batch
        const totalBatches = Math.ceil(nParticles / batchSize);
        
        for (let batch = 0; batch < totalBatches; batch++) {
            const startParticle = batch * batchSize;
            const endParticle = Math.min((batch + 1) * batchSize, nParticles);
            
            // Compute coefficients for this batch
            const batchCoeffs = this.computeSplineBatch(snapshots, startParticle, endParticle, times);
            
            // Store in IndexedDB
            const tx = this.splineDb.transaction('coefficients', 'readwrite');
            const store = tx.objectStore('coefficients');
            
            await new Promise((resolve, reject) => {
                const req = store.put({
                    key: `batch_${batch}`,
                    startParticle,
                    endParticle,
                    coefficients: batchCoeffs,
                    times: new Float32Array(times)
                });
                req.onsuccess = resolve;
                req.onerror = reject;
            });
            
            if (progressCallback) {
                const computeProgress = 30 + ((batch + 1) / totalBatches) * 70;  // 30-100%
                progressCallback(computeProgress, `Computing splines: batch ${batch + 1}/${totalBatches}...`);
            }
            
            // Yield to main thread
            await new Promise(r => setTimeout(r, 0));
        }
        
        // Store metadata
        const metaTx = this.splineDb.transaction('metadata', 'readwrite');
        await new Promise((resolve, reject) => {
            const req = metaTx.objectStore('metadata').put({
                id: 'splineInfo',
                snapshotCount: this.snapshotCount,
                nParticles,
                nBatches: totalBatches,
                batchSize,
                times: new Float32Array(times),
                computedAt: Date.now()
            });
            req.onsuccess = resolve;
            req.onerror = reject;
        });
        
        this.hasPrecomputedSplines = true;
        
        if (progressCallback) progressCallback(100, 'Spline computation complete!');
        
        console.log(`Spline precomputation complete: ${nParticles} particles, ${totalBatches} batches`);
        return true;
    }
    
    /**
     * Compute spline coefficients for a batch of particles.
     */
    computeSplineBatch(snapshots, startParticle, endParticle, times) {
        const nSnapshots = snapshots.length;
        const nSegments = nSnapshots - 1;
        const nParticles = endParticle - startParticle;
        
        // Output: coefficients for each particle, dimension, segment
        // Shape: [nParticles][3 dimensions][nSegments][4 coefficients (a,b,c,d)]
        const coeffs = {
            x: { a: [], b: [], c: [], d: [] },
            y: { a: [], b: [], c: [], d: [] },
            z: { a: [], b: [], c: [], d: [] }
        };
        
        // Pre-allocate arrays
        for (const dim of ['x', 'y', 'z']) {
            coeffs[dim].a = new Float32Array(nParticles * nSegments);
            coeffs[dim].b = new Float32Array(nParticles * nSegments);
            coeffs[dim].c = new Float32Array(nParticles * nSegments);
            coeffs[dim].d = new Float32Array(nParticles * nSegments);
        }
        
        // Compute h values (time differences)
        const h = new Float32Array(nSegments);
        for (let i = 0; i < nSegments; i++) {
            h[i] = times[i + 1] - times[i];
        }
        
        // Temporary arrays for spline computation
        const alpha = new Float32Array(nSegments);
        const l = new Float32Array(nSnapshots);
        const mu = new Float32Array(nSnapshots);
        const z = new Float32Array(nSnapshots);
        const cCoeff = new Float32Array(nSnapshots);
        
        // Process each particle
        for (let p = 0; p < nParticles; p++) {
            const particleIdx = startParticle + p;
            
            // Process each dimension
            const dims = ['x', 'y', 'z'];
            for (let d = 0; d < 3; d++) {
                const dimName = dims[d];
                
                // Extract values for this particle/dimension
                const values = new Float32Array(nSnapshots);
                for (let s = 0; s < nSnapshots; s++) {
                    values[s] = snapshots[s].positions[particleIdx * 3 + d];
                }
                
                // Natural cubic spline algorithm
                for (let i = 1; i < nSegments; i++) {
                    alpha[i] = (3 / h[i]) * (values[i + 1] - values[i]) - 
                               (3 / h[i - 1]) * (values[i] - values[i - 1]);
                }
                
                l[0] = 1;
                mu[0] = 0;
                z[0] = 0;
                
                for (let i = 1; i < nSegments; i++) {
                    l[i] = 2 * (times[i + 1] - times[i - 1]) - h[i - 1] * mu[i - 1];
                    mu[i] = h[i] / l[i];
                    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
                }
                
                l[nSnapshots - 1] = 1;
                z[nSnapshots - 1] = 0;
                cCoeff[nSnapshots - 1] = 0;
                
                for (let j = nSnapshots - 2; j >= 0; j--) {
                    cCoeff[j] = z[j] - mu[j] * cCoeff[j + 1];
                }
                
                // Store coefficients
                for (let s = 0; s < nSegments; s++) {
                    const idx = p * nSegments + s;
                    coeffs[dimName].a[idx] = values[s];
                    coeffs[dimName].b[idx] = (values[s + 1] - values[s]) / h[s] - 
                                             h[s] * (cCoeff[s + 1] + 2 * cCoeff[s]) / 3;
                    coeffs[dimName].c[idx] = cCoeff[s];
                    coeffs[dimName].d[idx] = (cCoeff[s + 1] - cCoeff[s]) / (3 * h[s]);
                }
            }
        }
        
        return coeffs;
    }
    
    /**
     * Get interpolated positions at a given time using precomputed splines.
     */
    async getSplineInterpolatedPositions(time) {
        if (!this.hasPrecomputedSplines) {
            return null;
        }
        
        await this.initSplineDb();
        
        // Get metadata
        const metaTx = this.splineDb.transaction('metadata', 'readonly');
        const metaStore = metaTx.objectStore('metadata');
        
        const info = await new Promise((resolve, reject) => {
            const req = metaStore.get('splineInfo');
            req.onsuccess = () => resolve(req.result);
            req.onerror = reject;
        });
        
        if (!info) return null;
        
        const times = info.times;
        const nParticles = info.nParticles;
        const nBatches = info.nBatches;
        const batchSize = info.batchSize;
        
        // Find the segment for this time
        let segmentIdx = 0;
        for (let i = 0; i < times.length - 1; i++) {
            if (time >= times[i] && time < times[i + 1]) {
                segmentIdx = i;
                break;
            }
            if (i === times.length - 2) {
                segmentIdx = i;  // Last segment
            }
        }
        
        const t0 = times[segmentIdx];
        const dt = time - t0;
        
        // Output positions
        const positions = new Float32Array(nParticles * 3);
        
        // Load and interpolate each batch
        const coeffTx = this.splineDb.transaction('coefficients', 'readonly');
        const coeffStore = coeffTx.objectStore('coefficients');
        
        for (let batch = 0; batch < nBatches; batch++) {
            const batchData = await new Promise((resolve, reject) => {
                const req = coeffStore.get(`batch_${batch}`);
                req.onsuccess = () => resolve(req.result);
                req.onerror = reject;
            });
            
            if (!batchData) continue;
            
            const startP = batchData.startParticle;
            const endP = batchData.endParticle;
            const coeffs = batchData.coefficients;
            const nSegments = times.length - 1;
            
            // Interpolate each particle in this batch
            for (let p = startP; p < endP; p++) {
                const localP = p - startP;
                const coeffIdx = localP * nSegments + segmentIdx;
                
                // x
                const ax = coeffs.x.a[coeffIdx];
                const bx = coeffs.x.b[coeffIdx];
                const cx = coeffs.x.c[coeffIdx];
                const dx = coeffs.x.d[coeffIdx];
                positions[p * 3] = ax + bx * dt + cx * dt * dt + dx * dt * dt * dt;
                
                // y
                const ay = coeffs.y.a[coeffIdx];
                const by = coeffs.y.b[coeffIdx];
                const cy = coeffs.y.c[coeffIdx];
                const dy = coeffs.y.d[coeffIdx];
                positions[p * 3 + 1] = ay + by * dt + cy * dt * dt + dy * dt * dt * dt;
                
                // z
                const az = coeffs.z.a[coeffIdx];
                const bz = coeffs.z.b[coeffIdx];
                const cz = coeffs.z.c[coeffIdx];
                const dz = coeffs.z.d[coeffIdx];
                positions[p * 3 + 2] = az + bz * dt + cz * dt * dt + dz * dt * dt * dt;
            }
        }
        
        return positions;
    }
    
    /**
     * Clear the spline cache.
     */
    async clearSplineCache() {
        try {
            await this.initSplineDb();
            
            const coeffTx = this.splineDb.transaction('coefficients', 'readwrite');
            await new Promise((resolve, reject) => {
                const req = coeffTx.objectStore('coefficients').clear();
                req.onsuccess = resolve;
                req.onerror = reject;
            });
            
            const metaTx = this.splineDb.transaction('metadata', 'readwrite');
            await new Promise((resolve, reject) => {
                const req = metaTx.objectStore('metadata').clear();
                req.onsuccess = resolve;
                req.onerror = reject;
            });
            
            this.hasPrecomputedSplines = false;
            console.log('Spline cache cleared');
        } catch (e) {
            console.error('Error clearing spline cache:', e);
        }
    }
    
    /**
     * Ensure h5wasm is loaded.
     */
    async ensureH5Wasm() {
        if (this.h5wasm) return;
        if (this.h5wasmReady) {
            await this.h5wasmReady;
            return;
        }
        
        this.h5wasmReady = new Promise(async (resolve, reject) => {
            try {
                const h5wasmModule = await import('https://cdn.jsdelivr.net/npm/h5wasm@0.7.4/dist/esm/hdf5_hl.js');
                await h5wasmModule.ready;
                this.h5wasm = h5wasmModule;
                console.log('h5wasm loaded for streaming loader');
                resolve();
            } catch (error) {
                console.error('Failed to load h5wasm:', error);
                reject(error);
            }
        });
        
        await this.h5wasmReady;
    }
    
    /**
     * Dispose and clean up resources.
     */
    dispose() {
        this.cache.clear();
        this.cacheOrder = [];
        this.files = [];
        this.fileMap.clear();
        this.currentCacheSize = 0;
        this.isLoaded = false;
        
        if (this.db) {
            this.db.close();
            this.db = null;
        }
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.StreamingDataLoader = StreamingDataLoader;
}
