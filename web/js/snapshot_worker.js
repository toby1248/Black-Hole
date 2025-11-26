/**
 * Snapshot Loading Web Worker
 * 
 * Handles HDF5/JSON file parsing in a background thread to prevent
 * blocking the main thread during playback.
 * 
 * Author: TDE-SPH Development Team
 * Date: 2025-11-26
 */

let h5wasm = null;
let h5wasmReady = null;

// Initialize h5wasm on first use
async function ensureH5Wasm() {
    if (h5wasm) return;
    if (h5wasmReady) {
        await h5wasmReady;
        return;
    }
    
    h5wasmReady = new Promise(async (resolve, reject) => {
        try {
            const module = await import('https://cdn.jsdelivr.net/npm/h5wasm@0.7.4/dist/esm/hdf5_hl.js');
            await module.ready;
            h5wasm = module;
            resolve();
        } catch (error) {
            reject(error);
        }
    });
    
    await h5wasmReady;
}

// Parse HDF5 file
function parseHDF5(arrayBuffer, filename) {
    const tempFilename = `worker_${filename}`;
    
    h5wasm.FS.writeFile(tempFilename, new Uint8Array(arrayBuffer));
    const f = new h5wasm.File(tempFilename, 'r');
    
    try {
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
                const attrs = obj.attrs;
                if (attrs && attrs[name] !== undefined) {
                    const val = attrs[name].value;
                    return Array.isArray(val) ? val[0] : val;
                }
            } catch (e) {}
            return null;
        };
        
        // Load particle data
        let positions = getDataset('/PartType0/Coordinates') || 
                       getDataset('/particles/coordinates') ||
                       getDataset('/positions');
        
        let density = getDataset('/PartType0/Density') ||
                     getDataset('/particles/density') ||
                     getDataset('/density');
        
        if (!positions || !density) {
            throw new Error('Missing required data fields');
        }
        
        let temperature = getDataset('/PartType0/Temperature') ||
                         getDataset('/particles/temperature') ||
                         getDataset('/temperature');
        
        let internalEnergy = getDataset('/PartType0/InternalEnergy') ||
                            getDataset('/particles/internal_energy') ||
                            getDataset('/internal_energy');
        
        let velocities = getDataset('/PartType0/Velocities') ||
                        getDataset('/particles/velocities') ||
                        getDataset('/velocities');
        
        let velocityMag = null;
        if (velocities) {
            const n = velocities.length / 3;
            velocityMag = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                const vx = velocities[i * 3];
                const vy = velocities[i * 3 + 1];
                const vz = velocities[i * 3 + 2];
                velocityMag[i] = Math.sqrt(vx*vx + vy*vy + vz*vz);
            }
        }
        
        let pressure = getDataset('/PartType0/Pressure') ||
                      getDataset('/particles/pressure') ||
                      getDataset('/pressure');
        
        let entropy = getDataset('/PartType0/Entropy') ||
                     getDataset('/particles/entropy') ||
                     getDataset('/entropy');
        
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
        
        const n = positions.length / 3;
        
        return {
            time: parseFloat(time),
            step: parseInt(step),
            n_particles: n,
            positions,
            density,
            temperature: temperature || new Float32Array(n),
            internal_energy: internalEnergy || new Float32Array(n),
            velocity_magnitude: velocityMag || new Float32Array(n),
            pressure: pressure || new Float32Array(n),
            entropy: entropy || new Float32Array(n)
        };
    } finally {
        f.close();
        try {
            h5wasm.FS.unlink(tempFilename);
        } catch (e) {}
    }
}

// Parse JSON file
function parseJSON(jsonText, filename) {
    const data = JSON.parse(jsonText);
    
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
    
    const n = data.n_particles || data.positions.length;
    
    return {
        time: data.time || 0,
        step: data.step || 0,
        n_particles: n,
        positions: flattenPositions(data.positions),
        density: new Float32Array(data.density),
        temperature: new Float32Array(data.temperature || data.internal_energy || new Array(n).fill(0)),
        internal_energy: new Float32Array(data.internal_energy || new Array(n).fill(0)),
        velocity_magnitude: new Float32Array(data.velocity_magnitude || new Array(n).fill(0)),
        pressure: new Float32Array(data.pressure || new Array(n).fill(0)),
        entropy: new Float32Array(data.entropy || new Array(n).fill(0))
    };
}

// Handle messages from main thread
self.onmessage = async function(e) {
    const { type, id, data } = e.data;
    
    try {
        if (type === 'loadSnapshot') {
            const { arrayBuffer, filename, isJson, jsonText } = data;
            
            let snapshot;
            if (isJson) {
                snapshot = parseJSON(jsonText, filename);
            } else {
                await ensureH5Wasm();
                snapshot = parseHDF5(arrayBuffer, filename);
            }
            
            // Transfer arrays back to main thread (zero-copy)
            const transferables = [
                snapshot.positions.buffer,
                snapshot.density.buffer,
                snapshot.temperature.buffer,
                snapshot.internal_energy.buffer,
                snapshot.velocity_magnitude.buffer,
                snapshot.pressure.buffer,
                snapshot.entropy.buffer
            ];
            
            self.postMessage({ type: 'snapshotLoaded', id, snapshot }, transferables);
        }
        else if (type === 'computeSplineCoefficients') {
            // Compute cubic spline coefficients for a set of snapshots
            const { snapshotData, particleIndex, dimension } = data;
            
            // snapshotData is array of { time, value } for one particle coordinate
            const n = snapshotData.length;
            const times = snapshotData.map(s => s.time);
            const values = snapshotData.map(s => s.value);
            
            // Natural cubic spline computation
            const h = new Float32Array(n - 1);
            const alpha = new Float32Array(n - 1);
            const l = new Float32Array(n);
            const mu = new Float32Array(n);
            const z = new Float32Array(n);
            const c = new Float32Array(n);
            const b = new Float32Array(n - 1);
            const d = new Float32Array(n - 1);
            
            for (let i = 0; i < n - 1; i++) {
                h[i] = times[i + 1] - times[i];
            }
            
            for (let i = 1; i < n - 1; i++) {
                alpha[i] = (3 / h[i]) * (values[i + 1] - values[i]) - 
                           (3 / h[i - 1]) * (values[i] - values[i - 1]);
            }
            
            l[0] = 1;
            mu[0] = 0;
            z[0] = 0;
            
            for (let i = 1; i < n - 1; i++) {
                l[i] = 2 * (times[i + 1] - times[i - 1]) - h[i - 1] * mu[i - 1];
                mu[i] = h[i] / l[i];
                z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
            }
            
            l[n - 1] = 1;
            z[n - 1] = 0;
            c[n - 1] = 0;
            
            for (let j = n - 2; j >= 0; j--) {
                c[j] = z[j] - mu[j] * c[j + 1];
                b[j] = (values[j + 1] - values[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
                d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
            }
            
            // Return coefficients (a = values[i], b, c, d for each segment)
            const coefficients = {
                particleIndex,
                dimension,
                a: new Float32Array(values),
                b,
                c: new Float32Array(c.slice(0, n - 1)),
                d,
                times: new Float32Array(times)
            };
            
            const transferables = [
                coefficients.a.buffer,
                coefficients.b.buffer,
                coefficients.c.buffer,
                coefficients.d.buffer,
                coefficients.times.buffer
            ];
            
            self.postMessage({ type: 'splineCoefficients', id, coefficients }, transferables);
        }
    } catch (error) {
        self.postMessage({ type: 'error', id, error: error.message });
    }
};
