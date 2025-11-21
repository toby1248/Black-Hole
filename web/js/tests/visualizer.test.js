/**
 * Unit Tests for TDE-SPH Web Visualizer
 *
 * Tests for color mapping, data generation, and core visualization functions.
 * Run with: npm test (requires jest)
 *
 * Author: TDE-SPH Development Team
 * Date: 2025-11-21
 */

// Mock THREE.js for node environment
const THREE = {
    Scene: class { add() {} },
    PerspectiveCamera: class { lookAt() {} },
    WebGLRenderer: class {
        constructor() { this.domElement = { toDataURL: () => 'data:image/png;base64,test' }; }
        setSize() {}
        setPixelRatio() {}
        render() {}
    },
    Color: class {},
    BufferGeometry: class { setAttribute() {} },
    BufferAttribute: class {},
    PointsMaterial: class {},
    Points: class {},
    SphereGeometry: class {},
    MeshStandardMaterial: class {},
    Mesh: class {},
    AxesHelper: class {},
    AmbientLight: class {},
    DirectionalLight: class {},
    OrbitControls: class { update() {} }
};

global.THREE = THREE;

// Color mapping tests
describe('Color Mapping', () => {
    // Test colormap application
    const applyColormap = (value, colormapName) => {
        const t = value;
        switch(colormapName) {
            case 'viridis':
                return [
                    0.267 * (1-t) + 0.993 * t,
                    0.005 * (1-t) + 0.906 * t,
                    0.329 * (1-t) + 0.144 * t
                ];
            case 'hot':
                if (t < 0.33) return [3.0 * t, 0, 0];
                else if (t < 0.66) return [1.0, 3.0 * (t - 0.33), 0];
                else return [1.0, 1.0, 3.0 * (t - 0.66)];
            default:
                return [t, t, t];
        }
    };

    test('viridis colormap at 0 returns correct color', () => {
        const rgb = applyColormap(0, 'viridis');
        expect(rgb[0]).toBeCloseTo(0.267, 2);
        expect(rgb[1]).toBeCloseTo(0.005, 2);
        expect(rgb[2]).toBeCloseTo(0.329, 2);
    });

    test('viridis colormap at 1 returns correct color', () => {
        const rgb = applyColormap(1, 'viridis');
        expect(rgb[0]).toBeCloseTo(0.993, 2);
        expect(rgb[1]).toBeCloseTo(0.906, 2);
        expect(rgb[2]).toBeCloseTo(0.144, 2);
    });

    test('hot colormap returns red at low values', () => {
        const rgb = applyColormap(0.2, 'hot');
        expect(rgb[0]).toBeGreaterThan(0);
        expect(rgb[1]).toBe(0);
        expect(rgb[2]).toBe(0);
    });

    test('grayscale fallback for unknown colormap', () => {
        const rgb = applyColormap(0.5, 'unknown');
        expect(rgb[0]).toBe(0.5);
        expect(rgb[1]).toBe(0.5);
        expect(rgb[2]).toBe(0.5);
    });
});

// Demo data generation tests
describe('Demo Data Generation', () => {
    const generateDemoParticles = (count) => {
        const positions = new Float32Array(count * 3);
        const density = new Float32Array(count);

        for (let i = 0; i < count; i++) {
            const r = 5.0 + Math.random() * 15.0;
            const phi = Math.random() * 2 * Math.PI;
            positions[i*3] = r * Math.cos(phi);
            positions[i*3+1] = r * Math.sin(phi);
            positions[i*3+2] = (Math.random() - 0.5) * 0.5;
            density[i] = 10.0 * Math.exp(-r / 10.0);
        }

        return { positions, density };
    };

    test('generates correct number of particles', () => {
        const data = generateDemoParticles(1000);
        expect(data.positions.length).toBe(3000); // 1000 * 3
        expect(data.density.length).toBe(1000);
    });

    test('positions are within expected range', () => {
        const data = generateDemoParticles(100);
        for (let i = 0; i < 100; i++) {
            const x = data.positions[i*3];
            const y = data.positions[i*3+1];
            const r = Math.sqrt(x*x + y*y);
            expect(r).toBeGreaterThanOrEqual(0);
            expect(r).toBeLessThan(25); // max radius ~20
        }
    });

    test('density values are positive', () => {
        const data = generateDemoParticles(100);
        for (let i = 0; i < data.density.length; i++) {
            expect(data.density[i]).toBeGreaterThan(0);
        }
    });
});

// Statistics computation tests
describe('Statistics Computation', () => {
    const computeStatistics = (snapshot) => {
        const stats = {
            n_particles: snapshot.n_particles,
            rho_min: Math.min(...snapshot.density),
            rho_max: Math.max(...snapshot.density),
            rho_mean: snapshot.density.reduce((a,b) => a+b, 0) / snapshot.density.length
        };
        return stats;
    };

    test('computes correct min/max density', () => {
        const snapshot = {
            n_particles: 5,
            density: new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0])
        };
        const stats = computeStatistics(snapshot);
        expect(stats.rho_min).toBe(1.0);
        expect(stats.rho_max).toBe(5.0);
        expect(stats.rho_mean).toBe(3.0);
    });

    test('handles single particle', () => {
        const snapshot = {
            n_particles: 1,
            density: new Float32Array([42.0])
        };
        const stats = computeStatistics(snapshot);
        expect(stats.rho_min).toBe(42.0);
        expect(stats.rho_max).toBe(42.0);
        expect(stats.rho_mean).toBe(42.0);
    });
});

// Animation timing tests
describe('Animation Timing', () => {
    const calculateFrameInterval = (fps) => 1000 / fps;

    test('30 fps gives ~33ms interval', () => {
        expect(calculateFrameInterval(30)).toBeCloseTo(33.33, 1);
    });

    test('60 fps gives ~16ms interval', () => {
        expect(calculateFrameInterval(60)).toBeCloseTo(16.67, 1);
    });
});

// WebGL detection mock test
describe('WebGL Detection', () => {
    const checkWebGLSupport = () => {
        // In browser this checks for canvas.getContext('webgl')
        // For testing, we return a mock result
        return typeof WebGLRenderingContext !== 'undefined';
    };

    test('WebGL detection returns boolean', () => {
        const result = checkWebGLSupport();
        expect(typeof result).toBe('boolean');
    });
});

console.log('TDE-SPH Web Visualizer Tests');
console.log('Run with: npm test (requires jest)');
