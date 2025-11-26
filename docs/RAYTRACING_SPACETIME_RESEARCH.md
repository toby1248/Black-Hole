# Research Report: Raytracing and Spacetime Curvature Rendering for Black Hole Visualization

## Executive Summary

This report investigates state-of-the-art techniques for rendering gravitational lensing, spacetime curvature, and volumetric effects in WebGL/Three.js environments. The goal is to identify potential enhancements for the TDE-SPH Web Visualizer beyond the current particle-based approach.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Gravitational Lensing Approaches](#2-gravitational-lensing-approaches)
3. [Key Open-Source Implementations](#3-key-open-source-implementations)
4. [Technical Deep Dive: Schwarzschild Geodesic Raytracing](#4-technical-deep-dive-schwarzschild-geodesic-raytracing)
5. [Volumetric Rendering Techniques](#5-volumetric-rendering-techniques)
6. [Performance Considerations](#6-performance-considerations)
7. [Integration Recommendations for TDE-SPH](#7-integration-recommendations-for-tde-sph)
8. [References](#8-references)

---

## 1. Introduction

Rendering black holes with physically accurate gravitational lensing requires solving the geodesic equations of General Relativity. Light rays near a black hole follow curved paths determined by the spacetime metric (Schwarzschild for non-rotating, Kerr for rotating black holes).

### Current TDE-SPH Approach
The current visualizer uses:
- **InstancedMesh** for particle rendering (1M+ particles at 30+ fps)
- **Emissive bloom** for visual enhancement
- **Blackbody colormap** for temperature-based coloring

### Enhancement Goals
- Add gravitational lensing effects around the central black hole
- Render spacetime curvature visualization
- Potentially integrate volumetric gas/accretion disk rendering

---

## 2. Gravitational Lensing Approaches

### 2.1 Post-Processing Screen-Space Distortion
**Concept**: Apply a distortion shader as a post-processing pass that warps the rendered scene based on the black hole position.

**Pros**:
- Simple to implement
- Low performance overhead
- Works with existing particle rendering

**Cons**:
- Not physically accurate
- Cannot show Einstein rings or multiple images
- Doesn't handle light bending around the back of the black hole

**Implementation**:
```glsl
// Simple gravitational lens distortion (screen-space)
vec2 distortUV(vec2 uv, vec2 bhPos, float mass) {
    vec2 delta = uv - bhPos;
    float r = length(delta);
    float rs = 2.0 * mass; // Schwarzschild radius
    float deflection = rs / (r * r);
    return uv + normalize(delta) * deflection;
}
```

### 2.2 Full Geodesic Ray Tracing
**Concept**: For each pixel, trace light rays backward from the camera, integrating the geodesic equations to determine where the light originated.

**Pros**:
- Physically accurate gravitational lensing
- Shows Einstein rings, multiple images, photon sphere effects
- Can render accretion disks with Doppler/beaming effects

**Cons**:
- Computationally expensive
- Requires many integration steps per pixel
- Complex implementation

---

## 3. Key Open-Source Implementations

### 3.1 Eric Bruneton's Black Hole Shader (Highest Quality)
**URL**: https://ebruneton.github.io/black_hole_shader/

**Features**:
- Real-time high-quality rendering of non-rotating black holes
- Precomputed lookup tables for constant-time ray tracing
- Includes accretion disk, relativistic Doppler/beaming effects
- Custom texture filtering for star rendering
- WebGL 2.0 implementation

**Technical Approach**:
- **Beam tracing** with precomputed intersection tables
- Algorithm 1: `Precompute` generates lookup tables
- Algorithm 2: `TraceRay` uses tables for O(1) intersection tests
- GLSL shader library in `black_hole/functions.glsl`

**Code Structure**:
```
black_hole/
├── preprocess/      # Generates lookup tables (C++)
├── definitions.glsl # Constants and uniforms
├── functions.glsl   # Core raytracing functions
└── model.glsl       # Scene composition
```

**Key Insight**: Precomputation trades memory for speed, enabling real-time performance.

### 3.2 oseiskar/black-hole (Three.js Integration)
**URL**: https://github.com/oseiskar/black-hole

**Features**:
- Schwarzschild geodesic integration via GLSL on GPU
- Three.js-based implementation
- Includes accretion disk and planet with Doppler effects
- ~30fps at 1080p on GTX 750 Ti

**Technical Approach**:
- Direct ODE integration using adaptive step sizes
- Leapfrog integration scheme for stability
- Maximum 2 revolutions around the black hole (limits computation)

**Key GLSL Code** (from `raytracer.glsl`):
```glsl
// Schwarzschild geodesic integration
float u = 1.0 / length(pos);  // u = 1/r
float du = -dot(ray, normal_vec) / dot(ray, tangent_vec) * u;

for (int j = 0; j < NSTEPS; j++) {
    // Adaptive step size
    float max_rel_u_change = (1.0 - log(u)) * 10.0 / float(NSTEPS);
    
    // Leapfrog integration
    u += du * step;
    float ddu = -u * (1.0 - 1.5 * u * u);  // Geodesic equation
    du += ddu * step;
    
    phi += step;
    pos = (cos(phi) * normal_vec + sin(phi) * tangent_vec) / u;
}
```

**Performance**: Uses `NSTEPS` = configurable (low/medium/high quality modes).

### 3.3 portsmouth/gravy (Gravitational Lensing Visualization)
**URL**: https://github.com/portsmouth/gravy

**Features**:
- WebGL simulation of gravitational lensing
- Custom 3D gravitational potential via GLSL
- Visualizes light ray paths explicitly
- Interactive API for scene authoring

**Technical Approach**:
- Treats gravity as refractive index variation
- Approximation valid for weak fields (breaks near event horizon)
- Draws ray paths for educational visualization

**Limitations**: Not suitable for strong-field black hole physics.

### 3.4 Other Notable Projects

| Project | URL | Notes |
|---------|-----|-------|
| vlwkaos/threejs-blackhole | github.com/vlwkaos/threejs-blackhole | Simple Three.js raytracer |
| zongzhengli/black-hole | github.com/zongzhengli/black-hole | WebGL with dynamic step sizes |
| sirxemic/Interstellar | github.com/sirxemic/Interstellar | Inspired by the movie |

---

## 4. Technical Deep Dive: Schwarzschild Geodesic Raytracing

### 4.1 The Geodesic Equation

For a Schwarzschild black hole (non-rotating, mass $M$), light follows null geodesics. Using the effective potential formulation:

$$\left(\frac{du}{d\phi}\right)^2 = \frac{1}{b^2} - u^2(1 - r_s u)$$

Where:
- $u = 1/r$ (inverse radius)
- $b$ = impact parameter
- $r_s = 2GM/c^2$ (Schwarzschild radius)

The second derivative gives the acceleration:

$$\frac{d^2u}{d\phi^2} = -u + \frac{3r_s u^2}{2}$$

### 4.2 Numerical Integration

**Leapfrog Method** (symplectic, stable):
```glsl
// Position update
u_new = u + du * step;

// Acceleration from geodesic equation
ddu = -u * (1.0 - 1.5 * rs * u * u);

// Velocity update
du_new = du + ddu * step;
```

**Adaptive Stepping**:
```glsl
// Smaller steps near the black hole
float step = min(base_step, max_change * u / abs(du));
```

### 4.3 Photon Sphere

At $r = 1.5 r_s$, light can orbit the black hole indefinitely. Rays passing close to this radius:
- May wrap multiple times around the BH
- Create secondary/tertiary images
- Require many integration steps

### 4.4 Event Horizon Detection

Rays with $u > 1/r_s$ (i.e., $r < r_s$) have crossed the event horizon:
```glsl
if (u > 1.0 / rs) {
    // Ray captured by black hole
    color = vec4(0.0);
    break;
}
```

---

## 5. Volumetric Rendering Techniques

### 5.1 Raymarching for Volumetric Effects

For accretion disks or gas clouds, volumetric raymarching can create realistic effects:

```glsl
vec4 raymarchVolume(vec3 rayOrigin, vec3 rayDir) {
    vec4 accumulated = vec4(0.0);
    float t = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 pos = rayOrigin + rayDir * t;
        
        float density = sampleDensity(pos);
        vec4 color = densityToColor(density);
        
        // Front-to-back compositing
        accumulated.rgb += (1.0 - accumulated.a) * color.a * color.rgb;
        accumulated.a += (1.0 - accumulated.a) * color.a;
        
        if (accumulated.a > 0.99) break;
        t += stepSize;
    }
    return accumulated;
}
```

### 5.2 3D Noise for Gas/Dust

Perlin/Simplex noise creates natural-looking gas distributions:
```glsl
float sampleDensity(vec3 pos) {
    float r = length(pos.xz);
    float diskProfile = exp(-abs(pos.y) / diskHeight);
    float radialProfile = exp(-(r - diskRadius) * (r - diskRadius) / diskWidth);
    float noise = fbm(pos * noiseScale);
    return diskProfile * radialProfile * (0.5 + 0.5 * noise);
}
```

### 5.3 GPGPU Particle Systems

For simulating millions of particles with GPU physics:
- **Compute shaders** (WebGPU) or **transform feedback** (WebGL2)
- Store particle state in textures
- Update positions/velocities entirely on GPU

**Performance**: WebGPU compute shaders can update 100,000+ particles in <2ms vs WebGL CPU bottleneck.

---

## 6. Performance Considerations

### 6.1 WebGL Limitations

| Technique | Particles | Rays/Pixel | FPS Target | Feasibility |
|-----------|-----------|------------|------------|-------------|
| InstancedMesh | 1M+ | N/A | 30+ | ✅ Current |
| Screen distortion | 1M+ | 1 | 30+ | ✅ Easy |
| Geodesic tracing | N/A | 50-200 | 30+ | ⚠️ Heavy |
| Combined | 100K | 50 | 30+ | ⚠️ Careful |

### 6.2 Optimization Strategies

1. **Resolution Scaling**: Render lensing at 1/2 or 1/4 resolution, then upscale
2. **Early Termination**: Stop tracing when ray escapes to infinity
3. **Precomputed Tables**: Bruneton's approach—precompute ray paths
4. **LOD for Particles**: Reduce particle count when lensing is active
5. **Temporal Reprojection**: Reuse previous frame's ray results

### 6.3 Hybrid Approach

Render in layers:
1. **Background**: Stars/skybox (static or precomputed lensed)
2. **Lensing Layer**: Low-res geodesic trace distortion
3. **Particles**: InstancedMesh at full resolution
4. **Post-processing**: Bloom, compositing

---

## 7. Integration Recommendations for TDE-SPH

### 7.1 Recommended Phased Approach

#### Phase 1: Screen-Space Distortion (Low Effort)
- Add a post-processing pass that distorts the scene based on BH position
- Use a simple lens equation (not physically accurate but visually effective)
- **Estimated effort**: 1-2 days

```javascript
// Post-processing shader uniform
lensShader.uniforms.bhPosition.value = blackHole.screenPosition;
lensShader.uniforms.bhMass.value = metadata.bh_mass;
lensShader.uniforms.eventHorizon.value = metadata.event_horizon_radius;
```

#### Phase 2: Background Lensing (Medium Effort)
- Raytrace only the background (skybox)
- Keep particles rendered normally
- Composite with alpha blending
- **Estimated effort**: 1 week

#### Phase 3: Full Integration (High Effort)
- Port Bruneton's or oseiskar's shader code
- Handle both background AND particles through geodesic tracing
- Requires significant architecture changes
- **Estimated effort**: 2-4 weeks

### 7.2 Quick Win Implementation

Add to `visualizer3d.js`:

```javascript
// Screen-space gravitational lensing (simplified)
const lensDistortionShader = {
    uniforms: {
        tDiffuse: { value: null },
        bhScreenPos: { value: new THREE.Vector2(0.5, 0.5) },
        lensStrength: { value: 0.1 },
        eventHorizonRadius: { value: 0.02 }
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform vec2 bhScreenPos;
        uniform float lensStrength;
        uniform float eventHorizonRadius;
        varying vec2 vUv;
        
        void main() {
            vec2 delta = vUv - bhScreenPos;
            float r = length(delta);
            
            // Event horizon - render black
            if (r < eventHorizonRadius) {
                gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
                return;
            }
            
            // Gravitational deflection (simplified)
            float deflection = lensStrength / (r * r);
            vec2 distortedUV = vUv + normalize(delta) * deflection;
            
            // Clamp to valid UV range
            distortedUV = clamp(distortedUV, 0.0, 1.0);
            
            gl_FragColor = texture2D(tDiffuse, distortedUV);
        }
    `
};
```

### 7.3 Future: WebGPU Migration Path

When WebGPU reaches broader support, migrate to:
- **Compute shaders** for particle physics
- **Ray queries** for proper geodesic tracing
- **150x performance improvement** for particle updates

---

## 8. References

### Academic/Technical
1. Bruneton, E. (2020). "A Real-time High-quality Black Hole Shader." https://ebruneton.github.io/black_hole_shader/
2. James, O. et al. (2015). "Gravitational Lensing by Spinning Black Holes in Astrophysics, and in the Movie Interstellar." Classical and Quantum Gravity.

### Open Source Implementations
3. oseiskar/black-hole: https://github.com/oseiskar/black-hole
4. portsmouth/gravy: https://github.com/portsmouth/gravy
5. vlwkaos/threejs-blackhole: https://github.com/vlwkaos/threejs-blackhole

### Three.js Resources
6. Three.js EffectComposer: https://threejs.org/docs/#examples/en/postprocessing/EffectComposer
7. Three.js ShaderMaterial: https://threejs.org/docs/#api/en/materials/ShaderMaterial

### Volumetric Rendering
8. Heckel, M. "Real-time Cloudscapes with Volumetric Raymarching." https://blog.maximeheckel.com/posts/real-time-cloudscapes-with-volumetric-raymarching/
9. FarazzShaikh/three-volumetric-clouds: https://github.com/FarazzShaikh/three-volumetric-clouds

### Performance
10. "WebGL vs WebGPU: Is It Time to Switch?" https://threejsroadmap.com/blog/webgl-vs-webgpu-explained

---

## Appendix A: Schwarzschild Metric Reference

The Schwarzschild metric in coordinates $(t, r, \theta, \phi)$:

$$ds^2 = -\left(1 - \frac{r_s}{r}\right)c^2 dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1} dr^2 + r^2 d\Omega^2$$

Where $r_s = \frac{2GM}{c^2}$ is the Schwarzschild radius.

For our simulation with $M = $ `bh_mass` (solar masses):
$$r_s = 2.95 \times 10^3 \cdot M \text{ meters} \approx 3 \cdot M \text{ km}$$

---

## Appendix B: Doppler and Beaming Effects

For a source moving with velocity $v$ toward/away from observer:

**Doppler Factor**:
$$D = \frac{1}{\gamma(1 - \beta \cos\theta)}$$

Where $\gamma = 1/\sqrt{1-\beta^2}$, $\beta = v/c$.

**Observed Temperature** (thermal radiation):
$$T_{obs} = T_{emit} \cdot D$$

**Intensity Beaming**:
$$I_{obs} = I_{emit} \cdot D^3$$ (for isotropic source)

---

*Report generated for TDE-SPH Black Hole Visualization Project*
*Date: December 2024*
