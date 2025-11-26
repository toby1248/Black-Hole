# CLAUDE Instructions — metric module

Role: Implement spacetime metrics and geodesic helpers for General Relativistic TDE simulations.

## Phase 2 Goals

Implement exact spacetime metrics (Minkowski, Schwarzschild, Kerr) for fixed background GR simulations following the hybrid approach of Tejeda et al. (2017) and the GRSPH framework of Liptai & Price (2019).

## Physics Background & Equations

### REQ-005: Spacetime Metrics

Implement three metrics in geometric units (G = c = M_BH = 1):

#### 1. Minkowski (Flat Spacetime)

For testing and validation:
```
ds² = -dt² + dx² + dy² + dz²
g_μν = diag(-1, 1, 1, 1)
```

#### 2. Schwarzschild (Non-Rotating Black Hole)

Boyer-Lindquist-like spherical coordinates (t, r, θ, φ):
```
ds² = -(1 - 2M/r) dt² + (1 - 2M/r)⁻¹ dr² + r² dθ² + r² sin²θ dφ²

g_tt = -(1 - 2M/r)
g_rr = (1 - 2M/r)⁻¹
g_θθ = r²
g_φφ = r² sin²θ

ISCO radius: r_ISCO = 6M
Event horizon: r_H = 2M
```

#### 3. Kerr (Rotating Black Hole)

Boyer-Lindquist coordinates with dimensionless spin a = J/M² ∈ [0, 1]:
```
Δ = r² - 2Mr + a²
Σ = r² + a² cos²θ
A = (r² + a²)² - a² Δ sin²θ

ds² = -(1 - 2Mr/Σ) dt² - (4Mra sin²θ/Σ) dt dφ
      + (Σ/Δ) dr² + Σ dθ² + (A sin²θ/Σ) dφ²

g_tt = -(1 - 2Mr/Σ)
g_tφ = -2Mra sin²θ/Σ
g_rr = Σ/Δ
g_θθ = Σ
g_φφ = A sin²θ/Σ

Prograde ISCO: r_ISCO = M(3 + Z₂ - √[(3 - Z₁)(3 + Z₁ + 2Z₂)])
  where Z₁ = 1 + (1-a²)^(1/3) [(1+a)^(1/3) + (1-a)^(1/3)]
        Z₂ = √(3a² + Z₁²)

Event horizon: r_+ = M + √(M² - a²)
```

**References:**
- Bardeen, Press & Teukolsky (1972), ApJ 178, 347 – Kerr geodesics
- Tejeda et al. (2017), MNRAS 469, 4483 [arXiv:1701.00303] – Hybrid GR TDE approach
- Liptai & Price (2019), MNRAS 485, 819 [arXiv:1901.08064] – GRSPH framework

### Christoffel Symbols

Compute via standard formula:
```
Γ^μ_νρ = (1/2) g^μσ (∂g_σν/∂x^ρ + ∂g_σρ/∂x^ν - ∂g_νρ/∂x^σ)
```

For **Schwarzschild**, non-zero components include:
```
Γ^r_tt = M(r-2M)/r³
Γ^t_tr = M/(r(r-2M))
Γ^r_rr = -M/(r(r-2M))
Γ^r_θθ = -(r-2M)
Γ^r_φφ = -(r-2M) sin²θ
... (see Misner, Thorne & Wheeler for complete list)
```

For **Kerr**, substantially more complex; use symbolic derivatives or tabulated expressions.

### Geodesic Equation

Particle motion governed by:
```
d²x^μ/dτ² + Γ^μ_νρ (dx^ν/dτ) (dx^ρ/dτ) = 0
```

**Implementation approach (TASK-013):**
- Provide `geodesic_acceleration(x, v)` returning spatial acceleration for SPH particles
- Input: position `x = (x, y, z)` in Cartesian, 4-velocity `v = (u^t, u^x, u^y, u^z)`
- Output: spatial acceleration `a = (a^x, a^y, a^z)`
- Handle coordinate transformations (Cartesian ↔ spherical) internally

**Numerical Stability:**
- Use **FP64** for metric tensor inversions and Christoffel symbols near r ~ 2M
- Implement coordinate regularization near poles (θ = 0, π)
- Detect and warn if particles cross event horizon

## Phase 2 Tasks for Metric Module Agents

### TASK-013: Implement Metric Subclasses

**TASK-013a**: Minkowski metric (`metric/minkowski.py`)
- Trivial flat spacetime for testing
- All Christoffel symbols = 0
- Geodesic acceleration = 0

**TASK-013b**: Schwarzschild metric (`metric/schwarzschild.py`)
- Implement Boyer-Lindquist Schwarzschild metric
- Methods: `metric_tensor(x)`, `inverse_metric(x)`, `christoffel_symbols(x)`, `geodesic_acceleration(x, v)`
- Cartesian ↔ spherical coordinate conversion
- Compute ISCO radius: 6M
- Handle r < 2M gracefully (warn, don't crash)

**TASK-013c**: Kerr metric (`metric/kerr.py`)
- Implement Boyer-Lindquist Kerr metric with spin parameter a
- Same methods as Schwarzschild
- Compute spin-dependent ISCO radius
- Handle ergosphere (r < r_+ + √(r_+² - a² cos²θ))
- Frame-dragging term g_tφ ≠ 0

**TASK-013d**: Coordinate utilities (`metric/coordinates.py`)
- Cartesian ↔ Boyer-Lindquist transformations
- Velocity transformations between coordinate systems
- Regularization near coordinate singularities

**TASK-013e**: Unit tests
- Verify g^μν g_νρ = δ^μ_ρ (metric inversion)
- Compare Schwarzschild geodesics to analytic Kepler orbits at large r
- Test Kerr → Schwarzschild limit as a → 0
- Validate epicyclic frequencies against Liptai & Price (2019) Appendix

## Architectural Risks

**RISK-METRIC-001**: Coordinate singularities at r=2M, θ=0,π
- **Mitigation**: Use Kerr-Schild coordinates near horizon (Phase 3); for now, warn and clamp

**RISK-METRIC-002**: Numerical precision loss in metric inversion near horizon
- **Mitigation**: Use FP64 for metric computations; condition number checks

**RISK-METRIC-003**: Kerr metric complexity
- **Mitigation**: Start with Schwarzschild; validate extensively before adding Kerr spin

**RISK-METRIC-004**: Frame-dragging coupling to SPH
- **Mitigation**: Document that Lense-Thirring precession requires careful velocity transformations

## Validation Strategy (TASK-019)

1. **Flat-space limit**: Verify Minkowski reproduces Newtonian results
2. **Weak-field limit**: Compare Schwarzschild at r >> M to Newtonian φ = -M/r
3. **Geodesic orbits**: Integrate test-particle trajectories and compare to analytic solutions
4. **Epicyclic frequencies**: Validate ω_r, ω_θ against Liptai & Price (2019) Eq. A1-A3
5. **ISCO**: Verify particles at r = r_ISCO have correct orbital properties
6. **Kerr → Schwarzschild**: Test a → 0 limit

## DO

- Work ONLY inside `tde_sph/metric`
- Implement Minkowski, Schwarzschild, Kerr metrics inheriting from `Metric` ABC
- Provide `metric_tensor`, `inverse_metric`, `christoffel_symbols`, `geodesic_acceleration`
- Use **FP64 for metric computations**, FP32 for particle arrays
- Handle coordinate transformations (Cartesian ↔ BL) robustly
- Document all equations with references to MTW, Tejeda et al., Liptai & Price
- Write extensive unit tests (metric inversion, geodesic tests, ISCO validation)

## DO NOT

- Implement self-gravity or SPH forces (belongs in `gravity/` and `sph/`)
- Hard-code simulation loops or I/O
- Open or modify anything under the `prompts/` folder
- Use FP32 for metric tensor operations (precision loss near horizon)
