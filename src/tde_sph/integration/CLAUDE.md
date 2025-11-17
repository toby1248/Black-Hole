# CLAUDE Instructions — integration module

Role: Time integration and timestep control for both Newtonian and GR dynamics.

## Phase 1 Status (Complete)
- ✅ `LeapfrogIntegrator` with kick-drift-kick scheme
- ✅ CFL timestep estimation (h / (c_s + |v|))
- ✅ Acceleration-based timestep (√(h / |a|))
- ✅ Internal energy evolution with viscous heating
- ✅ Functional for Newtonian SPH

## Phase 2 Goals: GR-Aware Integration

Implement Hamiltonian integrator for accurate GR geodesic motion near ISCO, following Liptai & Price (2019).

## Physics Background & Equations

### REQ-004 & REQ-010: Timestep Control in Strong Fields

In GR, orbital timescales vary dramatically with radius:
- **Far from BH** (r >> 6M): Newtonian leapfrog sufficient
- **Near ISCO** (r ~ 6M): Require symplectic/Hamiltonian integrator for energy conservation

#### Orbital Timescale

```
t_orb(r) ≈ 2π √(r³ / GM_BH)  [Newtonian approximation]

For Schwarzschild GR:
t_orb(r) = 2π √(r³ / M) / (1 - 3M/r)^(1/2)  [diverges at r = 3M]
```

#### Timestep Constraints

1. **CFL condition** (SPH):
   ```
   dt_CFL = C_CFL * min_i(h_i / (c_s,i + |v_i|))
   ```

2. **Acceleration constraint**:
   ```
   dt_acc = C_acc * min_i(√(h_i / |a_i|))
   ```

3. **Orbital constraint** (new for GR):
   ```
   dt_orb = C_orb * min_i(√(r_i³ / M))
   ```

4. **ISCO constraint** (near r = 6M):
   ```
   dt_ISCO = C_ISCO * (r_i / |v_i|)  for r_i < r_threshold
   ```

Take `dt = min(dt_CFL, dt_acc, dt_orb, dt_ISCO)` with safety factors C < 1.

### Hamiltonian Integrator for GR (TASK-016)

Following Liptai & Price (2019), use a **symplectic integrator** that preserves the Hamiltonian structure of geodesic motion.

#### GR Hamiltonian

For test particles in fixed metric:
```
H = (1/2) g^μν p_μ p_ν

where p_μ is the 4-momentum
```

Conserved along geodesics.

#### Integration Scheme

Use **Störmer-Verlet** or **leapfrog** in phase space (x, p):

```python
# Half-step momentum
p^(n+1/2) = p^n - (dt/2) * ∂H/∂x |_(x^n, p^(n+1/2))

# Full-step position
x^(n+1) = x^n + dt * ∂H/∂p |_(x^n, p^(n+1/2))

# Half-step momentum
p^(n+1) = p^(n+1/2) - (dt/2) * ∂H/∂x |_(x^(n+1), p^(n+1/2))
```

where `∂H/∂x` involves Christoffel symbols from `Metric`.

**Advantage:** Preserves phase-space volume (symplectic), conserves Hamiltonian to machine precision over long integrations.

**References:**
- Liptai & Price (2019), MNRAS 485, 819 [arXiv:1901.08064] – Hamiltonian GRSPH
- Hairer, Lubich & Wanner (2006) – Geometric Numerical Integration
- Leimkuhler & Reich (2004) – Simulating Hamiltonian Dynamics

### Radius-Dependent Integrator Switching (TASK-017)

**Algorithm:**
1. For each particle, compute `r_i = |x_i|`
2. If `r_i < r_threshold` (e.g., 10M) and `metric is not None`:
   - Use `HamiltonianIntegrator` for geodesic motion
   - Stricter timestep constraints
3. If `r_i >= r_threshold` or `metric is None`:
   - Use `LeapfrogIntegrator` (standard SPH)
   - Relaxed timesteps

**Configuration parameter:**
```python
config.isco_radius_threshold: float = 10.0  # Switch to Hamiltonian inside this radius
config.use_hamiltonian_integrator: bool = True
```

## Phase 2 Tasks for Integration Module Agents

### TASK-016: Hamiltonian Integrator

**TASK-016a**: Implement `HamiltonianIntegrator` class (`integration/hamiltonian.py`)
- Inherit from `TimeIntegrator` ABC
- Implement Störmer-Verlet scheme in phase space
- Use `Metric.christoffel_symbols()` for ∂H/∂x
- Verify Hamiltonian conservation over 1000s of orbits
- Support arbitrary metric (Schwarzschild, Kerr)

**TASK-016b**: Test against analytic geodesics
- Circular orbits at various radii
- Eccentric orbits (conserve E, L_z)
- Compare to Liptai & Price (2019) epicyclic frequency tests

**TASK-016c**: Hybrid SPH+GR integration
- Particles feel both geodesic acceleration and SPH/self-gravity forces
- Combine Hamiltonian step for GR part with force kick for SPH/self-gravity
- Document that this breaks strict symplectic structure (but acceptable for hybrid scheme)

### TASK-017: Radius-Dependent Timestep Strategy

**TASK-017a**: Implement `timestep_control.py`
- Function `estimate_timestep_gr(particles, metric, config)`
- Compute all constraints: CFL, acceleration, orbital, ISCO
- Return minimum with safety factors

**TASK-017b**: Adaptive integrator selection
- Add logic to `Simulation` to choose integrator per-particle or globally
- Option 1: Single integrator for all particles (simpler)
- Option 2: Per-particle integrator (more accurate, complex bookkeeping)
- Recommend Option 1 for Phase 2: use Hamiltonian if *any* particle near ISCO

**TASK-017c**: Logging and diagnostics
- Log timestep contributions (which constraint is limiting?)
- Warn if particles violate ISCO constraint
- Track fraction of time spent in Hamiltonian vs leapfrog regimes

### TASK-016d: Backward Compatibility

- `LeapfrogIntegrator` remains default for Newtonian mode
- Hamiltonian integrator only activated if `metric is not None` and `use_hamiltonian_integrator=True`
- All Phase 1 Newtonian tests must pass unchanged

## Architectural Risks

**RISK-INTEGRATION-001**: Hamiltonian integrator complexity
- **Context**: GR geodesic integration requires careful handling of 4-momentum, Christoffel symbols
- **Mitigation**: Start with simple circular orbits; validate against analytic solutions before adding SPH

**RISK-INTEGRATION-002**: Timestep explosion near ISCO
- **Context**: Orbital frequency → ∞ as r → 3M (Schwarzschild photon orbit)
- **Mitigation**: Enforce minimum timestep; warn if particles approach r < 3M

**RISK-INTEGRATION-003**: Hybrid scheme not strictly symplectic
- **Context**: SPH forces + GR geodesic motion don't come from single Hamiltonian
- **Mitigation**: Accept small energy drift; monitor and document

**RISK-INTEGRATION-004**: Per-particle integrators (future)
- **Context**: Different particles using different integrators complicates neighbor search
- **Mitigation**: Phase 2 uses single global integrator; defer per-particle to Phase 4

## Validation Strategy (TASK-019)

### Test 1: Hamiltonian Conservation
- Integrate test particle in Schwarzschild metric for 1000 orbits
- Verify `ΔH/H < 10⁻¹⁰` (machine precision)

### Test 2: Epicyclic Frequencies
- Place particle on circular orbit at r = 10M, 8M, 6.1M
- Perturb slightly and measure radial/vertical oscillation frequencies
- Compare to analytic: ω_r² = (1 - 6M/r) / r³, ω_θ² = (1/r³) (Schwarzschild)
- Reference: Liptai & Price (2019) Appendix A

### Test 3: Leapfrog ↔ Hamiltonian Consistency
- Integrate same orbit with both integrators at large r (r > 100M)
- Should agree to ~1% (leapfrog is O(dt²), Hamiltonian is exact symplectic)

### Test 4: ISCO Stability
- Place particle at r = 6M (marginally stable)
- Integrate for 100 orbits
- Verify orbit remains bounded (not plunging or escaping)

## GPU/CUDA Considerations (Phase 4 Preview)

Both leapfrog and Hamiltonian integrators are particle-local operations → highly parallelizable:

- Timestep estimation: parallel reduction (min over particles)
- Integration step: independent per-particle update
- Mixed precision: FP32 for positions/velocities, FP64 for Hamiltonian evaluation

Design for batched GPU tensor operations (PyTorch, CuPy).

## DO

- Work ONLY inside `tde_sph/integration`
- Implement leapfrog (done in Phase 1) and Hamiltonian integrators
- Implement global and ISCO-aware timestep control
- Support radius-dependent integrator switching
- Validate against analytic geodesics and Liptai & Price (2019) tests
- Accept force arrays from SPH, gravity, radiation modules

## DO NOT

- Implement physics forces (belongs in `sph/`, `gravity/`, `radiation/`)
- Own configuration files (accept from `core.Simulation`)
- Break backward compatibility with Phase 1 leapfrog
- Open or modify anything under the `prompts/` folder
