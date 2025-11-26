# CLAUDE Instructions — gravity module

Role: Provide gravity solvers (Newtonian and relativistic hybrid) behind a stable interface.

## Phase 1 Status (Complete)
- ✅ `NewtonianGravity` class implementing direct O(N²) self-gravity
- ✅ Softening length = smoothing length
- ✅ Potential and acceleration computation
- ✅ Functional for N ≲ 10⁵ particles

## Phase 2 Goals: Hybrid Relativistic Gravity

Implement the **hybrid approach** from Tejeda et al. (2017): exact GR motion in fixed BH spacetime + Newtonian self-gravity.

## Physics Background & Equations

### REQ-002 & REQ-006: Hybrid Relativistic Formulation

Following Tejeda et al. (2017) [arXiv:1701.00303]:

**Key insight:** For TDEs, the dominant gravitational effect is the *black hole's* spacetime curvature. Stellar self-gravity can be treated Newtonianly as a perturbation.

#### Total Acceleration

For particle i:
```
a_i^total = a_i^BH(GR) + a_i^self(Newtonian)
```

where:
- **a^BH(GR)**: Geodesic acceleration from `Metric.geodesic_acceleration(x_i, v_i)`
- **a^self(Newtonian)**: Self-gravity from other SPH particles (existing `NewtonianGravity`)

#### BH Acceleration (Schwarzschild Example)

In Cartesian coordinates, for Schwarzschild metric:
```
a^BH = -GM_BH/r³ * [r̂ + 3(v·r̂)² v̂ / c² + ...]
```

More precisely, use geodesic equation from `Metric` class.

#### Self-Gravity (Unchanged from Phase 1)

```
a_i^self = -Σ_{j≠i} G m_j (r_i - r_j) / (|r_i - r_j|² + ε²)^(3/2)
```

with softening ε = h_i (smoothing length).

**Key approximation:** Self-gravity remains Newtonian even in GR mode. This is justified because:
1. Stellar self-binding energy << BH gravitational binding energy
2. Internal stellar velocities << c
3. Validated in Tejeda et al. (2017), Rosswog et al. (2009)

### Pseudo-Newtonian Alternative (Optional, TASK-014b)

For comparison, implement Paczyński-Wiita pseudo-Newtonian potential:
```
φ_PN = -GM_BH / (r - r_S)

where r_S = 2GM_BH/c² (Schwarzschild radius)
```

This mimics GR ISCO but lacks frame-dragging. Use for validation/comparison only.

**References:**
- Tejeda et al. (2017), MNRAS 469, 4483 [arXiv:1701.00303] – Hybrid GR approach
- Tejeda & Rosswog (2013), MNRAS 433, 1930 – Pseudo-Newtonian potentials
- Paczyński & Wiita (1980), A&A 88, 23 – Original pseudo-Newtonian potential
- Liptai & Price (2019), MNRAS 485, 819 – GRSPH framework

## Phase 2 Tasks for Gravity Module Agents

### TASK-014: Hybrid Relativistic Acceleration

**TASK-014a**: `RelativisticGravitySolver` class (`gravity/relativistic_orbit.py`)
- Implement `GravitySolver` interface
- In `compute_acceleration()`:
  - If `metric is None`: fall back to pure Newtonian (backward compat)
  - If `metric is not None`:
    - Compute `a_BH = metric.geodesic_acceleration(positions, velocities)`
    - Compute `a_self = newtonian_gravity.compute_acceleration(...)`
    - Return `a_total = a_BH + a_self`
- Use existing `NewtonianGravity` for self-gravity component
- Support both Schwarzschild and Kerr metrics

**TASK-014b**: Pseudo-Newtonian solver (optional) (`gravity/pseudo_newtonian.py`)
- Implement Paczyński-Wiita potential for comparison
- Useful for algorithm development and validation
- Document limitations (no frame-dragging, approximate ISCO)

**TASK-014c**: Integration with core
- Ensure `Simulation` can switch between:
  - `NewtonianGravity` (pure Newtonian, mode="Newtonian")
  - `RelativisticGravitySolver` (hybrid GR, mode="GR")
- Pass `metric` from `Simulation` to solver

### TASK-018: Runtime Mode Toggle

**TASK-018a**: Wrapper logic in `RelativisticGravitySolver`
- If `config.mode == "Newtonian"` and `metric is None`:
  - Use `NewtonianGravity` for BH *and* self-gravity
  - No GR corrections
- If `config.mode == "GR"` and `metric is not None`:
  - Use `metric.geodesic_acceleration()` for BH
  - Use `NewtonianGravity` for self-gravity only

**TASK-018b**: Validation
- Ensure Newtonian mode reproduces Phase 1 results exactly
- Document energy non-conservation expected in hybrid scheme

## Architectural Risks

**RISK-GRAVITY-001**: Hybrid approximation validity
- **Context**: Newtonian self-gravity may fail for very compact stars or extreme tidal forces
- **Mitigation**: Document regime of validity (R_star >> R_S); compare to full GRSPH for extreme cases

**RISK-GRAVITY-002**: Coordinate system consistency
- **Context**: BH gravity in spherical, self-gravity in Cartesian
- **Mitigation**: Ensure `Metric` handles Cartesian ↔ BL conversions internally

**RISK-GRAVITY-003**: Energy non-conservation
- **Context**: Hybrid scheme not derived from single Hamiltonian
- **Mitigation**: Track energy drift; compare to pure Newtonian and test-particle GR limits

**RISK-GRAVITY-004**: Tree-based gravity future extension
- **Context**: O(N²) self-gravity doesn't scale to N > 10⁵
- **Mitigation**: Design interface to support tree-based replacement in Phase 4

## Validation Strategy (TASK-019, TASK-020)

### Test-Particle Orbits (No Self-Gravity)
1. Set all masses to zero except one
2. Integrate orbit in Schwarzschild/Kerr metric
3. Compare to analytic geodesics (conserved energy, angular momentum)
4. Verify periapsis precession matches GR predictions

### Newtonian Limit
1. Place star at r >> 6M
2. Compare hybrid GR to pure Newtonian
3. Should agree to < 1% at r > 100M

### ISCO Behavior
1. Place test particle at r = 6M (Schwarzschild ISCO)
2. Verify orbital stability properties match theory

### Kerr Frame-Dragging
1. Use inclined orbit in Kerr metric (a > 0)
2. Verify Lense-Thirring precession (orbit plane rotation)

## GPU/CUDA Considerations (Phase 4 Preview)

Current Phase 2 implementation can remain CPU-based, but structure for future GPU port:

- **Self-gravity**: Direct pairwise → GPU kernel with shared memory
- **BH gravity**: Metric evaluation is embarassingly parallel
- **Mixed precision**: FP32 for positions/masses, FP64 for metric near r ~ 2M

Design interface to accept GPU arrays (CuPy, PyTorch tensors) in addition to NumPy.

## DO

- Work ONLY inside `tde_sph/gravity`
- Implement Newtonian self-gravity (already done in Phase 1)
- Implement `RelativisticGravitySolver` combining BH GR + self-gravity Newtonian
- Respect `Metric` and `GravitySolver` interfaces from `tde_sph/core`
- Use `Metric` from `tde_sph/metric` for BH gravity
- Support seamless Newtonian ↔ GR mode switching
- Document hybrid approximation clearly
- Write tests comparing to analytic geodesics and Newtonian limits

## DO NOT

- Implement SPH kernels or hydrodynamic forces (belongs in `sph/`)
- Redefine spacetime metrics (use `tde_sph/metric`)
- Break backward compatibility with Phase 1 Newtonian mode
- Open or modify anything under the `prompts/` folder
