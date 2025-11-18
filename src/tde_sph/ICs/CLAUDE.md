# CLAUDE Instructions — initial conditions (ICs) module

Role: Generate initial particle configurations for stars, discs and orbits.

## Phase 1 & 2 Status (Complete)
- ✅ Polytropic stellar models in `polytrope.py`
- ✅ Support for γ=5/3 and γ=4/3 polytropes
- ✅ Orbital parameter setup (parabolic, eccentric orbits)
- ✅ Deterministic particle placement with seeded RNG

## Phase 3 Goals: Accretion Disc Initial Conditions

### TASK-034: Accretion Disc IC Generator

**Objective**: Implement `disc.py` for pre-existing accretion disc configurations.

**Disc Models to Support**:
1. **Thin Keplerian disc** (α-disc prescription):
   - Surface density: Σ(r) ∝ r^(-p) (p ~ 0.5-1.5)
   - Vertical structure: hydrostatic equilibrium H/r ~ c_s/v_K
   - Temperature profile: T(r) from viscous heating or power-law

2. **Torus** (thick disc):
   - Constant angular momentum or Fishbone-Moncrief solution
   - Pressure-supported vertical extent
   - Optional magnetic fields (for future MHD)

3. **Tilted/Warped disc**:
   - Arbitrary inclination and position angle relative to BH spin
   - Lense-Thirring precession initial velocities (GR mode)
   - For stream-disc collision studies

**Implementation Requirements**:
1. Create `DiscGenerator` class inheriting from `ICGenerator` interface
2. Provide methods:
   - `generate_thin_disc(r_in, r_out, M_disc, profile, **kwargs)` → particles
   - `generate_torus(r_max, l_specific, **kwargs)` → particles
   - `generate_tilted_disc(inclination, PA, **kwargs)` → particles
3. Support both Newtonian and GR (Kerr) equilibrium velocities
4. Particle placement: uniform in φ, stratified in r and z
5. Initial thermal structure consistent with EOS

**Physical Parameters**:
- Inner radius r_in (≥ ISCO for GR)
- Outer radius r_out
- Total disc mass M_disc
- Surface density profile index p
- Temperature normalization or α-viscosity parameter
- Inclination angle i (0° = equatorial, 90° = polar)
- Position angle PA (disc orientation)

**Tests**:
- Disc in equilibrium: no secular evolution without perturbations
- Mass and angular momentum consistent with inputs
- Vertical hydrostatic balance
- GR: circular orbit velocities match Kerr geodesics
- Deterministic: same seed → same particle configuration

**Cross-module Dependencies**:
- Uses `metric` for GR orbital velocities
- Uses `eos` for initial thermal structure
- Called by user scripts or `core/simulation.py` for setup

**Example Use Case**:
Stream-disc collision in tilted disc geometry (Bonnerot et al. 2023).

DO:
- Work ONLY inside `tde_sph/ICs`.
- Implement polytropic stellar models, misaligned discs, orbital parameterisation and mapping to SPH particle sets.
- Keep IC generators deterministic (seeded RNG) for reproducibility.
- Add comprehensive docstrings with physics references
- Include validation tests for equilibrium configurations

DO NOT:
- Implement runtime integration logic or physics updates.
- Change gravity, metric or EOS implementations.
- Open or modify anything under the `prompts/` folder.
- Modify files outside `tde_sph/ICs/`
