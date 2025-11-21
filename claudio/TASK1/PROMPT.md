## PROMPT
Systematically investigate and fix all bugs similar to the temperature issue: computed physical quantities that are not properly initialized or stored in the ParticleSystem. Search for defensive programming patterns (hasattr, getattr, try/except AttributeError), identify missing attributes, add them to ParticleSystem, and update Simulation to store computed values.

## COMPLEXITY
Medium

## CONTEXT REFERENCE
**For complete environment context, read:**
- `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md` - Contains full tech stack, architecture, project structure, coding conventions, and related code patterns

**You MUST read AI_PROMPT.md before executing this task to understand the environment.**

## TASK-SPECIFIC CONTEXT

### Investigation Methodology
1. **Automated search** for defensive patterns:
   - `grep -r "hasattr.*particles" src/`
   - `grep -r "getattr.*particles" src/`
   - `grep -r "try:.*particles\." src/` (manual review for AttributeError handling)

2. **Manual code review** of key files:
   - `src/tde_sph/sph/particles.py:__init__()` - What attributes are initialized?
   - `src/tde_sph/core/simulation.py` - What quantities are computed but not stored?
   - `src/tde_sph/eos/ideal_gas.py` - What does EOS return?
   - `src/tde_sph/io/hdf5.py` - What attributes are saved to snapshots?

### Files This Task Will Touch
**Investigation (read-only analysis):**
- `src/tde_sph/sph/particles.py` - Review attribute initialization
- `src/tde_sph/core/simulation.py` - Review quantity computation
- `src/tde_sph/eos/ideal_gas.py` - Review EOS outputs
- `src/tde_sph/metric/coordinates.py` - Review coordinate transforms
- `src/tde_sph/io/hdf5.py` - Review snapshot I/O
- `gui/simulation_thread.py` - Review diagnostic data access

**Modifications (based on findings):**
- `src/tde_sph/sph/particles.py` - Add missing attribute initializations
- `src/tde_sph/core/simulation.py` - Store computed quantities
- `src/tde_sph/io/hdf5.py` - Add attributes to snapshot schema (if needed)
- Remove defensive access patterns from any files using hasattr/getattr

**New test files:**
- `src/tde_sph/sph/particles.test.py` - Add tests for new attributes
- `src/tde_sph/io/hdf5.test.py` - Add round-trip tests

### Patterns to Follow
**For each missing attribute (follow temperature fix pattern):**

1. **Add to ParticleSystem.__init__():**
```python
# In src/tde_sph/sph/particles.py
self.velocity_magnitude = np.zeros(n_particles, dtype=np.float32)
self.entropy = np.zeros(n_particles, dtype=np.float32)
self.acceleration = np.zeros((n_particles, 3), dtype=np.float32)  # Note: 3D vector
self.du_dt = np.zeros(n_particles, dtype=np.float32)
```

2. **Add to _validate_shapes():**
```python
if self.velocity_magnitude.shape != (self.n_particles,):
    raise ValueError("velocity_magnitude shape mismatch")
```

3. **Store in Simulation:**
```python
# In src/tde_sph/core/simulation.py
# After computing velocity_magnitude:
self.particles.velocity_magnitude = np.linalg.norm(self.particles.velocity, axis=1)
```

4. **Remove defensive access:**
```python
# BEFORE (defensive):
vmag = getattr(particles, 'velocity_magnitude', 0.0)

# AFTER (direct):
vmag = particles.velocity_magnitude
```

### Likely Missing Attributes
Based on AI_PROMPT.md analysis, investigate these candidates:

1. **velocity_magnitude** (scalar):
   - Computed: `np.linalg.norm(particles.velocity, axis=1)`
   - Used for: Colormap visualization, statistics
   - Should be: Stored in ParticleSystem

2. **entropy** (scalar):
   - Computed: By EOS module (ideal_gas.py)
   - Used for: Thermodynamic diagnostics
   - Should be: Stored if EOS provides it

3. **acceleration** (3D vector):
   - Computed: By gravity solver
   - Used for: Debugging, analysis
   - Should be: Stored or at least accessible

4. **du_dt** (scalar):
   - Computed: Change in internal energy per timestep
   - Used for: Energy conservation diagnostics
   - Should be: Stored in ParticleSystem

### Integration Points
- **EOS module:** Returns temperature, pressure, sound_speed, possibly entropy
- **Gravity solver:** Computes acceleration, must be accessible
- **HDF5 I/O:** All attributes must be written/read for snapshots
- **GUI diagnostics:** Needs access to all computed quantities for display

### Coordinate Transform Bug Investigation
If coordinate transformation bugs exist (AI_PROMPT.md mentions this as possibility):

**Test approach:**
1. Create circular orbit at ISCO (Innermost Stable Circular Orbit)
2. Transform Cartesian → Boyer-Lindquist
3. Verify radius matches theoretical ISCO radius: `r_ISCO = 6M` (Schwarzschild)
4. Transform back: Boyer-Lindquist → Cartesian
5. Verify position matches original (round-trip test)

**Common bugs in GR coordinate transforms:**
- Sign errors in metric components
- Missing terms in Jacobian
- Incorrect handling of spin parameter (Kerr metric)

## EXTRA DOCUMENTATION

### Search Commands
Run these to find all defensive attribute access:
```bash
# From project root:
grep -rn "hasattr.*particles" src/ gui/
grep -rn "getattr.*particles" src/ gui/
grep -rn "AttributeError" src/ gui/ | grep particles
```

### Testing Strategy
**For each fixed attribute:**
1. Unit test: Initialization with correct shape/dtype
2. Unit test: Computation and storage in Simulation
3. Integration test: Value is physically reasonable (e.g., velocity_magnitude ≥ 0)
4. HDF5 test: Write snapshot, read back, verify attribute preserved

**Manual verification:**
1. Run simulation with fixed code
2. Check that new attributes are non-zero and reasonable
3. Verify HDF5 snapshot contains all attributes (use h5dump or h5py to inspect)

### Documentation Requirements
Create a summary of all bugs found and fixed. Example format:
```
FINDINGS SUMMARY:
1. velocity_magnitude - Not initialized, added to ParticleSystem
2. entropy - Computed by EOS but not stored, added storage in Simulation
3. acceleration - Accessible from gravity solver, added getter method
4. du_dt - Not computed, added calculation in energy update

COORDINATE BUGS:
- None found (transforms validated with ISCO test)

HDF5 BUGS:
- velocity_magnitude not written to snapshots, added to schema
```

## LAYER
0 (Foundation)

## PARALLELIZATION
Parallel with: []

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- All new attributes must use `dtype=np.float32` (GPU compatibility)
- HDF5 schema changes must be backward-compatible (old snapshots still load)
- Do NOT modify core physics algorithms (only add missing attribute storage)
- Do NOT change existing attribute names (preserve API compatibility)
- Test ONLY changed files (no global test suite)
- Document all findings clearly in code comments and task summary
- Follow PEP 8 and existing code conventions (NumPy docstrings, type hints)
