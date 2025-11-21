@dependencies [TASK0]
# Task: Investigate and Fix Similar Attribute Bugs

## Summary
Systematically search the codebase for bugs similar to the temperature issue: computed quantities that are not properly initialized or stored in ParticleSystem. The goal is to identify and fix all missing attributes that should be available but are accessed with defensive programming (hasattr, getattr, try/except).

**Investigation Strategy:**
1. Search for `hasattr()` and `getattr()` patterns in physics modules
2. Review `ParticleSystem.__init__()` for incomplete attribute initialization
3. Check `Simulation` update methods for computed-but-not-stored quantities
4. Verify all computed quantities are written to HDF5 snapshots

## Context Reference
**For complete environment context, see:**
- `../AI_PROMPT.md` - Contains full tech stack, architecture, coding conventions, and related code patterns

**Task-Specific Context:**
- **Files to investigate:**
  - `src/tde_sph/sph/particles.py` - Check for missing attribute initializations
  - `src/tde_sph/core/simulation.py` - Check for computed quantities not stored
  - `src/tde_sph/eos/ideal_gas.py` - Verify all EOS outputs are captured
  - `src/tde_sph/metric/coordinates.py` - Check coordinate transform completeness
  - `src/tde_sph/io/hdf5.py` - Verify all attributes written/read correctly
  - `gui/simulation_thread.py` - Look for defensive attribute access patterns
- **Likely missing attributes (to verify):**
  - `velocity_magnitude` - computed but not stored
  - `entropy` - may be computed by EOS but not stored
  - `acceleration` - computed in gravity solver, possibly not stored
  - `du_dt` - internal energy change rate (needed for diagnostics)
- **Pattern to detect:**
  - Code using `hasattr(particles, 'attr_name')` or `getattr(particles, 'attr_name', default)`
  - Try/except blocks catching AttributeError
  - Computed values that are not assigned back to `self.particles.attr_name`

## Complexity
Medium

## Dependencies
Depends on: [TASK0]
Blocks: [TASK2, TASK3, TASK4, TASK5]
Parallel with: []

## Detailed Steps
1. **Search for defensive attribute access:**
   - Grep for `hasattr.*particles` in all `.py` files
   - Grep for `getattr.*particles` in all `.py` files
   - Grep for `try:.*particles\.` followed by `except AttributeError` (multiline)
   - Create list of all defensively-accessed attributes

2. **Review ParticleSystem initialization:**
   - Open `src/tde_sph/sph/particles.py:__init__()`
   - Compare attributes initialized vs attributes accessed in simulation
   - Identify missing initializations (e.g., velocity_magnitude, entropy, acceleration)

3. **Review Simulation update methods:**
   - In `src/tde_sph/core/simulation.py`, find all quantity computation sections
   - Check if computed values are stored back to `self.particles`
   - Examples:
     - Velocity magnitude: `vmag = np.linalg.norm(particles.velocity, axis=1)` - is this stored?
     - Acceleration: computed in gravity solver - is it accessible?
     - Entropy: if EOS computes entropy, is it stored?

4. **Fix identified issues:**
   - For each missing attribute:
     - Add initialization to `ParticleSystem.__init__()`
     - Add shape validation to `_validate_shapes()`
     - Update `Simulation` to store computed values
     - Remove defensive access patterns (hasattr/getattr)
   - Follow same pattern as TASK0 (temperature fix)

5. **Verify HDF5 I/O consistency:**
   - Check `src/tde_sph/io/hdf5.py` write/read functions
   - Ensure all ParticleSystem attributes are saved and loaded
   - Add missing attributes to snapshot schema
   - Test round-trip: write snapshot, read back, verify all attributes match

6. **Create unit tests:**
   - For each fixed attribute, add test similar to temperature tests
   - Test initialization, computation, and storage
   - Test HDF5 round-trip for all attributes

7. **Document findings:**
   - Create a summary comment in code or commit message listing all bugs found and fixed
   - Format: "Fixed missing attributes: velocity_magnitude, entropy, acceleration, du_dt"

## Acceptance Criteria
- [ ] **T.4** All computed quantities verified and stored:
  - [ ] **T.4.1** Velocity magnitude: computed and stored if missing
  - [ ] **T.4.2** Entropy: computed and stored if EOS provides it
  - [ ] **T.4.3** Acceleration: accessible from gravity solver results
  - [ ] **T.4.4** du/dt: internal energy change rate stored for diagnostics
- [ ] **T.5** Coordinate transformation bugs fixed (if any found):
  - [ ] **T.5.1** Cartesian ↔ Boyer-Lindquist transformations verified
  - [ ] **T.5.2** Test with circular orbit at ISCO (known solution)
  - [ ] **T.5.3** Sign errors or missing terms corrected
- [ ] **T.6** HDF5 I/O bugs fixed:
  - [ ] **T.6.1** All particle attributes written to snapshots
  - [ ] **T.6.2** Attribute names consistent between writer and reader
  - [ ] **T.6.3** Round-trip test passes (write → read → compare)
- [ ] **T.7** All fixes documented in code comments
- [ ] No `hasattr()` or `getattr()` defensive patterns remain in physics code
- [ ] Unit tests added for all newly-initialized attributes
- [ ] All tests pass

## Code Review Checklist
- [ ] Search completed thoroughly (grep patterns covered all cases)
- [ ] Each identified bug has a clear fix (follows temperature pattern)
- [ ] No breaking changes to existing functionality
- [ ] HDF5 schema changes are backward-compatible (old snapshots still load)
- [ ] Docstrings updated to document new attributes
- [ ] Type hints added where appropriate
- [ ] Follows project conventions (snake_case, FP32 dtype, NumPy style)

## Reasoning Trace
**Why Layer 0 (Foundation):**
- Similar to temperature bug, these missing attributes block diagnostic features
- Must be fixed before Layer 1 (desktop diagnostics) can display comprehensive data
- Relatively independent from GUI code, can be tested in isolation

**Investigation Approach:**
- Start with automated search (grep) to find all defensive patterns
- Manual code review to understand which attributes are actually missing
- Systematic fix following established pattern (temperature fix)
- Comprehensive testing to prevent regressions

**Expected Findings:**
Based on code review in AI_PROMPT.md, likely missing attributes:
1. **velocity_magnitude** - Useful for coloring and statistics, probably computed but not stored
2. **entropy** - Thermodynamic quantity, EOS may compute but simulation doesn't store
3. **acceleration** - Needed for diagnostics and analysis, computed in gravity solver
4. **du_dt** - Energy change rate, important for conservation checks

**Potential Challenges:**
- Coordinate transform bugs may be complex (GR mathematics)
  - Solution: Test with known orbits, use literature values for validation
- HDF5 schema changes require careful backward compatibility
  - Solution: Make new attributes optional on read, use zeros as fallback for old files

**Trade-offs:**
- Could skip attributes that are never used, but completeness is preferred for scientific code
- All computed quantities should be accessible for post-processing and analysis
