@dependencies []
# Task: Fix Temperature Attribute Bug in ParticleSystem

## Summary
Fix the critical bug where `ParticleSystem` does not initialize the `temperature` attribute, causing temperature data to be unavailable throughout the simulation. This is a foundational bug that blocks temperature-related features in both desktop and web GUIs.

**Root Cause Analysis:**
- `src/tde_sph/sph/particles.py:114-117` initializes `density`, `pressure`, `sound_speed` but NOT `temperature`
- `src/tde_sph/core/simulation.py:517-523` computes temperature but only stores if attribute exists
- `gui/simulation_thread.py:262-263` uses `hasattr()` fallback to 0.0 to avoid crashes

**Expected Behavior:**
Temperature should be computed and accessible like density/pressure from the first simulation step onwards.

## Context Reference
**For complete environment context, see:**
- `../AI_PROMPT.md` - Contains full tech stack, architecture, coding conventions, and related code patterns

**Task-Specific Context:**
- **Files to modify:**
  - `src/tde_sph/sph/particles.py:114-117` - Add temperature array initialization
  - `src/tde_sph/sph/particles.py:_validate_shapes()` - Add temperature shape validation
  - `src/tde_sph/core/simulation.py:517-523` - Remove hasattr() check, always store temperature
  - `gui/simulation_thread.py:262-263` - Remove hasattr() fallback
- **Pattern to follow:**
  - Follow initialization pattern for density/pressure/sound_speed at particles.py:114-117
  - Use `np.zeros(n_particles, dtype=np.float32)` for GPU compatibility
- **Integration points:**
  - Temperature is computed by `tde_sph.eos.ideal_gas` module
  - Used by GUI diagnostics (data_display.py) and web visualizer

## Complexity
Low

## Dependencies
Depends on: []
Blocks: [TASK1, TASK2, TASK3, TASK4, TASK5, TASK6, TASK7, TASK8]
Parallel with: []

## Detailed Steps
1. **Add temperature attribute to ParticleSystem:**
   - In `src/tde_sph/sph/particles.py:__init__()`, after line 117, add:
     ```python
     self.temperature = np.zeros(n_particles, dtype=np.float32)
     ```
   - Add temperature to `_validate_shapes()` method (check shape matches n_particles)
   - Add getter method `get_temperature()` if pattern exists for other quantities

2. **Remove conditional storage in Simulation:**
   - In `src/tde_sph/core/simulation.py`, find the temperature computation section (around line 517-523)
   - Remove the `hasattr()` check
   - Change to: `self.particles.temperature = temperature` (unconditionally)

3. **Remove fallback in SimulationThread:**
   - In `gui/simulation_thread.py:262-263`, remove:
     ```python
     getattr(self.simulation.particles, 'temperature', 0.0)
     ```
   - Replace with direct access:
     ```python
     self.simulation.particles.temperature
     ```

4. **Create unit tests:**
   - In `src/tde_sph/sph/particles.test.py` (create if doesn't exist):
     - Test: ParticleSystem initializes temperature array
     - Test: Temperature has correct shape (n_particles,)
     - Test: Temperature has correct dtype (float32)
   - In `src/tde_sph/core/simulation.test.py`:
     - Integration test: Start simulation, verify temperature is computed in first step
     - Test: Temperature values are physically reasonable (>0 K)

5. **Verify fix:**
   - Run unit tests
   - Manually start simulation via desktop GUI
   - Check that temperature data appears in diagnostics from step 0

## Acceptance Criteria
- [ ] **T.1** ParticleSystem initializes temperature array in `__init__()`
- [ ] **T.1.1** Temperature array has shape `(n_particles,)` and dtype `float32`
- [ ] **T.1.2** Temperature shape validation added to `_validate_shapes()`
- [ ] **T.2** Simulation always computes and stores temperature (no hasattr() checks)
- [ ] **T.2.1** Temperature is available from first simulation step onwards
- [ ] **T.3** SimulationThread reports temperature correctly (no fallback to 0.0)
- [ ] **T.3.1** Mean and max temperature displayed in GUI diagnostics
- [ ] Unit tests pass with 100% coverage of modified code
- [ ] Manual test: Launch GUI, start simulation, verify temperature != 0.0 in diagnostics

## Code Review Checklist
- [ ] Clear naming: `temperature` attribute follows existing convention (matches `density`, `pressure`)
- [ ] No dead code: All hasattr() fallbacks for temperature removed
- [ ] Errors handled: Shape validation in `_validate_shapes()` will catch mismatches
- [ ] Follows conventions: Uses FP32 (not FP64) for GPU compatibility
- [ ] Type hints: Added if pattern exists in ParticleSystem class
- [ ] Docstrings: Updated to document temperature attribute
- [ ] No regressions: Existing attributes (density, pressure, etc.) still work

## Reasoning Trace
**Design Decision:**
- Initialize temperature to zeros rather than NaN or uninitialized because:
  - Consistent with existing pattern (density, pressure initialized to zeros)
  - Safe default (0 K is physically meaningful lower bound)
  - Avoids NaN propagation in calculations

**Why this is Layer 0 (Foundation):**
- All temperature-related features (diagnostics, visualization, unit conversion) depend on this fix
- Must complete before any Layer 1 or Layer 2 tasks can use temperature data
- Simple, atomic change with clear verification

**Trade-offs:**
- Could add temperature as optional feature with configuration flag, but simplicity is preferred
- Temperature is fundamental to astrophysical simulations, should always be available
