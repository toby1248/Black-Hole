## PROMPT
Fix the critical temperature attribute bug in the TDE-SPH particle system. The `ParticleSystem` class does not initialize the `temperature` attribute, causing temperature data to be unavailable throughout simulations. You must add the temperature array initialization, remove all conditional checks (hasattr), and ensure temperature is computed and stored from the first simulation step onwards.

## COMPLEXITY
Low

## CONTEXT REFERENCE
**For complete environment context, read:**
- `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md` - Contains full tech stack, architecture, project structure, coding conventions, and related code patterns

**You MUST read AI_PROMPT.md before executing this task to understand the environment.**

## TASK-SPECIFIC CONTEXT

### Root Cause
The temperature bug has three manifestations:
1. **Missing initialization:** `particles.py:114-117` initializes density/pressure/sound_speed but NOT temperature
2. **Conditional storage:** `simulation.py:517-523` only stores temperature if attribute exists
3. **Fallback workaround:** `simulation_thread.py:262-263` uses `getattr(..., 'temperature', 0.0)` to avoid crashes

### Files This Task Will Touch
**Primary modifications:**
- `src/tde_sph/sph/particles.py:114-117` - Add temperature initialization after sound_speed
- `src/tde_sph/sph/particles.py:_validate_shapes()` - Add temperature shape validation
- `src/tde_sph/core/simulation.py:517-523` - Remove hasattr() check, always assign temperature
- `gui/simulation_thread.py:262-263` - Replace getattr() with direct temperature access

**New test files:**
- `src/tde_sph/sph/particles.test.py` (create if missing) - Unit tests for temperature initialization
- `src/tde_sph/core/simulation.test.py` (create if missing) - Integration tests for temperature computation

### Patterns to Follow
**Initialization pattern (from particles.py:114-117):**
```python
self.density = np.zeros(n_particles, dtype=np.float32)
self.pressure = np.zeros(n_particles, dtype=np.float32)
self.sound_speed = np.zeros(n_particles, dtype=np.float32)
# ADD THIS:
self.temperature = np.zeros(n_particles, dtype=np.float32)
```

**Key conventions:**
- Use `dtype=np.float32` (NOT float64) for GPU compatibility
- Initialize to zeros (NOT NaN or uninitialized)
- Add to `_validate_shapes()` method for consistency
- Follow existing pattern for getter methods if they exist

### Integration Points
- **EOS module:** `tde_sph.eos.ideal_gas` computes temperature values
- **Simulation loop:** Temperature computed during each step in `simulation.py`
- **GUI display:** `data_display.py` and `simulation_thread.py` display temperature statistics
- **HDF5 I/O:** Temperature should be written to snapshots (verify in `tde_sph.io.hdf5`)

### Expected Behavior After Fix
1. ParticleSystem has `temperature` attribute from initialization
2. Simulation computes and stores temperature every step (no conditional logic)
3. GUI displays actual temperature values (not 0.0 fallback)
4. Temperature data available for visualization and diagnostics

## EXTRA DOCUMENTATION

### Testing Strategy
**Unit tests (required):**
1. Test ParticleSystem initialization includes temperature with correct shape/dtype
2. Test temperature shape validation catches mismatches
3. Test simulation stores temperature after first step

**Manual verification:**
1. Launch desktop GUI (`python gui/main_window.py`)
2. Load config: `configs/schwarzschild_tde.yaml`
3. Start simulation
4. Verify diagnostics panel shows temperature > 0.0 (not 0.0 fallback)

### Verification Checklist
- [ ] Temperature array initialized in ParticleSystem.__init__()
- [ ] Temperature validated in _validate_shapes()
- [ ] No hasattr() checks remain in simulation.py
- [ ] No getattr() fallbacks remain in simulation_thread.py
- [ ] Unit tests pass
- [ ] Manual GUI test confirms temperature data displays correctly

## LAYER
0 (Foundation)

## PARALLELIZATION
Parallel with: []

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Use `dtype=np.float32` for temperature array (GPU performance requirement)
- Do NOT change existing attributes (density, pressure, sound_speed)
- Do NOT modify temperature calculation logic in EOS module
- Must maintain backward compatibility (old snapshots without temperature should still load)
- Test ONLY changed files (particles.py, simulation.py, simulation_thread.py)
- Follow PEP 8 style guide and existing code conventions
- Add type hints and NumPy-style docstrings
