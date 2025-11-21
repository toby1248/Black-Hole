## PROMPT
Write and execute automated unit and integration tests for all modified code. Ensure >80% coverage for changed files and all tests pass. This is the automated quality assurance complement to manual testing.

## COMPLEXITY
Medium

## CONTEXT REFERENCE
**Read:** `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

## TASK-SPECIFIC CONTEXT
- **Python tests:** pytest or unittest, co-located with source
- **JavaScript tests:** Jest (if configured) or manual validation
- **Coverage:** A.1-A.4 from AI_PROMPT.md

### Test Examples
**Temperature test:**
```python
def test_particle_system_initializes_temperature():
    particles = ParticleSystem(n_particles=100, n_dim=3)
    assert hasattr(particles, 'temperature')
    assert particles.temperature.shape == (100,)
    assert particles.temperature.dtype == np.float32
```

**Unit conversion test:**
```python
def test_code_to_physical_length():
    config = {'black_hole_mass': 1e6}
    result = code_to_physical(10.0, 'length', config)
    # Verify expected conversion
```

## LAYER
3 (Integration/Testing)

## PARALLELIZATION
Parallel with: [TASK11]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Mock external dependencies (file I/O, network, time)
- Tests must be deterministic
- Coverage measurement recommended but not required
