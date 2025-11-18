# Continuous Integration (CI) for TDE-SPH

Automated testing and validation for the TDE-SPH framework using GitHub Actions.

## Overview

The CI pipeline runs on every push and pull request to ensure code quality and catch regressions early. Tests are executed across multiple Python versions to ensure broad compatibility.

## Workflows

### Main CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main`, `develop`, or `claude/**` branches
- Pull requests to `main` or `develop`
- Manual trigger via GitHub Actions UI

**Jobs:**

1. **Test** (Python 3.10, 3.11, 3.12 on Ubuntu):
   - Install dependencies
   - Run linting (flake8)
   - Run unit tests with coverage
   - Run regression tests
   - Upload coverage to Codecov

2. **Regression**:
   - Fast regression tests only
   - Validates core functionality with small N
   - Ensures numerical stability

3. **Examples**:
   - Run example scripts (dry run)
   - Ensures examples stay functional

4. **Export Tools**:
   - Test Blender/ParaView export tools
   - Validates all export formats (PLY, VTK, OBJ)

## Test Organization

### Unit Tests (`tests/test_*.py`)
Focused tests for individual modules:
- `test_config.py`: Configuration loading and validation
- `test_diagnostics.py`: Diagnostic output
- `test_disc.py`: Disc IC generator
- `test_energy_conservation.py`: Energy tracking
- `test_export_to_blender.py`: Export tools
- `test_gravity.py`: Gravity solvers
- `test_integration_gr.py`: GR time integration
- `test_io_visualization.py`: I/O and visualization
- `test_metric.py`: Spacetime metrics
- `test_radiation_gas_eos.py`: Radiation gas EOS
- `test_simple_cooling.py`: Cooling model

### Regression Tests (`tests/test_regression.py`)
End-to-end tests with small particle counts to catch behavior changes:
- **IC regression**: Polytrope and disc mass conservation, Keplerian velocities
- **Energy regression**: Conservation, scaling, static particles
- **EOS regression**: Ideal gas gamma, radiation pressure ratio
- **Metric regression**: Minkowski flatness, Schwarzschild horizon, Kerr → Schwarzschild
- **IO regression**: HDF5 roundtrip, diagnostic logging
- **Radiation regression**: Cooling reduces internal energy
- **Workflow regression**: Full IC → export pipeline
- **Numerical stability**: Edge cases (zero mass, extreme radii)

**Markers:**
- `regression`: All regression tests
- `fast`: Tests completing in < 5 seconds
- `slow`: Tests taking 5-30 seconds

**Run specific tests:**
```bash
pytest tests/test_regression.py -v -m regression          # All regression
pytest tests/test_regression.py -v -m "regression and fast"  # Fast only
```

## Local Testing

### Quick Test Script

Use the `run_tests.sh` helper script:

```bash
# Run all tests
./run_tests.sh

# Run only fast tests
./run_tests.sh fast

# Run regression tests
./run_tests.sh regression

# Run specific module tests
./run_tests.sh module energy_conservation

# Generate coverage report
./run_tests.sh coverage

# Show help
./run_tests.sh help
```

### Manual pytest Commands

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/tde_sph --cov-report=html

# Specific test file
pytest tests/test_energy_conservation.py -v

# Specific test function
pytest tests/test_energy_conservation.py::test_kinetic_energy_calculation -v

# Regression tests only
pytest tests/test_regression.py -v -m regression

# Fast tests only (across all files)
pytest tests/ -v -m fast

# Exclude slow tests
pytest tests/ -v -m "not slow"
```

## Coverage

Code coverage is tracked using `pytest-cov` and uploaded to Codecov.

**Current Coverage (as of Phase 3):**
- `energy_diagnostics.py`: 81%
- `disc.py`: 95%
- `export_to_blender.py`: (new, not yet measured)

**View coverage locally:**
```bash
pytest tests/ --cov=src/tde_sph --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Regression Test Philosophy

Regression tests are designed to:
1. **Catch numerical drift**: Ensure algorithms produce consistent results
2. **Validate physics**: Check conservation laws, scaling relations
3. **Test edge cases**: Zero mass, extreme parameters, boundary conditions
4. **Run fast**: Use small N (10-200 particles) for quick CI execution
5. **Be deterministic**: Use fixed seeds for reproducibility

**Example:**
```python
@pytest.mark.regression
@pytest.mark.fast
def test_polytrope_mass_conservation(self):
    """Regression: Polytrope mass should be conserved to high precision."""
    poly = Polytrope(n_particles=100, total_mass=1.0, ...)
    pos, vel, masses, u, h = poly.generate()
    total_mass = np.sum(masses)
    assert abs(total_mass - 1.0) / 1.0 < 1e-10
```

## Adding New Tests

### For New Features

1. **Unit test** in appropriate `tests/test_<module>.py`:
   ```python
   def test_new_feature():
       """Test new feature functionality."""
       # Setup
       # Execute
       # Assert
   ```

2. **Regression test** in `tests/test_regression.py`:
   ```python
   @pytest.mark.regression
   @pytest.mark.fast
   def test_new_feature_regression(self):
       """Regression: New feature should maintain stable behavior."""
       # Small N test
       # Physical validation
       # Numerical stability check
   ```

3. **Update CI workflow** if new dependencies are required

### Best Practices

- Use `pytest.approx()` for float comparisons: `assert value == pytest.approx(expected, rel=1e-6)`
- Mark slow tests: `@pytest.mark.slow`
- Use temporary directories for file I/O: `with tempfile.TemporaryDirectory() as tmpdir:`
- Test both success and failure cases
- Document expected behavior in docstrings
- Use descriptive assertion messages: `assert condition, "Helpful error message"`

## CI Status Badges

Add to README.md:
```markdown
![Tests](https://github.com/user/Black-Hole/workflows/TDE-SPH%20CI/badge.svg)
[![codecov](https://codecov.io/gh/user/Black-Hole/branch/main/graph/badge.svg)](https://codecov.io/gh/user/Black-Hole)
```

## Troubleshooting

### Tests pass locally but fail in CI
- Check Python version (CI uses 3.10-3.12)
- Verify all dependencies are in `requirements.txt` or `pyproject.toml`
- Ensure paths are relative, not absolute
- Check for platform-specific issues (CI uses Ubuntu)

### Flaky tests
- Use fixed random seeds: `np.random.seed(42)`
- Increase tolerances for numerical comparisons
- Add retries for network-dependent tests (if any)

### Slow CI
- Mark slow tests: `@pytest.mark.slow`
- Run fast tests in parallel CI jobs
- Use smaller N in regression tests
- Cache pip dependencies in workflow

## Future Enhancements

- **GPU tests**: Add CUDA-enabled tests when GPU runners available
- **Performance benchmarks**: Track execution time regressions
- **Nightly builds**: Run long-running tests overnight
- **Pre-commit hooks**: Run linting and fast tests before commit
- **Docker**: Containerized testing for reproducibility

---

**Maintainer:** TDE-SPH Development Team
**Last Updated:** 2025-11-18
