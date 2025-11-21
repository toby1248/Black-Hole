# Running TDE-SPH Tests

## Python Tests

### Prerequisites
```bash
pip install pytest
```

### Run All Tests
```bash
cd /path/to/Black-Hole
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Temperature and particle tests
pytest tests/test_particles_temperature.py tests/test_simulation_temperature.py -v

# Unit conversion tests
pytest tests/test_unit_conversion.py -v

# GUI widget tests
pytest tests/test_diagnostics_widget.py tests/test_preferences_dialog.py -v

# Attribute alias tests
pytest tests/test_attribute_aliases.py -v
```

### Test Coverage
```bash
pip install pytest-cov
pytest tests/ --cov=src/tde_sph --cov-report=html
# Open htmlcov/index.html to view coverage report
```

## JavaScript Tests

### Prerequisites
```bash
cd web/
npm install
```

### Run Tests
```bash
npm test
```

### Run with Watch Mode
```bash
npm test -- --watch
```

## Test File Locations

### Python Tests (tests/)
| Test File | Purpose | Task |
|-----------|---------|------|
| test_particles_temperature.py | Temperature attribute initialization | TASK0 |
| test_attribute_aliases.py | Attribute naming consistency | TASK1 |
| test_diagnostics_widget.py | DiagnosticsWidget functionality | TASK2 |
| test_unit_conversion.py | Metric unit conversion | TASK3 |
| test_preferences_dialog.py | Preferences persistence | TASK4 |
| test_simulation_temperature.py | Simulation integration | TASK0 |

### JavaScript Tests (web/js/tests/)
| Test File | Purpose | Task |
|-----------|---------|------|
| visualizer.test.js | Colormap, demo data, statistics | TASK6-9 |

## Test Conventions

1. **Co-located tests**: Test files are located near source files or in tests/ directory
2. **Self-contained mocks**: All mocks defined within test files
3. **Naming**: `test_<module>.py` or `<module>.test.js`
4. **Deterministic**: No timing-dependent or flaky tests

## Expected Test Results

All tests should pass with output similar to:
```
==================== test session starts ====================
collected 20 items

tests/test_unit_conversion.py::test_code_to_physical_length PASSED
tests/test_unit_conversion.py::test_physical_to_code_roundtrip PASSED
...
==================== 20 passed in 0.5s ====================
```

## Continuous Integration

The tests can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install pytest numpy scipy
      - name: Run tests
        run: pytest tests/ -v
```
