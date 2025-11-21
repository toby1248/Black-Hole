## PROMPT
Implement metric unit conversion system for the desktop diagnostics panel. Create a toggle that allows users to switch between dimensionless code units (geometrized G=c=1) and physical units (solar radii, solar masses, Kelvin, ergs). This is SECOND PRIORITY (user specified "A second priority") and should work seamlessly with the diagnostics display from TASK2.

## COMPLEXITY
Medium

## CONTEXT REFERENCE
**For complete environment context, read:**
- `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md` - Contains full tech stack, architecture, project structure, coding conventions, and related code patterns

**You MUST read AI_PROMPT.md before executing this task to understand the environment.**

## TASK-SPECIFIC CONTEXT

### Unit System Overview
**Code units (dimensionless, geometrized):**
- G = c = 1 (gravitational constant and speed of light set to unity)
- Lengths measured in units of M_BH (black hole mass sets length scale)
- Masses measured in units of M_BH
- Times measured in units of GM_BH/c³

**Physical units (target display):**
- Lengths: Solar radii (R☉) or gravitational radii (Rg)
- Masses: Solar masses (M☉)
- Times: Seconds or orbital periods
- Energies: Ergs
- Temperatures: Kelvin
- Velocities: km/s or fraction of c
- Densities: g/cm³

### Files This Task Will Touch
**New file to create:**
- `gui/unit_conversion.py` - Conversion utility module
  - Functions: `code_to_physical()`, `physical_to_code()`, `get_unit_label()`
  - Class: `UnitConverter` (optional, encapsulates conversion factors)

**Files to modify:**
- `gui/data_display.py` - Add toggle button to diagnostics widget
- All diagnostic display widgets created in TASK2:
  - ParticleStatsWidget - Convert density, temperature, etc.
  - PerformanceMetricsWidget - Convert times
  - CoordinateDataWidget - Convert distances

**Test file:**
- `gui/unit_conversion.test.py` - Unit tests for conversion functions

### Patterns to Follow

**Conversion module structure:**
```python
# gui/unit_conversion.py
import numpy as np
from scipy import constants

# Physical constants
M_SUN = 1.989e33  # grams
R_SUN = 6.96e10   # cm
G = constants.G   # cm^3 g^-1 s^-2
C = constants.c   # cm/s

def code_to_physical(value, unit_type, config):
    """
    Convert code units to physical units.

    Parameters:
    - value: numeric value in code units
    - unit_type: str, one of 'length', 'mass', 'time', 'energy', 'temperature', 'velocity', 'density'
    - config: dict, simulation config containing M_BH, M_star, R_star

    Returns:
    - Converted value in physical units
    """
    M_BH = config.get('black_hole_mass', 1.0)  # in M_sun

    if unit_type == 'length':
        # Code units: M_BH (gravitational radii)
        # Physical: R_sun
        Rg = G * M_BH * M_SUN / C**2  # gravitational radius in cm
        return value * Rg / R_SUN  # Convert to solar radii

    elif unit_type == 'mass':
        return value * M_BH  # Code mass unit is M_BH

    elif unit_type == 'time':
        t_g = G * M_BH * M_SUN / C**3  # Geometric time unit
        return value * t_g  # Convert to seconds

    # ... etc for other types

def get_unit_label(unit_type, use_physical):
    """Get display label for unit type"""
    if not use_physical:
        return ""  # Dimensionless

    labels = {
        'length': 'R☉',
        'mass': 'M☉',
        'time': 's',
        'energy': 'erg',
        'temperature': 'K',
        'velocity': 'km/s',
        'density': 'g/cm³',
    }
    return labels.get(unit_type, '')
```

**Toggle button integration (in DiagnosticsWidget):**
```python
class DiagnosticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}  # Will be set from main window
        self.use_physical = False  # Default: code units

        # ... create tabs

        # Add toggle button
        self.unit_toggle = QCheckBox("Show Physical Units")
        self.unit_toggle.toggled.connect(self.toggle_units)
        layout.addWidget(self.unit_toggle)

    def toggle_units(self, checked):
        self.use_physical = checked
        # Notify all sub-widgets to update display
        self.particle_stats_widget.set_unit_mode(self.use_physical, self.config)
        # ... notify other widgets

    def set_config(self, config):
        """Called when simulation starts, provides config for conversions"""
        self.config = config
```

**Value display with units:**
```python
def update_value(self, code_value, unit_type):
    if self.use_physical:
        phys_value = code_to_physical(code_value, unit_type, self.config)
        unit_label = get_unit_label(unit_type, True)
        self.label.setText(f"{phys_value:.3e} {unit_label}")
    else:
        self.label.setText(f"{code_value:.3e}")
```

### Conversion Formulas

**Key conversion factors:**
```
Rg = GM_BH/c² = 1.477 km × (M_BH/M☉)
R☉ = 6.96×10^10 cm
t_g = GM_BH/c³ = 4.926×10^-6 s × (M_BH/M☉)

Length: L_phys[R☉] = L_code × (Rg / R☉) = L_code × M_BH × 1.477 km / R☉
Mass: M_phys[M☉] = M_code × M_BH
Time: t_phys[s] = t_code × t_g
Energy: E_phys[erg] = E_code × M_BH × M☉ × c²
Velocity: v_phys[km/s] = v_code × c (since code velocities are fractions of c)
Temperature: T_phys[K] = T_code × (depends on EOS normalization)
Density: ρ_phys[g/cm³] = ρ_code × (M_BH × M☉) / (Rg)³
```

**Temperature conversion (requires EOS knowledge):**
- Ideal gas: T = (γ-1) × (internal energy per unit mass) × (mean molecular weight) × m_proton / k_B
- Dimensionless T_code → physical T_phys depends on normalization in EOS module
- Check `src/tde_sph/eos/ideal_gas.py` for temperature calculation

### Integration Points
- **Config source:** Simulation config YAML (loaded by main_window.py)
- **Pass config to diagnostics:** main_window.py calls `diagnostics_widget.set_config(config_dict)`
- **Toggle persistence:** Could use QSettings to remember user preference (optional)

### Config Parameters Needed
Extract from YAML:
```yaml
# configs/schwarzschild_tde.yaml
black_hole:
  mass: 1.0e6  # M_sun
star:
  mass: 1.0    # M_sun
  radius: 1.0  # R_sun
```

Access in code:
```python
M_BH = config.get('black_hole', {}).get('mass', 1.0)
M_star = config.get('star', {}).get('mass', 1.0)
R_star = config.get('star', {}).get('radius', 1.0)
```

## EXTRA DOCUMENTATION

### Testing Strategy
**Unit tests (gui/unit_conversion.test.py):**
1. Test code_to_physical for length: 1.0 code unit → expected R☉
2. Test code_to_physical for mass: 1.0 code unit → M_BH M☉
3. Test round-trip: code → physical → code (verify identity within tolerance)
4. Test get_unit_label returns correct strings for each type
5. Test edge case: missing M_BH in config (should use fallback)

**Integration test:**
1. Load config with known M_BH (e.g., 1e6 M☉)
2. Start simulation
3. Toggle physical units
4. Verify displayed values change (not just format)
5. Toggle back to code units
6. Verify values revert to original

**Manual verification:**
1. Calculate expected physical value manually for one quantity
2. Compare with GUI display
3. Example: If code length = 10.0, M_BH = 1e6 M☉:
   - Rg = 1.477 km × 1e6 = 1.477e9 km = 1.477e14 cm
   - R☉ = 6.96e10 cm
   - L_phys = 10.0 × (1.477e14 / 6.96e10) = 2.12e4 R☉
   - GUI should display ~2.12e4 R☉

### Implementation Checklist
- [ ] Create `gui/unit_conversion.py` module
- [ ] Implement `code_to_physical()` for all unit types
- [ ] Implement `physical_to_code()` (for completeness)
- [ ] Implement `get_unit_label()` with proper symbols
- [ ] Add toggle button to DiagnosticsWidget
- [ ] Connect toggle signal to update method
- [ ] Modify ParticleStatsWidget to use conversions
- [ ] Modify PerformanceMetricsWidget to use conversions
- [ ] Modify CoordinateDataWidget to use conversions
- [ ] Add `set_config()` method to receive config from main window
- [ ] Write unit tests for conversion functions
- [ ] Write integration test for toggle functionality
- [ ] Manual test with known config values

## LAYER
1 (Desktop GUI Enhancement - parallel)

## PARALLELIZATION
Parallel with: [TASK2, TASK4, TASK5]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Use scipy.constants for physical constants (no hardcoded magic numbers)
- Conversion functions must be pure (deterministic, no side effects)
- Handle missing config gracefully (use fallback values, don't crash)
- Physical constants should be accurate to at least 4 significant figures
- Unit labels should use Unicode symbols (R☉, M☉) not ASCII alternatives
- Toggle state should update display immediately (real-time update)
- Test ONLY changed files (unit_conversion.py, data_display.py)
- Follow PEP 8 style and existing code conventions
- Add type hints and NumPy-style docstrings
