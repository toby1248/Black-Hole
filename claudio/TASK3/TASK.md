@dependencies [TASK0, TASK1]
# Task: Implement Metric Unit Conversion

## Summary
Add metric/physical unit conversion to the desktop diagnostics panel. Allow users to toggle between dimensionless code units and physical units (solar radii, solar masses, Kelvin, etc.) for all displayed quantities. This is SECOND PRIORITY (per user request "A second priority").

**Current State:**
- All values displayed in dimensionless code units
- No unit conversion available

**Target State:**
- Toggle button: "Show Dimensionless / Show Physical Units"
- When physical units selected, convert all displayed values using config-derived factors
- Display units next to values (e.g., "1.23e6 K" instead of "1.23e-3")

## Context Reference
**For complete environment context, see:**
- `../AI_PROMPT.md` - Contains full tech stack, architecture, coding conventions, and related code patterns

**Task-Specific Context:**
- **New files to create:**
  - `gui/unit_conversion.py` - Conversion utility module
- **Files to modify:**
  - `gui/data_display.py` - Add toggle button to diagnostics panels
  - Every diagnostic display widget (particle stats, performance, etc.)
- **Conversion factors source:**
  - Simulation config YAML contains: black hole mass (M_BH), star mass/radius
  - Code units normalized to G=c=1 (geometrized units)
  - Physical units derived from M_BH

## Complexity
Medium

## Dependencies
Depends on: [TASK0, TASK1]
Blocks: []
Parallel with: [TASK2, TASK4, TASK5]

## Detailed Steps
1. **Create unit conversion module:**
   - Create `gui/unit_conversion.py` with conversion functions
   - Functions:
     ```python
     def code_to_physical(value, unit_type, config):
         """Convert code units to physical units"""
         # unit_type: 'length', 'mass', 'time', 'energy', 'temperature', etc.
         pass

     def physical_to_code(value, unit_type, config):
         """Convert physical units to code units"""
         pass

     def get_unit_label(unit_type, use_physical):
         """Get unit label for display (e.g., 'R☉', 'M☉', 'K')"""
         pass
     ```

2. **Derive conversion factors:**
   - From YAML config, extract: M_BH (black hole mass in solar masses), M_star, R_star
   - Compute conversion factors:
     ```
     # Geometrized units: G = c = 1, lengths in units of M_BH
     length_to_Rg = 1.0  # Code units are already in Rg (gravitational radii)
     length_to_R_sun = M_BH * Rg_to_R_sun  # where Rg = GM_BH/c^2
     mass_to_M_sun = M_BH  # Code mass unit
     time_to_sec = (M_BH * M_sun * G / c^3)  # Geometric time unit
     temperature_to_K = (use EOS parameters or dimensionless → Kelvin formula)
     energy_to_erg = (M_BH * M_sun * c^2)
     ```
   - Use physical constants from scipy.constants or define manually

3. **Add toggle button to diagnostics:**
   - In DiagnosticsWidget (created in TASK2), add QCheckBox or QPushButton
   - Label: "Show Physical Units" (checked) / "Show Dimensionless Units" (unchecked)
   - Connect signal: `toggled(bool).connect(self.toggle_units)`

4. **Update all display widgets to use conversions:**
   - In ParticleStatsWidget, PerformanceMetricsWidget, etc.:
   - Store `use_physical` flag (boolean)
   - When updating values, apply conversion:
     ```python
     if self.use_physical:
         value_display = code_to_physical(value, 'length', self.config)
         unit_label = get_unit_label('length', True)  # "R☉"
     else:
         value_display = value
         unit_label = ""  # Or "code units"
     self.label.setText(f"{value_display:.3e} {unit_label}")
     ```

5. **Implement conversion for each quantity type:**
   - **Lengths:** code units → R☉ (solar radii) or Rg (gravitational radii)
   - **Masses:** code units → M☉ (solar masses)
   - **Times:** code units → seconds or orbital periods
   - **Energies:** code units → ergs
   - **Temperatures:** code units → Kelvin (use EOS equation)
   - **Velocities:** code units → km/s or c (speed of light fraction)
   - **Densities:** code units → g/cm³

6. **Handle missing config parameters:**
   - If M_BH not in config, display warning and disable physical units
   - Fallback: assume M_BH = 1.0 M☉ for conversions

7. **Add unit tests:**
   - Test `code_to_physical()` with known values (e.g., 1.0 code length = Rg)
   - Test round-trip: code → physical → code (should be identity)
   - Test `get_unit_label()` returns correct strings

8. **Update diagnostics display:**
   - All numeric values should show units when physical mode enabled
   - Format: "1.23e6 K" or "5.67 R☉"

## Acceptance Criteria
- [ ] **A.1** Metric unit conversion toggle implemented:
  - [ ] Toggle button in diagnostics panel
  - [ ] State persists during session (stays toggled)
- [ ] **A.1.1** Conversion functions for all quantity types:
  - [ ] Lengths: code units ↔ solar radii (R☉) or gravitational radii (Rg)
  - [ ] Masses: code units ↔ solar masses (M☉)
  - [ ] Times: code units ↔ seconds or orbital periods
  - [ ] Energies: code units ↔ ergs
  - [ ] Temperatures: code units ↔ Kelvin
  - [ ] Velocities: code units ↔ km/s or c
  - [ ] Densities: code units ↔ g/cm³
- [ ] **A.1.2** Unit labels displayed next to values in physical mode
- [ ] **A.1.3** Conversion factors derived from simulation config (M_BH, M_star, R_star)
- [ ] Toggle works in real-time (no need to restart simulation)
- [ ] Missing config parameters handled gracefully (warning + fallback)
- [ ] Unit tests verify conversion accuracy (round-trip identity)
- [ ] All tests pass

## Code Review Checklist
- [ ] Conversion functions are pure (no side effects, deterministic)
- [ ] Physical constants accurate (use scipy.constants or verified values)
- [ ] Rounding/precision appropriate for scientific display (3-4 significant figures)
- [ ] Unit labels use proper symbols (R☉, M☉, K, erg, etc.)
- [ ] Error handling: missing config keys, invalid values
- [ ] Type hints for all functions
- [ ] Docstrings explain unit conventions
- [ ] No hardcoded conversion factors (derive from constants)

## Reasoning Trace
**Why Second Priority:**
- User explicitly stated "A second priority" (after diagnostics implementation)
- Enhances scientific usability (physical units more interpretable)
- Independent from other features, can implement in parallel

**Unit System Choice:**
- **Code units (geometrized):** G = c = 1, lengths in units of M_BH
- **Physical units:** CGS (cgs for density, erg for energy) + astronomical (M☉, R☉)
- **Rationale:** Astrophysics convention, familiar to target users

**Conversion Factor Derivation:**
- **Gravitational radius:** Rg = GM_BH/c² ≈ 1.477 km * (M_BH/M☉)
- **Solar radius:** R☉ ≈ 6.96e10 cm
- **Geometric time:** t_g = GM_BH/c³ ≈ 4.926e-6 s * (M_BH/M☉)
- **Temperature:** Depends on EOS, typically dimensionless T → Kelvin via mean molecular weight

**Design Decisions:**
- **Toggle at widget level (not global):** Allows different panels to use different units
  - *Revision:* Global toggle is simpler and more consistent, use that
- **Store config in widget:** Needed to access M_BH and other parameters
- **Real-time toggle:** Update display immediately on checkbox change (no simulation restart)

**Edge Cases:**
- **No config loaded:** Disable physical units or use default M_BH = 1.0 M☉
- **Newtonian simulation:** No black hole, use M_star for normalization instead
- **Missing parameters:** Warn user and fallback to dimensionless

**Performance:**
- Conversion is lightweight (simple arithmetic), no performance concern
- Can precompute conversion factors when config loads
