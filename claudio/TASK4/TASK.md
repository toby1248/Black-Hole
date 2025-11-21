@dependencies [TASK0]
# Task: Implement Preferences Dialog

## Summary
Implement the placeholder preferences dialog in the desktop GUI. Create a modal dialog with settings categories (General, Visualization, Performance) that persist via QSettings. This completes the desktop GUI button implementation requirements.

**Current State:**
- Preferences action exists in menu (`main_window.py:726`) but not implemented
- Action connected to empty slot or shows "Not implemented" message

**Target State:**
- Functional preferences dialog with tabbed categories
- Settings persist across sessions (QSettings already used in codebase)
- Apply/Cancel/OK buttons functional
- Settings affect application behavior

## Context Reference
**For complete environment context, see:**
- `../AI_PROMPT.md` - Contains full tech stack, architecture, coding conventions, and related code patterns

**Task-Specific Context:**
- **New file to create:**
  - `gui/preferences_dialog.py` - Preferences dialog implementation
- **File to modify:**
  - `gui/main_window.py:726` - Connect menu action to dialog
- **Existing QSettings usage:**
  - `main_window.py` already uses QSettings for window geometry
  - Follow same pattern for new preferences
- **Settings categories:**
  - General: Default config directory, auto-save interval
  - Visualization: Default colormap, point size, camera settings
  - Performance: GPU usage preference, thread count

## Complexity
Low

## Dependencies
Depends on: [TASK0]
Blocks: []
Parallel with: [TASK2, TASK3, TASK5]

## Detailed Steps
1. **Create PreferencesDialog class:**
   - In `gui/preferences_dialog.py`, create `PreferencesDialog(QDialog)`
   - Use QTabWidget for organizing categories
   - Three tabs: General, Visualization, Performance
   - Standard button box: OK, Cancel, Apply

2. **Implement General Settings tab:**
   - **Default config directory:** QLineEdit + browse button
   - **Auto-save interval:** QSpinBox (minutes, 0 = disabled)
   - **Show confirmation dialogs:** QCheckBox
   - Layout: QFormLayout

3. **Implement Visualization Settings tab:**
   - **Default colormap:** QComboBox (viridis, plasma, inferno, hot, cool)
   - **Default point size:** QDoubleSpinBox (0.5 - 10.0)
   - **Camera settings:** Default position (x, y, z as QDoubleSpinBox)
   - **Show axes by default:** QCheckBox
   - **Show black hole by default:** QCheckBox

4. **Implement Performance Settings tab:**
   - **Prefer GPU (CUDA):** QCheckBox (if available)
   - **Thread count:** QSpinBox (1 - CPU count, 0 = auto)
   - **Progress update frequency:** QSpinBox (steps between updates)
   - **Memory limit:** QSpinBox (MB, 0 = unlimited)

5. **Load settings from QSettings:**
   - In dialog `__init__()`, load current values:
     ```python
     settings = QSettings('TDE-SPH', 'Simulator')
     default_dir = settings.value('default_config_dir', os.getcwd())
     self.config_dir_edit.setText(default_dir)
     # ... etc for all settings
     ```

6. **Save settings on Apply/OK:**
   - Connect Apply button to `apply_settings()` method
   - Connect OK button to `apply_settings()` then `accept()`
   - Save all settings:
     ```python
     settings = QSettings('TDE-SPH', 'Simulator')
     settings.setValue('default_config_dir', self.config_dir_edit.text())
     # ... etc
     settings.sync()
     ```

7. **Connect to main window menu:**
   - In `main_window.py:726`, replace placeholder with:
     ```python
     def show_preferences(self):
         dialog = PreferencesDialog(self)
         if dialog.exec() == QDialog.Accepted:
             # Settings already saved by dialog
             # Optionally refresh UI to reflect changes
             pass
     ```

8. **Apply settings to application:**
   - Main window reads settings on startup (already does this for geometry)
   - Update config file open dialog to use default directory
   - Update visualization widgets to use default colormap/point size
   - Pass performance settings to SimulationThread

9. **Add validation:**
   - Validate paths exist (config directory)
   - Validate numeric ranges (point size > 0, thread count > 0)
   - Show error message if invalid

10. **Create unit tests:**
    - Test settings save and load correctly
    - Test validation catches invalid inputs
    - Mock QSettings to avoid file I/O in tests

## Acceptance Criteria
- [ ] **B.1** Preferences dialog opens from menu (Edit → Preferences)
- [ ] **B.1.1** General settings implemented:
  - [ ] Default config directory (with browse button)
  - [ ] Auto-save interval (minutes)
  - [ ] Confirmation dialog toggle
- [ ] **B.1.2** Visualization settings implemented:
  - [ ] Default colormap selection
  - [ ] Default point size
  - [ ] Default camera position
  - [ ] Show axes/black hole defaults
- [ ] **B.1.3** Performance settings implemented:
  - [ ] GPU preference toggle
  - [ ] Thread count setting
  - [ ] Progress update frequency
  - [ ] Memory limit
- [ ] **B.1.4** Settings persisted via QSettings
- [ ] **B.1.5** Apply/Cancel/OK buttons functional
- [ ] Settings loaded on application startup
- [ ] Settings affect application behavior (e.g., default directory actually used)
- [ ] Input validation prevents invalid settings
- [ ] Unit tests pass

## Code Review Checklist
- [ ] Dialog is modal (blocks main window until closed)
- [ ] Settings keys are namespaced consistently ('tde_sph/default_config_dir', etc.)
- [ ] All widgets have sensible default values
- [ ] Validation provides clear error messages
- [ ] Settings sync() called after save
- [ ] No hardcoded paths or magic numbers
- [ ] Follows PyQt conventions (signals/slots, layouts)
- [ ] Type hints and docstrings added

## Reasoning Trace
**Why Low Complexity:**
- Standard PyQt dialog pattern (QDialog + QTabWidget + QSettings)
- No complex logic, mostly UI construction and data binding
- Well-established pattern in existing codebase (window geometry persistence)

**Design Decisions:**
- **QTabWidget over single scrolling view:** Organizes settings by category, reduces clutter
- **QFormLayout for settings:** Standard key-value layout, clear and compact
- **Apply button in addition to OK:** Allows users to test settings without closing dialog
- **QSettings over JSON config file:** Already used in project, cross-platform, simple API

**Settings Organization:**
- **General:** Application-wide behavior (file paths, dialogs)
- **Visualization:** Default rendering settings (affects new visualizations)
- **Performance:** Resource usage preferences (affects simulation execution)

**Integration Strategy:**
- Settings loaded in main_window.__init__() (already does this for geometry)
- Settings passed to relevant components:
  - Default directory → file dialog initial path
  - Colormap/point size → web viewer initialization
  - GPU/thread settings → SimulationThread constructor

**Edge Cases:**
- **Config directory doesn't exist:** Validate on save, show error, don't save invalid path
- **Thread count > CPU count:** Warn but allow (may be intentional for testing)
- **GPU not available but preference enabled:** Ignore preference, use CPU, show warning in performance panel
