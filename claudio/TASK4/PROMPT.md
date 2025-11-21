## PROMPT
Implement the preferences dialog for the desktop GUI. Create a modal dialog with three tabs (General, Visualization, Performance) containing application settings that persist via QSettings. Connect the menu action at `main_window.py:726` to open this dialog.

## COMPLEXITY
Low

## CONTEXT REFERENCE
**For complete environment context, read:**
- `J:\AI\vibes\black hole 3D\Black-Hole\.claudiomiro\AI_PROMPT.md`

**You MUST read AI_PROMPT.md before executing this task.**

## TASK-SPECIFIC CONTEXT

### Files This Task Will Touch
- **New:** `gui/preferences_dialog.py` - Complete dialog implementation
- **Modify:** `gui/main_window.py:726` - Connect menu action to dialog
- **Test:** `gui/preferences_dialog.test.py` - Unit tests

### Patterns to Follow
**QSettings usage (from main_window.py):**
```python
settings = QSettings('TDE-SPH', 'Simulator')
value = settings.value('key', default_value)
settings.setValue('key', value)
settings.sync()
```

**Dialog structure:**
```python
class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self.create_general_tab(), "General")
        tabs.addTab(self.create_visualization_tab(), "Visualization")
        tabs.addTab(self.create_performance_tab(), "Performance")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self.accept_and_save)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        layout.addWidget(buttons)

        self.load_settings()
```

### Settings to Implement
**General tab:**
- Default config directory: `QLineEdit` + browse button (`QFileDialog.getExistingDirectory`)
- Auto-save interval: `QSpinBox` (0-60 minutes, 0 = disabled)
- Show confirmation dialogs: `QCheckBox`

**Visualization tab:**
- Default colormap: `QComboBox` (viridis, plasma, inferno, hot, cool, rainbow)
- Default point size: `QDoubleSpinBox` (0.5 - 10.0)
- Default camera position: 3x `QDoubleSpinBox` (x, y, z)
- Show axes by default: `QCheckBox`
- Show black hole by default: `QCheckBox`

**Performance tab:**
- Prefer GPU (CUDA): `QCheckBox` (disabled if CUDA not available)
- Thread count: `QSpinBox` (1 - os.cpu_count(), 0 = auto)
- Progress update frequency: `QSpinBox` (steps between UI updates)
- Memory limit: `QSpinBox` (MB, 0 = unlimited)

## EXTRA DOCUMENTATION

### Implementation Checklist
- [ ] Create `PreferencesDialog` class in new file
- [ ] Create three tab widgets (General, Visualization, Performance)
- [ ] Add all settings widgets with proper defaults
- [ ] Implement `load_settings()` from QSettings
- [ ] Implement `save_settings()` to QSettings
- [ ] Add input validation (paths exist, ranges valid)
- [ ] Connect to main_window.py menu action
- [ ] Write unit tests (mock QSettings)
- [ ] Manual test: open dialog, change settings, verify persistence

## LAYER
1 (Desktop GUI Enhancement)

## PARALLELIZATION
Parallel with: [TASK2, TASK3, TASK5]

## CONSTRAINTS
- IMPORTANT: Do not perform any git commit or git push
- Dialog must be modal (blocks main window)
- QSettings keys must be consistent ('tde_sph/setting_name')
- Validate all inputs before saving
- Follow PyQt6 (or PyQt5) conventions
- Test ONLY changed files
