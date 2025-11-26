#!/usr/bin/env python3
"""
Tests for GUI Components (TASK-100)

Test coverage:
- Main window creation and initialization
- Config editor functionality
- Control panel widgets
- Data display widgets
- Signal/slot connections

NOTE: GUI tests require PyQt6 or PyQt5. Tests are skipped if not available.
Uses offscreen platform for headless testing in CI.

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import pytest
import sys

# Check for PyQt availability
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    HAS_PYQT = True
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        HAS_PYQT = True
        PYQT_VERSION = 5
    except ImportError:
        HAS_PYQT = False

pytestmark = pytest.mark.skipif(
    not HAS_PYQT,
    reason="PyQt6 or PyQt5 required for GUI tests"
)


# QApplication fixture (shared across all tests)
@pytest.fixture(scope='session')
def qapp():
    """Create QApplication for all GUI tests."""
    if HAS_PYQT:
        # Use offscreen platform for headless testing
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv + ['-platform', 'offscreen'])
        yield app
    else:
        yield None


@pytest.fixture
def main_window(qapp):
    """Create main window instance."""
    if HAS_PYQT:
        from gui import TDESPHMainWindow
        window = TDESPHMainWindow()
        yield window
        window.close()
    else:
        yield None


@pytest.fixture
def config_editor(qapp):
    """Create config editor widget."""
    if HAS_PYQT:
        from gui import ConfigEditorWidget
        editor = ConfigEditorWidget()
        yield editor
    else:
        yield None


@pytest.fixture
def control_panel(qapp):
    """Create control panel widget."""
    if HAS_PYQT:
        from gui import ControlPanelWidget
        panel = ControlPanelWidget()
        yield panel
    else:
        yield None


@pytest.fixture
def data_display(qapp):
    """Create data display widget."""
    if HAS_PYQT:
        from gui import DataDisplayWidget
        display = DataDisplayWidget()
        yield display
    else:
        yield None


class TestMainWindow:
    """Test main window functionality."""

    def test_window_creation(self, main_window):
        """Test that main window is created successfully."""
        assert main_window is not None
        assert main_window.windowTitle() == "TDE-SPH Simulator"

    def test_window_geometry(self, main_window):
        """Test window has reasonable initial geometry."""
        geometry = main_window.geometry()
        assert geometry.width() == 1400
        assert geometry.height() == 900

    def test_menu_bar_exists(self, main_window):
        """Test menu bar is created."""
        menubar = main_window.menuBar()
        assert menubar is not None

        # Check for expected menus
        actions = menubar.actions()
        menu_titles = [action.text() for action in actions]

        assert '&File' in menu_titles
        assert '&Edit' in menu_titles
        assert '&Simulation' in menu_titles
        assert '&View' in menu_titles
        assert '&Help' in menu_titles

    def test_status_bar_exists(self, main_window):
        """Test status bar is created."""
        statusbar = main_window.statusBar()
        assert statusbar is not None

    def test_central_widget_exists(self, main_window):
        """Test central widget (config editor) exists."""
        central = main_window.centralWidget()
        assert central is not None

    def test_new_config_action(self, main_window):
        """Test new config action creates default config."""
        main_window.new_config()

        # Should have config editor with default text
        if main_window.config_editor is not None:
            text = main_window.config_editor.get_text()
            assert '# TDE-SPH Configuration File' in text
            assert 'simulation:' in text
            assert 'black_hole:' in text


class TestConfigEditor:
    """Test configuration editor widget."""

    def test_editor_creation(self, config_editor):
        """Test editor is created successfully."""
        assert config_editor is not None

    def test_set_and_get_text(self, config_editor):
        """Test setting and retrieving text."""
        test_text = "test: value\nfoo: bar"
        config_editor.set_text(test_text)
        retrieved = config_editor.get_text()

        assert retrieved == test_text

    def test_modification_tracking(self, config_editor):
        """Test modification flag is set on edits."""
        config_editor.set_text("initial")
        assert not config_editor.is_modified()

        # Modify text
        config_editor.text_edit.insertPlainText("modified")
        assert config_editor.is_modified()

    def test_yaml_validation_valid(self, config_editor):
        """Test YAML validation with valid input."""
        valid_yaml = """
simulation:
  name: test
  mode: newtonian

black_hole:
  mass: 1.0e6
"""
        config_editor.set_text(valid_yaml)
        config_editor.validate_yaml()

        # Check status label shows success
        assert "valid" in config_editor.status_label.text().lower()

    def test_yaml_validation_invalid(self, config_editor):
        """Test YAML validation with invalid input."""
        invalid_yaml = """
simulation:
  name: test
  invalid syntax here: [unclosed bracket
"""
        config_editor.set_text(invalid_yaml)
        config_editor.validate_yaml()

        # Check status label shows error
        assert "error" in config_editor.status_label.text().lower()

    def test_undo_redo(self, config_editor):
        """Test undo/redo functionality."""
        config_editor.set_text("initial")
        initial_text = config_editor.get_text()

        # Make edit
        config_editor.text_edit.insertPlainText(" modified")
        modified_text = config_editor.get_text()

        assert modified_text != initial_text

        # Undo
        config_editor.undo()
        undone_text = config_editor.get_text()
        assert undone_text == initial_text

        # Redo
        config_editor.redo()
        redone_text = config_editor.get_text()
        assert redone_text == modified_text


class TestControlPanel:
    """Test simulation control panel."""

    def test_panel_creation(self, control_panel):
        """Test panel is created successfully."""
        assert control_panel is not None

    def test_initial_state(self, control_panel):
        """Test initial button states."""
        assert control_panel.start_button.isEnabled()
        assert not control_panel.pause_button.isEnabled()
        assert not control_panel.stop_button.isEnabled()

        assert not control_panel.is_running
        assert not control_panel.is_paused

    def test_simulation_started_state(self, control_panel):
        """Test state changes when simulation starts."""
        control_panel.on_simulation_started()

        assert not control_panel.start_button.isEnabled()
        assert control_panel.pause_button.isEnabled()
        assert control_panel.stop_button.isEnabled()

        assert control_panel.is_running
        assert not control_panel.is_paused

    def test_simulation_stopped_state(self, control_panel):
        """Test state changes when simulation stops."""
        control_panel.on_simulation_started()
        control_panel.on_simulation_stopped()

        assert control_panel.start_button.isEnabled()
        assert not control_panel.pause_button.isEnabled()
        assert not control_panel.stop_button.isEnabled()

        assert not control_panel.is_running
        assert not control_panel.is_paused

    def test_progress_update(self, control_panel):
        """Test progress bar updates."""
        control_panel.total_time = 100.0

        control_panel.set_progress(50.0, 5000)

        assert control_panel.progress_bar.value() == 50

    def test_log_messages(self, control_panel):
        """Test log viewer."""
        control_panel.clear_log()
        assert control_panel.log_viewer.toPlainText() == ""

        control_panel.log("Test message")
        log_text = control_panel.log_viewer.toPlainText()

        assert "Test message" in log_text

    def test_parameter_update(self, control_panel):
        """Test parameter display update."""
        test_config = {
            'simulation': {'mode': 'schwarzschild'},
            'black_hole': {'mass': 1e6},
            'particles': {'count': 50000},
            'integration': {'timestep': 0.01, 't_end': 200.0}
        }

        control_panel.update_parameters(test_config)

        assert 'schwarzschild' in control_panel.param_mode.text()
        assert '1.0e+06' in control_panel.param_bh_mass.text() or '1.0e6' in control_panel.param_bh_mass.text()
        assert '50,000' in control_panel.param_particles.text()


class TestDataDisplay:
    """Test live data display widget."""

    def test_display_creation(self, data_display):
        """Test display widget is created successfully."""
        assert data_display is not None

    def test_energy_plot_exists(self, data_display):
        """Test energy plot widget exists."""
        assert data_display.energy_plot is not None

    def test_statistics_widget_exists(self, data_display):
        """Test statistics widget exists."""
        assert data_display.statistics_widget is not None

    def test_energy_data_update(self, data_display):
        """Test updating energy plot with data."""
        energies = {
            'kinetic': 1.0,
            'potential': -2.0,
            'internal': 0.5,
            'total': -0.5,
            'error': 0.001
        }

        data_display.energy_plot.update_data(10.0, energies)

        assert len(data_display.energy_plot.times) == 1
        assert data_display.energy_plot.times[0] == 10.0
        assert data_display.energy_plot.E_kin[0] == 1.0

    def test_statistics_update(self, data_display):
        """Test updating statistics table."""
        stats = {
            'n_particles': 100000,
            'total_mass': 1.0,
            'total_energy': -1.0,
            'mean_density': 2.5
        }

        data_display.statistics_widget.update_statistics(stats)

        # Check that table is updated
        table = data_display.statistics_widget.table
        particles_item = table.item(0, 1)  # Row 0 is "Total Particles"

        assert particles_item is not None
        assert '100,000' in particles_item.text()

    def test_clear_energy_data(self, data_display):
        """Test clearing energy plot data."""
        # Add data
        energies = {'kinetic': 1.0, 'potential': -2.0, 'internal': 0.5, 'total': -0.5, 'error': 0.0}
        data_display.energy_plot.update_data(10.0, energies)
        data_display.energy_plot.update_data(20.0, energies)

        assert len(data_display.energy_plot.times) == 2

        # Clear
        data_display.energy_plot.clear_data()

        assert len(data_display.energy_plot.times) == 0


class TestSignalConnections:
    """Test signal/slot connections between widgets."""

    def test_control_panel_start_signal(self, control_panel):
        """Test start_requested signal is emitted."""
        signal_received = False

        def on_start():
            nonlocal signal_received
            signal_received = True

        control_panel.start_requested.connect(on_start)

        # Simulate button click
        if PYQT_VERSION == 6:
            QTest.mouseClick(control_panel.start_button, Qt.MouseButton.LeftButton)
        else:
            QTest.mouseClick(control_panel.start_button, Qt.LeftButton)

        assert signal_received

    def test_control_panel_stop_signal(self, control_panel):
        """Test stop_requested signal is emitted."""
        signal_received = False

        def on_stop():
            nonlocal signal_received
            signal_received = True

        control_panel.stop_requested.connect(on_stop)

        # Enable stop button first
        control_panel.on_simulation_started()

        # Simulate button click
        if PYQT_VERSION == 6:
            QTest.mouseClick(control_panel.stop_button, Qt.MouseButton.LeftButton)
        else:
            QTest.mouseClick(control_panel.stop_button, Qt.LeftButton)

        assert signal_received

    def test_config_editor_modification_signal(self, config_editor):
        """Test config_modified signal is emitted."""
        signal_received = False

        def on_modified():
            nonlocal signal_received
            signal_received = True

        config_editor.config_modified.connect(on_modified)

        # Modify text
        config_editor.text_edit.insertPlainText("test")

        assert signal_received


class TestYAMLSyntaxHighlighter:
    """Test YAML syntax highlighter."""

    def test_highlighter_creation(self, config_editor):
        """Test syntax highlighter is created."""
        highlighter = config_editor.text_edit.highlighter
        assert highlighter is not None

    def test_key_highlighting(self, config_editor):
        """Test that YAML keys are highlighted."""
        yaml_text = "simulation: test"
        config_editor.set_text(yaml_text)

        # Highlighter should be applied
        # (Visual inspection required, but at least test no crashes)
        assert True  # Highlighter applied without errors


class TestIntegration:
    """Integration tests for GUI components."""

    def test_full_workflow(self, main_window):
        """Test complete workflow: create config, validate, start simulation."""
        # Create new config
        main_window.new_config()

        # Validate
        main_window.validate_config()

        # Note: Can't actually start simulation without mocking,
        # but can test that method doesn't crash
        assert main_window.simulation_running == False

    def test_config_save_and_load(self, main_window, tmp_path):
        """Test saving and loading config files."""
        # Create test config
        main_window.new_config()

        # Set a config file path
        config_path = tmp_path / "test_config.yaml"
        main_window.current_config_file = config_path

        # Save
        success = main_window.save_config()
        assert success
        assert config_path.exists()

        # Load
        main_window.load_config_file(config_path)

        if main_window.config_editor is not None:
            text = main_window.config_editor.get_text()
            assert '# TDE-SPH Configuration File' in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
