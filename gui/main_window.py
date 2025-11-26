#!/usr/bin/env python3
"""
TDE-SPH GUI: Main Application Window (TASK-100)

A comprehensive PyQt6-based graphical interface for configuring and running
TDE-SPH simulations. Provides:
- YAML configuration editor with syntax highlighting
- Simulation control panel (start/stop/pause, progress tracking)
- Live data visualization (energy evolution, particle statistics)
- File browser for initial conditions and outputs

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

from typing import Optional
from pathlib import Path
import sys

# Ensure src is in sys.path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QApplication, QDockWidget, QMenuBar, QMenu,
        QStatusBar, QToolBar, QFileDialog, QMessageBox, QWidget,
        QVBoxLayout, QLabel, QTabWidget
    )
    from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal
    from PyQt6.QtGui import QAction, QIcon, QKeySequence
    HAS_PYQT6 = True
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QMainWindow, QApplication, QDockWidget, QMenuBar, QMenu,
            QStatusBar, QToolBar, QFileDialog, QMessageBox, QWidget,
            QVBoxLayout, QLabel
        )
        from PyQt5.QtCore import Qt, QSettings, QTimer, pyqtSignal
        from PyQt5.QtGui import QAction, QIcon, QKeySequence
        HAS_PYQT6 = False
        PYQT_VERSION = 5
    except ImportError:
        raise ImportError(
            "PyQt6 or PyQt5 required for GUI. Install with:\n"
            "  pip install PyQt6   # Recommended\n"
            "  pip install PyQt5   # Alternative"
        )

# Import GUI components
try:
    # Try relative import (when running as package)
    from .config_editor import ConfigEditorWidget
    from .control_panel import ControlPanelWidget
    from .data_display import DataDisplayWidget
    from .simulation_thread import SimulationThread
    from .web_viewer import WebViewerWidget
except ImportError:
    try:
        # Try absolute import (when running as script from gui folder)
        from config_editor import ConfigEditorWidget
        from control_panel import ControlPanelWidget
        from data_display import DataDisplayWidget
        from simulation_thread import SimulationThread
        from web_viewer import WebViewerWidget
    except ImportError:
        try:
            # Try import from gui package (when running from root)
            from gui.config_editor import ConfigEditorWidget
            from gui.control_panel import ControlPanelWidget
            from gui.data_display import DataDisplayWidget
            from gui.simulation_thread import SimulationThread
            from gui.web_viewer import WebViewerWidget
        except ImportError:
            # Fallback
            ConfigEditorWidget = None
            ControlPanelWidget = None
            DataDisplayWidget = None
            SimulationThread = None
            WebViewerWidget = None


class TDESPHMainWindow(QMainWindow):
    """
    Main application window for TDE-SPH GUI.

    Provides a comprehensive interface for:
    - Editing YAML configuration files
    - Launching and controlling simulations
    - Monitoring live simulation data
    - Visualizing results

    Architecture:
    - Central widget: Config editor (YAML editing with syntax highlighting)
    - Left dock: Control panel (start/stop/progress)
    - Right dock: Data display (energy plots, statistics)
    - Menu bar: File, Edit, Simulation, View, Help
    - Status bar: Simulation status and messages

    Attributes:
        config_editor (ConfigEditorWidget): YAML configuration editor
        control_panel (ControlPanelWidget): Simulation control interface
        data_display (DataDisplayWidget): Live data visualization
        current_config_file (Path): Currently loaded config file path
        settings (QSettings): Persistent application settings
    """

    # Signals
    config_changed = pyqtSignal(str)  # Emitted when config file changes
    simulation_started = pyqtSignal()
    simulation_stopped = pyqtSignal()

    def __init__(self):
        """Initialize the main window and all components."""
        super().__init__()

        # Application metadata
        self.setWindowTitle("TDE-SPH Simulator")
        self.setGeometry(100, 100, 1400, 900)

        # Persistent settings
        self.settings = QSettings("TDE-SPH", "Simulator")

        # Current state
        self.current_config_file: Optional[Path] = None
        self.simulation_running = False
        self.sim_thread: Optional[SimulationThread] = None

        # Build UI components
        self._create_widgets()
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_status_bar()
        self._create_dock_widgets()
        self._connect_signals()
        self._restore_settings()

        # Status message
        self.statusBar().showMessage("Ready", 3000)

    def _create_widgets(self):
        """Create central widget and dock widgets."""
        # Central widget: Tab widget with Config Editor and 3D Viewer
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Configuration Editor
        if ConfigEditorWidget is not None:
            self.config_editor = ConfigEditorWidget(self)
            self.tabs.addTab(self.config_editor, "Configuration")
        else:
            # Fallback if config_editor.py not created yet
            placeholder = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Configuration Editor (placeholder)"))
            placeholder.setLayout(layout)
            self.setCentralWidget(placeholder) # Fallback to simple widget if tabs fail? No, just add to tab.
            self.config_editor = None
            self.tabs.addTab(placeholder, "Configuration")

        # Tab 2: 3D Viewer
        if WebViewerWidget is not None:
            self.web_viewer = WebViewerWidget(self)
            self.tabs.addTab(self.web_viewer, "3D Visualization")
        else:
            self.web_viewer = None

        # Dock widgets
        if ControlPanelWidget is not None:
            self.control_panel = ControlPanelWidget(self)
        else:
            self.control_panel = None

        if DataDisplayWidget is not None:
            self.data_display = DataDisplayWidget(self)
        else:
            self.data_display = None

    def _create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Config", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setStatusTip("Create a new configuration file")
        new_action.triggered.connect(self.new_config)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Config...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open an existing configuration file")
        open_action.triggered.connect(self.open_config)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Config", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setStatusTip("Save the current configuration")
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Config &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.setStatusTip("Save configuration to a new file")
        save_as_action.triggered.connect(self.save_config_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        preferences_action = QAction("&Preferences...", self)
        preferences_action.setStatusTip("Open preferences dialog")
        preferences_action.triggered.connect(self.show_preferences)
        edit_menu.addAction(preferences_action)

        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")

        validate_action = QAction("&Validate Config", self)
        validate_action.setShortcut(QKeySequence("Ctrl+Shift+V"))
        validate_action.setStatusTip("Validate current configuration")
        validate_action.triggered.connect(self.validate_config)
        sim_menu.addAction(validate_action)

        start_action = QAction("&Start Simulation", self)
        start_action.setShortcut(QKeySequence("Ctrl+R"))
        start_action.setStatusTip("Start the simulation with current config")
        start_action.triggered.connect(self.start_simulation)
        sim_menu.addAction(start_action)

        stop_action = QAction("S&top Simulation", self)
        stop_action.setShortcut(QKeySequence("Ctrl+T"))
        stop_action.setStatusTip("Stop the running simulation")
        stop_action.triggered.connect(self.stop_simulation)
        sim_menu.addAction(stop_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Will add dock widget visibility toggles here
        self.view_menu = view_menu  # Store reference for dock toggles

        # Help menu
        help_menu = menubar.addMenu("&Help")

        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        docs_action.setStatusTip("Open documentation")
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

        about_action = QAction("&About TDE-SPH", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _create_tool_bar(self):
        """Create application toolbar."""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)

        # File operations
        open_action = QAction("Open", self)
        open_action.setStatusTip("Open configuration file")
        open_action.triggered.connect(self.open_config)
        toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setStatusTip("Save configuration")
        save_action.triggered.connect(self.save_config)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # Simulation control
        start_action = QAction("Start", self)
        start_action.setStatusTip("Start simulation")
        start_action.triggered.connect(self.start_simulation)
        toolbar.addAction(start_action)

        stop_action = QAction("Stop", self)
        stop_action.setStatusTip("Stop simulation")
        stop_action.triggered.connect(self.stop_simulation)
        toolbar.addAction(stop_action)

    def _create_status_bar(self):
        """Create application status bar."""
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _create_dock_widgets(self):
        """Create and configure dock widgets."""
        # Left dock: Control panel
        if self.control_panel is not None:
            control_dock = QDockWidget("Simulation Control", self)
            control_dock.setWidget(self.control_panel)
            control_dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
            )
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, control_dock)

            # Add to View menu
            self.view_menu.addAction(control_dock.toggleViewAction())

        # Right dock: Data display
        if self.data_display is not None:
            data_dock = QDockWidget("Live Data", self)
            data_dock.setWidget(self.data_display)
            data_dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
            )
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, data_dock)

            # Add to View menu
            self.view_menu.addAction(data_dock.toggleViewAction())

    def _connect_signals(self):
        """Connect signals between widgets."""
        if self.control_panel is not None:
            self.control_panel.start_requested.connect(self.start_simulation)
            self.control_panel.stop_requested.connect(self.stop_simulation)
            self.control_panel.pause_requested.connect(self.pause_simulation)
            self.control_panel.resume_requested.connect(self.resume_simulation)
            self.control_panel.chart_updates_toggled.connect(self._on_chart_updates_toggled)
            self.control_panel.detailed_diagnostics_toggled.connect(self._on_detailed_diagnostics_toggled)
            self.control_panel.live_data_interval_changed.connect(self._on_live_data_interval_changed)

        if self.config_editor is not None:
            self.config_editor.config_modified.connect(self._on_config_modified)

    def _restore_settings(self):
        """Restore window geometry and state from previous session."""
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        state = self.settings.value("windowState")
        if state is not None:
            self.restoreState(state)

        # Restore last opened config file
        last_config = self.settings.value("last_config_file")
        if last_config and Path(last_config).exists():
            self.load_config_file(Path(last_config))

    def _save_settings(self):
        """Save window geometry and state."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

        if self.current_config_file:
            self.settings.setValue("last_config_file", str(self.current_config_file))

    # -------------------------------------------------------------------------
    # File operations
    # -------------------------------------------------------------------------

    def new_config(self):
        """Create a new configuration file."""
        if not self._check_unsaved_changes():
            return

        # Load default template
        default_config = """# TDE-SPH Configuration File
# Autogenerated by TDE-SPH GUI

simulation:
  name: "tde_simulation"
  output_dir: "outputs"
  mode: "schwarzschild"  # newtonian, schwarzschild, kerr

black_hole:
  mass: 1.0e6  # Solar masses
  spin: 0.0    # a/M, 0 for Schwarzschild

star:
  mass: 1.0      # Solar masses
  radius: 1.0    # Solar radii
  polytropic_index: 1.5

orbit:
  pericentre: 10.0  # Gravitational radii
  eccentricity: 0.95
  inclination: 0.0  # Degrees

particles:
  count: 100000
  sph_kernel: "cubic_spline"

integration:
  timestep: 0.01
  t_end: 100000.0
  cfl_factor: 1.0
  output_interval: 1.0

physics:
  self_gravity: true
  viscosity: "artificial"
  eos: "ideal_gas"
  radiation: false
"""

        if self.config_editor is not None:
            self.config_editor.set_text(default_config)

        self.current_config_file = None
        self.setWindowTitle("TDE-SPH Simulator - New Configuration*")
        self.statusBar().showMessage("New configuration created", 3000)

    def open_config(self):
        """Open an existing configuration file."""
        if not self._check_unsaved_changes():
            return

        last_dir = self.settings.value("last_directory", str(Path.home()))

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Configuration File",
            last_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if file_path:
            self.load_config_file(Path(file_path))
            self.settings.setValue("last_directory", str(Path(file_path).parent))

    def load_config_file(self, file_path: Path):
        """Load a configuration file into the editor."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            if self.config_editor is not None:
                self.config_editor.set_text(content)

            self.current_config_file = file_path
            self.setWindowTitle(f"TDE-SPH Simulator - {file_path.name}")
            self.statusBar().showMessage(f"Loaded: {file_path}", 3000)
            self.config_changed.emit(str(file_path))

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading File",
                f"Could not load configuration file:\n{file_path}\n\nError: {e}"
            )

    def save_config(self):
        """Save the current configuration."""
        if self.current_config_file is None:
            return self.save_config_as()

        try:
            if self.config_editor is not None:
                content = self.config_editor.get_text()
                with open(self.current_config_file, 'w') as f:
                    f.write(content)

            self.setWindowTitle(f"TDE-SPH Simulator - {self.current_config_file.name}")
            self.statusBar().showMessage(f"Saved: {self.current_config_file}", 3000)
            return True

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving File",
                f"Could not save configuration file:\n{self.current_config_file}\n\nError: {e}"
            )
            return False

    def save_config_as(self):
        """Save configuration to a new file."""
        last_dir = self.settings.value("last_directory", str(Path.home()))

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration As",
            last_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if file_path:
            self.current_config_file = Path(file_path)
            self.settings.setValue("last_directory", str(Path(file_path).parent))
            return self.save_config()

        return False

    def _check_unsaved_changes(self) -> bool:
        """
        Check for unsaved changes and prompt user.

        Returns:
            True if it's safe to proceed (no changes or user chose to discard)
            False if operation should be cancelled
        """
        if self.config_editor is not None and self.config_editor.is_modified():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "The current configuration has unsaved changes.\n"
                "Do you want to save them?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Save:
                return self.save_config()
            elif reply == QMessageBox.StandardButton.Cancel:
                return False

        return True

    # -------------------------------------------------------------------------
    # Edit operations
    # -------------------------------------------------------------------------

    def _undo(self):
        """Undo last edit in config editor."""
        if self.config_editor is not None:
            self.config_editor.undo()

    def _redo(self):
        """Redo last undone edit."""
        if self.config_editor is not None:
            self.config_editor.redo()

    # -------------------------------------------------------------------------
    # Simulation operations
    # -------------------------------------------------------------------------

    def validate_config(self):
        """Validate the current configuration."""
        try:
            import yaml

            if self.config_editor is not None:
                content = self.config_editor.get_text()
                config = yaml.safe_load(content)

                # Basic validation
                required_keys = ['simulation', 'black_hole', 'star', 'orbit', 'particles', 'integration']
                missing_keys = [key for key in required_keys if key not in config]

                if missing_keys:
                    QMessageBox.warning(
                        self,
                        "Validation Warning",
                        f"Configuration is missing required sections:\n{', '.join(missing_keys)}"
                    )
                else:
                    QMessageBox.information(
                        self,
                        "Validation Success",
                        "Configuration file is valid!"
                    )
                    self.statusBar().showMessage("Configuration validated successfully", 3000)

        except yaml.YAMLError as e:
            QMessageBox.critical(
                self,
                "YAML Syntax Error",
                f"Configuration has syntax errors:\n\n{e}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Validation Error",
                f"Error during validation:\n{e}"
            )

    def start_simulation(self):
        """Start the simulation with current configuration."""
        if self.simulation_running:
            QMessageBox.warning(
                self,
                "Simulation Running",
                "A simulation is already running. Stop it before starting a new one."
            )
            return

        # Validate config first
        try:
            import yaml

            if self.config_editor is not None:
                content = self.config_editor.get_text()
                config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            QMessageBox.critical(
                self,
                "Invalid Configuration",
                f"Cannot start simulation with invalid YAML:\n\n{e}"
            )
            return

        # Save config if modified
        if self.current_config_file is None:
            if not self.save_config_as():
                return
        elif self.config_editor is not None and self.config_editor.is_modified():
            self.save_config()

        # Start simulation thread
        if SimulationThread is not None:
            self.sim_thread = SimulationThread(config)
            self.sim_thread.progress_updated.connect(self._on_progress_updated)
            self.sim_thread.live_data_updated.connect(self._on_live_data_updated)
            self.sim_thread.simulation_finished.connect(self._on_simulation_finished)
            self.sim_thread.simulation_stopped.connect(self._on_simulation_stopped)
            self.sim_thread.simulation_error.connect(self._on_simulation_error)
            self.sim_thread.log_message.connect(self._on_log_message)
            
            # Apply current performance settings
            if self.control_panel:
                self.sim_thread.set_detailed_diagnostics(
                    self.control_panel.detailed_diagnostics_checkbox.isChecked()
                )
                self.sim_thread.set_live_data_interval(
                    self.control_panel.live_data_interval_spinbox.value()
                )
            
            # Check if data display is visible
            if self.data_display:
                # Assume visible initially (can be enhanced with actual visibility check)
                self.sim_thread.set_data_display_visible(True)
            
            self.sim_thread.start()
            
            self.simulation_running = True
            self.simulation_started.emit()

            if self.control_panel is not None:
                self.control_panel.on_simulation_started()
                self.control_panel.update_parameters(config)

            self.statusBar().showMessage("Simulation started", 3000)
        else:
            QMessageBox.critical(self, "Error", "SimulationThread not available (check imports)")

    def stop_simulation(self):
        """Stop the running simulation."""
        if not self.simulation_running:
            return

        reply = QMessageBox.question(
            self,
            "Stop Simulation",
            "Are you sure you want to stop the simulation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.sim_thread and self.sim_thread.isRunning():
                self.sim_thread.stop()
                self.sim_thread.wait()
            
            # UI update handled by signal connection to _on_simulation_stopped
            self.statusBar().showMessage("Stopping simulation...", 3000)

    def pause_simulation(self):
        """Pause the running simulation."""
        if self.sim_thread:
            self.sim_thread.pause()
            self.statusBar().showMessage("Simulation paused", 3000)

    def resume_simulation(self):
        """Resume the paused simulation."""
        if self.sim_thread:
            self.sim_thread.resume()
            self.statusBar().showMessage("Simulation resumed", 3000)

    def _on_progress_updated(self, time, step, progress_info):
        """Handle lightweight progress updates from simulation thread."""
        if self.control_panel:
            self.control_panel.set_progress(time, step, progress_info)
    
    def _on_live_data_updated(self, time, step, energies, stats):
        """Handle expensive live data updates from simulation thread."""
        if self.data_display:
            self.data_display.update_live_data(time, energies, stats)

    def _on_simulation_finished(self):
        """Handle simulation completion."""
        self.simulation_running = False
        if self.control_panel:
            self.control_panel.on_simulation_finished()
        self.statusBar().showMessage("Simulation finished", 5000)

    def _on_simulation_stopped(self):
        """Handle simulation stopped by user."""
        self.simulation_running = False
        self.simulation_stopped.emit()
        if self.control_panel:
            self.control_panel.on_simulation_stopped()
        self.statusBar().showMessage("Simulation stopped", 3000)

    def _on_simulation_error(self, error_msg):
        """Handle simulation error."""
        self.simulation_running = False
        if self.control_panel:
            self.control_panel.on_simulation_stopped()
            self.control_panel.log(f"ERROR: {error_msg}")
        
        QMessageBox.critical(self, "Simulation Error", f"An error occurred:\n{error_msg}")

    def _on_log_message(self, msg):
        """Handle log message from thread."""
        if self.control_panel:
            self.control_panel.log(msg)

    # -------------------------------------------------------------------------
    # Other menu actions
    # -------------------------------------------------------------------------

    def show_preferences(self):
        """Open preferences dialog."""
        # TODO: Implement preferences dialog
        QMessageBox.information(self, "Preferences", "Preferences dialog (coming soon)")

    def show_documentation(self):
        """Open documentation in browser."""
        import webbrowser
        webbrowser.open("https://github.com/your-repo/tde-sph/wiki")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About TDE-SPH",
            "<h2>TDE-SPH Simulator</h2>"
            "<p>Version 1.0.0</p>"
            "<p>Relativistic SPH framework for tidal disruption events around supermassive black holes.</p>"
            "<p><b>Authors:</b> TDE-SPH Development Team</p>"
            "<p><b>License:</b> MIT</p>"
            "<p>For more information, visit "
            "<a href='https://github.com/your-repo/tde-sph'>the project page</a>.</p>"
        )

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def _on_config_modified(self):
        """Handle config editor modification."""
        if self.current_config_file:
            self.setWindowTitle(f"TDE-SPH Simulator - {self.current_config_file.name}*")
        else:
            self.setWindowTitle("TDE-SPH Simulator - New Configuration*")

    def _on_chart_updates_toggled(self, enabled: bool):
        """Handle chart updates toggle."""
        if self.data_display is not None:
            self.data_display.set_chart_updates_enabled(enabled)
    
    def _on_detailed_diagnostics_toggled(self, enabled: bool):
        """Handle detailed diagnostics toggle."""
        if self.sim_thread:
            self.sim_thread.set_detailed_diagnostics(enabled)
    
    def _on_live_data_interval_changed(self, interval: int):
        """Handle live data interval change."""
        if self.sim_thread:
            self.sim_thread.set_live_data_interval(interval)

    def closeEvent(self, event):
        """Handle window close event."""
        if not self._check_unsaved_changes():
            event.ignore()
            return

        if self.simulation_running:
            reply = QMessageBox.question(
                self,
                "Simulation Running",
                "A simulation is running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        self._save_settings()
        event.accept()


def main():
    """Run the TDE-SPH GUI application."""
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("TDE-SPH Simulator")
    app.setOrganizationName("TDE-SPH")

    window = TDESPHMainWindow()
    window.show()

    sys.exit(app.exec() if PYQT_VERSION == 6 else app.exec_())


if __name__ == "__main__":
    main()
