"""
TDE-SPH GUI Module (TASK-100)

A comprehensive PyQt6/PyQt5-based graphical user interface for configuring
and running TDE-SPH simulations.

Components:
- TDESPHMainWindow: Main application window with menus, toolbars, docks
- ConfigEditorWidget: YAML configuration editor with syntax highlighting
- ControlPanelWidget: Simulation control (start/stop/pause, progress tracking)
- DataDisplayWidget: Live data visualization (energy plots, statistics)

Usage:
    from tde_sph.gui import launch_gui, TDESPHMainWindow

    # Launch the GUI application
    launch_gui()

    # Or create window programmatically
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = TDESPHMainWindow()
    window.show()
    sys.exit(app.exec())

Requirements:
    PyQt6 (recommended) or PyQt5
    matplotlib (for embedded plots)
    pyyaml (for config validation)

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

# Check for PyQt availability
try:
    from PyQt6.QtWidgets import QApplication
    HAS_PYQT6 = True
    HAS_PYQT = True
except ImportError:
    HAS_PYQT6 = False
    try:
        from PyQt5.QtWidgets import QApplication
        HAS_PYQT = True
    except ImportError:
        HAS_PYQT = False

if HAS_PYQT:
    from tde_sph.gui.main_window import TDESPHMainWindow, main as launch_gui
    from tde_sph.gui.config_editor import ConfigEditorWidget, YAMLSyntaxHighlighter
    from tde_sph.gui.control_panel import ControlPanelWidget
    from tde_sph.gui.data_display import (
        DataDisplayWidget, EnergyPlotWidget, StatisticsWidget,
        DiagnosticsWidget, ParticleStatsTable, PerformanceMetricsWidget,
        CoordinateMetricWidget
    )
    from tde_sph.gui.simulation_thread import SimulationThread
else:
    # Provide placeholders if PyQt not available
    TDESPHMainWindow = None
    ConfigEditorWidget = None
    ControlPanelWidget = None
    DataDisplayWidget = None
    EnergyPlotWidget = None
    StatisticsWidget = None
    DiagnosticsWidget = None
    ParticleStatsTable = None
    PerformanceMetricsWidget = None
    CoordinateMetricWidget = None
    SimulationThread = None
    YAMLSyntaxHighlighter = None
    launch_gui = None

__all__ = [
    'TDESPHMainWindow',
    'ConfigEditorWidget',
    'ControlPanelWidget',
    'DataDisplayWidget',
    'EnergyPlotWidget',
    'StatisticsWidget',
    'DiagnosticsWidget',
    'ParticleStatsTable',
    'PerformanceMetricsWidget',
    'CoordinateMetricWidget',
    'SimulationThread',
    'YAMLSyntaxHighlighter',
    'launch_gui',
    'HAS_PYQT',
    'HAS_PYQT6',
]
