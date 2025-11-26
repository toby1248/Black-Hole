#!/usr/bin/env python3
"""
Simulation Control Panel Widget (TASK-100)

A control panel for starting, stopping, and monitoring TDE-SPH simulations.

Features:
- Start/Stop/Pause buttons
- Progress bar with time/step information
- Simulation status display
- Quick parameter overview
- Log output viewer

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
        QProgressBar, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox,
        QFormLayout, QCheckBox
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QFont
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
        QProgressBar, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox,
        QFormLayout, QCheckBox
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont
    PYQT_VERSION = 5


class ControlPanelWidget(QWidget):
    """
    Control panel for simulation execution and monitoring.

    Provides:
    - Start/Stop/Pause controls
    - Progress tracking (time, step, percentage)
    - Real-time status updates
    - Quick parameter display
    - Log message viewer

    Signals:
        start_requested: Emitted when user clicks Start
        stop_requested: Emitted when user clicks Stop
        pause_requested: Emitted when user clicks Pause
    """

    # Signals
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    chart_updates_toggled = pyqtSignal(bool)  # Emitted when chart update checkbox is toggled
    detailed_diagnostics_toggled = pyqtSignal(bool)  # Emitted when detailed diagnostics is toggled
    live_data_interval_changed = pyqtSignal(int)  # Emitted when live data interval changes

    def __init__(self, parent=None):
        super().__init__(parent)

        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.current_time = 0.0
        self.current_step = 0
        self.total_time = 100.0
        self.total_steps = 10000
        self.is_simulated = False

        # Update timer (for simulated progress)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_progress)

        # Build UI
        self._create_ui()

    def _create_ui(self):
        """Create the control panel UI."""
        layout = QVBoxLayout()

        # Control buttons group
        control_group = self._create_control_group()
        layout.addWidget(control_group)

        # Progress group
        progress_group = self._create_progress_group()
        layout.addWidget(progress_group)

        # Parameters group
        params_group = self._create_parameters_group()
        layout.addWidget(params_group)

        # Log viewer group
        log_group = self._create_log_group()
        layout.addWidget(log_group)

        layout.addStretch()

        self.setLayout(layout)

    def _create_control_group(self) -> QGroupBox:
        """Create control buttons group."""
        group = QGroupBox("Simulation Control")
        layout = QVBoxLayout()

        # Start button
        self.start_button = QPushButton("Start Simulation")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_button)

        # Pause button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.setStyleSheet(
            "QPushButton { background-color: #FFA500; color: white; font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #FF8C00; }"
        )
        self.pause_button.clicked.connect(self._on_pause_clicked)
        layout.addWidget(self.pause_button)

        # Stop button
        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        self.stop_button.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self.stop_button)

        # Chart update toggle
        self.chart_update_checkbox = QCheckBox("Enable Live Charts")
        self.chart_update_checkbox.setChecked(True)
        self.chart_update_checkbox.setToolTip(
            "Disable to improve performance during long simulations.\n"
            "Charts will not update while disabled."
        )
        self.chart_update_checkbox.stateChanged.connect(self._on_chart_toggle_changed)
        layout.addWidget(self.chart_update_checkbox)
        
        # Detailed diagnostics toggle
        self.detailed_diagnostics_checkbox = QCheckBox("Detailed Diagnostics")
        self.detailed_diagnostics_checkbox.setChecked(True)
        self.detailed_diagnostics_checkbox.setToolTip(
            "Disable to skip expensive calculations:\n"
            "• Percentile computations (distance, energy)\n"
            "• Per-particle potential energy (O(N²))\n"
            "Can improve performance by 10-100× for large N"
        )
        self.detailed_diagnostics_checkbox.stateChanged.connect(self._on_detailed_diagnostics_changed)
        layout.addWidget(self.detailed_diagnostics_checkbox)
        
        # Live data update interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Live Data Every:"))
        self.live_data_interval_spinbox = QSpinBox()
        self.live_data_interval_spinbox.setRange(1, 1000)
        self.live_data_interval_spinbox.setValue(10)
        self.live_data_interval_spinbox.setSuffix(" steps")
        self.live_data_interval_spinbox.setToolTip(
            "How often to update live data display.\n"
            "Higher values = faster simulation, less frequent updates."
        )
        self.live_data_interval_spinbox.valueChanged.connect(self._on_live_data_interval_changed)
        interval_layout.addWidget(self.live_data_interval_spinbox)
        interval_layout.addStretch()
        layout.addLayout(interval_layout)

        group.setLayout(layout)
        return group

    def _create_progress_group(self) -> QGroupBox:
        """Create progress monitoring group."""
        group = QGroupBox("Progress")
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Status: Idle")
        font = QFont()
        font.setBold(True)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Time and step info
        info_layout = QFormLayout()

        self.time_label = QLabel("0.00 / 100.00")
        info_layout.addRow("Time:", self.time_label)

        self.step_label = QLabel("0 / 10000")
        info_layout.addRow("Step:", self.step_label)

        self.eta_label = QLabel("N/A")
        info_layout.addRow("ETA:", self.eta_label)

        layout.addLayout(info_layout)

        group.setLayout(layout)
        return group

    def _create_parameters_group(self) -> QGroupBox:
        """Create quick parameters display group."""
        group = QGroupBox("Quick Parameters")
        layout = QFormLayout()

        # Display key simulation parameters
        self.param_mode = QLabel("schwarzschild")
        layout.addRow("Mode:", self.param_mode)

        self.param_bh_mass = QLabel("1.0e6 M☉")
        layout.addRow("BH Mass:", self.param_bh_mass)

        self.param_particles = QLabel("100000")
        layout.addRow("Particles:", self.param_particles)

        self.param_timestep = QLabel("0.01")
        layout.addRow("Timestep:", self.param_timestep)

        group.setLayout(layout)
        return group

    def _create_log_group(self) -> QGroupBox:
        """Create log viewer group."""
        group = QGroupBox("Simulation Log")
        layout = QVBoxLayout()

        # Log text area
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(150)
        self.log_viewer.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        layout.addWidget(self.log_viewer)

        # Clear button
        clear_button = QPushButton("Clear Log")
        clear_button.clicked.connect(self.clear_log)
        layout.addWidget(clear_button)

        group.setLayout(layout)
        return group

    # -------------------------------------------------------------------------
    # Control button handlers
    # -------------------------------------------------------------------------

    def _on_start_clicked(self):
        """Handle Start button click."""
        if self.is_paused:
            # Resume from pause
            self.is_paused = False
            self.update_timer.start(100)  # Update every 100ms
            self.status_label.setText("Status: Running")
            self.pause_button.setText("Pause")
            self.log("Simulation resumed")
        else:
            # Start new simulation
            self.start_requested.emit()

    def _on_pause_clicked(self):
        """Handle Pause button click."""
        if self.is_paused:
            # Resume
            self.is_paused = False
            if self.is_simulated:
                self.update_timer.start(100)
            self.status_label.setText("Status: Running")
            self.pause_button.setText("Pause")
            self.log("Simulation resumed")
            self.resume_requested.emit()
        else:
            # Pause
            self.is_paused = True
            self.update_timer.stop()
            self.status_label.setText("Status: Paused")
            self.pause_button.setText("Resume")
            self.log("Simulation paused")
            self.pause_requested.emit()

    def _on_stop_clicked(self):
        """Handle Stop button click."""
        self.stop_requested.emit()

    def _on_chart_toggle_changed(self, state):
        """Handle chart update checkbox toggle."""
        enabled = (state == Qt.CheckState.Checked.value if PYQT_VERSION == 6 else state == Qt.Checked)
        self.chart_updates_toggled.emit(enabled)
        status = "enabled" if enabled else "disabled"
        self.log(f"Live chart updates {status}")
    
    def _on_detailed_diagnostics_changed(self, state):
        """Handle detailed diagnostics checkbox toggle."""
        enabled = (state == Qt.CheckState.Checked.value if PYQT_VERSION == 6 else state == Qt.Checked)
        self.detailed_diagnostics_toggled.emit(enabled)
        status = "enabled" if enabled else "disabled"
        self.log(f"Detailed diagnostics {status}")
    
    def _on_live_data_interval_changed(self, value):
        """Handle live data interval change."""
        self.live_data_interval_changed.emit(value)
        self.log(f"Live data interval: every {value} steps")

    # -------------------------------------------------------------------------
    # Public methods (called by main window)
    # -------------------------------------------------------------------------

    def on_simulation_started(self, simulated=False):
        """Called when simulation starts."""
        self.is_running = True
        self.is_paused = False
        self.is_simulated = simulated
        self.current_time = 0.0
        self.current_step = 0

        # Update UI
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

        self.status_label.setText("Status: Running")
        self.progress_bar.setValue(0)

        # Start update timer (simulated progress)
        if self.is_simulated:
            self.update_timer.start(100)  # Update every 100ms

        self.log("Simulation started")

    def on_simulation_stopped(self):
        """Called when simulation stops."""
        self.is_running = False
        self.is_paused = False

        # Stop update timer
        self.update_timer.stop()

        # Update UI
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(False)

        self.status_label.setText("Status: Stopped")

        self.log("Simulation stopped")

    def on_simulation_finished(self):
        """Called when simulation finishes normally."""
        self.is_running = False
        self.is_paused = False

        # Stop update timer
        self.update_timer.stop()

        # Update UI
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.status_label.setText("Status: Finished")
        self.progress_bar.setValue(100)

        self.log("Simulation finished successfully")

    def update_parameters(self, config: dict):
        """Update parameter display from configuration."""
        try:
            mode = config.get('simulation', {}).get('mode', 'unknown')
            self.param_mode.setText(mode)

            bh_mass = config.get('black_hole', {}).get('mass', 0)
            self.param_bh_mass.setText(f"{bh_mass:.1e} M☉")

            n_particles = config.get('particles', {}).get('count', 0)
            self.param_particles.setText(f"{n_particles:,}")

            dt = config.get('integration', {}).get('timestep', 0)
            self.param_timestep.setText(f"{dt:.4f}")

            t_end = config.get('integration', {}).get('t_end', 100.0)
            self.total_time = t_end

            # Estimate total steps
            self.total_steps = int(t_end / dt) if dt > 0 else 10000

        except Exception as e:
            self.log(f"Warning: Could not parse all parameters ({e})")

    def log(self, message: str):
        """Add a message to the log viewer."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_viewer.append(f"[{timestamp}] {message}")

    def clear_log(self):
        """Clear the log viewer."""
        self.log_viewer.clear()

    # -------------------------------------------------------------------------
    # Progress updates (simulated for demonstration)
    # -------------------------------------------------------------------------

    def _update_progress(self):
        """Update progress indicators (simulated)."""
        # Simulate time advancement
        # In real implementation, this would read from simulation output
        self.current_time += 0.1
        self.current_step += 10

        # Update progress bar
        progress = int(100 * self.current_time / self.total_time)
        self.progress_bar.setValue(min(progress, 100))

        # Update labels
        self.time_label.setText(f"{self.current_time:.2f} / {self.total_time:.2f}")
        self.step_label.setText(f"{self.current_step} / {self.total_steps}")

        # Calculate ETA (simulated)
        if self.current_time > 0:
            elapsed_real = self.current_time * 0.1  # Simulated elapsed time
            time_per_unit = elapsed_real / self.current_time if self.current_time > 0 else 0
            remaining = (self.total_time - self.current_time) * time_per_unit
            self.eta_label.setText(f"{remaining:.1f} s")
        else:
            self.eta_label.setText("N/A")

        # Check if finished
        if self.current_time >= self.total_time:
            self.on_simulation_finished()

    def set_progress(self, time: float, step: int, progress_info: dict = None):
        """
        Manually set progress (called from actual simulation).

        Parameters:
            time: Current simulation time
            step: Current step number
            progress_info: Dictionary with timing and progress information
        """
        self.current_time = time
        self.current_step = step

        # Update from progress_info if provided
        if progress_info:
            if 'total_time' in progress_info:
                self.total_time = progress_info['total_time']
            if 'timing_ms' in progress_info:
                timings = progress_info['timing_ms']
                # Update timing display in diagnostics table if it exists
                # (will be handled by data_display widget)

        progress = int(100 * time / self.total_time) if self.total_time > 0 else 0
        self.progress_bar.setValue(min(progress, 100))

        self.time_label.setText(f"{time:.2f} / {self.total_time:.2f}")
        self.step_label.setText(f"{step}")
