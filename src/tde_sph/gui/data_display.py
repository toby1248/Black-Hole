#!/usr/bin/env python3
"""
Live Data Display Widget (TASK-100)

Real-time visualization of simulation data including energy evolution,
particle statistics, and diagnostic plots.

Features:
- Energy evolution plot (matplotlib embedded)
- Particle statistics table
- Snapshot preview
- Export data functionality

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

from typing import List, Dict, Optional
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
        QLabel, QTableWidget, QTableWidgetItem, QTabWidget
    )
    from PyQt6.QtCore import Qt, QTimer
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
        QLabel, QTableWidget, QTableWidgetItem, QTabWidget
    )
    from PyQt5.QtCore import Qt, QTimer
    PYQT_VERSION = 5

# Matplotlib integration
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class EnergyPlotWidget(FigureCanvas):
    """
    Matplotlib canvas for live energy evolution plot.

    Plots:
    - Kinetic, potential, internal, and total energy vs time
    - Energy conservation error vs time
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        # Two subplots: energies and conservation error
        self.ax_energy = self.fig.add_subplot(211)
        self.ax_error = self.fig.add_subplot(212, sharex=self.ax_energy)

        super().__init__(self.fig)
        self.setParent(parent)

        # Data storage
        self.times: List[float] = []
        self.E_kin: List[float] = []
        self.E_pot: List[float] = []
        self.E_int: List[float] = []
        self.E_tot: List[float] = []
        self.E_error: List[float] = []

        # Initialize plots
        self._setup_plots()

    def _setup_plots(self):
        """Set up plot styling and labels."""
        # Energy plot
        self.ax_energy.set_ylabel('Energy', fontsize=10)
        self.ax_energy.set_title('Energy Evolution', fontsize=11, fontweight='bold')
        self.ax_energy.grid(True, alpha=0.3)
        self.ax_energy.tick_params(labelsize=8)

        # Conservation error plot
        self.ax_error.set_xlabel('Time', fontsize=10)
        self.ax_error.set_ylabel('ΔE / E₀', fontsize=10)
        self.ax_error.set_title('Conservation Error', fontsize=10)
        self.ax_error.grid(True, alpha=0.3)
        self.ax_error.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_error.tick_params(labelsize=8)

        self.fig.tight_layout()

    def update_data(self, time: float, energies: Dict[str, float]):
        """
        Add new data point and update plots.

        Parameters:
            time: Current simulation time
            energies: Dictionary with keys 'kinetic', 'potential', 'internal', 'total', 'error'
        """
        # Append data
        self.times.append(time)
        self.E_kin.append(energies.get('kinetic', 0.0))
        self.E_pot.append(energies.get('potential', 0.0))
        self.E_int.append(energies.get('internal', 0.0))
        self.E_tot.append(energies.get('total', 0.0))
        self.E_error.append(energies.get('error', 0.0))

        # Limit data points (keep last 1000)
        if len(self.times) > 1000:
            self.times = self.times[-1000:]
            self.E_kin = self.E_kin[-1000:]
            self.E_pot = self.E_pot[-1000:]
            self.E_int = self.E_int[-1000:]
            self.E_tot = self.E_tot[-1000:]
            self.E_error = self.E_error[-1000:]

        # Redraw plots
        self._redraw()

    def _redraw(self):
        """Redraw the plots with current data."""
        # Clear previous plots
        self.ax_energy.clear()
        self.ax_error.clear()

        # Re-setup styling
        self._setup_plots()

        if len(self.times) > 0:
            # Plot energy components
            self.ax_energy.plot(self.times, self.E_kin, 'r-', label='Kinetic', linewidth=1.5)
            self.ax_energy.plot(self.times, self.E_pot, 'b-', label='Potential', linewidth=1.5)
            self.ax_energy.plot(self.times, self.E_int, 'm-', label='Internal', linewidth=1.5)
            self.ax_energy.plot(self.times, self.E_tot, 'k--', label='Total', linewidth=2)
            self.ax_energy.legend(fontsize=8, loc='best')

            # Plot conservation error
            self.ax_error.plot(self.times, self.E_error, 'k-', linewidth=1.5)

        self.fig.tight_layout()
        self.draw()

    def clear_data(self):
        """Clear all data and reset plots."""
        self.times = []
        self.E_kin = []
        self.E_pot = []
        self.E_int = []
        self.E_tot = []
        self.E_error = []
        self._redraw()


class StatisticsWidget(QWidget):
    """
    Widget for displaying particle statistics in tabular format.

    Shows:
    - Total particles
    - Mass, energy, momentum statistics
    - Min/max/mean values
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # Create table
        self.table = QTableWidget(10, 2)
        self.table.setHorizontalHeaderLabels(['Quantity', 'Value'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(300)

        # Initialize rows
        quantities = [
            'Total Particles',
            'Total Mass',
            'Total Energy',
            'Kinetic Energy',
            'Potential Energy',
            'Internal Energy',
            'Mean Density',
            'Max Density',
            'Mean Temperature',
            'Max Temperature'
        ]

        for i, qty in enumerate(quantities):
            self.table.setItem(i, 0, QTableWidgetItem(qty))
            self.table.setItem(i, 1, QTableWidgetItem('N/A'))

        layout.addWidget(self.table)

        self.setLayout(layout)

    def update_statistics(self, stats: Dict[str, float]):
        """
        Update statistics display.

        Parameters:
            stats: Dictionary of statistic name -> value
        """
        stat_mapping = {
            'n_particles': 0,
            'total_mass': 1,
            'total_energy': 2,
            'kinetic_energy': 3,
            'potential_energy': 4,
            'internal_energy': 5,
            'mean_density': 6,
            'max_density': 7,
            'mean_temperature': 8,
            'max_temperature': 9
        }

        for key, value in stats.items():
            if key in stat_mapping:
                row = stat_mapping[key]
                formatted_value = self._format_value(key, value)
                self.table.setItem(row, 1, QTableWidgetItem(formatted_value))

    def _format_value(self, key: str, value: float) -> str:
        """Format value for display."""
        if key == 'n_particles':
            return f"{int(value):,}"
        elif 'energy' in key or 'mass' in key:
            return f"{value:.4e}"
        else:
            return f"{value:.4f}"


class DataDisplayWidget(QWidget):
    """
    Main data display widget with tabs for different views.

    Tabs:
    1. Energy Evolution: Live energy plots
    2. Statistics: Particle statistics table
    3. Diagnostics: Additional diagnostic information
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create layout
        layout = QVBoxLayout()

        # Tab widget
        tabs = QTabWidget()

        # Tab 1: Energy plot
        energy_tab = QWidget()
        energy_layout = QVBoxLayout()

        self.energy_plot = EnergyPlotWidget(width=5, height=6)
        energy_layout.addWidget(self.energy_plot)

        # Control buttons
        button_layout = QHBoxLayout()

        clear_button = QPushButton("Clear Data")
        clear_button.clicked.connect(self.energy_plot.clear_data)
        button_layout.addWidget(clear_button)

        export_button = QPushButton("Export to CSV")
        export_button.clicked.connect(self._export_energy_data)
        button_layout.addWidget(export_button)

        button_layout.addStretch()

        energy_layout.addLayout(button_layout)
        energy_tab.setLayout(energy_layout)

        tabs.addTab(energy_tab, "Energy Evolution")

        # Tab 2: Statistics
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()

        self.statistics_widget = StatisticsWidget()
        stats_layout.addWidget(self.statistics_widget)
        stats_layout.addStretch()

        stats_tab.setLayout(stats_layout)
        tabs.addTab(stats_tab, "Statistics")

        # Tab 3: Diagnostics (placeholder)
        diagnostics_tab = QWidget()
        diag_layout = QVBoxLayout()
        diag_layout.addWidget(QLabel("Diagnostic information will be displayed here."))
        diag_layout.addStretch()
        diagnostics_tab.setLayout(diag_layout)

        tabs.addTab(diagnostics_tab, "Diagnostics")

        layout.addWidget(tabs)

        self.setLayout(layout)

        # Auto-update timer (for demo)
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self._demo_update)
        self.demo_time = 0.0

    def start_demo_mode(self):
        """Start demo mode with simulated data updates."""
        self.demo_time = 0.0
        self.demo_timer.start(100)  # Update every 100ms

    def stop_demo_mode(self):
        """Stop demo mode."""
        self.demo_timer.stop()

    def _demo_update(self):
        """Generate demo data for testing."""
        self.demo_time += 0.1

        # Simulated energy data
        E0 = -1.0
        noise = np.random.randn() * 0.001

        energies = {
            'kinetic': 1.0 + np.sin(self.demo_time * 0.1) * 0.1,
            'potential': -2.5 + np.cos(self.demo_time * 0.1) * 0.2,
            'internal': 0.5 + noise,
            'total': E0 + noise,
            'error': noise / E0
        }

        self.energy_plot.update_data(self.demo_time, energies)

        # Simulated statistics
        stats = {
            'n_particles': 100000,
            'total_mass': 1.0,
            'total_energy': E0,
            'kinetic_energy': energies['kinetic'],
            'potential_energy': energies['potential'],
            'internal_energy': energies['internal'],
            'mean_density': 1.5 + np.random.rand() * 0.1,
            'max_density': 10.0 + np.random.rand() * 2.0,
            'mean_temperature': 5000 + np.random.rand() * 500,
            'max_temperature': 50000 + np.random.rand() * 5000
        }

        self.statistics_widget.update_statistics(stats)

        # Stop after 100 time units
        if self.demo_time >= 100:
            self.stop_demo_mode()

    def update_live_data(self, time: float, energies: Dict[str, float], stats: Dict[str, float]):
        """
        Update with live simulation data.

        Parameters:
            time: Current simulation time
            energies: Energy dictionary
            stats: Statistics dictionary
        """
        self.energy_plot.update_data(time, energies)
        self.statistics_widget.update_statistics(stats)

    def _export_energy_data(self):
        """Export energy data to CSV file."""
        try:
            from PyQt6.QtWidgets import QFileDialog
        except ImportError:
            from PyQt5.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Energy Data",
            "energy_evolution.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            # Write CSV
            with open(file_path, 'w') as f:
                f.write("time,kinetic,potential,internal,total,error\n")
                for i in range(len(self.energy_plot.times)):
                    f.write(
                        f"{self.energy_plot.times[i]},"
                        f"{self.energy_plot.E_kin[i]},"
                        f"{self.energy_plot.E_pot[i]},"
                        f"{self.energy_plot.E_int[i]},"
                        f"{self.energy_plot.E_tot[i]},"
                        f"{self.energy_plot.E_error[i]}\n"
                    )

            # Show confirmation
            try:
                from PyQt6.QtWidgets import QMessageBox
            except ImportError:
                from PyQt5.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Export Successful",
                f"Energy data exported to:\n{file_path}"
            )
