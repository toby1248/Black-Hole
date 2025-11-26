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
    from PyQt6.QtGui import QFont
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
        QLabel, QTableWidget, QTableWidgetItem, QTabWidget
    )
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
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

        # Always call tight_layout and draw, even if no data (prevents crash)
        try:
            self.fig.tight_layout()
            self.draw()
        except Exception as e:
            # Silently handle any plotting errors
            pass

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
        self.table = QTableWidget(12, 2)
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
            'Potential (BH)',
            'Internal Energy',
            'Median Distance (BH)',
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
            'potential_bh': 5,
            'internal_energy': 6,
            'median_distance_from_bh': 7,
            'mean_density': 8,
            'max_density': 9,
            'mean_temperature': 10,
            'max_temperature': 11
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


class PercentilePlotWidget(FigureCanvas):
    """Matplotlib canvas for percentile evolution plots."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax_distance = self.fig.add_subplot(211)
        self.ax_energy = self.fig.add_subplot(212, sharex=self.ax_distance)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Data storage
        self.times: List[float] = []
        self.dist_p01: List[float] = []
        self.dist_p10: List[float] = []
        self.dist_p25: List[float] = []
        self.dist_p50: List[float] = []
        self.dist_p75: List[float] = []
        self.dist_p90: List[float] = []
        self.dist_p99: List[float] = []
        
        self.eng_p01: List[float] = []
        self.eng_p10: List[float] = []
        self.eng_p25: List[float] = []
        self.eng_p50: List[float] = []
        self.eng_p75: List[float] = []
        self.eng_p90: List[float] = []
        self.eng_p99: List[float] = []
        
        self._setup_plots()
    
    def _setup_plots(self):
        """Set up plot styling."""
        self.ax_distance.set_ylabel('Distance from BH', fontsize=10)
        self.ax_distance.set_title('Distance Percentiles vs Time', fontsize=11, fontweight='bold')
        self.ax_distance.grid(True, alpha=0.3)
        self.ax_distance.tick_params(labelsize=8)
        
        self.ax_energy.set_xlabel('Time', fontsize=10)
        self.ax_energy.set_ylabel('Specific Energy', fontsize=10)
        self.ax_energy.set_title('Energy Percentiles vs Time', fontsize=10)
        self.ax_energy.grid(True, alpha=0.3)
        self.ax_energy.tick_params(labelsize=8)
        
        self.fig.tight_layout()
    
    def update_data(self, time: float, dist_pct: List[float], eng_pct: List[float]):
        """Add new percentile data."""
        self.times.append(time)
        self.dist_p01.append(dist_pct[0])
        self.dist_p10.append(dist_pct[1])
        self.dist_p25.append(dist_pct[2])
        self.dist_p50.append(dist_pct[3])
        self.dist_p75.append(dist_pct[4])
        self.dist_p90.append(dist_pct[5])
        self.dist_p99.append(dist_pct[6])
        
        self.eng_p01.append(eng_pct[0])
        self.eng_p10.append(eng_pct[1])
        self.eng_p25.append(eng_pct[2])
        self.eng_p50.append(eng_pct[3])
        self.eng_p75.append(eng_pct[4])
        self.eng_p90.append(eng_pct[5])
        self.eng_p99.append(eng_pct[6])
        
        # Limit data
        if len(self.times) > 1000:
            self.times = self.times[-1000:]
            self.dist_p01 = self.dist_p01[-1000:]
            self.dist_p10 = self.dist_p10[-1000:]
            self.dist_p25 = self.dist_p25[-1000:]
            self.dist_p50 = self.dist_p50[-1000:]
            self.dist_p75 = self.dist_p75[-1000:]
            self.dist_p90 = self.dist_p90[-1000:]
            self.dist_p99 = self.dist_p99[-1000:]
            self.eng_p01 = self.eng_p01[-1000:]
            self.eng_p10 = self.eng_p10[-1000:]
            self.eng_p25 = self.eng_p25[-1000:]
            self.eng_p50 = self.eng_p50[-1000:]
            self.eng_p75 = self.eng_p75[-1000:]
            self.eng_p90 = self.eng_p90[-1000:]
            self.eng_p99 = self.eng_p99[-1000:]
        
        self._redraw()
    
    def _redraw(self):
        """Redraw plots."""
        self.ax_distance.clear()
        self.ax_energy.clear()
        self._setup_plots()
        
        if len(self.times) > 0:
            # Distance percentiles with 3 shaded bands
            self.ax_distance.fill_between(self.times, self.dist_p01, self.dist_p99, alpha=0.1, color='blue', label='1-99%')
            self.ax_distance.fill_between(self.times, self.dist_p10, self.dist_p90, alpha=0.2, color='blue', label='10-90%')
            self.ax_distance.fill_between(self.times, self.dist_p25, self.dist_p75, alpha=0.3, color='blue', label='25-75%')
            self.ax_distance.plot(self.times, self.dist_p50, 'b-', linewidth=2, label='Median')
            self.ax_distance.legend(fontsize=8, loc='best')
            
            # Energy percentiles with 3 shaded bands
            self.ax_energy.fill_between(self.times, self.eng_p01, self.eng_p99, alpha=0.1, color='red', label='1-99%')
            self.ax_energy.fill_between(self.times, self.eng_p10, self.eng_p90, alpha=0.2, color='red', label='10-90%')
            self.ax_energy.fill_between(self.times, self.eng_p25, self.eng_p75, alpha=0.3, color='red', label='25-75%')
            self.ax_energy.plot(self.times, self.eng_p50, 'r-', linewidth=2, label='Median')
            self.ax_energy.legend(fontsize=8, loc='best')
        
        # Always call tight_layout and draw, even if no data (prevents crash)
        try:
            self.fig.tight_layout()
            self.draw()
        except Exception as e:
            # Silently handle any plotting errors
            pass
    
    def clear_data(self):
        """Clear data."""
        self.times = []
        self.dist_p01 = []
        self.dist_p10 = []
        self.dist_p25 = []
        self.dist_p50 = []
        self.dist_p75 = []
        self.dist_p90 = []
        self.eng_p10 = []
        self.eng_p25 = []
        self.eng_p50 = []
        self.eng_p75 = []
        self.eng_p90 = []
        self._redraw()


class TimingDiagnosticsWidget(QWidget):
    """Widget for module execution timing information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        
        # Create table for timing data (13 rows now)
        self.table = QTableWidget(13, 2)
        self.table.setHorizontalHeaderLabels(['Module', 'Time (ms)'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(420)
        
        modules = [
            'Compute Forces (Total)',
            'Gravity Solver',
            'SPH Density',
            'Smoothing Lengths',
            'SPH Pressure Forces',
            'Time Integration',
            'Thermodynamics (EOS)',
            'Energy Computation',
            'Timestep Estimation',
            'GPU Transfer',
            'I/O Overhead',
            'Other',
            'Total Timestep'
        ]
        
        for i, mod in enumerate(modules):
            self.table.setItem(i, 0, QTableWidgetItem(mod))
            self.table.setItem(i, 1, QTableWidgetItem('N/A'))
        
        layout.addWidget(QLabel("Module Execution Timing (per timestep)"))
        layout.addWidget(self.table)
        
        # Performance summary
        self.summary_label = QLabel("Performance: N/A")
        font = QFont()
        font.setBold(True)
        self.summary_label.setFont(font)
        layout.addWidget(self.summary_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_timings(self, timings: Dict[str, float]):
        """Update timing display."""
        timing_map = {
            'compute_forces': 0,
            'gravity': 1,
            'sph_density': 2,
            'smoothing_lengths': 3,
            'sph_pressure': 4,
            'integration': 5,
            'thermodynamics': 6,
            'energy_computation': 7,
            'timestep_estimation': 8,
            'gpu_transfer': 9,
            'io_overhead': 10,
            'other': 11,
            'total': 12
        }
        
        for key, value in timings.items():
            if key in timing_map:
                row = timing_map[key]
                self.table.setItem(row, 1, QTableWidgetItem(f"{value*1000:.2f}"))
        
        # Update summary
        total = timings.get('total', 0)
        if total > 0:
            steps_per_sec = 1.0 / total
            self.summary_label.setText(f"Performance: {steps_per_sec:.1f} steps/sec ({total*1000:.1f} ms/step)")


class DataDisplayWidget(QWidget):
    """
    Main data display widget with tabs for different views.

    Tabs:
    1. Energy Evolution: Live energy plots
    2. Statistics: Particle statistics table
    3. Percentiles: Distance and energy percentile evolution
    4. Timing: Module execution timing diagnostics
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Chart update control
        self.charts_enabled = True

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

        # Tab 3: Percentile plots
        percentile_tab = QWidget()
        percentile_layout = QVBoxLayout()
        
        self.percentile_plot = PercentilePlotWidget(width=5, height=6)
        percentile_layout.addWidget(self.percentile_plot)
        
        pct_button_layout = QHBoxLayout()
        clear_pct_button = QPushButton("Clear Data")
        clear_pct_button.clicked.connect(self.percentile_plot.clear_data)
        pct_button_layout.addWidget(clear_pct_button)
        pct_button_layout.addStretch()
        
        percentile_layout.addLayout(pct_button_layout)
        percentile_tab.setLayout(percentile_layout)
        tabs.addTab(percentile_tab, "Percentiles")

        # Tab 4: Timing Diagnostics
        timing_tab = QWidget()
        timing_layout = QVBoxLayout()
        
        self.timing_widget = TimingDiagnosticsWidget()
        timing_layout.addWidget(self.timing_widget)
        
        timing_tab.setLayout(timing_layout)
        tabs.addTab(timing_tab, "Timing")

        layout.addWidget(tabs)

        self.setLayout(layout)

    def update_live_data(self, time: float, energies: Dict[str, float], stats: Dict[str, float]):
        """
        Update with live simulation data.

        Parameters:
            time: Current simulation time
            energies: Energy dictionary (includes 'potential_bh')
            stats: Statistics dictionary (includes percentiles and timings)
        """
        # Always update statistics and timing (lightweight)
        self.statistics_widget.update_statistics(stats)
        
        # Update BH potential in statistics if available
        if 'potential_bh' in energies:
            stats['potential_bh'] = energies['potential_bh']
            self.statistics_widget.update_statistics(stats)
        
        # Update timing diagnostics if available
        if 'timings' in stats:
            self.timing_widget.update_timings(stats['timings'])
        
        # Only update charts if enabled (expensive matplotlib redraws)
        if self.charts_enabled:
            self.energy_plot.update_data(time, energies)
            
            # Update percentile plots if data available
            if 'distance_percentiles' in stats and 'energy_percentiles' in stats:
                self.percentile_plot.update_data(
                    time,
                    stats['distance_percentiles'],
                    stats['energy_percentiles']
                )

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

    def set_chart_updates_enabled(self, enabled: bool):
        """
        Enable or disable live chart updates.
        
        Parameters:
            enabled: True to enable chart updates, False to disable
        """
        self.charts_enabled = enabled

