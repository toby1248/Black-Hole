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
        QLabel, QTableWidget, QTableWidgetItem, QTabWidget,
        QFormLayout, QGridLayout, QSplitter, QFrame, QHeaderView
    )
    from PyQt6.QtCore import Qt, QTimer
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
        QLabel, QTableWidget, QTableWidgetItem, QTabWidget,
        QFormLayout, QGridLayout, QSplitter, QFrame, QHeaderView
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


class ParticleStatsTable(QWidget):
    """
    Comprehensive particle statistics table (TASK2).

    Displays min/max/mean/stddev for all particle quantities
    in a multi-column table optimized for ultrawide displays.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Particle Statistics")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Create table with 5 columns: Quantity | Min | Max | Mean | Std Dev
        self.table = QTableWidget(8, 5)
        self.table.setHorizontalHeaderLabels(['Quantity', 'Min', 'Max', 'Mean', 'Std Dev'])

        # Configure header
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        # Initialize rows
        quantities = [
            'Density (ρ)',
            'Pressure (P)',
            'Temperature (T)',
            'Sound Speed (cs)',
            'Velocity Magnitude (|v|)',
            'Smoothing Length (h)',
            'Internal Energy (u)',
            'Specific Entropy (s)'
        ]

        for i, qty in enumerate(quantities):
            self.table.setItem(i, 0, QTableWidgetItem(qty))
            for j in range(1, 5):
                self.table.setItem(i, j, QTableWidgetItem('N/A'))

        layout.addWidget(self.table)
        self.setLayout(layout)

    def update_stats(self, stats: Dict[str, Dict[str, float]]):
        """
        Update particle statistics.

        Parameters:
            stats: Dict with quantity names as keys, each containing
                   {'min': ..., 'max': ..., 'mean': ..., 'std': ...}
        """
        quantity_mapping = {
            'density': 0,
            'pressure': 1,
            'temperature': 2,
            'sound_speed': 3,
            'velocity_magnitude': 4,
            'smoothing_length': 5,
            'internal_energy': 6,
            'entropy': 7
        }

        for key, values in stats.items():
            if key in quantity_mapping:
                row = quantity_mapping[key]
                if isinstance(values, dict):
                    self.table.setItem(row, 1, QTableWidgetItem(f"{values.get('min', 0):.4e}"))
                    self.table.setItem(row, 2, QTableWidgetItem(f"{values.get('max', 0):.4e}"))
                    self.table.setItem(row, 3, QTableWidgetItem(f"{values.get('mean', 0):.4e}"))
                    self.table.setItem(row, 4, QTableWidgetItem(f"{values.get('std', 0):.4e}"))


class PerformanceMetricsWidget(QWidget):
    """
    Performance metrics display (TASK2).

    Shows wall-clock time, steps/sec, GPU status, memory usage.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Performance Metrics")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Use form layout for key-value pairs
        form = QFormLayout()

        # Create labels for values (will be updated)
        self.wall_time_label = QLabel("00:00:00")
        self.sim_time_label = QLabel("0.000")
        self.steps_per_sec_label = QLabel("N/A")
        self.gpu_status_label = QLabel("Unknown")
        self.memory_label = QLabel("N/A")
        self.step_count_label = QLabel("0")
        self.dt_label = QLabel("N/A")

        form.addRow("Wall-clock Time:", self.wall_time_label)
        form.addRow("Simulation Time:", self.sim_time_label)
        form.addRow("Current Step:", self.step_count_label)
        form.addRow("Timestep (dt):", self.dt_label)
        form.addRow("Steps/second:", self.steps_per_sec_label)
        form.addRow("GPU Status:", self.gpu_status_label)
        form.addRow("Memory Usage:", self.memory_label)

        layout.addLayout(form)
        layout.addStretch()
        self.setLayout(layout)

    def update_metrics(self, metrics: Dict[str, float]):
        """
        Update performance metrics.

        Parameters:
            metrics: Dict with keys like 'wall_time', 'sim_time', 'steps_per_sec', etc.
        """
        if 'wall_time' in metrics:
            # Format as HH:MM:SS
            t = int(metrics['wall_time'])
            hours, remainder = divmod(t, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.wall_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        if 'sim_time' in metrics:
            self.sim_time_label.setText(f"{metrics['sim_time']:.6f}")

        if 'step_count' in metrics:
            self.step_count_label.setText(f"{int(metrics['step_count']):,}")

        if 'dt' in metrics:
            self.dt_label.setText(f"{metrics['dt']:.4e}")

        if 'steps_per_sec' in metrics:
            self.steps_per_sec_label.setText(f"{metrics['steps_per_sec']:.2f}")

        if 'gpu_available' in metrics:
            if metrics['gpu_available']:
                self.gpu_status_label.setText("CUDA Available")
                self.gpu_status_label.setStyleSheet("color: green;")
            else:
                self.gpu_status_label.setText("CPU Only")
                self.gpu_status_label.setStyleSheet("color: orange;")

        if 'memory_mb' in metrics:
            self.memory_label.setText(f"{metrics['memory_mb']:.1f} MB")


class CoordinateMetricWidget(QWidget):
    """
    Coordinate and metric data display for GR simulations (TASK2).

    Shows black hole parameters, particle distances, ISCO statistics.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Coordinate/Metric Data (GR Mode)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Use form layout
        form = QFormLayout()

        self.metric_type_label = QLabel("N/A")
        self.coord_system_label = QLabel("N/A")
        self.bh_mass_label = QLabel("N/A")
        self.bh_spin_label = QLabel("N/A")
        self.r_min_label = QLabel("N/A")
        self.r_max_label = QLabel("N/A")
        self.r_mean_label = QLabel("N/A")
        self.isco_radius_label = QLabel("N/A")
        self.particles_in_isco_label = QLabel("N/A")

        form.addRow("Metric Type:", self.metric_type_label)
        form.addRow("Coordinate System:", self.coord_system_label)
        form.addRow("Black Hole Mass (M):", self.bh_mass_label)
        form.addRow("Black Hole Spin (a):", self.bh_spin_label)
        form.addRow("Min Distance (r_min):", self.r_min_label)
        form.addRow("Max Distance (r_max):", self.r_max_label)
        form.addRow("Mean Distance (r_mean):", self.r_mean_label)
        form.addRow("ISCO Radius:", self.isco_radius_label)
        form.addRow("Particles < ISCO:", self.particles_in_isco_label)

        layout.addLayout(form)
        layout.addStretch()
        self.setLayout(layout)

    def update_data(self, data: Dict[str, float]):
        """
        Update coordinate/metric data.

        Parameters:
            data: Dict with GR-specific data like 'metric_type', 'bh_mass', etc.
        """
        if 'metric_type' in data:
            self.metric_type_label.setText(str(data['metric_type']))

        if 'coordinate_system' in data:
            self.coord_system_label.setText(str(data['coordinate_system']))

        if 'bh_mass' in data:
            self.bh_mass_label.setText(f"{data['bh_mass']:.4e} M☉")

        if 'bh_spin' in data:
            self.bh_spin_label.setText(f"{data['bh_spin']:.4f}")

        if 'r_min' in data:
            self.r_min_label.setText(f"{data['r_min']:.4f} R_g")

        if 'r_max' in data:
            self.r_max_label.setText(f"{data['r_max']:.4f} R_g")

        if 'r_mean' in data:
            self.r_mean_label.setText(f"{data['r_mean']:.4f} R_g")

        if 'isco_radius' in data:
            self.isco_radius_label.setText(f"{data['isco_radius']:.4f} R_g")

        if 'particles_within_isco' in data:
            self.particles_in_isco_label.setText(f"{int(data['particles_within_isco']):,}")


class DiagnosticsWidget(QWidget):
    """
    Comprehensive diagnostics widget (TASK2).

    Combines particle statistics, performance metrics, and coordinate data
    in a multi-panel layout optimized for ultrawide monitors.

    Features:
    - Particle statistics: min/max/mean/stddev for all quantities
    - Performance metrics: wall time, steps/sec, GPU status
    - Coordinate/metric data: GR-specific information (conditional)
    - Layout optimized for 2560px+ width displays
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout with splitter for resizable panels
        main_layout = QHBoxLayout()

        # Create splitter for side-by-side panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Particle Statistics
        self.particle_stats = ParticleStatsTable()
        splitter.addWidget(self.particle_stats)

        # Middle panel: Performance Metrics
        self.performance_metrics = PerformanceMetricsWidget()
        splitter.addWidget(self.performance_metrics)

        # Right panel: Coordinate/Metric Data (GR mode)
        self.coordinate_metric = CoordinateMetricWidget()
        splitter.addWidget(self.coordinate_metric)

        # Set initial sizes (roughly equal)
        splitter.setSizes([400, 300, 300])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Set minimum width hint for ultrawide optimization
        self.setMinimumWidth(1200)

    def update_diagnostics(self, data: Dict):
        """
        Update all diagnostic panels.

        Parameters:
            data: Dictionary containing all diagnostic data:
                - 'particle_stats': dict of quantity stats
                - 'performance': dict of performance metrics
                - 'coordinate_metric': dict of GR data
        """
        # Update particle statistics
        if 'particle_stats' in data:
            self.particle_stats.update_stats(data['particle_stats'])

        # Update performance metrics
        if 'performance' in data:
            self.performance_metrics.update_metrics(data['performance'])

        # Update coordinate/metric data
        if 'coordinate_metric' in data:
            self.coordinate_metric.update_data(data['coordinate_metric'])

    def set_demo_data(self):
        """Set demo data for testing the widget."""
        demo_data = {
            'particle_stats': {
                'density': {'min': 1e-10, 'max': 1e-5, 'mean': 1e-7, 'std': 1e-8},
                'pressure': {'min': 1e-12, 'max': 1e-8, 'mean': 1e-10, 'std': 1e-11},
                'temperature': {'min': 1e3, 'max': 1e6, 'mean': 5e4, 'std': 1e4},
                'sound_speed': {'min': 0.01, 'max': 0.1, 'mean': 0.05, 'std': 0.01},
                'velocity_magnitude': {'min': 0.001, 'max': 0.5, 'mean': 0.1, 'std': 0.05},
                'smoothing_length': {'min': 0.05, 'max': 0.2, 'mean': 0.1, 'std': 0.02},
                'internal_energy': {'min': 1e5, 'max': 1e8, 'mean': 1e6, 'std': 1e5},
            },
            'performance': {
                'wall_time': 3661,  # 1 hour, 1 minute, 1 second
                'sim_time': 0.5,
                'step_count': 10000,
                'dt': 1e-5,
                'steps_per_sec': 15.3,
                'gpu_available': True,
                'memory_mb': 256.5,
            },
            'coordinate_metric': {
                'metric_type': 'Schwarzschild',
                'coordinate_system': 'Boyer-Lindquist',
                'bh_mass': 1e6,
                'bh_spin': 0.0,
                'r_min': 3.5,
                'r_max': 100.0,
                'r_mean': 25.0,
                'isco_radius': 6.0,
                'particles_within_isco': 150,
            }
        }
        self.update_diagnostics(demo_data)


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

        # Tab 3: Comprehensive Diagnostics (TASK2)
        self.diagnostics_widget = DiagnosticsWidget()
        tabs.addTab(self.diagnostics_widget, "Diagnostics")

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

    def update_live_data(self, time: float, energies: Dict[str, float], stats: Dict[str, float],
                         diagnostics: Optional[Dict] = None):
        """
        Update with live simulation data.

        Parameters:
            time: Current simulation time
            energies: Energy dictionary
            stats: Statistics dictionary
            diagnostics: Optional diagnostics data for TASK2 comprehensive panel
        """
        self.energy_plot.update_data(time, energies)
        self.statistics_widget.update_statistics(stats)
        if diagnostics is not None:
            self.diagnostics_widget.update_diagnostics(diagnostics)

    def update_diagnostics(self, diagnostics: Dict):
        """
        Update diagnostics panel (TASK2).

        Parameters:
            diagnostics: Dictionary containing particle_stats, performance, coordinate_metric
        """
        self.diagnostics_widget.update_diagnostics(diagnostics)

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
