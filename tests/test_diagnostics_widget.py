"""
Tests for TASK2 Diagnostics Widget and SimulationThread.

Tests cover:
- DiagnosticsWidget initialization and updates
- ParticleStatsTable statistics display
- PerformanceMetricsWidget metrics display
- CoordinateMetricWidget GR data display
- SimulationThread stats dict construction

References
----------
- TASK2: Implement Comprehensive Diagnostics Tab

Test Coverage
-------------
1. DiagnosticsWidget initializes with three panels
2. ParticleStatsTable displays min/max/mean/std
3. PerformanceMetricsWidget displays wall time, steps/sec
4. CoordinateMetricWidget displays GR-specific data
5. SimulationThread builds correct stats dict
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestSimulationThreadStatsDict:
    """Test SimulationThread stats dictionary construction."""

    def test_build_stats_dict_basic_structure(self):
        """Test that stats dict has expected top-level keys."""
        # Import here to allow test to run even without PyQt
        try:
            from tde_sph.gui.simulation_thread import SimulationThread
        except ImportError:
            pytest.skip("PyQt not available")

        config = {
            'integration': {'t_end': 10.0, 'timestep': 0.01},
            'particles': {'count': 100},
            'simulation': {'mode': 'schwarzschild'},
            'black_hole': {'mass': 1e6, 'spin': 0.0},
        }

        thread = SimulationThread(config)

        # Mock particles
        mock_particles = MagicMock()
        mock_particles.n_particles = 100
        mock_particles.masses = np.ones(100, dtype=np.float32) / 100
        mock_particles.density = np.ones(100, dtype=np.float32) * 1e-7
        mock_particles.pressure = np.ones(100, dtype=np.float32) * 1e-10
        mock_particles.temperature = np.ones(100, dtype=np.float32) * 5000
        mock_particles.sound_speed = np.ones(100, dtype=np.float32) * 0.05
        mock_particles.smoothing_length = np.ones(100, dtype=np.float32) * 0.1
        mock_particles.internal_energy = np.ones(100, dtype=np.float32) * 1e6
        mock_particles.velocities = np.random.randn(100, 3).astype(np.float32) * 0.1
        mock_particles.kinetic_energy = MagicMock(return_value=1.0)
        mock_particles.thermal_energy = MagicMock(return_value=0.5)

        # Mock simulation
        mock_sim = MagicMock()
        mock_sim.particles = mock_particles
        mock_sim.current_time = 1.0
        mock_sim.dt = 0.01

        # Build stats
        stats = thread._build_stats_dict(mock_sim, mock_particles, 10.0)

        # Verify structure
        assert 'performance' in stats, "Stats should have 'performance' key"
        assert 'particle_stats' in stats, "Stats should have 'particle_stats' key"
        assert 'n_particles' in stats, "Stats should have 'n_particles' key"

    def test_compute_particle_stats(self):
        """Test particle statistics computation."""
        try:
            from tde_sph.gui.simulation_thread import SimulationThread
        except ImportError:
            pytest.skip("PyQt not available")

        config = {'integration': {'t_end': 10.0}}
        thread = SimulationThread(config)

        # Create mock particles with known values
        mock_particles = MagicMock()
        mock_particles.density = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        mock_particles.pressure = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        mock_particles.temperature = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
        mock_particles.sound_speed = np.ones(5, dtype=np.float32) * 0.5
        mock_particles.smoothing_length = np.ones(5, dtype=np.float32) * 0.1
        mock_particles.internal_energy = np.ones(5, dtype=np.float32) * 1e6
        mock_particles.velocities = np.zeros((5, 3), dtype=np.float32)

        stats = thread._compute_particle_stats(mock_particles)

        # Verify density stats
        assert 'density' in stats
        assert stats['density']['min'] == pytest.approx(1.0)
        assert stats['density']['max'] == pytest.approx(5.0)
        assert stats['density']['mean'] == pytest.approx(3.0)

        # Verify temperature stats
        assert 'temperature' in stats
        assert stats['temperature']['min'] == pytest.approx(100.0)
        assert stats['temperature']['max'] == pytest.approx(500.0)
        assert stats['temperature']['mean'] == pytest.approx(300.0)

    def test_performance_metrics(self):
        """Test performance metrics computation."""
        try:
            from tde_sph.gui.simulation_thread import SimulationThread
        except ImportError:
            pytest.skip("PyQt not available")

        config = {'integration': {'t_end': 10.0}}
        thread = SimulationThread(config)
        thread._step_count = 1000

        mock_sim = MagicMock()
        mock_sim.current_time = 5.0
        mock_sim.dt = 0.005

        mock_particles = MagicMock()
        mock_particles.n_particles = 10000

        stats = thread._build_stats_dict(mock_sim, mock_particles, 100.0)  # 100s wall time

        assert stats['performance']['wall_time'] == pytest.approx(100.0)
        assert stats['performance']['step_count'] == 1000
        assert stats['performance']['steps_per_sec'] == pytest.approx(10.0)

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        try:
            from tde_sph.gui.simulation_thread import SimulationThread
        except ImportError:
            pytest.skip("PyQt not available")

        config = {}
        thread = SimulationThread(config)

        mock_particles = MagicMock()
        mock_particles.n_particles = 100000  # 100k particles

        memory_mb = thread._estimate_memory_usage(mock_particles)

        # 100k particles * 13 floats * 4 bytes = 5.2 MB
        expected_mb = 100000 * 13 * 4 / (1024 * 1024)
        assert memory_mb == pytest.approx(expected_mb)

    def test_coordinate_data_schwarzschild(self):
        """Test coordinate data for Schwarzschild metric."""
        try:
            from tde_sph.gui.simulation_thread import SimulationThread
        except ImportError:
            pytest.skip("PyQt not available")

        config = {}
        thread = SimulationThread(config)

        mock_sim = MagicMock()
        mock_sim.metric_type = 'Schwarzschild'
        mock_sim.coordinate_system = 'Boyer-Lindquist'
        mock_sim.black_hole_mass = 1e6
        mock_sim.black_hole_spin = 0.0

        mock_particles = MagicMock()
        # Particles at various distances
        mock_particles.positions = np.array([
            [5.0, 0.0, 0.0],  # Inside ISCO (r=5)
            [10.0, 0.0, 0.0],  # Outside ISCO (r=10)
            [20.0, 0.0, 0.0],  # Far out (r=20)
        ], dtype=np.float32)

        data = thread._compute_coordinate_data(mock_sim, mock_particles)

        assert data['metric_type'] == 'Schwarzschild'
        assert data['bh_spin'] == pytest.approx(0.0)
        assert data['isco_radius'] == pytest.approx(6.0)  # Schwarzschild ISCO
        assert data['r_min'] == pytest.approx(5.0)
        assert data['r_max'] == pytest.approx(20.0)
        assert data['particles_within_isco'] == 1  # Only the r=5 particle


class TestMockSimulation:
    """Test MockSimulation for SimulationThread testing."""

    def test_mock_simulation_initialization(self):
        """Test MockSimulation initializes correctly."""
        try:
            from tde_sph.gui.simulation_thread import MockSimulation
        except ImportError:
            pytest.skip("PyQt not available")

        config = {
            'integration': {'timestep': 0.01, 't_end': 10.0},
            'particles': {'count': 500},
            'simulation': {'mode': 'kerr'},
            'black_hole': {'mass': 2e6, 'spin': 0.7},
        }

        sim = MockSimulation(config)

        assert sim.current_time == 0.0
        assert sim.dt == 0.01
        assert sim.particles.n_particles == 500
        assert sim.metric_type == 'Kerr'
        assert sim.black_hole_spin == 0.7

    def test_mock_simulation_step(self):
        """Test MockSimulation advances time correctly."""
        try:
            from tde_sph.gui.simulation_thread import MockSimulation
        except ImportError:
            pytest.skip("PyQt not available")

        config = {
            'integration': {'timestep': 0.1},
            'particles': {'count': 100},
        }

        sim = MockSimulation(config)
        initial_time = sim.current_time

        sim.step()

        assert sim.current_time == pytest.approx(initial_time + 0.1)

    def test_mock_particle_system_energies(self):
        """Test MockParticleSystem energy calculations."""
        try:
            from tde_sph.gui.simulation_thread import MockParticleSystem
        except ImportError:
            pytest.skip("PyQt not available")

        particles = MockParticleSystem(1000)

        # Energies should be positive
        ke = particles.kinetic_energy()
        te = particles.thermal_energy()

        assert ke >= 0, "Kinetic energy should be non-negative"
        assert te >= 0, "Thermal energy should be non-negative"


class TestDiagnosticsWidgetIntegration:
    """Integration tests for DiagnosticsWidget."""

    def test_diagnostics_widget_demo_data(self):
        """Test DiagnosticsWidget with demo data."""
        try:
            from tde_sph.gui.data_display import DiagnosticsWidget
        except ImportError:
            pytest.skip("PyQt not available")

        # This test requires a QApplication
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            try:
                from PyQt5.QtWidgets import QApplication
            except ImportError:
                pytest.skip("No PyQt available")

        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        widget = DiagnosticsWidget()

        # Set demo data - should not raise
        widget.set_demo_data()

        # Verify widget exists
        assert widget is not None

    def test_particle_stats_table_update(self):
        """Test ParticleStatsTable updates correctly."""
        try:
            from tde_sph.gui.data_display import ParticleStatsTable
        except ImportError:
            pytest.skip("PyQt not available")

        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            try:
                from PyQt5.QtWidgets import QApplication
            except ImportError:
                pytest.skip("No PyQt available")

        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        table = ParticleStatsTable()

        # Update with stats
        stats = {
            'density': {'min': 1e-10, 'max': 1e-5, 'mean': 1e-7, 'std': 1e-8},
            'temperature': {'min': 1000, 'max': 100000, 'mean': 50000, 'std': 10000},
        }

        table.update_stats(stats)

        # Should not raise
        assert table is not None

    def test_performance_metrics_update(self):
        """Test PerformanceMetricsWidget updates correctly."""
        try:
            from tde_sph.gui.data_display import PerformanceMetricsWidget
        except ImportError:
            pytest.skip("PyQt not available")

        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            try:
                from PyQt5.QtWidgets import QApplication
            except ImportError:
                pytest.skip("No PyQt available")

        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        widget = PerformanceMetricsWidget()

        metrics = {
            'wall_time': 3661,  # 1 hour, 1 minute, 1 second
            'sim_time': 0.5,
            'step_count': 10000,
            'steps_per_sec': 15.3,
            'gpu_available': True,
            'memory_mb': 256.5,
        }

        widget.update_metrics(metrics)

        # Check wall time formatting
        assert widget.wall_time_label.text() == "01:01:01"
        assert widget.step_count_label.text() == "10,000"

    def test_coordinate_metric_update(self):
        """Test CoordinateMetricWidget updates correctly."""
        try:
            from tde_sph.gui.data_display import CoordinateMetricWidget
        except ImportError:
            pytest.skip("PyQt not available")

        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            try:
                from PyQt5.QtWidgets import QApplication
            except ImportError:
                pytest.skip("No PyQt available")

        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        widget = CoordinateMetricWidget()

        data = {
            'metric_type': 'Kerr',
            'coordinate_system': 'Boyer-Lindquist',
            'bh_mass': 1e6,
            'bh_spin': 0.9,
            'r_min': 2.0,
            'r_max': 100.0,
            'isco_radius': 2.32,
            'particles_within_isco': 50,
        }

        widget.update_data(data)

        assert widget.metric_type_label.text() == "Kerr"
        assert "0.9" in widget.bh_spin_label.text()
