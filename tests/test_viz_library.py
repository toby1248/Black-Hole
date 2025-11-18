#!/usr/bin/env python3
"""
Tests for Visualization Library (TASK-101)

Test coverage:
- Matplotlib plotting functions
- Convenience functions
- Error handling
- Plot generation (smoke tests, no display)

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from tde_sph.visualization.viz_library import (
    MatplotlibVisualizer,
    quick_density_slice,
    quick_energy_plot,
    quick_phase_diagram,
    quick_radial_profile
)


class TestMatplotlibVisualizer:
    """Test Matplotlib visualizer methods."""

    def test_plot_density_slice(self):
        """Test density slice plotting."""
        # Create test data
        x = np.linspace(-1, 1, 64)
        y = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        density = np.exp(-R**2)

        # Plot
        fig, ax = MatplotlibVisualizer.plot_density_slice(
            density, X, Y, title="Test Slice"
        )

        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        assert "Test Slice" in ax.get_title()

        plt.close(fig)

    def test_plot_density_slice_log_scale(self):
        """Test density slice with log scale."""
        x = np.linspace(-1, 1, 32)
        y = np.linspace(-1, 1, 32)
        X, Y = np.meshgrid(x, y)
        density = np.ones((32, 32)) * 1.5

        fig, ax = MatplotlibVisualizer.plot_density_slice(
            density, X, Y, log_scale=True
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_density_slice_linear_scale(self):
        """Test density slice with linear scale."""
        x = np.linspace(-1, 1, 32)
        y = np.linspace(-1, 1, 32)
        X, Y = np.meshgrid(x, y)
        density = np.ones((32, 32)) * 1.5

        fig, ax = MatplotlibVisualizer.plot_density_slice(
            density, X, Y, log_scale=False
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_energy_evolution(self):
        """Test energy evolution plotting."""
        times = np.linspace(0, 10, 50)
        energies = {
            'kinetic': np.ones(50) * 1.0,
            'potential_bh': np.ones(50) * -2.0,
            'potential_self': np.ones(50) * -0.5,
            'internal_thermal': np.ones(50) * 0.5,
            'total': np.ones(50) * -1.0,
            'conservation_error': np.zeros(50)
        }

        fig, (ax1, ax2) = MatplotlibVisualizer.plot_energy_evolution(
            times, energies, title="Test Energy"
        )

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert "Test Energy" in ax1.get_title()

        plt.close(fig)

    def test_plot_phase_diagram(self):
        """Test phase diagram plotting."""
        N = 100
        x_data = np.random.lognormal(0, 1, N)
        y_data = np.random.lognormal(0, 0.5, N)
        weights = np.ones(N) * 0.01

        fig, ax = MatplotlibVisualizer.plot_phase_diagram(
            x_data, y_data, weights=weights,
            x_label='Density', y_label='Temperature'
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_phase_diagram_no_weights(self):
        """Test phase diagram without weights."""
        N = 100
        x_data = np.random.randn(N)
        y_data = np.random.randn(N)

        fig, ax = MatplotlibVisualizer.plot_phase_diagram(
            x_data, y_data, weights=None,
            log_x=False, log_y=False
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_radial_profile(self):
        """Test radial profile plotting."""
        N = 500
        radii = np.linspace(0.1, 10, N)
        quantity = np.exp(-radii)  # Exponential decay

        fig, ax = MatplotlibVisualizer.plot_radial_profile(
            radii, quantity, quantity_label='Density'
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_radial_profile_median(self):
        """Test radial profile with median binning."""
        N = 500
        radii = np.linspace(0.1, 10, N)
        quantity = np.random.rand(N)

        fig, ax = MatplotlibVisualizer.plot_radial_profile(
            radii, quantity, method='median', bins=20
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_radial_profile_linear_scale(self):
        """Test radial profile with linear scales."""
        radii = np.linspace(0.1, 10, 100)
        quantity = radii**2

        fig, ax = MatplotlibVisualizer.plot_radial_profile(
            radii, quantity, log_x=False, log_y=False
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_histogram(self):
        """Test histogram plotting."""
        data = np.random.randn(1000)

        fig, ax = MatplotlibVisualizer.plot_histogram(
            data, label='Quantity', bins=50
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_histogram_weighted(self):
        """Test weighted histogram."""
        data = np.random.randn(1000)
        weights = np.random.rand(1000)

        fig, ax = MatplotlibVisualizer.plot_histogram(
            data, weights=weights, log_y=True
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_histogram_with_nan(self):
        """Test histogram handles NaN values."""
        data = np.random.randn(100)
        data[::10] = np.nan  # Every 10th value is NaN

        fig, ax = MatplotlibVisualizer.plot_histogram(data, bins=20)

        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory_2d(self):
        """Test 2D trajectory plotting."""
        t = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)
        positions = np.column_stack([x, y])

        fig, ax = MatplotlibVisualizer.plot_trajectory(
            positions, title="Circular Orbit"
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_trajectory_3d_input(self):
        """Test trajectory plotting with 3D input (uses XY projection)."""
        t = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)
        z = np.zeros(100)
        positions = np.column_stack([x, y, z])

        fig, ax = MatplotlibVisualizer.plot_trajectory(positions)

        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory_colored_by_time(self):
        """Test trajectory with color-coded time."""
        t = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)
        positions = np.column_stack([x, y])

        fig, ax = MatplotlibVisualizer.plot_trajectory(
            positions, color=t, colorbar_label='Time'
        )

        assert fig is not None
        plt.close(fig)


class TestConvenienceFunctions:
    """Test convenience plotting functions."""

    def test_quick_density_slice(self):
        """Test quick density slice function."""
        x = np.linspace(-1, 1, 32)
        y = np.linspace(-1, 1, 32)
        X, Y = np.meshgrid(x, y)
        density = np.exp(-X**2 - Y**2)

        fig, ax = quick_density_slice(density, X, Y)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_quick_energy_plot(self):
        """Test quick energy plot function."""
        times = np.linspace(0, 10, 50)
        energies = {
            'time': times,
            'kinetic': np.ones(50),
            'potential_bh': np.ones(50) * -2.0,
            'potential_self': np.ones(50) * -0.5,
            'internal_thermal': np.ones(50) * 0.5,
            'total': np.ones(50) * -1.0,
            'conservation_error': np.zeros(50)
        }

        fig, (ax1, ax2) = quick_energy_plot(energies)

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        plt.close(fig)

    def test_quick_phase_diagram(self):
        """Test quick phase diagram function."""
        density = np.random.lognormal(0, 1, 100)
        temperature = np.random.lognormal(0, 0.5, 100)
        masses = np.ones(100) * 0.01

        fig, ax = quick_phase_diagram(density, temperature, masses)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_quick_radial_profile(self):
        """Test quick radial profile function."""
        positions = np.random.randn(500, 3)
        quantity = np.random.rand(500)

        fig, ax = quick_radial_profile(positions, quantity, 'Density')

        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test plotting with empty data."""
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        density = np.zeros((10, 10))

        fig, ax = MatplotlibVisualizer.plot_density_slice(density, X, Y)

        assert fig is not None
        plt.close(fig)

    def test_single_value_data(self):
        """Test plotting with constant data."""
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        density = np.ones((20, 20)) * 5.0

        fig, ax = MatplotlibVisualizer.plot_density_slice(density, X, Y)

        assert fig is not None
        plt.close(fig)

    def test_histogram_with_inf(self):
        """Test histogram handles infinite values."""
        data = np.random.randn(100)
        data[0] = np.inf
        data[1] = -np.inf

        fig, ax = MatplotlibVisualizer.plot_histogram(data, bins=20)

        assert fig is not None
        plt.close(fig)

    def test_radial_profile_invalid_method(self):
        """Test radial profile raises error for invalid method."""
        radii = np.linspace(0.1, 10, 100)
        quantity = np.random.rand(100)

        with pytest.raises(ValueError, match="Unknown method"):
            MatplotlibVisualizer.plot_radial_profile(
                radii, quantity, method='invalid'
            )


class TestIntegration:
    """Integration tests with realistic data."""

    def test_full_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Simulate particle data
        N = 500
        positions = np.random.randn(N, 3) * 0.5
        radii = np.linalg.norm(positions, axis=1)
        density = np.exp(-radii)
        temperature = density * 2.0
        masses = np.ones(N) * 0.002

        # Radial profile
        fig1, ax1 = MatplotlibVisualizer.plot_radial_profile(
            radii, density, quantity_label='Density'
        )
        assert fig1 is not None
        plt.close(fig1)

        # Phase diagram
        fig2, ax2 = MatplotlibVisualizer.plot_phase_diagram(
            density, temperature, weights=masses
        )
        assert fig2 is not None
        plt.close(fig2)

        # Histogram
        fig3, ax3 = MatplotlibVisualizer.plot_histogram(
            density, label='Density', weights=masses
        )
        assert fig3 is not None
        plt.close(fig3)

    def test_energy_tracking_workflow(self):
        """Test energy tracking visualization workflow."""
        # Simulate energy history
        times = np.linspace(0, 20, 200)
        E0 = -1.0

        # Add small conservation error
        E_total = E0 * (1.0 + np.random.randn(200) * 0.001)

        energies = {
            'time': times,
            'kinetic': np.ones(200) * 1.0 + np.random.randn(200) * 0.01,
            'potential_bh': np.ones(200) * -2.5 + np.random.randn(200) * 0.02,
            'potential_self': np.ones(200) * -0.5 + np.random.randn(200) * 0.01,
            'internal_thermal': np.ones(200) * 1.0 + np.random.randn(200) * 0.01,
            'total': E_total,
            'conservation_error': (E_total - E0) / E0
        }

        fig, (ax1, ax2) = MatplotlibVisualizer.plot_energy_evolution(
            times, energies
        )

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None

        # Check conservation error is small
        assert np.max(np.abs(energies['conservation_error'])) < 0.01

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
