#!/usr/bin/env python3
"""
Tests for Spatial/Temporal Interpolation (TASK-102)

Test coverage:
- SPH kernel functions (cubic spline, Wendland C2, Gaussian)
- 3D grid interpolation
- 2D slice interpolation
- Temporal interpolation between snapshots
- Smoothing filters
- Edge cases (empty particles, constant fields)

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import pytest
import numpy as np
from tde_sph.visualization.interpolation import (
    SPHInterpolator,
    TemporalInterpolator,
    SmoothingFilters,
    particles_to_grid_3d,
    particles_to_slice_2d
)


class TestSPHKernels:
    """Test SPH kernel functions."""

    def test_cubic_spline_kernel_at_origin(self):
        """Test cubic spline kernel at q=0 (maximum)."""
        interp = SPHInterpolator(kernel='cubic_spline')
        q = np.array([0.0])
        W = interp._cubic_spline_kernel(q)
        assert W[0] == pytest.approx(1.0, abs=1e-6)

    def test_cubic_spline_kernel_at_boundary(self):
        """Test cubic spline kernel at q=2 (support boundary)."""
        interp = SPHInterpolator(kernel='cubic_spline')
        q = np.array([2.0])
        W = interp._cubic_spline_kernel(q)
        assert W[0] == pytest.approx(0.0, abs=1e-6)

    def test_cubic_spline_kernel_beyond_support(self):
        """Test cubic spline kernel beyond q=2 is zero."""
        interp = SPHInterpolator(kernel='cubic_spline')
        q = np.array([2.5, 3.0, 5.0])
        W = interp._cubic_spline_kernel(q)
        assert np.all(W == 0.0)

    def test_cubic_spline_kernel_continuity(self):
        """Test cubic spline kernel is continuous at q=1."""
        interp = SPHInterpolator(kernel='cubic_spline')
        q = np.array([0.999, 1.0, 1.001])
        W = interp._cubic_spline_kernel(q)
        # Should be approximately continuous
        assert abs(W[0] - W[1]) < 0.01
        assert abs(W[1] - W[2]) < 0.01

    def test_wendland_c2_kernel_at_origin(self):
        """Test Wendland C2 kernel at q=0."""
        interp = SPHInterpolator(kernel='wendland_c2')
        q = np.array([0.0])
        W = interp._wendland_c2_kernel(q)
        assert W[0] == pytest.approx(1.0, abs=1e-6)

    def test_wendland_c2_kernel_at_boundary(self):
        """Test Wendland C2 kernel at q=1 (compact support)."""
        interp = SPHInterpolator(kernel='wendland_c2')
        q = np.array([1.0])
        W = interp._wendland_c2_kernel(q)
        assert W[0] == pytest.approx(0.0, abs=1e-6)

    def test_wendland_c2_kernel_beyond_support(self):
        """Test Wendland C2 kernel beyond q=1 is zero."""
        interp = SPHInterpolator(kernel='wendland_c2')
        q = np.array([1.5, 2.0])
        W = interp._wendland_c2_kernel(q)
        assert np.all(W == 0.0)

    def test_gaussian_kernel_at_origin(self):
        """Test Gaussian kernel at q=0."""
        interp = SPHInterpolator(kernel='gaussian')
        q = np.array([0.0])
        W = interp._gaussian_kernel(q)
        assert W[0] == pytest.approx(1.0, abs=1e-6)

    def test_gaussian_kernel_truncation(self):
        """Test Gaussian kernel truncated at q=3."""
        interp = SPHInterpolator(kernel='gaussian')
        q = np.array([3.0, 3.5])
        W = interp._gaussian_kernel(q)
        assert np.all(W == 0.0)

    def test_gaussian_kernel_decay(self):
        """Test Gaussian kernel decays as exp(-q²)."""
        interp = SPHInterpolator(kernel='gaussian')
        q = np.array([1.0, 2.0])
        W = interp._gaussian_kernel(q)
        # W(1) = exp(-1) ≈ 0.368
        # W(2) = exp(-4) ≈ 0.018
        assert W[0] == pytest.approx(np.exp(-1.0), rel=1e-3)
        assert W[1] == pytest.approx(np.exp(-4.0), rel=1e-3)

    def test_invalid_kernel_raises_error(self):
        """Test invalid kernel name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            SPHInterpolator(kernel='invalid_kernel')


class TestSPHInterpolation3D:
    """Test SPH interpolation to 3D grids."""

    def test_interpolate_single_particle_to_grid(self):
        """Test interpolating single particle to grid."""
        interp = SPHInterpolator(kernel='cubic_spline')

        # Single particle at origin with constant density
        positions = np.array([[0.0, 0.0, 0.0]])
        field_values = np.array([1.0])
        smoothing_lengths = np.array([0.1])

        grid, (X, Y, Z) = interp.interpolate_to_grid(
            positions, field_values, smoothing_lengths,
            grid_shape=(16, 16, 16),
            bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
        )

        # Grid should be non-zero near origin
        assert grid.shape == (16, 16, 16)
        assert np.max(grid) > 0.0

        # Maximum should be at or near center
        center_value = grid[8, 8, 8]
        assert center_value > 0.0

    def test_interpolate_constant_field(self):
        """Test interpolating constant density field."""
        interp = SPHInterpolator(kernel='cubic_spline')

        # 100 particles with constant density
        N = 100
        positions = np.random.randn(N, 3) * 0.2
        field_values = np.ones(N) * 2.0
        smoothing_lengths = np.ones(N) * 0.1

        grid, (X, Y, Z) = interp.interpolate_to_grid(
            positions, field_values, smoothing_lengths,
            grid_shape=(32, 32, 32)
        )

        # Grid should have positive values
        assert np.max(grid) > 0.0

    def test_interpolate_gaussian_blob(self):
        """Test interpolating Gaussian density profile."""
        interp = SPHInterpolator(kernel='cubic_spline')

        # Particles with Gaussian density profile
        N = 500
        positions = np.random.randn(N, 3) * 0.3
        r_squared = np.sum(positions**2, axis=1)
        field_values = np.exp(-r_squared)  # Gaussian profile
        smoothing_lengths = np.ones(N) * 0.1

        grid, (X, Y, Z) = interp.interpolate_to_grid(
            positions, field_values, smoothing_lengths,
            grid_shape=(32, 32, 32),
            bounds=((-1, 1), (-1, 1), (-1, 1))
        )

        # Maximum should be near center
        assert np.max(grid) > 0.0

        # Grid should decay away from center
        center_val = grid[16, 16, 16]
        edge_val = grid[0, 0, 0]
        assert center_val > edge_val

    def test_interpolate_vector_field(self):
        """Test interpolating vector field (velocity)."""
        interp = SPHInterpolator(kernel='cubic_spline')

        # Particles with velocity field
        N = 100
        positions = np.random.randn(N, 3) * 0.2
        velocities = np.random.randn(N, 3) * 0.1
        smoothing_lengths = np.ones(N) * 0.1

        grid, (X, Y, Z) = interp.interpolate_to_grid(
            positions, velocities, smoothing_lengths,
            grid_shape=(16, 16, 16)
        )

        # Grid should have shape (16, 16, 16, 3) for 3D vector
        assert grid.shape == (16, 16, 16, 3)

    def test_interpolate_with_custom_bounds(self):
        """Test interpolation with custom grid bounds."""
        interp = SPHInterpolator(kernel='cubic_spline')

        positions = np.array([[0.5, 0.5, 0.5]])
        field_values = np.array([1.0])
        smoothing_lengths = np.array([0.1])

        # Custom bounds
        bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        grid, (X, Y, Z) = interp.interpolate_to_grid(
            positions, field_values, smoothing_lengths,
            grid_shape=(10, 10, 10),
            bounds=bounds
        )

        assert grid.shape == (10, 10, 10)
        assert X.min() == pytest.approx(0.0)
        assert X.max() == pytest.approx(1.0)

    def test_interpolate_mismatched_lengths_raises_error(self):
        """Test mismatched array lengths raise ValueError."""
        interp = SPHInterpolator(kernel='cubic_spline')

        positions = np.random.randn(10, 3)
        field_values = np.random.randn(5)  # Wrong length
        smoothing_lengths = np.ones(10)

        with pytest.raises(ValueError, match="Field values length"):
            interp.interpolate_to_grid(positions, field_values, smoothing_lengths)


class TestSPHInterpolationSlice:
    """Test SPH interpolation to 2D slices."""

    def test_interpolate_to_xy_slice(self):
        """Test interpolation to XY slice at z=0."""
        interp = SPHInterpolator(kernel='cubic_spline')

        # Particles in XY plane
        N = 100
        positions = np.random.randn(N, 3) * 0.2
        positions[:, 2] *= 0.1  # Concentrate near z=0
        field_values = np.ones(N)
        smoothing_lengths = np.ones(N) * 0.1

        slice_grid, (X, Y) = interp.interpolate_to_slice(
            positions, field_values, smoothing_lengths,
            slice_axis='z', slice_position=0.0,
            grid_shape=(32, 32)
        )

        assert slice_grid.shape == (32, 32)
        assert np.max(slice_grid) > 0.0

    def test_interpolate_to_xz_slice(self):
        """Test interpolation to XZ slice at y=0."""
        interp = SPHInterpolator(kernel='cubic_spline')

        N = 50
        positions = np.random.randn(N, 3) * 0.2
        field_values = np.ones(N)
        smoothing_lengths = np.ones(N) * 0.1

        slice_grid, (X, Z) = interp.interpolate_to_slice(
            positions, field_values, smoothing_lengths,
            slice_axis='y', slice_position=0.0,
            grid_shape=(32, 32)
        )

        assert slice_grid.shape == (32, 32)

    def test_slice_with_no_nearby_particles_warns(self):
        """Test slice with no particles nearby issues warning."""
        interp = SPHInterpolator(kernel='cubic_spline')

        # Particles far from slice
        positions = np.array([[0.0, 0.0, 10.0]])
        field_values = np.array([1.0])
        smoothing_lengths = np.array([0.1])

        # Slice at z=0, far from particles
        with pytest.warns(UserWarning, match="No particles found near slice"):
            slice_grid, (X, Y) = interp.interpolate_to_slice(
                positions, field_values, smoothing_lengths,
                slice_axis='z', slice_position=0.0
            )

        # Should return zero grid
        assert np.all(slice_grid == 0.0)

    def test_slice_invalid_axis_raises_error(self):
        """Test invalid slice axis raises ValueError."""
        interp = SPHInterpolator(kernel='cubic_spline')

        positions = np.random.randn(10, 3)
        field_values = np.ones(10)
        smoothing_lengths = np.ones(10) * 0.1

        with pytest.raises(ValueError, match="slice_axis must be"):
            interp.interpolate_to_slice(
                positions, field_values, smoothing_lengths,
                slice_axis='w',  # Invalid
                slice_position=0.0
            )


class TestTemporalInterpolation:
    """Test temporal interpolation between snapshots."""

    def test_interpolate_linear_positions(self):
        """Test linear interpolation of positions."""
        interp = TemporalInterpolator(method='linear')

        # Two snapshots
        snap_before = {
            'time': 0.0,
            'positions': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            'velocities': np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            'masses': np.array([1.0, 1.0]),
            'density': np.array([1.0, 1.0]),
            'internal_energy': np.array([0.5, 0.5]),
            'smoothing_length': np.array([0.1, 0.1])
        }

        snap_after = {
            'time': 1.0,
            'positions': np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            'velocities': np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            'masses': np.array([1.0, 1.0]),
            'density': np.array([1.0, 1.0]),
            'internal_energy': np.array([0.5, 0.5]),
            'smoothing_length': np.array([0.1, 0.1])
        }

        # Interpolate at t=0.5
        snap_interp = interp.interpolate_snapshots(snap_before, snap_after, t_interp=0.5)

        assert snap_interp['time'] == 0.5
        # Particle 0: (0,0,0) → (1,0,0), at t=0.5 should be (0.5,0,0)
        assert np.allclose(snap_interp['positions'][0], [0.5, 0.0, 0.0])
        # Particle 1: (1,0,0) → (2,0,0), at t=0.5 should be (1.5,0,0)
        assert np.allclose(snap_interp['positions'][1], [1.5, 0.0, 0.0])

    def test_interpolate_scalar_fields(self):
        """Test interpolation of scalar fields (density, internal energy)."""
        interp = TemporalInterpolator(method='linear')

        snap_before = {
            'time': 0.0,
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'velocities': np.array([[0.0, 0.0, 0.0]]),
            'masses': np.array([1.0]),
            'density': np.array([1.0]),
            'internal_energy': np.array([0.0]),
            'smoothing_length': np.array([0.1])
        }

        snap_after = {
            'time': 1.0,
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'velocities': np.array([[0.0, 0.0, 0.0]]),
            'masses': np.array([1.0]),
            'density': np.array([2.0]),
            'internal_energy': np.array([1.0]),
            'smoothing_length': np.array([0.1])
        }

        snap_interp = interp.interpolate_snapshots(snap_before, snap_after, t_interp=0.5)

        # Density: 1.0 → 2.0, at t=0.5 should be 1.5
        assert snap_interp['density'][0] == pytest.approx(1.5)

        # Internal energy: 0.0 → 1.0, at t=0.5 should be 0.5
        assert snap_interp['internal_energy'][0] == pytest.approx(0.5)

    def test_interpolate_at_boundary_times(self):
        """Test interpolation at t=t0 returns first snapshot."""
        interp = TemporalInterpolator(method='linear')

        snap_before = {
            'time': 0.0,
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'velocities': np.array([[1.0, 0.0, 0.0]]),
            'masses': np.array([1.0]),
            'density': np.array([1.0]),
            'internal_energy': np.array([0.5]),
            'smoothing_length': np.array([0.1])
        }

        snap_after = {
            'time': 1.0,
            'positions': np.array([[1.0, 0.0, 0.0]]),
            'velocities': np.array([[1.0, 0.0, 0.0]]),
            'masses': np.array([1.0]),
            'density': np.array([2.0]),
            'internal_energy': np.array([1.0]),
            'smoothing_length': np.array([0.1])
        }

        # Interpolate at t=0 (boundary)
        snap_interp = interp.interpolate_snapshots(snap_before, snap_after, t_interp=0.0)

        assert snap_interp['time'] == 0.0
        assert np.allclose(snap_interp['positions'], snap_before['positions'])
        assert snap_interp['density'][0] == snap_before['density'][0]

    def test_interpolate_out_of_range_raises_error(self):
        """Test interpolation outside time range raises ValueError."""
        interp = TemporalInterpolator(method='linear')

        snap_before = {'time': 0.0, 'positions': np.array([[0.0, 0.0, 0.0]])}
        snap_after = {'time': 1.0, 'positions': np.array([[1.0, 0.0, 0.0]])}

        with pytest.raises(ValueError, match="not in range"):
            interp.interpolate_snapshots(snap_before, snap_after, t_interp=-0.5)

        with pytest.raises(ValueError, match="not in range"):
            interp.interpolate_snapshots(snap_before, snap_after, t_interp=1.5)

    def test_generate_interpolated_sequence(self):
        """Test generating smooth sequence with interpolated frames."""
        interp = TemporalInterpolator(method='linear')

        # 3 original snapshots
        snapshots = [
            {'time': 0.0, 'positions': np.array([[0.0, 0.0, 0.0]])},
            {'time': 1.0, 'positions': np.array([[1.0, 0.0, 0.0]])},
            {'time': 2.0, 'positions': np.array([[2.0, 0.0, 0.0]])}
        ]

        # Add 2 interpolated frames between each pair
        smooth_sequence = interp.generate_interpolated_sequence(
            snapshots, frames_per_interval=2
        )

        # Should have: 3 original + 2×2 interpolated = 7 frames
        assert len(smooth_sequence) == 7

        # Check times are monotonic
        times = [snap['time'] for snap in smooth_sequence]
        assert times == sorted(times)

    def test_generate_sequence_with_single_snapshot(self):
        """Test sequence generation with single snapshot returns unchanged."""
        interp = TemporalInterpolator(method='linear')

        snapshots = [{'time': 0.0, 'positions': np.array([[0.0, 0.0, 0.0]])}]

        smooth_sequence = interp.generate_interpolated_sequence(snapshots, frames_per_interval=10)

        assert len(smooth_sequence) == 1
        assert smooth_sequence[0]['time'] == 0.0


class TestSmoothingFilters:
    """Test smoothing filters."""

    def test_gaussian_filter_3d(self):
        """Test 3D Gaussian filter."""
        # Create test data with spike
        data = np.zeros((16, 16, 16), dtype=np.float32)
        data[8, 8, 8] = 1.0

        smoothed = SmoothingFilters.gaussian_filter_3d(data, sigma=1.0)

        # Smoothed data should spread out spike
        assert smoothed.shape == data.shape
        assert smoothed[8, 8, 8] < 1.0  # Peak reduced
        assert smoothed[7, 8, 8] > 0.0  # Neighbors increased

    def test_median_filter_3d(self):
        """Test 3D median filter."""
        # Create data with outlier
        data = np.ones((10, 10, 10), dtype=np.float32)
        data[5, 5, 5] = 100.0  # Outlier

        filtered = SmoothingFilters.median_filter_3d(data, size=3)

        # Outlier should be suppressed
        assert filtered.shape == data.shape
        assert filtered[5, 5, 5] < 100.0

    def test_bilateral_filter_2d(self):
        """Test 2D bilateral filter."""
        # Create step function (edge)
        data = np.zeros((32, 32), dtype=np.float32)
        data[:, 16:] = 1.0

        filtered = SmoothingFilters.bilateral_filter_2d(
            data, spatial_sigma=1.0, intensity_sigma=0.1
        )

        # Should smooth while preserving edge
        assert filtered.shape == data.shape


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_particles_to_grid_3d_function(self):
        """Test particles_to_grid_3d convenience function."""
        N = 50
        positions = np.random.randn(N, 3) * 0.2
        field_values = np.ones(N)
        smoothing_lengths = np.ones(N) * 0.1

        grid, (X, Y, Z) = particles_to_grid_3d(
            positions, field_values, smoothing_lengths,
            grid_shape=(32, 32, 32),
            kernel='cubic_spline'
        )

        assert grid.shape == (32, 32, 32)
        assert np.max(grid) > 0.0

    def test_particles_to_slice_2d_function(self):
        """Test particles_to_slice_2d convenience function."""
        N = 50
        positions = np.random.randn(N, 3) * 0.2
        field_values = np.ones(N)
        smoothing_lengths = np.ones(N) * 0.1

        slice_grid, (X, Y) = particles_to_slice_2d(
            positions, field_values, smoothing_lengths,
            slice_axis='z', slice_position=0.0,
            grid_shape=(64, 64),
            kernel='cubic_spline'
        )

        assert slice_grid.shape == (64, 64)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_particle_array(self):
        """Test interpolation with empty particle array."""
        interp = SPHInterpolator(kernel='cubic_spline')

        positions = np.zeros((0, 3))
        field_values = np.zeros(0)
        smoothing_lengths = np.zeros(0)

        grid, (X, Y, Z) = interp.interpolate_to_grid(
            positions, field_values, smoothing_lengths,
            grid_shape=(8, 8, 8)
        )

        # Should return zero grid
        assert grid.shape == (8, 8, 8)
        assert np.all(grid == 0.0)

    def test_single_particle_different_kernels(self):
        """Test single particle interpolation with different kernels."""
        positions = np.array([[0.0, 0.0, 0.0]])
        field_values = np.array([1.0])
        smoothing_lengths = np.array([0.1])

        for kernel in ['cubic_spline', 'wendland_c2', 'gaussian']:
            interp = SPHInterpolator(kernel=kernel)
            grid, _ = interp.interpolate_to_grid(
                positions, field_values, smoothing_lengths,
                grid_shape=(16, 16, 16),
                bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
            )

            assert grid.shape == (16, 16, 16)
            assert np.max(grid) > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
