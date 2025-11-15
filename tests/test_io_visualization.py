"""
Tests for I/O and visualization modules.

Tests the HDF5Writer and Plotly3DVisualizer implementations.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import h5py

from tde_sph.io.hdf5 import HDF5Writer, write_snapshot, read_snapshot
from tde_sph.visualization.plotly_3d import Plotly3DVisualizer, quick_plot


class TestHDF5Writer:
    """Test suite for HDF5Writer."""

    @pytest.fixture
    def sample_particles(self):
        """Create sample particle data for testing."""
        n = 1000
        return {
            'positions': np.random.randn(n, 3).astype(np.float32),
            'velocities': np.random.randn(n, 3).astype(np.float32),
            'masses': np.ones(n, dtype=np.float32) * 1e-4,
            'density': np.random.rand(n).astype(np.float32) * 1e3,
            'internal_energy': np.random.rand(n).astype(np.float32) * 1e5,
            'smoothing_length': np.ones(n, dtype=np.float32) * 0.1,
        }

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            'bh_mass': 1e6,
            'bh_spin': 0.5,
            'stellar_mass': 1.0,
            'metric_type': 'Kerr',
            'simulation_name': 'test_run',
        }

    def test_write_and_read_snapshot(self, sample_particles, sample_metadata, tmp_path):
        """Test writing and reading a snapshot."""
        writer = HDF5Writer()
        filename = tmp_path / "test_snapshot.h5"

        # Write snapshot
        writer.write_snapshot(
            str(filename),
            sample_particles,
            time=1.5,
            metadata=sample_metadata
        )

        # Verify file exists
        assert filename.exists()

        # Read snapshot
        data = writer.read_snapshot(str(filename))

        # Check basic fields
        assert 'particles' in data
        assert 'time' in data
        assert 'n_particles' in data
        assert 'metadata' in data

        # Check time
        assert np.isclose(data['time'], 1.5)

        # Check particle count
        assert data['n_particles'] == len(sample_particles['masses'])

        # Check particle arrays
        for key in sample_particles.keys():
            assert key in data['particles']
            np.testing.assert_array_almost_equal(
                data['particles'][key],
                sample_particles[key],
                decimal=5
            )

        # Check metadata
        for key, value in sample_metadata.items():
            if isinstance(value, (int, float, str)):
                assert data['metadata'][key] == value

    def test_write_with_compression(self, sample_particles, tmp_path):
        """Test that compression is applied."""
        writer = HDF5Writer(compression='gzip', compression_level=9)
        filename = tmp_path / "compressed.h5"

        writer.write_snapshot(str(filename), sample_particles, time=0.0)

        # Check that compression is set
        with h5py.File(filename, 'r') as f:
            pos_dataset = f['particles']['positions']
            assert pos_dataset.compression == 'gzip'
            assert pos_dataset.compression_opts == 9

    def test_convenience_functions(self, sample_particles, tmp_path):
        """Test write_snapshot and read_snapshot convenience functions."""
        filename = tmp_path / "convenient.h5"

        # Write using convenience function
        write_snapshot(str(filename), sample_particles, time=2.0)

        # Read using convenience function
        data = read_snapshot(str(filename))

        assert np.isclose(data['time'], 2.0)
        assert data['n_particles'] == len(sample_particles['masses'])

    def test_missing_required_fields(self, tmp_path):
        """Test that missing required fields raise an error."""
        writer = HDF5Writer()
        filename = tmp_path / "incomplete.h5"

        # Missing 'velocities'
        incomplete_particles = {
            'positions': np.random.randn(100, 3).astype(np.float32),
            'masses': np.ones(100, dtype=np.float32),
        }

        with pytest.raises(ValueError, match="Missing required particle fields"):
            writer.write_snapshot(str(filename), incomplete_particles, time=0.0)

    def test_get_snapshot_info(self, sample_particles, tmp_path):
        """Test getting snapshot info without loading full data."""
        writer = HDF5Writer()
        filename = tmp_path / "info_test.h5"

        writer.write_snapshot(str(filename), sample_particles, time=3.5)

        info = writer.get_snapshot_info(str(filename))

        assert info['time'] == 3.5
        assert info['n_particles'] == len(sample_particles['masses'])
        assert 'particle_fields' in info
        assert 'positions' in info['particle_fields']

    def test_list_snapshots(self, sample_particles, tmp_path):
        """Test listing snapshot files in a directory."""
        writer = HDF5Writer()

        # Create multiple snapshots
        for i in range(5):
            filename = tmp_path / f"snapshot_{i:04d}.h5"
            writer.write_snapshot(str(filename), sample_particles, time=float(i))

        # List snapshots
        snapshots = writer.list_snapshots(str(tmp_path))

        assert len(snapshots) == 5
        assert all(f.suffix == '.h5' for f in snapshots)

    def test_read_nonexistent_file(self):
        """Test that reading nonexistent file raises error."""
        writer = HDF5Writer()

        with pytest.raises(FileNotFoundError):
            writer.read_snapshot("nonexistent.h5")


class TestPlotly3DVisualizer:
    """Test suite for Plotly3DVisualizer."""

    @pytest.fixture
    def sample_positions(self):
        """Create sample particle positions."""
        n = 500
        return np.random.randn(n, 3).astype(np.float32)

    @pytest.fixture
    def sample_quantities(self):
        """Create sample quantities for visualization."""
        n = 500
        return {
            'density': np.random.rand(n).astype(np.float32) * 1e3,
            'temperature': np.random.rand(n).astype(np.float32) * 1e4,
            'velocity_magnitude': np.random.rand(n).astype(np.float32) * 1e2,
        }

    def test_visualizer_init(self):
        """Test visualizer initialization."""
        viz = Plotly3DVisualizer()
        assert viz.max_particles_plot == 100_000
        assert viz.default_marker_size == 2.0
        assert viz.colorscale == 'Viridis'

        # Custom initialization
        viz_custom = Plotly3DVisualizer(
            max_particles_plot=50_000,
            default_marker_size=3.0,
            colorscale='Plasma'
        )
        assert viz_custom.max_particles_plot == 50_000
        assert viz_custom.colorscale == 'Plasma'

    def test_plot_particles_basic(self, sample_positions):
        """Test basic particle plotting."""
        viz = Plotly3DVisualizer()
        fig = viz.plot_particles(sample_positions)

        assert fig is not None
        assert len(fig.data) == 1  # One scatter trace
        assert fig.data[0].type == 'scatter3d'

    def test_plot_particles_with_color(self, sample_positions, sample_quantities):
        """Test plotting with color mapping."""
        viz = Plotly3DVisualizer()

        # Plot with density coloring
        fig = viz.plot_particles(
            sample_positions,
            quantities=sample_quantities,
            color_by='density',
            log_scale=True
        )

        assert fig is not None
        assert len(fig.data) == 1
        scatter = fig.data[0]
        assert scatter.marker.color is not None
        assert len(scatter.marker.color) == len(sample_positions)

    def test_plot_particles_linear_scale(self, sample_positions, sample_quantities):
        """Test plotting with linear color scale."""
        viz = Plotly3DVisualizer()

        fig = viz.plot_particles(
            sample_positions,
            quantities=sample_quantities,
            color_by='temperature',
            log_scale=False
        )

        assert fig is not None

    def test_plot_with_custom_marker_size(self, sample_positions):
        """Test plotting with custom marker sizes."""
        viz = Plotly3DVisualizer()

        # Uniform size
        fig = viz.plot_particles(sample_positions, marker_size=5.0)
        assert fig.data[0].marker.size == 5.0

        # Variable size
        sizes = np.random.rand(len(sample_positions)).astype(np.float32) * 10
        fig = viz.plot_particles(sample_positions, marker_size=sizes)
        np.testing.assert_array_equal(fig.data[0].marker.size, sizes)

    def test_downsampling(self):
        """Test automatic downsampling for large datasets."""
        viz = Plotly3DVisualizer(max_particles_plot=1000)

        # Create large dataset
        large_positions = np.random.randn(10000, 3).astype(np.float32)

        with pytest.warns(UserWarning, match="Downsampled from"):
            fig = viz.plot_particles(large_positions, downsample=True)

        # Should have fewer particles in the plot
        assert len(fig.data[0].x) == 1000

    def test_no_downsampling_when_disabled(self):
        """Test that downsampling can be disabled."""
        viz = Plotly3DVisualizer(max_particles_plot=1000)

        # Create dataset larger than limit
        positions = np.random.randn(2000, 3).astype(np.float32)

        fig = viz.plot_particles(positions, downsample=False)

        # Should have all particles
        assert len(fig.data[0].x) == 2000

    def test_animate(self, sample_positions, sample_quantities):
        """Test animation creation."""
        viz = Plotly3DVisualizer()

        # Create snapshots
        snapshots = [
            {
                'positions': sample_positions + i * 0.1,
                'quantities': sample_quantities,
                'time': float(i)
            }
            for i in range(3)
        ]

        fig = viz.animate(snapshots)

        assert fig is not None
        assert len(fig.frames) == 3
        assert fig.layout.updatemenus is not None  # Has play/pause buttons
        assert fig.layout.sliders is not None  # Has time slider

    def test_animate_empty_list(self):
        """Test that animate raises error for empty snapshot list."""
        viz = Plotly3DVisualizer()

        with pytest.raises(ValueError, match="snapshots list is empty"):
            viz.animate([])

    def test_quick_plot(self, sample_positions):
        """Test quick_plot convenience function."""
        density = np.random.rand(len(sample_positions)).astype(np.float32)

        fig = quick_plot(sample_positions, color_by=density)

        assert fig is not None
        assert len(fig.data) == 1

    def test_quick_plot_no_color(self, sample_positions):
        """Test quick_plot without color data."""
        fig = quick_plot(sample_positions)

        assert fig is not None
        assert len(fig.data) == 1


class TestIntegration:
    """Integration tests combining I/O and visualization."""

    def test_save_and_visualize(self, tmp_path):
        """Test saving data and visualizing it."""
        # Create particle data
        n = 1000
        particles = {
            'positions': np.random.randn(n, 3).astype(np.float32),
            'velocities': np.random.randn(n, 3).astype(np.float32),
            'masses': np.ones(n, dtype=np.float32) * 1e-4,
            'density': np.random.rand(n).astype(np.float32) * 1e3,
            'internal_energy': np.random.rand(n).astype(np.float32) * 1e5,
            'smoothing_length': np.ones(n, dtype=np.float32) * 0.1,
            'temperature': np.random.rand(n).astype(np.float32) * 1e4,
        }

        # Save snapshot
        filename = tmp_path / "integration_test.h5"
        write_snapshot(str(filename), particles, time=1.0)

        # Read snapshot
        data = read_snapshot(str(filename))

        # Visualize
        viz = Plotly3DVisualizer()
        fig = viz.plot_particles(
            data['particles']['positions'],
            quantities={
                'density': data['particles']['density'],
                'temperature': data['particles']['temperature'],
            },
            color_by='density'
        )

        assert fig is not None

    def test_animate_from_snapshots(self, tmp_path):
        """Test creating animation from saved snapshots."""
        writer = HDF5Writer()

        # Create and save multiple snapshots
        snapshots = []
        for i in range(3):
            n = 500
            particles = {
                'positions': np.random.randn(n, 3).astype(np.float32) + i * 0.5,
                'velocities': np.random.randn(n, 3).astype(np.float32),
                'masses': np.ones(n, dtype=np.float32) * 1e-4,
                'density': np.random.rand(n).astype(np.float32) * 1e3,
                'internal_energy': np.random.rand(n).astype(np.float32) * 1e5,
                'smoothing_length': np.ones(n, dtype=np.float32) * 0.1,
            }

            filename = tmp_path / f"snap_{i:04d}.h5"
            writer.write_snapshot(str(filename), particles, time=float(i))

            # Read back for animation
            data = read_snapshot(str(filename))
            snapshots.append({
                'positions': data['particles']['positions'],
                'quantities': {'density': data['particles']['density']},
                'time': data['time']
            })

        # Create animation
        viz = Plotly3DVisualizer()
        fig = viz.animate(snapshots)

        assert fig is not None
        assert len(fig.frames) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
