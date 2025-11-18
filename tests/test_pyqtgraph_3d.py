#!/usr/bin/env python3
"""
Tests for PyQtGraph 3D Visualizer (TASK-037)

Test coverage:
- Snapshot loading and caching
- Colormap computation
- UI controls (play/pause, step, slider)
- Frame navigation
- Color and size updates
- Edge cases (single frame, empty snapshots)

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import h5py

# Test if PyQt is available
try:
    from tde_sph.visualization.pyqtgraph_3d import (
        SnapshotLoader,
        ParticleVisualizer3D,
        visualize_snapshots
    )
    PYQT_AVAILABLE = True
except ImportError as e:
    PYQT_AVAILABLE = False
    PYQT_IMPORT_ERROR = str(e)


def create_test_snapshot(filepath: str, time: float, n_particles: int = 100):
    """Helper: create a test HDF5 snapshot."""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('time', data=time)

        # Particle data
        particles_group = f.create_group('particles')
        particles_group.create_dataset(
            'positions',
            data=np.random.randn(n_particles, 3).astype(np.float32)
        )
        particles_group.create_dataset(
            'velocities',
            data=np.random.randn(n_particles, 3).astype(np.float32) * 0.1
        )
        particles_group.create_dataset(
            'masses',
            data=np.ones(n_particles, dtype=np.float32) * 0.01
        )
        particles_group.create_dataset(
            'density',
            data=np.random.rand(n_particles).astype(np.float32) + 0.5
        )
        particles_group.create_dataset(
            'internal_energy',
            data=np.random.rand(n_particles).astype(np.float32) * 0.5
        )
        particles_group.create_dataset(
            'smoothing_length',
            data=np.ones(n_particles, dtype=np.float32) * 0.1
        )

        # Metadata
        f.attrs['bh_mass'] = 1.0
        f.attrs['mode'] = 'Newtonian'


@pytest.mark.skipif(not PYQT_AVAILABLE, reason=f"PyQt not available: {PYQT_IMPORT_ERROR if not PYQT_AVAILABLE else ''}")
class TestSnapshotLoader:
    """Test snapshot loading and caching."""

    def test_loader_initialization(self):
        """Test snapshot loader can be initialized with file list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 test snapshots
            snapshot_files = []
            for i in range(3):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.1)
                snapshot_files.append(str(filepath))

            loader = SnapshotLoader(snapshot_files)
            assert len(loader) == 3

    def test_loader_requires_files(self):
        """Test loader raises error with empty file list."""
        with pytest.raises(ValueError, match="No snapshot files provided"):
            SnapshotLoader([])

    def test_loader_validates_file_existence(self):
        """Test loader validates all files exist."""
        with pytest.raises(FileNotFoundError):
            SnapshotLoader(["nonexistent_file.h5"])

    def test_load_snapshot_by_index(self):
        """Test loading individual snapshots by index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 2 test snapshots
            snapshot_files = []
            for i in range(2):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.5, n_particles=50)
                snapshot_files.append(str(filepath))

            loader = SnapshotLoader(snapshot_files)

            # Load first snapshot
            snapshot0 = loader.load_snapshot(0)
            assert snapshot0['time'] == 0.0
            assert snapshot0['positions'].shape == (50, 3)
            assert 'velocities' in snapshot0
            assert 'masses' in snapshot0
            assert 'density' in snapshot0
            assert 'internal_energy' in snapshot0
            assert 'smoothing_length' in snapshot0
            assert 'metadata' in snapshot0

            # Load second snapshot
            snapshot1 = loader.load_snapshot(1)
            assert snapshot1['time'] == 0.5

    def test_load_snapshot_out_of_range(self):
        """Test loading with invalid index raises IndexError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0)

            loader = SnapshotLoader([str(filepath)])

            with pytest.raises(IndexError):
                loader.load_snapshot(-1)

            with pytest.raises(IndexError):
                loader.load_snapshot(1)

    def test_snapshot_caching(self):
        """Test snapshot caching mechanism."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 snapshots
            snapshot_files = []
            for i in range(3):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.1)
                snapshot_files.append(str(filepath))

            loader = SnapshotLoader(snapshot_files, cache_size=2)

            # Load all 3 snapshots
            snapshot0 = loader.load_snapshot(0)
            snapshot1 = loader.load_snapshot(1)
            snapshot2 = loader.load_snapshot(2)

            # Cache should contain only 2 most recent (1 and 2)
            assert 0 not in loader._cache
            assert 1 in loader._cache
            assert 2 in loader._cache

            # Load snapshot 0 again (should evict 1)
            snapshot0_again = loader.load_snapshot(0)
            assert 0 in loader._cache
            assert 1 not in loader._cache
            assert 2 in loader._cache

    def test_get_time_array(self):
        """Test getting array of all snapshot times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 5 snapshots with known times
            snapshot_files = []
            expected_times = [0.0, 0.1, 0.25, 0.4, 0.6]
            for i, t in enumerate(expected_times):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=t)
                snapshot_files.append(str(filepath))

            loader = SnapshotLoader(snapshot_files)
            times = loader.get_time_array()

            assert len(times) == 5
            assert np.allclose(times, expected_times)


@pytest.mark.skipif(not PYQT_AVAILABLE, reason=f"PyQt not available: {PYQT_IMPORT_ERROR if not PYQT_AVAILABLE else ''}")
class TestParticleVisualizer3D:
    """Test 3D visualizer functionality."""

    def test_visualizer_initialization(self, qtbot):
        """Test visualizer can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test snapshot
            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)],
                color_by='density',
                point_size=2.0
            )
            qtbot.addWidget(visualizer)

            assert visualizer.current_frame == 0
            assert not visualizer.is_playing
            assert visualizer.fps == 10

    def test_visualizer_loads_initial_frame(self, qtbot):
        """Test visualizer loads initial frame on startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.5, n_particles=20)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)],
                color_by='density'
            )
            qtbot.addWidget(visualizer)

            assert visualizer.current_snapshot is not None
            assert visualizer.current_snapshot['time'] == 0.5
            assert visualizer.current_snapshot['positions'].shape[0] == 20

    def test_colormap_computation(self, qtbot):
        """Test colormap computation for different fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0, n_particles=50)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)],
                color_by='density'
            )
            qtbot.addWidget(visualizer)

            # Test density colors
            colors_density = visualizer._compute_colors()
            assert colors_density.shape == (50, 4)
            assert np.all(colors_density >= 0.0)
            assert np.all(colors_density <= 1.0)

            # Test internal energy colors
            visualizer.color_by = 'internal_energy'
            colors_energy = visualizer._compute_colors()
            assert colors_energy.shape == (50, 4)

            # Test velocity magnitude colors
            visualizer.color_by = 'velocity_magnitude'
            colors_velocity = visualizer._compute_colors()
            assert colors_velocity.shape == (50, 4)

    def test_apply_colormap_viridis(self, qtbot):
        """Test viridis colormap application."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0, n_particles=10)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)],
                color_by='density'
            )
            qtbot.addWidget(visualizer)

            # Test viridis colormap
            values = np.linspace(0, 1, 10)
            colors = visualizer._apply_colormap(values, 'viridis')

            assert colors.shape == (10, 4)
            assert np.all(colors[:, 3] == 1.0)  # Alpha = 1

            # Check color progression (low values -> dark, high values -> bright)
            assert np.sum(colors[0, :3]) < np.sum(colors[-1, :3])

    def test_apply_colormap_plasma(self, qtbot):
        """Test plasma colormap application."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0, n_particles=10)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)]
            )
            qtbot.addWidget(visualizer)

            values = np.linspace(0, 1, 10)
            colors = visualizer._apply_colormap(values, 'plasma')

            assert colors.shape == (10, 4)
            assert np.all(colors >= 0.0)
            assert np.all(colors <= 1.0)

    def test_apply_colormap_hot(self, qtbot):
        """Test hot colormap application."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0, n_particles=10)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)]
            )
            qtbot.addWidget(visualizer)

            values = np.linspace(0, 1, 10)
            colors = visualizer._apply_colormap(values, 'hot')

            # Hot colormap: black -> red -> yellow -> white
            # Low values should be red-ish (R > G, B ≈ 0)
            assert colors[0, 0] > colors[0, 1]
            assert colors[0, 2] < 0.1

            # High values should be white-ish (R ≈ G ≈ B ≈ 1)
            assert colors[-1, 0] > 0.9
            assert colors[-1, 1] > 0.9

    def test_step_forward_backward(self, qtbot):
        """Test stepping through frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 snapshots
            snapshot_files = []
            for i in range(3):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.1)
                snapshot_files.append(str(filepath))

            visualizer = ParticleVisualizer3D(
                snapshot_files=snapshot_files
            )
            qtbot.addWidget(visualizer)

            # Initially at frame 0
            assert visualizer.current_frame == 0

            # Step forward
            visualizer._step_forward()
            assert visualizer.current_frame == 1

            visualizer._step_forward()
            assert visualizer.current_frame == 2

            # Can't step beyond last frame
            visualizer._step_forward()
            assert visualizer.current_frame == 2

            # Step backward
            visualizer._step_backward()
            assert visualizer.current_frame == 1

            visualizer._step_backward()
            assert visualizer.current_frame == 0

            # Can't step before first frame
            visualizer._step_backward()
            assert visualizer.current_frame == 0

    def test_play_pause_animation(self, qtbot):
        """Test play/pause animation controls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 2 snapshots
            snapshot_files = []
            for i in range(2):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.1)
                snapshot_files.append(str(filepath))

            visualizer = ParticleVisualizer3D(
                snapshot_files=snapshot_files
            )
            qtbot.addWidget(visualizer)

            # Initially not playing
            assert not visualizer.is_playing
            assert visualizer.play_button.text() == "▶ Play"

            # Start playing
            visualizer._play()
            assert visualizer.is_playing
            assert visualizer.play_button.text() == "⏸ Pause"
            assert visualizer.animation_timer.isActive()

            # Pause
            visualizer._pause()
            assert not visualizer.is_playing
            assert visualizer.play_button.text() == "▶ Play"
            assert not visualizer.animation_timer.isActive()

    def test_animation_loops(self, qtbot):
        """Test animation loops back to start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 2 snapshots
            snapshot_files = []
            for i in range(2):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.1)
                snapshot_files.append(str(filepath))

            visualizer = ParticleVisualizer3D(
                snapshot_files=snapshot_files
            )
            qtbot.addWidget(visualizer)

            # Go to last frame
            visualizer._load_frame(1)
            assert visualizer.current_frame == 1

            # Advance should loop back to 0
            visualizer._advance_frame()
            assert visualizer.current_frame == 0

    def test_reset_view(self, qtbot):
        """Test reset view returns to initial state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 snapshots
            snapshot_files = []
            for i in range(3):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.1)
                snapshot_files.append(str(filepath))

            visualizer = ParticleVisualizer3D(
                snapshot_files=snapshot_files
            )
            qtbot.addWidget(visualizer)

            # Go to frame 2
            visualizer._load_frame(2)
            assert visualizer.current_frame == 2

            # Reset
            visualizer._reset_view()
            assert visualizer.current_frame == 0
            assert not visualizer.is_playing

    def test_fps_change(self, qtbot):
        """Test changing FPS updates timer interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)]
            )
            qtbot.addWidget(visualizer)

            # Initial FPS
            assert visualizer.fps == 10

            # Change FPS
            visualizer._on_fps_changed(30)
            assert visualizer.fps == 30

    def test_color_by_change(self, qtbot):
        """Test changing color-by field updates display."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0, n_particles=20)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)],
                color_by='density'
            )
            qtbot.addWidget(visualizer)

            assert visualizer.color_by == 'density'

            # Change to internal energy
            visualizer._on_color_changed('internal_energy')
            assert visualizer.color_by == 'internal_energy'

            # Scatter plot should be updated (non-None)
            assert visualizer.scatter_plot is not None

    def test_point_size_change(self, qtbot):
        """Test changing point size updates display."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)],
                point_size=2.0
            )
            qtbot.addWidget(visualizer)

            assert visualizer.point_size == 2.0

            # Change size
            visualizer._on_size_changed(5.0)
            assert visualizer.point_size == 5.0

    def test_slider_navigation(self, qtbot):
        """Test timeline slider updates frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 snapshots
            snapshot_files = []
            for i in range(3):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.1)
                snapshot_files.append(str(filepath))

            visualizer = ParticleVisualizer3D(
                snapshot_files=snapshot_files
            )
            qtbot.addWidget(visualizer)

            # Slider should be at 0
            assert visualizer.timeline_slider.value() == 0

            # Change slider value
            visualizer._on_slider_changed(2)
            assert visualizer.current_frame == 2


@pytest.mark.skipif(not PYQT_AVAILABLE, reason=f"PyQt not available: {PYQT_IMPORT_ERROR if not PYQT_AVAILABLE else ''}")
class TestVisualizationAPI:
    """Test high-level visualization API."""

    def test_visualize_snapshots_function(self, qtbot):
        """Test visualize_snapshots() convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 2 snapshots
            snapshot_files = []
            for i in range(2):
                filepath = tmpdir / f"snapshot_{i:04d}.h5"
                create_test_snapshot(str(filepath), time=i * 0.1)
                snapshot_files.append(str(filepath))

            app, window = visualize_snapshots(
                snapshot_files=snapshot_files,
                color_by='density',
                point_size=3.0,
                background='black',
                title="Test Visualizer"
            )
            qtbot.addWidget(window)

            assert window is not None
            assert window.windowTitle() == "Test Visualizer"
            assert window.color_by == 'density'
            assert window.point_size == 3.0


@pytest.mark.skipif(not PYQT_AVAILABLE, reason=f"PyQt not available: {PYQT_IMPORT_ERROR if not PYQT_AVAILABLE else ''}")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_snapshot(self, qtbot):
        """Test visualizer works with single snapshot (no animation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"
            create_test_snapshot(str(filepath), time=0.0)

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)]
            )
            qtbot.addWidget(visualizer)

            assert len(visualizer.snapshot_loader) == 1
            assert visualizer.current_frame == 0

            # Step forward should stay at 0
            visualizer._step_forward()
            assert visualizer.current_frame == 0

    def test_constant_field_colormap(self, qtbot):
        """Test colormap handles constant field values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"

            # Create snapshot with constant density
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('time', data=0.0)
                particles_group = f.create_group('particles')
                n = 10
                particles_group.create_dataset('positions', data=np.zeros((n, 3), dtype=np.float32))
                particles_group.create_dataset('velocities', data=np.zeros((n, 3), dtype=np.float32))
                particles_group.create_dataset('masses', data=np.ones(n, dtype=np.float32))
                particles_group.create_dataset('density', data=np.ones(n, dtype=np.float32) * 1.0)  # Constant
                particles_group.create_dataset('internal_energy', data=np.ones(n, dtype=np.float32))
                particles_group.create_dataset('smoothing_length', data=np.ones(n, dtype=np.float32))

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)],
                color_by='density'
            )
            qtbot.addWidget(visualizer)

            colors = visualizer._compute_colors()

            # Should handle constant field gracefully (all mid-range color)
            assert colors.shape == (10, 4)
            assert np.all(colors >= 0.0)
            assert np.all(colors <= 1.0)

    def test_zero_particles(self, qtbot):
        """Test visualizer handles empty snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            filepath = tmpdir / "snapshot_0000.h5"

            # Create snapshot with 0 particles
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('time', data=0.0)
                particles_group = f.create_group('particles')
                particles_group.create_dataset('positions', data=np.zeros((0, 3), dtype=np.float32))
                particles_group.create_dataset('velocities', data=np.zeros((0, 3), dtype=np.float32))
                particles_group.create_dataset('masses', data=np.zeros(0, dtype=np.float32))
                particles_group.create_dataset('density', data=np.zeros(0, dtype=np.float32))
                particles_group.create_dataset('internal_energy', data=np.zeros(0, dtype=np.float32))
                particles_group.create_dataset('smoothing_length', data=np.zeros(0, dtype=np.float32))

            visualizer = ParticleVisualizer3D(
                snapshot_files=[str(filepath)]
            )
            qtbot.addWidget(visualizer)

            # Should not crash, just show empty view
            assert visualizer.current_snapshot is not None
            assert len(visualizer.current_snapshot['positions']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
