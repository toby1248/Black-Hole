"""
Tests for Blender/ParaView export tool (TASK-038).

Tests PLY, VTK, and OBJ export formats, color mapping, and batch processing.
"""

import pytest
import numpy as np
import h5py
import tempfile
from pathlib import Path
import sys

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.export_to_blender import SnapshotExporter


class MockHDF5Snapshot:
    """Helper class to create mock HDF5 snapshots for testing."""

    @staticmethod
    def create(
        filename: str,
        n_particles: int = 100,
        include_smoothing: bool = True,
        include_density: bool = True,
        include_velocities: bool = True
    ):
        """
        Create a mock HDF5 snapshot file.

        Parameters
        ----------
        filename : str
            Output HDF5 file path.
        n_particles : int
            Number of particles.
        include_smoothing : bool
            Include smoothing_length field.
        include_density : bool
            Include density field.
        include_velocities : bool
            Include velocities field.
        """
        with h5py.File(filename, 'w') as f:
            # Create groups
            particles_grp = f.create_group('particles')
            metadata_grp = f.create_group('metadata')

            # Particle data
            positions = np.random.randn(n_particles, 3).astype(np.float32)
            masses = np.ones(n_particles, dtype=np.float32) * 0.01

            particles_grp.create_dataset('positions', data=positions)
            particles_grp.create_dataset('masses', data=masses)

            if include_density:
                density = np.random.uniform(0.1, 10.0, n_particles).astype(np.float32)
                particles_grp.create_dataset('density', data=density)

            if include_velocities:
                velocities = np.random.randn(n_particles, 3).astype(np.float32) * 0.1
                particles_grp.create_dataset('velocities', data=velocities)

            if include_smoothing:
                smoothing_length = np.random.uniform(0.01, 0.1, n_particles).astype(np.float32)
                particles_grp.create_dataset('smoothing_length', data=smoothing_length)

            internal_energy = np.random.uniform(0.5, 2.0, n_particles).astype(np.float32)
            particles_grp.create_dataset('internal_energy', data=internal_energy)

            # Metadata
            metadata_grp.attrs['time'] = 0.0
            metadata_grp.attrs['bh_mass'] = 1.0
            metadata_grp.attrs['n_particles'] = n_particles


# ============================================================================
# Test: Basic Functionality
# ============================================================================

class TestBasicFunctionality:
    """Test basic read and export operations."""

    def test_snapshot_exporter_initialization(self):
        """Test SnapshotExporter initialization."""
        exporter = SnapshotExporter(normalize_colors=True, verbose=False)
        assert exporter.normalize_colors is True
        assert exporter.verbose is False

    def test_read_snapshot(self):
        """Test reading HDF5 snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            MockHDF5Snapshot.create(str(snapshot_file), n_particles=50)

            exporter = SnapshotExporter(verbose=False)
            data = exporter.read_snapshot(str(snapshot_file))

            assert 'particles' in data
            assert 'metadata' in data
            assert 'positions' in data['particles']
            assert len(data['particles']['positions']) == 50

    def test_read_snapshot_missing_file(self):
        """Test reading non-existent snapshot raises error."""
        exporter = SnapshotExporter(verbose=False)
        with pytest.raises(FileNotFoundError):
            exporter.read_snapshot("nonexistent_file.h5")


# ============================================================================
# Test: Color Computation
# ============================================================================

class TestColorComputation:
    """Test color mapping from scalar fields."""

    def test_compute_colors_viridis(self):
        """Test viridis color map."""
        exporter = SnapshotExporter(verbose=False)
        field = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        colors = exporter.compute_colors(field, cmap="viridis")

        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8
        assert np.all(colors >= 0) and np.all(colors <= 255)

    def test_compute_colors_plasma(self):
        """Test plasma color map."""
        exporter = SnapshotExporter(verbose=False)
        field = np.linspace(0, 1, 10, dtype=np.float32)
        colors = exporter.compute_colors(field, cmap="plasma")

        assert colors.shape == (10, 3)
        assert colors.dtype == np.uint8

    def test_compute_colors_hot(self):
        """Test hot color map."""
        exporter = SnapshotExporter(verbose=False)
        field = np.linspace(0, 1, 10, dtype=np.float32)
        colors = exporter.compute_colors(field, cmap="hot")

        assert colors.shape == (10, 3)

    def test_compute_colors_cool(self):
        """Test cool color map."""
        exporter = SnapshotExporter(verbose=False)
        field = np.linspace(0, 1, 10, dtype=np.float32)
        colors = exporter.compute_colors(field, cmap="cool")

        assert colors.shape == (10, 3)

    def test_compute_colors_custom_range(self):
        """Test color mapping with custom vmin/vmax."""
        exporter = SnapshotExporter(verbose=False)
        field = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        colors = exporter.compute_colors(field, vmin=0.0, vmax=10.0)

        # Colors should vary across the range
        assert colors.shape == (3, 3)
        # Colors should be different (at least one channel should differ)
        assert not np.array_equal(colors[0], colors[2])
        # Middle value should be between extremes (sum of RGB channels)
        assert np.sum(colors[0]) != np.sum(colors[2])

    def test_compute_colors_constant_field(self):
        """Test color mapping with constant field (vmin == vmax)."""
        exporter = SnapshotExporter(verbose=False)
        field = np.ones(10, dtype=np.float32) * 5.0
        colors = exporter.compute_colors(field)

        # Should not crash, return some valid color
        assert colors.shape == (10, 3)
        assert colors.dtype == np.uint8


# ============================================================================
# Test: PLY Export
# ============================================================================

class TestPLYExport:
    """Test PLY point cloud export."""

    def test_export_ply_basic(self):
        """Test basic PLY export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            ply_file = Path(tmpdir) / "output.ply"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_ply(str(snapshot_file), str(ply_file), color_by="density")

            assert ply_file.exists()

            # Check file format
            with open(ply_file, 'r') as f:
                lines = f.readlines()
                assert lines[0].strip() == "ply"
                assert "format ascii 1.0" in lines[1]
                assert any("element vertex 10" in line for line in lines)
                assert any("property float x" in line for line in lines)
                assert any("property uchar red" in line for line in lines)
                assert any("end_header" in line for line in lines)

            # Check data lines (header + 10 particles)
            with open(ply_file, 'r') as f:
                content = f.read()
                # Should have 10 data lines (vertices)
                data_lines = [line for line in content.split('\n') if line and not line.startswith('#') and 'ply' not in line and 'property' not in line and 'element' not in line and 'format' not in line and 'end_header' not in line and 'comment' not in line]
                assert len(data_lines) == 10

    def test_export_ply_with_smoothing_length(self):
        """Test PLY export includes smoothing length as radius."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            ply_file = Path(tmpdir) / "output.ply"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=5, include_smoothing=True)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_ply(str(snapshot_file), str(ply_file))

            # Check for radius property
            with open(ply_file, 'r') as f:
                content = f.read()
                assert "property float radius" in content

    def test_export_ply_velocity_magnitude(self):
        """Test PLY export with velocity magnitude coloring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            ply_file = Path(tmpdir) / "output.ply"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10, include_velocities=True)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_ply(str(snapshot_file), str(ply_file), color_by="velocity_magnitude")

            assert ply_file.exists()

    def test_export_ply_missing_positions(self):
        """Test PLY export fails gracefully without positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"

            # Create snapshot without positions
            with h5py.File(snapshot_file, 'w') as f:
                particles_grp = f.create_group('particles')
                particles_grp.create_dataset('masses', data=np.ones(10, dtype=np.float32))

            exporter = SnapshotExporter(verbose=False)

            with pytest.raises(KeyError, match="positions"):
                exporter.export_ply(str(snapshot_file), "output.ply")

    def test_export_ply_missing_color_field(self):
        """Test PLY export falls back to density if color field missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            ply_file = Path(tmpdir) / "output.ply"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10, include_density=True)

            exporter = SnapshotExporter(verbose=False)
            # Request non-existent field, should fall back
            exporter.export_ply(str(snapshot_file), str(ply_file), color_by="temperature")

            assert ply_file.exists()


# ============================================================================
# Test: VTK Export
# ============================================================================

class TestVTKExport:
    """Test VTK unstructured grid export."""

    def test_export_vtk_basic(self):
        """Test basic VTK export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            vtk_file = Path(tmpdir) / "output.vtk"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_vtk(str(snapshot_file), str(vtk_file))

            assert vtk_file.exists()

            # Check file format
            with open(vtk_file, 'r') as f:
                lines = f.readlines()
                assert "# vtk DataFile" in lines[0]
                assert "ASCII" in lines[2]
                assert "DATASET UNSTRUCTURED_GRID" in lines[3]
                assert any("POINTS 10" in line for line in lines)
                assert any("CELLS 10" in line for line in lines)
                assert any("CELL_TYPES 10" in line for line in lines)
                assert any("POINT_DATA 10" in line for line in lines)

    def test_export_vtk_scalars(self):
        """Test VTK export with scalar fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            vtk_file = Path(tmpdir) / "output.vtk"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_vtk(str(snapshot_file), str(vtk_file), scalars=["density", "internal_energy"])

            # Check for scalar fields
            with open(vtk_file, 'r') as f:
                content = f.read()
                assert "SCALARS density" in content
                assert "SCALARS internal_energy" in content

    def test_export_vtk_vectors(self):
        """Test VTK export with vector fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            vtk_file = Path(tmpdir) / "output.vtk"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10, include_velocities=True)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_vtk(str(snapshot_file), str(vtk_file), vectors=["velocities"])

            # Check for vector fields
            with open(vtk_file, 'r') as f:
                content = f.read()
                assert "VECTORS velocities" in content

    def test_export_vtk_missing_field_warning(self):
        """Test VTK export skips missing fields gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            vtk_file = Path(tmpdir) / "output.vtk"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)

            exporter = SnapshotExporter(verbose=False)
            # Request non-existent field, should skip silently
            exporter.export_vtk(str(snapshot_file), str(vtk_file), scalars=["nonexistent_field"])

            assert vtk_file.exists()


# ============================================================================
# Test: OBJ Export
# ============================================================================

class TestOBJExport:
    """Test OBJ point cloud export."""

    def test_export_obj_basic(self):
        """Test basic OBJ export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            obj_file = Path(tmpdir) / "output.obj"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_obj(str(snapshot_file), str(obj_file))

            assert obj_file.exists()

            # Check file format
            with open(obj_file, 'r') as f:
                lines = f.readlines()
                assert lines[0].startswith("#")  # Comment line
                vertex_lines = [line for line in lines if line.startswith("v ")]
                assert len(vertex_lines) == 10

    def test_export_obj_vertex_format(self):
        """Test OBJ vertex format is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            obj_file = Path(tmpdir) / "output.obj"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=5)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_obj(str(snapshot_file), str(obj_file))

            with open(obj_file, 'r') as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        assert len(parts) == 4  # "v" + x + y + z
                        assert parts[0] == "v"
                        # Check that x, y, z are floats
                        float(parts[1])
                        float(parts[2])
                        float(parts[3])


# ============================================================================
# Test: Batch Export
# ============================================================================

class TestBatchExport:
    """Test batch export of multiple snapshots."""

    def test_batch_export_ply(self):
        """Test batch export to PLY."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 snapshots
            snapshots = []
            for i in range(3):
                snapshot_file = Path(tmpdir) / f"snapshot_{i:04d}.h5"
                MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)
                snapshots.append(str(snapshot_file))

            output_dir = Path(tmpdir) / "ply_output"

            exporter = SnapshotExporter(verbose=False)
            exporter.batch_export(snapshots, str(output_dir), format="ply", color_by="density")

            # Check all output files exist
            assert output_dir.exists()
            assert (output_dir / "snapshot_0000.ply").exists()
            assert (output_dir / "snapshot_0001.ply").exists()
            assert (output_dir / "snapshot_0002.ply").exists()

    def test_batch_export_vtk(self):
        """Test batch export to VTK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2 snapshots
            snapshots = []
            for i in range(2):
                snapshot_file = Path(tmpdir) / f"snapshot_{i:04d}.h5"
                MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)
                snapshots.append(str(snapshot_file))

            output_dir = Path(tmpdir) / "vtk_output"

            exporter = SnapshotExporter(verbose=False)
            exporter.batch_export(snapshots, str(output_dir), format="vtk")

            # Check output
            assert (output_dir / "snapshot_0000.vtk").exists()
            assert (output_dir / "snapshot_0001.vtk").exists()

    def test_batch_export_obj(self):
        """Test batch export to OBJ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2 snapshots
            snapshots = []
            for i in range(2):
                snapshot_file = Path(tmpdir) / f"snapshot_{i:04d}.h5"
                MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)
                snapshots.append(str(snapshot_file))

            output_dir = Path(tmpdir) / "obj_output"

            exporter = SnapshotExporter(verbose=False)
            exporter.batch_export(snapshots, str(output_dir), format="obj")

            # Check output
            assert (output_dir / "snapshot_0000.obj").exists()
            assert (output_dir / "snapshot_0001.obj").exists()

    def test_batch_export_invalid_format(self):
        """Test batch export raises error for invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "snapshot_0000.h5"
            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10)

            exporter = SnapshotExporter(verbose=False)

            with pytest.raises(ValueError, match="Unknown format"):
                exporter.batch_export([str(snapshot_file)], tmpdir, format="invalid_format")


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_export_very_small_snapshot(self):
        """Test export with very few particles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            ply_file = Path(tmpdir) / "output.ply"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=1)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_ply(str(snapshot_file), str(ply_file))

            assert ply_file.exists()

    def test_export_large_snapshot(self):
        """Test export with many particles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            vtk_file = Path(tmpdir) / "output.vtk"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=1000)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_vtk(str(snapshot_file), str(vtk_file))

            assert vtk_file.exists()

    def test_export_with_custom_point_size(self):
        """Test PLY export with custom point size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"
            ply_file = Path(tmpdir) / "output.ply"

            MockHDF5Snapshot.create(str(snapshot_file), n_particles=10, include_smoothing=False)

            exporter = SnapshotExporter(verbose=False)
            exporter.export_ply(str(snapshot_file), str(ply_file), point_size=0.5)

            # Check that radius property exists
            with open(ply_file, 'r') as f:
                content = f.read()
                assert "property float radius" in content
                # Check that at least one line has 0.5 as radius
                assert "0.500000" in content
