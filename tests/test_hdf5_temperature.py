"""
Tests for HDF5 I/O with temperature attribute (TASK1).

Tests cover:
- Temperature is written to HDF5 snapshots
- Temperature is read back from HDF5 snapshots
- Round-trip consistency (write -> read -> verify)

References
----------
- TASK1: Fix HDF5 I/O to include temperature in snapshots

Test Coverage
-------------
1. Temperature included in snapshot writes
2. Temperature loaded correctly from snapshots
3. Round-trip test: write and read produce identical data
4. Backward compatibility: old snapshots without temperature load successfully
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


class TestHDF5TemperatureIO:
    """Test HDF5 I/O includes temperature."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_particle_data(self):
        """Create sample particle data with temperature."""
        n_particles = 100
        return {
            'positions': np.random.randn(n_particles, 3).astype(np.float32),
            'velocities': np.random.randn(n_particles, 3).astype(np.float32) * 0.1,
            'masses': np.ones(n_particles, dtype=np.float32) / n_particles,
            'density': np.ones(n_particles, dtype=np.float32) * 0.1,
            'internal_energy': np.ones(n_particles, dtype=np.float32),
            'smoothing_length': np.ones(n_particles, dtype=np.float32) * 0.1,
            'pressure': np.ones(n_particles, dtype=np.float32) * 0.01,
            'sound_speed': np.ones(n_particles, dtype=np.float32) * 1.0,
            'temperature': np.ones(n_particles, dtype=np.float32) * 5000.0,
        }

    def test_temperature_written_to_hdf5(self, temp_dir, sample_particle_data):
        """Test that temperature is written to HDF5 snapshots."""
        try:
            from tde_sph.io.hdf5 import write_snapshot
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        filename = temp_dir / "test_snapshot.h5"

        # Write snapshot
        write_snapshot(str(filename), sample_particle_data, time=0.0)

        # Read back and verify temperature exists
        with h5py.File(filename, 'r') as f:
            assert 'particles' in f, "particles group should exist"
            assert 'temperature' in f['particles'], "temperature dataset should exist"

            temperature = f['particles/temperature'][:]
            np.testing.assert_array_almost_equal(
                temperature,
                sample_particle_data['temperature']
            )

    def test_temperature_round_trip(self, temp_dir, sample_particle_data):
        """Test temperature survives write -> read round trip."""
        try:
            from tde_sph.io.hdf5 import write_snapshot, read_snapshot
        except ImportError:
            pytest.skip("h5py not available")

        filename = temp_dir / "test_roundtrip.h5"

        # Write snapshot
        write_snapshot(str(filename), sample_particle_data, time=0.0)

        # Read snapshot back
        loaded_data = read_snapshot(str(filename))

        # Verify temperature matches
        assert 'temperature' in loaded_data, "temperature should be in loaded data"
        np.testing.assert_array_almost_equal(
            loaded_data['temperature'],
            sample_particle_data['temperature'],
            err_msg="Temperature should match after round-trip"
        )

    def test_all_particle_attributes_round_trip(self, temp_dir, sample_particle_data):
        """Test that all particle attributes survive round-trip."""
        try:
            from tde_sph.io.hdf5 import write_snapshot, read_snapshot
        except ImportError:
            pytest.skip("h5py not available")

        filename = temp_dir / "test_all_attributes.h5"

        # Write snapshot
        write_snapshot(str(filename), sample_particle_data, time=1.5)

        # Read snapshot back
        loaded_data = read_snapshot(str(filename))

        # Verify all attributes
        for key in sample_particle_data.keys():
            assert key in loaded_data, f"{key} should be in loaded data"
            np.testing.assert_array_almost_equal(
                loaded_data[key],
                sample_particle_data[key],
                err_msg=f"{key} should match after round-trip"
            )

        # Verify time
        assert loaded_data['time'] == 1.5, "time should match"

    def test_temperature_dtype_float32(self, temp_dir, sample_particle_data):
        """Test that temperature maintains float32 dtype in HDF5."""
        try:
            from tde_sph.io.hdf5 import write_snapshot, read_snapshot
        except ImportError:
            pytest.skip("h5py not available")

        filename = temp_dir / "test_dtype.h5"

        # Write snapshot
        write_snapshot(str(filename), sample_particle_data, time=0.0)

        # Read back
        loaded_data = read_snapshot(str(filename))

        # Temperature should be float32
        assert loaded_data['temperature'].dtype == np.float32, \
            "Temperature should be float32 for GPU compatibility"

    def test_backward_compatibility_optional_temperature(self, temp_dir):
        """Test that snapshots without temperature can still be loaded."""
        try:
            from tde_sph.io.hdf5 import write_snapshot, read_snapshot
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        filename = temp_dir / "test_no_temperature.h5"

        # Create particle data WITHOUT temperature (old format)
        n_particles = 50
        old_format_data = {
            'positions': np.random.randn(n_particles, 3).astype(np.float32),
            'velocities': np.random.randn(n_particles, 3).astype(np.float32) * 0.1,
            'masses': np.ones(n_particles, dtype=np.float32) / n_particles,
            'density': np.ones(n_particles, dtype=np.float32) * 0.1,
            'internal_energy': np.ones(n_particles, dtype=np.float32),
            'smoothing_length': np.ones(n_particles, dtype=np.float32) * 0.1,
        }

        # Write old format snapshot
        write_snapshot(str(filename), old_format_data, time=0.0)

        # Should be able to read it without errors
        loaded_data = read_snapshot(str(filename))

        # Required fields should exist
        assert 'positions' in loaded_data
        assert 'velocities' in loaded_data
        assert 'masses' in loaded_data

        # Temperature may or may not exist (backward compatibility)
        # Code should handle both cases gracefully


class TestSimulationSnapshotWithTemperature:
    """Test that Simulation class writes temperature in snapshots."""

    def test_simulation_write_snapshot_includes_temperature(self):
        """Test that Simulation.write_snapshot() includes temperature."""
        # This is a documentation/integration test
        # The actual test would require a full simulation setup
        # We verify the code structure instead

        import inspect
        try:
            from tde_sph.core.simulation import Simulation
        except ImportError:
            pytest.skip("Simulation not available")

        # Get the source code of write_snapshot method
        source = inspect.getsource(Simulation.write_snapshot)

        # Verify temperature is in particle_data dict
        assert "'temperature'" in source, \
            "Simulation.write_snapshot() should include 'temperature' in particle_data"
