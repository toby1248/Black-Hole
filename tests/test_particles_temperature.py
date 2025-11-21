"""
Tests for ParticleSystem temperature attribute (TASK0).

Tests cover:
- Temperature array initialization
- Temperature shape validation
- Temperature dtype (float32)
- Temperature getter/setter methods
- Integration with simulation temperature computation

References
----------
- TASK0: Fix temperature attribute bug in ParticleSystem

Test Coverage
-------------
1. ParticleSystem initializes temperature array
2. Temperature has correct shape (n_particles,)
3. Temperature has correct dtype (float32)
4. Temperature shape validation catches mismatches
5. Temperature getter/setter methods work correctly
"""

import pytest
import numpy as np
from tde_sph.sph.particles import ParticleSystem


class TestParticleSystemTemperatureInitialization:
    """Test temperature attribute initialization in ParticleSystem."""

    def test_temperature_array_initialized(self):
        """Test that temperature array is initialized in __init__."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Temperature attribute should exist
        assert hasattr(particles, 'temperature'), "ParticleSystem should have temperature attribute"

        # Temperature should be a numpy array
        assert isinstance(particles.temperature, np.ndarray), "Temperature should be a numpy array"

        # Temperature should be initialized to zeros
        np.testing.assert_array_equal(
            particles.temperature,
            np.zeros(n_particles, dtype=np.float32),
            err_msg="Temperature should be initialized to zeros"
        )

    def test_temperature_correct_shape(self):
        """Test that temperature has correct shape (n_particles,)."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        assert particles.temperature.shape == (n_particles,), \
            f"Temperature shape should be ({n_particles},), got {particles.temperature.shape}"

    def test_temperature_correct_dtype(self):
        """Test that temperature has correct dtype (float32)."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        assert particles.temperature.dtype == np.float32, \
            f"Temperature dtype should be float32, got {particles.temperature.dtype}"

    def test_temperature_shape_validation(self):
        """Test that _validate_shapes() validates temperature shape."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Should not raise with correct shape
        particles._validate_shapes()

        # Should raise with incorrect shape
        particles.temperature = np.zeros(50, dtype=np.float32)  # Wrong shape
        with pytest.raises(AssertionError, match="temperature shape mismatch"):
            particles._validate_shapes()

    def test_temperature_with_different_sizes(self):
        """Test temperature initialization with different particle counts."""
        for n in [10, 100, 1000, 10000]:
            particles = ParticleSystem(n_particles=n)
            assert particles.temperature.shape == (n,), f"Failed for n={n}"
            assert particles.temperature.dtype == np.float32, f"Failed for n={n}"
            assert len(particles.temperature) == n, f"Failed for n={n}"


class TestParticleSystemTemperatureGetterSetter:
    """Test temperature getter and setter methods."""

    def test_get_temperature(self):
        """Test get_temperature() method."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Should have get_temperature method
        assert hasattr(particles, 'get_temperature'), "ParticleSystem should have get_temperature method"

        # Should return temperature array
        temp = particles.get_temperature()
        np.testing.assert_array_equal(temp, particles.temperature)

    def test_set_temperature(self):
        """Test set_temperature() method."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Should have set_temperature method
        assert hasattr(particles, 'set_temperature'), "ParticleSystem should have set_temperature method"

        # Set new temperature values
        new_temp = np.ones(n_particles, dtype=np.float32) * 5000.0
        particles.set_temperature(new_temp)

        # Verify temperature was set
        np.testing.assert_array_almost_equal(particles.temperature, new_temp)

    def test_set_temperature_validates_shape(self):
        """Test that set_temperature() validates shape."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Should raise with wrong shape
        wrong_temp = np.ones(50, dtype=np.float32)
        with pytest.raises(AssertionError):
            particles.set_temperature(wrong_temp)

    def test_set_temperature_converts_dtype(self):
        """Test that set_temperature() converts to float32."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Set with float64
        temp_f64 = np.ones(n_particles, dtype=np.float64) * 5000.0
        particles.set_temperature(temp_f64)

        # Should be converted to float32
        assert particles.temperature.dtype == np.float32, \
            f"Temperature should be converted to float32, got {particles.temperature.dtype}"


class TestParticleSystemTemperatureIntegration:
    """Integration tests for temperature with other particle properties."""

    def test_temperature_alongside_other_properties(self):
        """Test that temperature works alongside other particle properties."""
        n_particles = 100

        positions = np.random.randn(n_particles, 3).astype(np.float32)
        velocities = np.random.randn(n_particles, 3).astype(np.float32)
        masses = np.ones(n_particles, dtype=np.float32)
        internal_energy = np.ones(n_particles, dtype=np.float32)

        particles = ParticleSystem(
            n_particles=n_particles,
            positions=positions,
            velocities=velocities,
            masses=masses,
            internal_energy=internal_energy
        )

        # All properties should exist
        assert hasattr(particles, 'positions')
        assert hasattr(particles, 'velocities')
        assert hasattr(particles, 'masses')
        assert hasattr(particles, 'internal_energy')
        assert hasattr(particles, 'density')
        assert hasattr(particles, 'pressure')
        assert hasattr(particles, 'sound_speed')
        assert hasattr(particles, 'temperature')

        # Validate shapes should pass
        particles._validate_shapes()

    def test_temperature_in_repr(self):
        """Test that ParticleSystem string representation still works with temperature."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Should not raise
        repr_str = repr(particles)
        assert isinstance(repr_str, str)
        assert 'ParticleSystem' in repr_str
        assert f'n_particles={n_particles}' in repr_str
