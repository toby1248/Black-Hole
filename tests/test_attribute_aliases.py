"""
Tests for ParticleSystem attribute aliases (TASK1).

Tests cover:
- Plural property aliases for backward compatibility
- smoothing_lengths <-> smoothing_length
- sound_speeds <-> sound_speed

References
----------
- TASK1: Fix similar attribute bugs - added plural aliases for backward compatibility

Test Coverage
-------------
1. smoothing_lengths property reads from smoothing_length
2. smoothing_lengths setter writes to smoothing_length
3. sound_speeds property reads from sound_speed
4. sound_speeds setter writes to sound_speed
5. Both singular and plural forms work identically
"""

import pytest
import numpy as np
from tde_sph.sph.particles import ParticleSystem


class TestSmoothingLengthsAlias:
    """Test smoothing_lengths property alias."""

    def test_smoothing_lengths_reads_from_smoothing_length(self):
        """Test that smoothing_lengths property returns smoothing_length."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Both should be the same reference
        assert particles.smoothing_lengths is particles.smoothing_length, \
            "smoothing_lengths should reference smoothing_length"

        # Values should be identical
        np.testing.assert_array_equal(
            particles.smoothing_lengths,
            particles.smoothing_length
        )

    def test_smoothing_lengths_setter_updates_smoothing_length(self):
        """Test that setting smoothing_lengths updates smoothing_length."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Set via plural form
        new_values = np.ones(n_particles, dtype=np.float32) * 0.5
        particles.smoothing_lengths = new_values

        # Should update singular form
        np.testing.assert_array_almost_equal(
            particles.smoothing_length,
            new_values
        )

        # And plural form should read the new values
        np.testing.assert_array_almost_equal(
            particles.smoothing_lengths,
            new_values
        )

    def test_smoothing_length_setter_reflects_in_plural(self):
        """Test that setting smoothing_length is visible via smoothing_lengths."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Set via singular form
        new_values = np.ones(n_particles, dtype=np.float32) * 0.3
        particles.set_smoothing_length(new_values)

        # Should be visible via plural form
        np.testing.assert_array_almost_equal(
            particles.smoothing_lengths,
            new_values
        )

    def test_smoothing_lengths_dtype_float32(self):
        """Test that smoothing_lengths maintains float32 dtype."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Set with float64
        new_values_f64 = np.ones(n_particles, dtype=np.float64) * 0.5
        particles.smoothing_lengths = new_values_f64

        # Should be converted to float32
        assert particles.smoothing_length.dtype == np.float32
        assert particles.smoothing_lengths.dtype == np.float32


class TestSoundSpeedsAlias:
    """Test sound_speeds property alias."""

    def test_sound_speeds_reads_from_sound_speed(self):
        """Test that sound_speeds property returns sound_speed."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Both should be the same reference
        assert particles.sound_speeds is particles.sound_speed, \
            "sound_speeds should reference sound_speed"

        # Values should be identical
        np.testing.assert_array_equal(
            particles.sound_speeds,
            particles.sound_speed
        )

    def test_sound_speeds_setter_updates_sound_speed(self):
        """Test that setting sound_speeds updates sound_speed."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Set via plural form
        new_values = np.ones(n_particles, dtype=np.float32) * 1.5
        particles.sound_speeds = new_values

        # Should update singular form
        np.testing.assert_array_almost_equal(
            particles.sound_speed,
            new_values
        )

        # And plural form should read the new values
        np.testing.assert_array_almost_equal(
            particles.sound_speeds,
            new_values
        )

    def test_sound_speed_setter_reflects_in_plural(self):
        """Test that setting sound_speed is visible via sound_speeds."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Set via singular form
        new_values = np.ones(n_particles, dtype=np.float32) * 2.0
        particles.set_sound_speed(new_values)

        # Should be visible via plural form
        np.testing.assert_array_almost_equal(
            particles.sound_speeds,
            new_values
        )

    def test_sound_speeds_dtype_float32(self):
        """Test that sound_speeds maintains float32 dtype."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Set with float64
        new_values_f64 = np.ones(n_particles, dtype=np.float64) * 1.5
        particles.sound_speeds = new_values_f64

        # Should be converted to float32
        assert particles.sound_speed.dtype == np.float32
        assert particles.sound_speeds.dtype == np.float32


class TestAttributeAliasesIntegration:
    """Integration tests for attribute aliases."""

    def test_both_forms_work_identically(self):
        """Test that singular and plural forms work identically."""
        n_particles = 100
        particles1 = ParticleSystem(n_particles=n_particles)
        particles2 = ParticleSystem(n_particles=n_particles)

        # Set smoothing lengths using different forms
        values_h = np.random.rand(n_particles).astype(np.float32)
        particles1.smoothing_length = values_h.copy()
        particles2.smoothing_lengths = values_h.copy()

        # Should produce identical results
        np.testing.assert_array_equal(
            particles1.smoothing_length,
            particles2.smoothing_length
        )
        np.testing.assert_array_equal(
            particles1.smoothing_lengths,
            particles2.smoothing_lengths
        )

        # Set sound speeds using different forms
        values_cs = np.random.rand(n_particles).astype(np.float32)
        particles1.sound_speed = values_cs.copy()
        particles2.sound_speeds = values_cs.copy()

        # Should produce identical results
        np.testing.assert_array_equal(
            particles1.sound_speed,
            particles2.sound_speed
        )
        np.testing.assert_array_equal(
            particles1.sound_speeds,
            particles2.sound_speeds
        )

    def test_aliases_with_simulation_workflow(self):
        """Test that aliases work in typical simulation workflow."""
        n_particles = 100
        particles = ParticleSystem(n_particles=n_particles)

        # Simulate typical workflow: code uses plural forms
        particles.smoothing_lengths = np.ones(n_particles, dtype=np.float32) * 0.1
        particles.sound_speeds = np.ones(n_particles, dtype=np.float32) * 1.0

        # Verify attributes are accessible via both forms
        assert len(particles.smoothing_length) == n_particles
        assert len(particles.smoothing_lengths) == n_particles
        assert len(particles.sound_speed) == n_particles
        assert len(particles.sound_speeds) == n_particles

        # All should be float32
        assert particles.smoothing_length.dtype == np.float32
        assert particles.smoothing_lengths.dtype == np.float32
        assert particles.sound_speed.dtype == np.float32
        assert particles.sound_speeds.dtype == np.float32
