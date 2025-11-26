import numpy as np

from tde_sph.sph import ParticleSystem


def test_particle_system_initializes_temperature_and_velocity_arrays():
    n = 16
    particles = ParticleSystem(n)

    assert particles.temperature.shape == (n,)
    assert particles.temperature.dtype == np.float32
    assert np.all(particles.temperature == 0.0)

    assert particles.velocity_magnitude.shape == (n,)
    assert particles.velocity_magnitude.dtype == np.float32
    assert np.all(particles.velocity_magnitude == 0.0)


def test_particle_system_provides_plural_aliases():
    n = 4
    particles = ParticleSystem(n)

    smoothing = np.full(n, 0.2, dtype=np.float32)
    particles.smoothing_lengths = smoothing

    np.testing.assert_array_equal(particles.smoothing_length, smoothing)
    np.testing.assert_array_equal(particles.smoothing_lengths, smoothing)

    sound_speed = np.linspace(0.1, 0.4, n, dtype=np.float32)
    particles.sound_speeds = sound_speed

    np.testing.assert_array_equal(particles.sound_speed, sound_speed)
    np.testing.assert_array_equal(particles.sound_speeds, sound_speed)
