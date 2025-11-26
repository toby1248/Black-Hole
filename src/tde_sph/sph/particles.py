"""
Particle system management for SPH simulations.

This module implements the ParticleSystem class that manages all SPH particle data,
including positions, velocities, thermodynamic quantities, and derived properties.

Design follows REQ-001 (SPH core) and CON-002 (FP32 performance optimization).
"""

from typing import Optional
import numpy as np
import numpy.typing as npt

# Type alias for clarity
NDArrayFloat = npt.NDArray[np.float32]


class ParticleSystem:
    """
    Container for SPH particle data and properties.

    Manages particle state arrays using numpy arrays with float32 precision
    for optimal GPU performance (CON-002: FP32 executes 64x faster on RTX 4090).

    Attributes
    ----------
    n_particles : int
        Number of particles in the system.
    positions : NDArrayFloat, shape (N, 3)
        Cartesian coordinates (x, y, z) in code units.
    velocities : NDArrayFloat, shape (N, 3)
        Velocity components (vx, vy, vz) in code units.
    masses : NDArrayFloat, shape (N,)
        Particle masses.
    internal_energy : NDArrayFloat, shape (N,)
        Specific internal energy u (energy per unit mass).
    density : NDArrayFloat, shape (N,)
        Mass density ρ computed from SPH summation.
    smoothing_length : NDArrayFloat, shape (N,)
        Adaptive smoothing lengths h for each particle.
    pressure : NDArrayFloat, shape (N,)
        Pressure P computed from EOS.
    sound_speed : NDArrayFloat, shape (N,)
        Local sound speed cs from EOS.

    Notes
    -----
    Following modern SPH practice from Price (2012) [1]_ and PHANTOM code [2]_,
    this implementation uses:
    - Adaptive smoothing lengths updated to maintain constant neighbour count
    - Float32 precision for GPU efficiency
    - Contiguous memory layout for vectorization

    References
    ----------
    .. [1] Price, D. J. (2012), "Smoothed particle hydrodynamics and
           magnetohydrodynamics", Journal of Computational Physics, 231, 759.
    .. [2] Price, D. J. et al. (2018), "PHANTOM: A Smoothed Particle
           Hydrodynamics and Magnetohydrodynamics Code for Astrophysics",
           PASA, 35, e031.
    """

    def __init__(
        self,
        n_particles: int,
        positions: Optional[NDArrayFloat] = None,
        velocities: Optional[NDArrayFloat] = None,
        masses: Optional[NDArrayFloat] = None,
        internal_energy: Optional[NDArrayFloat] = None,
        smoothing_length: Optional[NDArrayFloat] = None
    ):
        """
        Initialize particle system.

        Parameters
        ----------
        n_particles : int
            Number of particles.
        positions : NDArrayFloat, shape (N, 3), optional
            Initial positions. If None, initialized to zeros.
        velocities : NDArrayFloat, shape (N, 3), optional
            Initial velocities. If None, initialized to zeros.
        masses : NDArrayFloat, shape (N,), optional
            Particle masses. If None, initialized to equal mass.
        internal_energy : NDArrayFloat, shape (N,), optional
            Specific internal energies. If None, initialized to zeros.
        smoothing_length : NDArrayFloat, shape (N,), optional
            Smoothing lengths. If None, initialized to 0.1.
        """
        self.n_particles = n_particles

        # Primary state variables
        self.positions = (
            positions.astype(np.float32, copy=False) if positions is not None
            else np.zeros((n_particles, 3), dtype=np.float32)
        )
        self.velocities = (
            velocities.astype(np.float32, copy=False) if velocities is not None
            else np.zeros((n_particles, 3), dtype=np.float32)
        )
        self.masses = (
            masses.astype(np.float32, copy=False) if masses is not None
            else np.ones(n_particles, dtype=np.float32) / n_particles
        )
        self.internal_energy = (
            internal_energy.astype(np.float32, copy=False) if internal_energy is not None
            else np.zeros(n_particles, dtype=np.float32)
        )
        self.smoothing_length = (
            smoothing_length.astype(np.float32, copy=False) if smoothing_length is not None
            else np.full(n_particles, 0.1, dtype=np.float32)
        )

        # Derived quantities (computed during simulation)
        self.density = np.zeros(n_particles, dtype=np.float32)
        self.pressure = np.zeros(n_particles, dtype=np.float32)
        self.sound_speed = np.zeros(n_particles, dtype=np.float32)
        self.temperature = np.zeros(n_particles, dtype=np.float32)
        self.velocity_magnitude = np.zeros(n_particles, dtype=np.float32)

        # Validate shapes
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        """Validate that all arrays have consistent shapes."""
        n = self.n_particles
        assert self.positions.shape == (n, 3), f"positions shape mismatch: {self.positions.shape}"
        assert self.velocities.shape == (n, 3), f"velocities shape mismatch: {self.velocities.shape}"
        assert self.masses.shape == (n,), f"masses shape mismatch: {self.masses.shape}"
        assert self.internal_energy.shape == (n,), f"internal_energy shape mismatch: {self.internal_energy.shape}"
        assert self.smoothing_length.shape == (n,), f"smoothing_length shape mismatch: {self.smoothing_length.shape}"
        assert self.density.shape == (n,), f"density shape mismatch: {self.density.shape}"
        assert self.pressure.shape == (n,), f"pressure shape mismatch: {self.pressure.shape}"
        assert self.sound_speed.shape == (n,), f"sound_speed shape mismatch: {self.sound_speed.shape}"
        assert self.temperature.shape == (n,), f"temperature shape mismatch: {self.temperature.shape}"
        assert self.velocity_magnitude.shape == (n,), f"velocity_magnitude shape mismatch: {self.velocity_magnitude.shape}"

    def get_positions(self) -> NDArrayFloat:
        """Get particle positions."""
        return self.positions

    def get_velocities(self) -> NDArrayFloat:
        """Get particle velocities."""
        return self.velocities

    def get_masses(self) -> NDArrayFloat:
        """Get particle masses."""
        return self.masses

    def get_internal_energy(self) -> NDArrayFloat:
        """Get specific internal energies."""
        return self.internal_energy

    def get_density(self) -> NDArrayFloat:
        """Get densities."""
        return self.density

    def get_pressure(self) -> NDArrayFloat:
        """Get pressures."""
        return self.pressure

    def get_sound_speed(self) -> NDArrayFloat:
        """Get sound speeds."""
        return self.sound_speed

    def get_temperature(self) -> NDArrayFloat:
        """Get temperatures."""
        return self.temperature

    def get_velocity_magnitude(self) -> NDArrayFloat:
        """Get velocity magnitudes."""
        return self.velocity_magnitude

    def get_smoothing_length(self) -> NDArrayFloat:
        """Get smoothing lengths."""
        return self.smoothing_length

    def set_positions(self, positions: NDArrayFloat) -> None:
        """Set particle positions."""
        assert positions.shape == (self.n_particles, 3)
        self.positions = positions.astype(np.float32, copy=False)

    def set_velocities(self, velocities: NDArrayFloat) -> None:
        """Set particle velocities."""
        assert velocities.shape == (self.n_particles, 3)
        self.velocities = velocities.astype(np.float32, copy=False)

    def set_internal_energy(self, internal_energy: NDArrayFloat) -> None:
        """Set specific internal energies."""
        assert internal_energy.shape == (self.n_particles,)
        self.internal_energy = internal_energy.astype(np.float32, copy=False)

    def set_density(self, density: NDArrayFloat) -> None:
        """Set densities."""
        assert density.shape == (self.n_particles,)
        self.density = density.astype(np.float32, copy=False)

    def set_pressure(self, pressure: NDArrayFloat) -> None:
        """Set pressures."""
        assert pressure.shape == (self.n_particles,)
        self.pressure = pressure.astype(np.float32, copy=False)

    def set_sound_speed(self, sound_speed: NDArrayFloat) -> None:
        """Set sound speeds."""
        assert sound_speed.shape == (self.n_particles,)
        self.sound_speed = sound_speed.astype(np.float32, copy=False)

    def set_smoothing_length(self, smoothing_length: NDArrayFloat) -> None:
        """Set smoothing lengths."""
        assert smoothing_length.shape == (self.n_particles,)
        self.smoothing_length = smoothing_length.astype(np.float32, copy=False)

    def set_temperature(self, temperature: NDArrayFloat) -> None:
        """Set temperatures."""
        assert temperature.shape == (self.n_particles,)
        self.temperature = temperature.astype(np.float32, copy=False)

    def set_velocity_magnitude(self, velocity_magnitude: NDArrayFloat) -> None:
        """Set velocity magnitudes."""
        assert velocity_magnitude.shape == (self.n_particles,)
        self.velocity_magnitude = velocity_magnitude.astype(np.float32, copy=False)

    @property
    def smoothing_lengths(self) -> NDArrayFloat:
        """Alias for smoothing length array (plural for API consistency)."""
        return self.smoothing_length

    @smoothing_lengths.setter
    def smoothing_lengths(self, value: NDArrayFloat) -> None:
        self.set_smoothing_length(value)

    @property
    def sound_speeds(self) -> NDArrayFloat:
        """Alias for sound speed array (plural for legacy code)."""
        return self.sound_speed

    @sound_speeds.setter
    def sound_speeds(self, value: NDArrayFloat) -> None:
        self.set_sound_speed(value)

    def kinetic_energy(self) -> float:
        """
        Compute total kinetic energy of the system.

        Returns
        -------
        E_kin : float
            Total kinetic energy: ∑ (1/2) m v².
        """
        v_squared = np.sum(self.velocities**2, axis=1)
        return 0.5 * np.sum(self.masses * v_squared)

    def thermal_energy(self) -> float:
        """
        Compute total thermal (internal) energy of the system.

        Returns
        -------
        E_thermal : float
            Total thermal energy: ∑ m u.
        """
        return np.sum(self.masses * self.internal_energy)

    def total_mass(self) -> float:
        """
        Compute total mass of the system.

        Returns
        -------
        M_total : float
            Total mass: ∑ m.
        """
        return np.sum(self.masses)

    def center_of_mass(self) -> NDArrayFloat:
        """
        Compute center of mass position.

        Returns
        -------
        r_com : NDArrayFloat, shape (3,)
            Center of mass position.
        """
        total_mass = self.total_mass()
        if total_mass == 0:
            return np.zeros(3, dtype=np.float32)
        return np.sum(self.masses[:, np.newaxis] * self.positions, axis=0) / total_mass

    def center_of_mass_velocity(self) -> NDArrayFloat:
        """
        Compute center of mass velocity.

        Returns
        -------
        v_com : NDArrayFloat, shape (3,)
            Center of mass velocity.
        """
        total_mass = self.total_mass()
        if total_mass == 0:
            return np.zeros(3, dtype=np.float32)
        return np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0) / total_mass

    def __repr__(self) -> str:
        """String representation of particle system."""
        return (
            f"ParticleSystem(n_particles={self.n_particles}, "
            f"total_mass={self.total_mass():.3e}, "
            f"E_kin={self.kinetic_energy():.3e}, "
            f"E_thermal={self.thermal_energy():.3e})"
        )
