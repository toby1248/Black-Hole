"""
Abstract base classes defining interfaces for pluggable TDE-SPH modules.

This module establishes the contract that all physics modules must implement,
enabling clean separation of concerns and easy swapping of implementations.

Design principle (GUD-001): All physics modules must implement clearly specified
Python interfaces so they can be replaced with alternative implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt


# Type aliases for clarity
NDArrayFloat = npt.NDArray[np.float32]


class Metric(ABC):
    """
    Abstract base class for spacetime metrics.

    Implementations: Minkowski, Schwarzschild, Kerr (Boyer-Lindquist, Kerr-Schild).
    Supports REQ-005: spacetime modelling with metric tensor, inverse, Christoffel symbols.
    """

    @abstractmethod
    def metric_tensor(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Compute metric tensor g_μν at position(s) x.

        Parameters
        ----------
        x : NDArrayFloat, shape (N, 3) or (3,)
            Spatial coordinates (x, y, z) or (r, θ, φ) depending on coordinate system.

        Returns
        -------
        g : NDArrayFloat, shape (N, 4, 4) or (4, 4)
            Metric tensor components including time-time component.
        """
        pass

    @abstractmethod
    def inverse_metric(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Compute inverse metric tensor g^μν at position(s) x.

        Parameters
        ----------
        x : NDArrayFloat, shape (N, 3) or (3,)
            Spatial coordinates.

        Returns
        -------
        g_inv : NDArrayFloat, shape (N, 4, 4) or (4, 4)
            Inverse metric tensor components.
        """
        pass

    @abstractmethod
    def christoffel_symbols(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Compute Christoffel symbols Γ^μ_νρ at position(s) x.

        Parameters
        ----------
        x : NDArrayFloat, shape (N, 3) or (3,)
            Spatial coordinates.

        Returns
        -------
        gamma : NDArrayFloat, shape (N, 4, 4, 4) or (4, 4, 4)
            Christoffel symbols.
        """
        pass

    @abstractmethod
    def geodesic_acceleration(self, x: NDArrayFloat, v: NDArrayFloat) -> NDArrayFloat:
        """
        Compute geodesic acceleration d²x^μ/dτ² from position and 4-velocity.

        This is the core method for relativistic particle motion (REQ-002).

        Parameters
        ----------
        x : NDArrayFloat, shape (N, 3) or (3,)
            Spatial position.
        v : NDArrayFloat, shape (N, 4) or (4,)
            4-velocity (u^μ).

        Returns
        -------
        a : NDArrayFloat, shape (N, 3) or (3,)
            Spatial acceleration.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable metric name."""
        pass


class GravitySolver(ABC):
    """
    Abstract base class for gravity solvers.

    Implementations: Newtonian (tree/direct), RelativisticOrbit (hybrid GR+Newtonian).
    Supports REQ-002, REQ-006.
    """

    @abstractmethod
    def compute_acceleration(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        metric: Optional[Metric] = None
    ) -> NDArrayFloat:
        """
        Compute gravitational acceleration on all particles.

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        smoothing_lengths : NDArrayFloat, shape (N,)
            Smoothing lengths (used for softening).
        metric : Optional[Metric]
            Spacetime metric for relativistic solvers; None for Newtonian.

        Returns
        -------
        accel : NDArrayFloat, shape (N, 3)
            Gravitational acceleration on each particle.
        """
        pass

    @abstractmethod
    def compute_potential(
        self,
        positions: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute gravitational potential energy per particle.

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        smoothing_lengths : NDArrayFloat, shape (N,)
            Smoothing lengths.

        Returns
        -------
        potential : NDArrayFloat, shape (N,)
            Gravitational potential at each particle.
        """
        pass


class EOS(ABC):
    """
    Abstract base class for equation of state.

    Implementations: IdealGas, RadiationGas (gas + radiation pressure).
    Supports REQ-008: thermodynamics & EOS.
    """

    @abstractmethod
    def pressure(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute pressure from density and internal energy.

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u.
        **kwargs : additional EOS parameters (e.g., temperature for radiation).

        Returns
        -------
        pressure : NDArrayFloat, shape (N,)
            Pressure P.
        """
        pass

    @abstractmethod
    def sound_speed(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute sound speed from density and internal energy.

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u.
        **kwargs : additional EOS parameters.

        Returns
        -------
        cs : NDArrayFloat, shape (N,)
            Sound speed.
        """
        pass

    @abstractmethod
    def temperature(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute temperature from density and internal energy.

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u.
        **kwargs : additional EOS parameters.

        Returns
        -------
        T : NDArrayFloat, shape (N,)
            Temperature.
        """
        pass


class RadiationModel(ABC):
    """
    Abstract base class for radiation/cooling models.

    Implementations: SimpleCooling, FluxLimitedDiffusion.
    Supports REQ-009, REQ-011: energy accounting & radiative transfer.
    """

    @abstractmethod
    def cooling_rate(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute radiative cooling rate du/dt.

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density.
        temperature : NDArrayFloat, shape (N,)
            Temperature.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy.
        **kwargs : model-specific parameters.

        Returns
        -------
        du_dt : NDArrayFloat, shape (N,)
            Rate of change of internal energy due to cooling (negative for cooling).
        """
        pass

    @abstractmethod
    def luminosity(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        masses: NDArrayFloat,
        **kwargs
    ) -> float:
        """
        Compute total luminosity.

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density.
        temperature : NDArrayFloat, shape (N,)
            Temperature.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy.
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        **kwargs : model-specific parameters.

        Returns
        -------
        L : float
            Total luminosity.
        """
        pass


class TimeIntegrator(ABC):
    """
    Abstract base class for time integration schemes.

    Implementations: Leapfrog, Hamiltonian (for GR orbits), RK4.
    Supports REQ-004, REQ-010: dynamic timesteps and integration strategies.
    """

    @abstractmethod
    def step(
        self,
        particles: Any,  # ParticleSystem type
        dt: float,
        forces: Dict[str, NDArrayFloat],
        **kwargs
    ) -> None:
        """
        Advance particle system by one timestep.

        Parameters
        ----------
        particles : ParticleSystem
            Particle system to evolve.
        dt : float
            Timestep.
        forces : Dict[str, NDArrayFloat]
            Dictionary of force contributions (gravity, hydro, etc.).
        **kwargs : integrator-specific parameters.
        """
        pass

    @abstractmethod
    def estimate_timestep(
        self,
        particles: Any,
        cfl_factor: float = 0.3,
        **kwargs
    ) -> float:
        """
        Estimate appropriate timestep based on CFL and other criteria.

        Parameters
        ----------
        particles : ParticleSystem
            Particle system.
        cfl_factor : float
            CFL safety factor (< 1).
        **kwargs : additional criteria (e.g., orbital timescale for GR).

        Returns
        -------
        dt : float
            Suggested timestep.
        """
        pass


class ICGenerator(ABC):
    """
    Abstract base class for initial conditions generators.

    Implementations: Polytrope, MESAStar, Disc.
    Supports REQ-007: stellar models.
    """

    @abstractmethod
    def generate(
        self,
        n_particles: int,
        **kwargs
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Generate initial particle distribution.

        Parameters
        ----------
        n_particles : int
            Number of particles to generate.
        **kwargs : model-specific parameters (mass, radius, gamma, etc.).

        Returns
        -------
        positions : NDArrayFloat, shape (n_particles, 3)
            Initial positions.
        velocities : NDArrayFloat, shape (n_particles, 3)
            Initial velocities.
        masses : NDArrayFloat, shape (n_particles,)
            Particle masses.
        internal_energies : NDArrayFloat, shape (n_particles,)
            Specific internal energies.
        densities : NDArrayFloat, shape (n_particles,)
            Initial densities.
        """
        pass


class Visualizer(ABC):
    """
    Abstract base class for visualization backends.

    Implementations: Plotly3D, Matplotlib3D.
    Supports CON-004: visualization requirements.
    """

    @abstractmethod
    def plot_particles(
        self,
        positions: NDArrayFloat,
        quantities: Optional[Dict[str, NDArrayFloat]] = None,
        **kwargs
    ) -> Any:
        """
        Create 3D visualization of particle distribution.

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.
        quantities : Optional[Dict[str, NDArrayFloat]]
            Dictionary of quantities to visualize (density, temperature, etc.).
        **kwargs : backend-specific parameters.

        Returns
        -------
        fig : Any
            Figure object (type depends on backend).
        """
        pass

    @abstractmethod
    def animate(
        self,
        snapshots: list,
        **kwargs
    ) -> Any:
        """
        Create animation from sequence of snapshots.

        Parameters
        ----------
        snapshots : list
            List of snapshot dictionaries.
        **kwargs : backend-specific parameters.

        Returns
        -------
        anim : Any
            Animation object.
        """
        pass
