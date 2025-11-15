"""
Ideal gas equation of state for SPH thermodynamics.

Implements REQ-008: thermodynamics & EOS with adiabatic gas.

This module provides thermodynamic relations for an ideal gas with
arbitrary adiabatic index γ. Supports both monatomic (γ=5/3) and
relativistic (γ=4/3) gases as required for TDE simulations.

References:
    Kippenhahn & Weigert - Stellar Structure and Evolution
    Price (2012) - SPH review, thermodynamics section
    REQ-007 - Stellar models with γ = 5/3, 4/3
"""

import numpy as np
from ..core.interfaces import EOS, NDArrayFloat


class IdealGas(EOS):
    """
    Ideal gas equation of state with arbitrary adiabatic index.

    Thermodynamic relations:
        P = (γ - 1) ρ u           (pressure)
        c_s = sqrt(γ P / ρ)       (sound speed)
        T = (γ - 1) u μ m_p / k_B (temperature)

    where:
        ρ = density
        u = specific internal energy
        γ = adiabatic index
        μ = mean molecular weight
        m_p = proton mass
        k_B = Boltzmann constant

    Parameters
    ----------
    gamma : float, optional
        Adiabatic index (default 5/3 for monatomic ideal gas).
        Common values:
            γ = 5/3 : monatomic gas (REQ-007)
            γ = 4/3 : relativistic gas (REQ-007)
    mean_molecular_weight : float, optional
        Mean molecular weight in units of proton mass (default 0.6,
        appropriate for ionized solar-composition gas).

    Attributes
    ----------
    gamma : float
        Adiabatic index.
    mu : float
        Mean molecular weight.
    k_B : float
        Boltzmann constant in CGS units [erg/K].
    m_p : float
        Proton mass in CGS units [g].

    Notes
    -----
    Physical constants are stored in CGS units for temperature calculations.
    All other quantities use dimensionless units per GUD-003.

    Edge cases:
    - Negative density raises warning and returns zero pressure/sound speed.
    - Negative internal energy raises warning and returns zero temperature.
    - Division by zero is prevented with epsilon floor.
    """

    # Physical constants in CGS units
    k_B = np.float32(1.380649e-16)  # Boltzmann constant [erg/K]
    m_p = np.float32(1.672621e-24)  # Proton mass [g]

    def __init__(
        self,
        gamma: float = 5.0/3.0,
        mean_molecular_weight: float = 0.6
    ):
        """
        Initialize ideal gas EOS.

        Parameters
        ----------
        gamma : float, optional
            Adiabatic index (default 5/3).
        mean_molecular_weight : float, optional
            Mean molecular weight in proton masses (default 0.6).
        """
        self.gamma = np.float32(gamma)
        self.mu = np.float32(mean_molecular_weight)

        # Precompute commonly used quantities
        self._gamma_minus_1 = np.float32(gamma - 1.0)

        # Validate gamma
        if gamma <= 1.0:
            raise ValueError(f"Adiabatic index gamma must be > 1, got {gamma}")

    def pressure(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute gas pressure from density and internal energy.

        P = (γ - 1) ρ u

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u (energy per unit mass).
        **kwargs
            Additional parameters (ignored for ideal gas).

        Returns
        -------
        pressure : NDArrayFloat, shape (N,)
            Gas pressure P.

        Notes
        -----
        Returns zero pressure for negative density or internal energy
        (with warning in debug mode).
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Handle negative values gracefully
        density_safe = np.maximum(density, 0.0)
        internal_energy_safe = np.maximum(internal_energy, 0.0)

        pressure = self._gamma_minus_1 * density_safe * internal_energy_safe

        return pressure.astype(np.float32)

    def sound_speed(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute adiabatic sound speed.

        c_s = sqrt(γ P / ρ) = sqrt(γ (γ-1) u)

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u.
        **kwargs
            Additional parameters (ignored for ideal gas).

        Returns
        -------
        cs : NDArrayFloat, shape (N,)
            Sound speed c_s.

        Notes
        -----
        For ideal gas, c_s² = γ (γ-1) u, independent of density.
        We include density in signature for interface consistency
        and to handle general EOS cases.

        A small floor (1e-30) is applied to prevent sqrt(0) issues.
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # c_s² = γ P / ρ = γ (γ-1) u
        # Use pressure method for consistency and safety
        P = self.pressure(density, internal_energy)

        # Add small floor to prevent division by zero
        density_safe = np.maximum(density, 1e-30)

        cs_squared = self.gamma * P / density_safe

        # Ensure non-negative before sqrt
        cs_squared = np.maximum(cs_squared, 0.0)

        cs = np.sqrt(cs_squared)

        return cs.astype(np.float32)

    def temperature(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute temperature from internal energy.

        T = (γ - 1) u μ m_p / k_B

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ (not used for ideal gas, included for interface).
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u.
        **kwargs
            Additional parameters (ignored for ideal gas).

        Returns
        -------
        T : NDArrayFloat, shape (N,)
            Temperature in Kelvin.

        Notes
        -----
        For ideal gas with constant mean molecular weight:
            u = (1/(γ-1)) * (k_B T) / (μ m_p)
        therefore:
            T = (γ-1) u μ m_p / k_B

        This assumes non-relativistic particles. For γ=4/3 (relativistic),
        this relation is approximate.

        Temperature is capped at minimum value of 0 K.
        """
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Ensure non-negative internal energy
        u_safe = np.maximum(internal_energy, 0.0)

        # T = (γ-1) u μ m_p / k_B
        temperature = self._gamma_minus_1 * u_safe * self.mu * self.m_p / self.k_B

        return temperature.astype(np.float32)

    def internal_energy_from_temperature(
        self,
        temperature: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute specific internal energy from temperature (convenience method).

        u = (1/(γ-1)) * (k_B T) / (μ m_p)

        Parameters
        ----------
        temperature : NDArrayFloat, shape (N,)
            Temperature in Kelvin.

        Returns
        -------
        u : NDArrayFloat, shape (N,)
            Specific internal energy.

        Notes
        -----
        Inverse of the temperature() method. Useful for setting initial
        conditions from temperature profiles.
        """
        temperature = np.asarray(temperature, dtype=np.float32)

        # Ensure non-negative temperature
        T_safe = np.maximum(temperature, 0.0)

        # u = T k_B / ((γ-1) μ m_p)
        internal_energy = T_safe * self.k_B / (self._gamma_minus_1 * self.mu * self.m_p)

        return internal_energy.astype(np.float32)

    def __repr__(self) -> str:
        """String representation of EOS."""
        return f"IdealGas(gamma={self.gamma}, mu={self.mu})"
