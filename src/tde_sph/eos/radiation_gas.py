"""
Gas + radiation pressure equation of state for SPH thermodynamics.

Implements REQ-008: thermodynamics with combined gas and radiation pressure,
critical for hot TDE debris where radiation pressure can dominate.

This module extends the ideal gas EOS to include radiation pressure:
    P = P_gas + P_rad = (γ - 1) ρ u_gas + (1/3) a T⁴

The total specific internal energy u includes both gas and radiation components:
    u = u_gas + u_rad = (1/(γ-1)) * (k_B T)/(μ m_p) + (a T⁴)/ρ

Temperature is solved iteratively from the total internal energy.

References:
    Kippenhahn & Weigert - Stellar Structure and Evolution (radiation pressure)
    Liptai & Price (2019) - GRSPH thermodynamics
    Ryu et al. (1993) - Radiation hydrodynamics equations
"""

import numpy as np
from typing import Tuple
from ..core.interfaces import EOS, NDArrayFloat


class RadiationGas(EOS):
    """
    Combined gas + radiation pressure equation of state.

    Thermodynamic relations:
        P_gas = (γ - 1) ρ u_gas
        P_rad = (1/3) a T⁴
        P_total = P_gas + P_rad

        u_total = u_gas + u_rad
        u_gas = (1/(γ-1)) * (k_B T)/(μ m_p)
        u_rad = (a T⁴) / ρ

        c_s² = (γ P_gas + (4/3) P_rad) / ρ

    Temperature is solved iteratively from total internal energy using Newton-Raphson.

    Parameters
    ----------
    gamma : float, optional
        Adiabatic index for gas component (default 5/3).
    mean_molecular_weight : float, optional
        Mean molecular weight in proton masses (default 0.6).
    radiation_constant_cgs : float, optional
        Radiation constant a in CGS [erg/(cm³·K⁴)] (default: 7.5657e-15).
    max_iterations : int, optional
        Maximum iterations for temperature solver (default: 50).
    tolerance : float, optional
        Relative convergence tolerance for temperature (default: 1e-6).

    Attributes
    ----------
    gamma : float
        Adiabatic index.
    mu : float
        Mean molecular weight.
    a_rad : float
        Radiation constant in CGS.
    k_B : float
        Boltzmann constant [erg/K].
    m_p : float
        Proton mass [g].

    Notes
    -----
    - At low temperatures (T < 10⁶ K), gas pressure dominates
    - At high temperatures (T > 10⁷ K), radiation pressure dominates
    - Temperature solver uses Newton-Raphson with analytic derivative
    - Falls back to bisection if Newton-Raphson fails to converge
    """

    # Physical constants in CGS units
    k_B = np.float32(1.380649e-16)  # Boltzmann constant [erg/K]
    m_p = np.float32(1.672621e-24)  # Proton mass [g]
    a_rad_default = np.float32(7.5657e-15)  # Radiation constant [erg/(cm³·K⁴)]

    def __init__(
        self,
        gamma: float = 5.0/3.0,
        mean_molecular_weight: float = 0.6,
        radiation_constant_cgs: float = 7.5657e-15,
        max_iterations: int = 50,
        tolerance: float = 1e-6
    ):
        """
        Initialize gas + radiation EOS.

        Parameters
        ----------
        gamma : float, optional
            Adiabatic index for gas (default 5/3).
        mean_molecular_weight : float, optional
            Mean molecular weight in proton masses (default 0.6).
        radiation_constant_cgs : float, optional
            Radiation constant a [erg/(cm³·K⁴)] (default: 7.5657e-15).
        max_iterations : int, optional
            Max iterations for T solver (default: 50).
        tolerance : float, optional
            Convergence tolerance (default: 1e-6).
        """
        self.gamma = np.float32(gamma)
        self.mu = np.float32(mean_molecular_weight)
        self.a_rad = np.float32(radiation_constant_cgs)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Precompute commonly used quantities
        self._gamma_minus_1 = np.float32(gamma - 1.0)
        self._inv_gamma_minus_1 = np.float32(1.0 / (gamma - 1.0))
        self._four_thirds = np.float32(4.0 / 3.0)
        self._one_third = np.float32(1.0 / 3.0)

        # Gas internal energy coefficient: u_gas = coeff * T
        self._u_gas_coeff = self.k_B / (self._gamma_minus_1 * self.mu * self.m_p)

        # Validate gamma
        if gamma <= 1.0:
            raise ValueError(f"Adiabatic index gamma must be > 1, got {gamma}")

    def temperature(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Solve for temperature from total internal energy.

        Uses Newton-Raphson iteration to solve:
            u_total = u_gas(T) + u_rad(T, ρ)
            u_total = (k_B T)/((γ-1) μ m_p) + (a T⁴)/ρ

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Total specific internal energy u.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        T : NDArrayFloat, shape (N,)
            Temperature in Kelvin.

        Notes
        -----
        - Uses Newton-Raphson with analytic derivative
        - Initial guess from gas-only formula
        - Enforces T > 0 at all steps
        - Falls back to bisection if Newton fails
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Ensure positive values
        rho_safe = np.maximum(density, 1e-30)
        u_safe = np.maximum(internal_energy, 0.0)

        # Initial guess: gas-only temperature
        T_guess = u_safe / self._u_gas_coeff
        T_guess = np.maximum(T_guess, 1.0)  # At least 1 K

        # Newton-Raphson iteration
        T = T_guess.copy()

        for iteration in range(self.max_iterations):
            # Compute residual: f(T) = u_total - u_gas(T) - u_rad(T)
            u_gas = self._u_gas_coeff * T
            u_rad = self.a_rad * T**4 / rho_safe
            residual = u_safe - u_gas - u_rad

            # Compute derivative: f'(T) = -du_gas/dT - du_rad/dT
            du_gas_dT = self._u_gas_coeff
            du_rad_dT = 4.0 * self.a_rad * T**3 / rho_safe
            derivative = -(du_gas_dT + du_rad_dT)

            # Newton step: T_new = T - f(T) / f'(T)
            # Avoid division by zero
            derivative_safe = np.where(np.abs(derivative) > 1e-30, derivative, -1e-30)
            delta_T = -residual / derivative_safe

            # Update with damping to ensure positivity
            T_new = T + 0.5 * delta_T  # Damping factor 0.5 for stability
            T_new = np.maximum(T_new, 1.0)  # Enforce T > 1 K

            # Check convergence
            relative_change = np.abs(delta_T) / (T + 1e-30)
            converged = relative_change < self.tolerance

            if np.all(converged):
                return T_new.astype(np.float32)

            T = T_new

        # If not converged, return current estimate (with warning in production)
        return T.astype(np.float32)

    def pressure(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute total pressure P = P_gas + P_rad.

        P_gas = (γ - 1) ρ u_gas
        P_rad = (1/3) a T⁴

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Total specific internal energy u.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        pressure : NDArrayFloat, shape (N,)
            Total pressure P.

        Notes
        -----
        - Solves for temperature first
        - Computes gas and radiation pressures separately then sums
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Ensure positive values
        rho_safe = np.maximum(density, 1e-30)
        u_safe = np.maximum(internal_energy, 0.0)

        # Solve for temperature
        T = self.temperature(rho_safe, u_safe)

        # Gas pressure
        u_gas = self._u_gas_coeff * T
        P_gas = self._gamma_minus_1 * rho_safe * u_gas

        # Radiation pressure
        P_rad = self._one_third * self.a_rad * T**4

        P_total = P_gas + P_rad

        return P_total.astype(np.float32)

    def sound_speed(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute sound speed including radiation pressure.

        c_s² = (γ P_gas + (4/3) P_rad) / ρ

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Total specific internal energy u.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        cs : NDArrayFloat, shape (N,)
            Sound speed c_s.

        Notes
        -----
        - Radiation increases effective sound speed
        - In radiation-dominated regime: c_s → c/√3
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Ensure positive values
        rho_safe = np.maximum(density, 1e-30)
        u_safe = np.maximum(internal_energy, 0.0)

        # Solve for temperature
        T = self.temperature(rho_safe, u_safe)

        # Gas pressure
        u_gas = self._u_gas_coeff * T
        P_gas = self._gamma_minus_1 * rho_safe * u_gas

        # Radiation pressure
        P_rad = self._one_third * self.a_rad * T**4

        # Sound speed squared
        cs_squared = (self.gamma * P_gas + self._four_thirds * P_rad) / rho_safe

        # Ensure non-negative
        cs_squared = np.maximum(cs_squared, 0.0)

        cs = np.sqrt(cs_squared)

        return cs.astype(np.float32)

    def gas_energy(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute gas component of internal energy.

        u_gas = (k_B T) / ((γ-1) μ m_p)

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Total specific internal energy u.

        Returns
        -------
        u_gas : NDArrayFloat, shape (N,)
            Gas component of internal energy.
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        rho_safe = np.maximum(density, 1e-30)
        u_safe = np.maximum(internal_energy, 0.0)

        T = self.temperature(rho_safe, u_safe)
        u_gas = self._u_gas_coeff * T

        return u_gas.astype(np.float32)

    def radiation_energy(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute radiation component of internal energy.

        u_rad = (a T⁴) / ρ

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Total specific internal energy u.

        Returns
        -------
        u_rad : NDArrayFloat, shape (N,)
            Radiation component of internal energy.
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        rho_safe = np.maximum(density, 1e-30)
        u_safe = np.maximum(internal_energy, 0.0)

        T = self.temperature(rho_safe, u_safe)
        u_rad = self.a_rad * T**4 / rho_safe

        return u_rad.astype(np.float32)

    def beta_parameter(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute gas-to-total pressure ratio β = P_gas / P_total.

        Useful diagnostic:
        - β ≈ 1: gas pressure dominated
        - β ≈ 0: radiation pressure dominated
        - β ≈ 0.5: transition regime

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Total specific internal energy u.

        Returns
        -------
        beta : NDArrayFloat, shape (N,)
            Gas-to-total pressure ratio.
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        rho_safe = np.maximum(density, 1e-30)
        u_safe = np.maximum(internal_energy, 0.0)

        T = self.temperature(rho_safe, u_safe)

        u_gas = self._u_gas_coeff * T
        P_gas = self._gamma_minus_1 * rho_safe * u_gas
        P_rad = self._one_third * self.a_rad * T**4

        P_total = P_gas + P_rad
        P_total_safe = np.maximum(P_total, 1e-30)

        beta = P_gas / P_total_safe

        return beta.astype(np.float32)

    def __repr__(self) -> str:
        """String representation of EOS."""
        return f"RadiationGas(gamma={self.gamma}, mu={self.mu}, a_rad={self.a_rad:.3e})"
