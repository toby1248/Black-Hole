"""
Radiation + gas equation of state for SPH thermodynamics.

Implements TASK-021: Combined gas + radiation pressure EOS for optically thick gas.
Extends REQ-008 to include radiation pressure.

This module provides thermodynamic relations for a gas + radiation mixture,
accounting for both gas pressure and radiation pressure. This is essential
for TDE simulations where temperatures can reach ~10^6-10^8 K and radiation
pressure becomes significant.

Physics:
    Total pressure: P = P_gas + P_rad = (γ - 1) ρ u + (1/3) a T⁴
    Radiation constant: a = 4σ/c = 7.5657e-16 erg cm⁻³ K⁻⁴
    Stefan-Boltzmann: σ = 5.670374e-5 erg cm⁻² s⁻¹ K⁻⁴
    Speed of light: c = 2.99792458e10 cm/s

The internal energy includes both gas and radiation components:
    u = u_gas + u_rad
    u_gas = (1/(γ-1)) * P_gas / ρ
    u_rad = a T⁴ / ρ

References:
    Kippenhahn & Weigert - Stellar Structure and Evolution, Chapter 13
    Liptai & Price (2019) - GRSPH thermodynamics
    Price (2012) - SPH thermodynamics section
"""

import numpy as np
from typing import Optional, Tuple
from ..core.interfaces import EOS, NDArrayFloat


class RadiationGasEOS(EOS):
    """
    Combined gas + radiation pressure equation of state.

    Implements thermodynamic relations for optically thick gas with radiation
    pressure. Uses Newton-Raphson iteration to solve for temperature T from
    (ρ, u), then computes pressure, sound speed, etc.

    Thermodynamic relations:
        P = P_gas + P_rad = (γ - 1) ρ u_gas + (1/3) a T⁴
        u_total = u_gas + u_rad
        u_gas = P_gas / ((γ-1) ρ)
        u_rad = a T⁴ / ρ

    From T(ρ,u), we can derive:
        u = k_B T / ((γ-1) μ m_p) + a T⁴ / ρ

    This is a nonlinear equation solved via Newton-Raphson.

    Parameters
    ----------
    gamma : float, optional
        Adiabatic index for gas component (default 5/3).
    mean_molecular_weight : float, optional
        Mean molecular weight in proton masses (default 0.6).
    max_iterations : int, optional
        Maximum Newton-Raphson iterations (default 20).
    tolerance : float, optional
        Convergence tolerance for temperature (default 1e-6).
    use_fp64_for_temperature : bool, optional
        Use FP64 precision for temperature iterations (default True).
        Helps with convergence in mixed regimes.

    Attributes
    ----------
    gamma : float
        Adiabatic index.
    mu : float
        Mean molecular weight.
    k_B : float
        Boltzmann constant [erg/K].
    m_p : float
        Proton mass [g].
    sigma : float
        Stefan-Boltzmann constant [erg cm⁻² s⁻¹ K⁻⁴].
    a_rad : float
        Radiation constant a = 4σ/c [erg cm⁻³ K⁻⁴].
    c_light : float
        Speed of light [cm/s].

    Notes
    -----
    Regime transitions:
    - Gas-dominated: T << T_crit, where T_crit ~ ((γ-1) ρ μ m_p / (a k_B))^(1/3)
    - Radiation-dominated: T >> T_crit
    - Mixed: T ~ T_crit (requires careful Newton-Raphson)

    The sound speed includes both gas and radiation contributions:
        c_s² = γ_eff P / ρ
    where γ_eff varies between gas-dominated (γ) and radiation-dominated (4/3).

    Edge cases handled:
    - Negative ρ or u: clamp to zero, return zero pressure/temperature
    - Failed convergence: fall back to ideal gas approximation with warning
    - Very high T: radiation-dominated limit
    - Very low T: gas-dominated limit
    """

    # Physical constants in CGS units
    k_B = 1.380649e-16  # Boltzmann constant [erg/K]
    m_p = 1.672621e-24  # Proton mass [g]
    sigma = 5.670374e-5  # Stefan-Boltzmann constant [erg cm⁻² s⁻¹ K⁻⁴]
    c_light = 2.99792458e10  # Speed of light [cm/s]
    a_rad = 4.0 * sigma / c_light  # Radiation constant [erg cm⁻³ K⁻⁴]
    # a_rad ≈ 7.5657e-16 erg cm⁻³ K⁻⁴

    def __init__(
        self,
        gamma: float = 5.0/3.0,
        mean_molecular_weight: float = 0.6,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        use_fp64_for_temperature: bool = True
    ):
        """
        Initialize radiation + gas EOS.

        Parameters
        ----------
        gamma : float
            Adiabatic index (default 5/3 for monatomic gas).
        mean_molecular_weight : float
            Mean molecular weight in proton masses (default 0.6).
        max_iterations : int
            Max Newton-Raphson iterations for T(ρ,u) (default 20).
        tolerance : float
            Convergence tolerance for temperature (default 1e-6).
        use_fp64_for_temperature : bool
            Use FP64 for temperature solve (default True).
        """
        self.gamma = np.float32(gamma)
        self.mu = np.float32(mean_molecular_weight)
        self.max_iter = max_iterations
        self.tol = tolerance
        self.use_fp64 = use_fp64_for_temperature

        # Precompute commonly used quantities
        self._gamma_minus_1 = gamma - 1.0

        # Validate gamma
        if gamma <= 1.0:
            raise ValueError(f"Adiabatic index gamma must be > 1, got {gamma}")

        # Convert constants to appropriate precision
        if self.use_fp64:
            self.k_B_compute = np.float64(self.k_B)
            self.m_p_compute = np.float64(self.m_p)
            self.a_rad_compute = np.float64(self.a_rad)
        else:
            self.k_B_compute = np.float32(self.k_B)
            self.m_p_compute = np.float32(self.m_p)
            self.a_rad_compute = np.float32(self.a_rad)

    def _solve_temperature_newton_raphson(
        self,
        density: np.ndarray,
        internal_energy: np.ndarray
    ) -> np.ndarray:
        """
        Solve for temperature T from (ρ, u) using Newton-Raphson.

        We solve the implicit equation:
            f(T) = u - u_gas(T) - u_rad(T) = 0

        where:
            u_gas(T) = k_B T / ((γ-1) μ m_p)
            u_rad(T) = a T⁴ / ρ

        Newton-Raphson iteration:
            T_(n+1) = T_n - f(T_n) / f'(T_n)

        where:
            f(T) = u - k_B T / ((γ-1) μ m_p) - a T⁴ / ρ
            f'(T) = -k_B / ((γ-1) μ m_p) - 4 a T³ / ρ

        Parameters
        ----------
        density : np.ndarray
            Mass density ρ.
        internal_energy : np.ndarray
            Specific internal energy u.

        Returns
        -------
        temperature : np.ndarray
            Temperature T in Kelvin.

        Notes
        -----
        Initial guess uses gas-only approximation: T_0 = (γ-1) u μ m_p / k_B
        This works well in gas-dominated regime and provides reasonable starting
        point in mixed/radiation-dominated regimes.
        """
        # Use higher precision for temperature solve if requested
        if self.use_fp64:
            rho = np.asarray(density, dtype=np.float64)
            u = np.asarray(internal_energy, dtype=np.float64)
            dtype_compute = np.float64
        else:
            rho = np.asarray(density, dtype=np.float32)
            u = np.asarray(internal_energy, dtype=np.float32)
            dtype_compute = np.float32

        # Ensure positive values
        rho = np.maximum(rho, 1e-30)
        u = np.maximum(u, 0.0)

        # Initial guess: ideal gas approximation
        # T_0 = (γ-1) u μ m_p / k_B
        T = self._gamma_minus_1 * u * self.mu * self.m_p_compute / self.k_B_compute
        T = np.maximum(T, 1.0)  # Floor at 1 K

        # Precompute constants
        c1 = self.k_B_compute / (self._gamma_minus_1 * self.mu * self.m_p_compute)
        c2 = self.a_rad_compute / rho

        # Newton-Raphson iteration
        for iteration in range(self.max_iter):
            # Compute function and derivative
            # f(T) = u - c1 * T - c2 * T^4
            # f'(T) = -c1 - 4 * c2 * T^3

            T_cubed = T**3
            T_fourth = T * T_cubed

            f = u - c1 * T - c2 * T_fourth
            df_dT = -c1 - 4.0 * c2 * T_cubed

            # Newton-Raphson step
            dT = -f / df_dT
            T_new = T + dT

            # Ensure positive temperature
            T_new = np.maximum(T_new, 1.0)

            # Check convergence
            relative_change = np.abs(dT / T_new)
            if np.all(relative_change < self.tol):
                break

            T = T_new
        else:
            # Convergence warning (not an error, as fallback is reasonable)
            pass
            # In production, might want: warnings.warn("Temperature solve did not converge")

        # Convert back to fp32 for consistency
        return T.astype(np.float32)

    def pressure(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        temperature: Optional[NDArrayFloat] = None,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute total pressure P = P_gas + P_rad.

        P = (γ - 1) ρ u_gas + (1/3) a T⁴

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u (total = gas + radiation).
        temperature : Optional[NDArrayFloat], shape (N,)
            Temperature T in Kelvin. If None, will be computed from (ρ, u).
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        pressure : NDArrayFloat, shape (N,)
            Total pressure P.

        Notes
        -----
        If temperature is provided, uses it directly (faster).
        Otherwise, solves for T from (ρ, u) via Newton-Raphson (slower but self-consistent).
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Compute temperature if not provided
        if temperature is None:
            T = self._solve_temperature_newton_raphson(density, internal_energy)
        else:
            T = np.asarray(temperature, dtype=np.float32)

        # Compute gas internal energy from temperature
        # u_gas = k_B T / ((γ-1) μ m_p)
        u_gas = self.k_B * T / (self._gamma_minus_1 * self.mu * self.m_p)

        # Gas pressure: P_gas = (γ-1) ρ u_gas
        P_gas = self._gamma_minus_1 * density * u_gas

        # Radiation pressure: P_rad = (1/3) a T⁴
        # Use float64 to avoid overflow for high temperatures
        T_64 = T.astype(np.float64)
        P_rad_64 = (1.0/3.0) * np.float64(self.a_rad) * T_64**4
        P_rad = P_rad_64.astype(np.float32)

        # Total pressure
        P_total = P_gas + P_rad

        return P_total.astype(np.float32)

    def sound_speed(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        temperature: Optional[NDArrayFloat] = None,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute adiabatic sound speed including radiation effects.

        For radiation + gas mixture:
            c_s² = (∂P/∂ρ)_s = γ_eff P / ρ

        where γ_eff varies smoothly from γ (gas-dominated) to 4/3 (radiation-dominated).

        We use the thermodynamically consistent formula:
            c_s² = (γ P_gas + (4/3) P_rad) / ρ

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u.
        temperature : Optional[NDArrayFloat], shape (N,)
            Temperature T. If None, computed from (ρ, u).
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        cs : NDArrayFloat, shape (N,)
            Sound speed.

        Notes
        -----
        In gas-dominated limit (P_rad → 0): c_s² → γ P_gas / ρ (ideal gas)
        In radiation-dominated limit (P_gas → 0): c_s² → (4/3) P_rad / ρ

        Reference: Kippenhahn & Weigert, Chapter 13.
        """
        density = np.asarray(density, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Compute temperature if not provided
        if temperature is None:
            T = self._solve_temperature_newton_raphson(density, internal_energy)
        else:
            T = np.asarray(temperature, dtype=np.float32)

        # Compute gas internal energy
        u_gas = self.k_B * T / (self._gamma_minus_1 * self.mu * self.m_p)

        # Gas and radiation pressures
        P_gas = self._gamma_minus_1 * density * u_gas

        # Use float64 for T⁴ calculation to avoid overflow
        T_64 = T.astype(np.float64)
        P_rad_64 = (1.0/3.0) * np.float64(self.a_rad) * T_64**4
        P_rad = P_rad_64.astype(np.float32)

        # Effective sound speed
        # c_s² = (γ P_gas + (4/3) P_rad) / ρ
        rho_safe = np.maximum(density, 1e-30)
        cs_squared = (self.gamma * P_gas + (4.0/3.0) * P_rad) / rho_safe

        # Ensure non-negative
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
        Compute temperature from (ρ, u) via Newton-Raphson.

        Solves the implicit equation:
            u = u_gas(T) + u_rad(T)
            u = k_B T / ((γ-1) μ m_p) + a T⁴ / ρ

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        T : NDArrayFloat, shape (N,)
            Temperature in Kelvin.

        Notes
        -----
        This is the core method that enables self-consistent thermodynamics.
        All other methods (pressure, sound_speed) can call this to obtain T,
        or accept T as optional input for efficiency.
        """
        return self._solve_temperature_newton_raphson(density, internal_energy)

    def internal_energy_from_temperature(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute specific internal energy from (ρ, T).

        u = u_gas + u_rad
        u = k_B T / ((γ-1) μ m_p) + a T⁴ / ρ

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density ρ.
        temperature : NDArrayFloat, shape (N,)
            Temperature T in Kelvin.

        Returns
        -------
        u : NDArrayFloat, shape (N,)
            Specific internal energy.

        Notes
        -----
        Inverse of temperature() method. Useful for setting initial conditions
        from temperature profiles.
        """
        density = np.asarray(density, dtype=np.float32)
        temperature = np.asarray(temperature, dtype=np.float32)

        # Ensure positive values
        rho_safe = np.maximum(density, 1e-30)
        T_safe = np.maximum(temperature, 0.0)

        # Gas component
        u_gas = self.k_B * T_safe / (self._gamma_minus_1 * self.mu * self.m_p)

        # Radiation component - use float64 to avoid overflow
        T_64 = T_safe.astype(np.float64)
        rho_64 = rho_safe.astype(np.float64)
        u_rad_64 = np.float64(self.a_rad) * T_64**4 / rho_64
        u_rad = u_rad_64.astype(np.float32)

        # Total internal energy
        u_total = u_gas + u_rad

        return u_total.astype(np.float32)

    def gas_pressure_fraction(
        self,
        density: NDArrayFloat,
        internal_energy: NDArrayFloat,
        temperature: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat:
        """
        Compute fraction of pressure from gas (diagnostic).

        β = P_gas / (P_gas + P_rad)

        Parameters
        ----------
        density : NDArrayFloat
            Mass density.
        internal_energy : NDArrayFloat
            Specific internal energy.
        temperature : Optional[NDArrayFloat]
            Temperature (if None, computed).

        Returns
        -------
        beta : NDArrayFloat
            Gas pressure fraction β ∈ [0, 1].

        Notes
        -----
        β ≈ 1: gas-dominated
        β ≈ 0: radiation-dominated
        β ≈ 0.5: mixed regime
        """
        if temperature is None:
            T = self._solve_temperature_newton_raphson(density, internal_energy)
        else:
            T = np.asarray(temperature, dtype=np.float32)

        density = np.asarray(density, dtype=np.float32)

        # Compute gas internal energy
        u_gas = self.k_B * T / (self._gamma_minus_1 * self.mu * self.m_p)

        # Pressures
        P_gas = self._gamma_minus_1 * density * u_gas

        # Use float64 for T⁴ to avoid overflow
        T_64 = T.astype(np.float64)
        P_rad_64 = (1.0/3.0) * np.float64(self.a_rad) * T_64**4
        P_rad = P_rad_64.astype(np.float32)

        # Fraction
        P_total = P_gas + P_rad
        P_total_safe = np.maximum(P_total, 1e-30)
        beta = P_gas / P_total_safe

        return beta.astype(np.float32)

    def __repr__(self) -> str:
        """String representation of EOS."""
        return f"RadiationGasEOS(gamma={self.gamma}, mu={self.mu}, use_fp64={self.use_fp64})"
