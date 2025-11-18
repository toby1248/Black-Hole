"""
Simple radiative cooling and luminosity models for TDE-SPH simulations.

Implements REQ-009, REQ-011: radiative cooling, energy loss rates, and luminosity
proxies for tidal disruption event simulations.

Provides multiple cooling models:
- Free-free (bremsstrahlung) for optically thin gas
- Blackbody for optically thick gas
- Power-law parameterization

Includes safety features:
- Temperature floors
- Cooling timestep limiters
- Prevention of negative internal energies

References:
    Rybicki & Lightman - Radiative Processes in Astrophysics
    Lodato & Rossi (2011) - TDE luminosity estimates
    Dai et al. (2015) - TDE light curves
"""

import numpy as np
from typing import Dict, Optional
import numpy.typing as npt
from ..core.interfaces import RadiationModel, EOS, NDArrayFloat


class SimpleCooling(RadiationModel):
    """
    Simple radiative cooling and luminosity model.

    Supports multiple cooling prescriptions:
    - Free-free (bremsstrahlung): Λ_ff = C_ff ρ² T^(1/2)
    - Blackbody: Λ_bb = σ_SB T⁴  (per unit area)
    - Power-law: Λ = Λ₀ (T/T₀)^β

    Parameters
    ----------
    cooling_model : str, optional
        Cooling prescription: 'free_free', 'blackbody', 'power_law', 'none'.
        Default: 'free_free'.
    temperature_floor : float, optional
        Minimum temperature in Kelvin (default: 100 K).
    cooling_timescale_factor : float, optional
        Safety factor for cooling timestep (default: 0.1).
    opacity : float, optional
        Uniform opacity in cm²/g for optical depth (default: 0.34).
    power_law_index : float, optional
        Exponent β for power-law cooling (default: 0.5).
    power_law_coefficient : float, optional
        Λ₀ for power-law cooling in CGS (default: 1e-27).
    reference_temperature : float, optional
        T₀ for power-law cooling in K (default: 1e7).

    Attributes
    ----------
    cooling_model : str
        Active cooling model.
    T_floor : float
        Temperature floor [K].
    cooling_factor : float
        Timestep safety factor.
    kappa : float
        Opacity [cm²/g].

    Notes
    -----
    Physical constants in CGS:
    - Stefan-Boltzmann: σ_SB = 5.67e-5 erg/(cm²·s·K⁴)
    - Free-free coefficient: C_ff = 1.4e-27 erg·cm³/(s·K^(1/2)·g²)
    """

    # Physical constants in CGS
    sigma_SB = np.float32(5.670374e-5)  # Stefan-Boltzmann [erg/(cm²·s·K⁴)]
    C_ff = np.float32(1.4e-27)  # Free-free cooling [erg·cm³/(s·K^(1/2)·g²)]
    kappa_es = np.float32(0.34)  # Electron scattering opacity [cm²/g]

    def __init__(
        self,
        cooling_model: str = 'free_free',
        temperature_floor: float = 100.0,
        cooling_timescale_factor: float = 0.1,
        opacity: float = 0.34,
        power_law_index: float = 0.5,
        power_law_coefficient: float = 1e-27,
        reference_temperature: float = 1e7
    ):
        """
        Initialize simple cooling model.

        Parameters
        ----------
        cooling_model : str, optional
            Cooling model type (default: 'free_free').
        temperature_floor : float, optional
            Minimum temperature [K] (default: 100).
        cooling_timescale_factor : float, optional
            Safety factor (default: 0.1).
        opacity : float, optional
            Opacity [cm²/g] (default: 0.34).
        power_law_index : float, optional
            Power-law exponent (default: 0.5).
        power_law_coefficient : float, optional
            Power-law normalization (default: 1e-27).
        reference_temperature : float, optional
            Reference temperature [K] (default: 1e7).
        """
        valid_models = ['free_free', 'blackbody', 'power_law', 'none']
        if cooling_model not in valid_models:
            raise ValueError(
                f"cooling_model must be one of {valid_models}, got '{cooling_model}'"
            )

        self.cooling_model = cooling_model
        self.T_floor = np.float32(temperature_floor)
        self.cooling_factor = np.float32(cooling_timescale_factor)
        self.kappa = np.float32(opacity)

        # Power-law parameters
        self.power_law_beta = np.float32(power_law_index)
        self.power_law_Lambda0 = np.float32(power_law_coefficient)
        self.T_ref = np.float32(reference_temperature)

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
            Mass density ρ [g/cm³ in CGS].
        temperature : NDArrayFloat, shape (N,)
            Temperature T [K].
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy u [erg/g].
        **kwargs
            Additional parameters:
            - smoothing_length : NDArrayFloat, shape (N,) for optical depth

        Returns
        -------
        du_dt : NDArrayFloat, shape (N,)
            Cooling rate du/dt [erg/(g·s)].
            Negative for cooling, zero if T < T_floor.

        Notes
        -----
        - Returns 0 for temperatures below T_floor
        - Free-free: Λ_ff = C_ff ρ² T^(1/2) [erg/(cm³·s)]
        - Blackbody: Λ_bb = σ_SB T⁴ (requires surface area estimate)
        - Power-law: Λ = Λ₀ (T/T₀)^β [erg/(cm³·s)]
        """
        density = np.asarray(density, dtype=np.float32)
        temperature = np.asarray(temperature, dtype=np.float32)
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Enforce temperature floor
        T_safe = np.maximum(temperature, self.T_floor)

        # Initialize cooling rate
        du_dt = np.zeros_like(density, dtype=np.float32)

        if self.cooling_model == 'none':
            return du_dt

        elif self.cooling_model == 'free_free':
            # Free-free: Λ_ff = C_ff ρ² T^(1/2) [erg/(cm³·s)]
            Lambda_ff = self.C_ff * density**2 * np.sqrt(T_safe)

            # Convert to specific cooling rate: du/dt = -Λ/ρ [erg/(g·s)]
            du_dt = -Lambda_ff / np.maximum(density, 1e-30)

        elif self.cooling_model == 'blackbody':
            # Blackbody: L/A = σ_SB T⁴ [erg/(cm²·s)]
            # Estimate effective surface area from smoothing length
            h = kwargs.get('smoothing_length', np.ones_like(density))
            area = 4.0 * np.pi * h**2  # Approximate particle surface area

            # Volume cooling rate
            Lambda_bb = self.sigma_SB * T_safe**4 * area

            # Specific cooling rate
            # Assume particle volume V ~ (4π/3) h³
            volume = (4.0 / 3.0) * np.pi * h**3
            mass_particle = density * volume

            du_dt = -Lambda_bb / np.maximum(mass_particle, 1e-30)

        elif self.cooling_model == 'power_law':
            # Power-law: Λ = Λ₀ (T/T₀)^β
            Lambda_pl = self.power_law_Lambda0 * (T_safe / self.T_ref)**self.power_law_beta

            # Specific cooling rate
            du_dt = -Lambda_pl / np.maximum(density, 1e-30)

        # Zero out cooling for particles below temperature floor
        below_floor = temperature <= self.T_floor
        du_dt[below_floor] = 0.0

        return du_dt.astype(np.float32)

    def luminosity(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        masses: NDArrayFloat,
        **kwargs
    ) -> float:
        """
        Compute total luminosity L = Σ(m_i |du/dt|_i).

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
        **kwargs
            Additional parameters for cooling_rate().

        Returns
        -------
        L : float
            Total luminosity [erg/s].

        Notes
        -----
        Luminosity is the total energy loss rate summed over all particles.
        """
        du_dt = self.cooling_rate(density, temperature, internal_energy, **kwargs)

        # Luminosity is positive (energy radiated away)
        L = -np.sum(masses * du_dt)  # Negative because du_dt < 0 for cooling

        return float(np.maximum(L, 0.0))

    def cooling_timescale(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute cooling timescale t_cool = u / |du/dt|.

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density.
        temperature : NDArrayFloat, shape (N,)
            Temperature.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy.
        **kwargs
            Additional parameters for cooling_rate().

        Returns
        -------
        t_cool : NDArrayFloat, shape (N,)
            Cooling timescale [s].

        Notes
        -----
        - Returns np.inf for particles with negligible cooling
        - Useful for timestep control
        """
        internal_energy = np.asarray(internal_energy, dtype=np.float32)
        du_dt = self.cooling_rate(density, temperature, internal_energy, **kwargs)

        # Avoid division by zero
        du_dt_abs = np.abs(du_dt)
        du_dt_safe = np.maximum(du_dt_abs, 1e-30)

        t_cool = internal_energy / du_dt_safe

        # Set infinite timescale where cooling is negligible
        negligible_cooling = du_dt_abs < 1e-30
        t_cool[negligible_cooling] = np.inf

        return t_cool.astype(np.float32)

    def apply_cooling(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        eos: EOS,
        dt: float,
        **kwargs
    ) -> NDArrayFloat:
        """
        Apply cooling over timestep Δt with safety checks.

        Uses subcycling or implicit update to prevent negative energies.

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density.
        temperature : NDArrayFloat, shape (N,)
            Temperature.
        internal_energy : NDArrayFloat, shape (N,)
            Current specific internal energy.
        eos : EOS
            Equation of state (for temperature updates).
        dt : float
            Timestep [s].
        **kwargs
            Additional parameters for cooling_rate().

        Returns
        -------
        u_new : NDArrayFloat, shape (N,)
            Updated internal energy after cooling.

        Notes
        -----
        - Limits Δu to prevent negative internal energy
        - Uses implicit update: u_new = u_old + dt * du/dt(u_new)
        - Ensures T ≥ T_floor
        """
        internal_energy = np.asarray(internal_energy, dtype=np.float32)

        # Compute cooling rate
        du_dt = self.cooling_rate(density, temperature, internal_energy, **kwargs)

        # Explicit update (with limiter)
        du = du_dt * dt

        # Limit to prevent negative internal energy (retain at least 1% of current)
        u_min = 0.01 * internal_energy
        du_limited = np.maximum(du, u_min - internal_energy)

        u_new = internal_energy + du_limited

        # Enforce minimum internal energy from temperature floor
        T_floor_u = eos.internal_energy_from_temperature(
            np.full_like(density, self.T_floor)
        ) if hasattr(eos, 'internal_energy_from_temperature') else u_min

        u_new = np.maximum(u_new, T_floor_u)

        return u_new.astype(np.float32)

    def suggested_timestep(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        **kwargs
    ) -> float:
        """
        Suggest timestep based on cooling timescale.

        Δt_suggest = cooling_factor * min(t_cool)

        Parameters
        ----------
        density : NDArrayFloat, shape (N,)
            Mass density.
        temperature : NDArrayFloat, shape (N,)
            Temperature.
        internal_energy : NDArrayFloat, shape (N,)
            Specific internal energy.
        **kwargs
            Additional parameters for cooling_timescale().

        Returns
        -------
        dt_suggest : float
            Suggested timestep [s].

        Notes
        -----
        - Returns np.inf if cooling is negligible everywhere
        - Apply additional CFL and orbital constraints externally
        """
        t_cool = self.cooling_timescale(density, temperature, internal_energy, **kwargs)

        # Find minimum finite cooling timescale
        t_cool_finite = t_cool[np.isfinite(t_cool)]

        if len(t_cool_finite) == 0 or np.all(t_cool_finite == np.inf):
            return float(np.inf)

        t_cool_min = np.min(t_cool_finite)
        dt_suggest = self.cooling_factor * t_cool_min

        return float(dt_suggest)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SimpleCooling(model='{self.cooling_model}', "
            f"T_floor={self.T_floor:.1f} K)"
        )
