"""
Simple radiative cooling and luminosity model for TDE-SPH (TASK-023).

Implements local cooling functions and luminosity proxies for optically thin/thick gas.
Supports REQ-009: energy accounting & luminosity, REQ-011: energy transport.

This module provides simple but physically motivated cooling prescriptions suitable
for TDE simulations, balancing computational efficiency with physical accuracy.

Cooling options:
1. Optically thin bremsstrahlung (free-free emission)
2. Optically thick diffusion approximation
3. Blackbody photosphere escape
4. Viscous heating from artificial viscosity

Luminosity diagnostics:
- Total bolometric luminosity L_bol
- Luminosity vs radius L(r)
- Spectral energy distribution (simple energy bins)

References:
    Liptai & Price (2019) - GRSPH thermodynamics
    Lodato & Rossi (2011) - TDE luminosity estimates
    Dai et al. (2015) - TDE light curves
    Kippenhahn & Weigert - Stellar structure, radiative transfer
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float32]


@dataclass
class CoolingRates:
    """
    Container for cooling/heating rates.

    Attributes
    ----------
    du_dt_cooling : NDArrayFloat
        Cooling rate du/dt < 0 per particle [energy/mass/time].
    du_dt_heating : NDArrayFloat
        Heating rate du/dt > 0 per particle [energy/mass/time].
    du_dt_net : NDArrayFloat
        Net rate du/dt = heating - cooling per particle.
    luminosity_total : float
        Total bolometric luminosity [energy/time].
    """
    du_dt_cooling: NDArrayFloat
    du_dt_heating: NDArrayFloat
    du_dt_net: NDArrayFloat
    luminosity_total: float


class SimpleCoolingModel:
    """
    Simple radiative cooling and luminosity model for TDE-SPH.

    Implements multiple cooling prescriptions:
    - Optically thin free-free (bremsstrahlung) cooling
    - Optically thick radiative diffusion
    - Blackbody photosphere escape
    - Viscous heating tracking

    Parameters
    ----------
    cooling_mode : str, optional
        Cooling prescription: "bremsstrahlung", "diffusion", "blackbody", "none" (default "bremsstrahlung").
    enable_viscous_heating : bool, optional
        Track viscous heating from artificial viscosity (default True).
    cooling_timescale_limit : float, optional
        Minimum cooling timescale t_cool = u / |du/dt| (default 0.1 in code units).
        Prevents runaway cooling.
    radiation_constant : float, optional
        Radiation constant a [erg cm⁻³ K⁻⁴] (default 7.5657e-16).
    stefan_boltzmann : float, optional
        Stefan-Boltzmann constant σ [erg cm⁻² s⁻¹ K⁻⁴] (default 5.670374e-5).

    Attributes
    ----------
    cooling_mode : str
        Active cooling prescription.
    enable_viscous_heating : bool
        Whether viscous heating is tracked.
    t_cool_min : float
        Minimum cooling timescale.
    a_rad : float
        Radiation constant.
    sigma_sb : float
        Stefan-Boltzmann constant.
    cumulative_radiated_energy : float
        Total energy radiated away over simulation.

    Notes
    -----
    **Cooling prescriptions**:

    1. **Bremsstrahlung** (optically thin):
       Λ(T) ≈ 10^-27 T^(1/2) erg cm³ s⁻¹ (approximate)
       du/dt = -Λ(T) ρ / m (where m is particle mass)

    2. **Radiative diffusion** (optically thick):
       du/dt = -∇·F_rad ≈ -(16 σ T³ / (3 κ ρ)) ∇T
       Approximate as local: du/dt ∝ -(T - T_bg) / t_diff

    3. **Blackbody photosphere**:
       L ≈ 4π R_ph² σ T_ph⁴
       Requires photosphere radius estimate

    4. **None**: No cooling (adiabatic evolution)

    **Viscous heating**:
    - Tracks artificial viscosity dissipation: dE/dt = Σ Π_ij v_ij
    - Adds to internal energy as heating term

    **Luminosity**:
    - L_bol = Σ m_i |du/dt_i|_cooling (sum over cooling particles)
    - Can be binned by radius, energy, etc.
    """

    # Physical constants (CGS)
    sigma_sb_default = 5.670374e-5  # Stefan-Boltzmann [erg cm⁻² s⁻¹ K⁻⁴]
    a_rad_default = 7.5657e-16  # Radiation constant [erg cm⁻³ K⁻⁴]
    k_B = 1.380649e-16  # Boltzmann constant [erg/K]
    m_p = 1.672621e-24  # Proton mass [g]
    c_light = 2.99792458e10  # Speed of light [cm/s]

    def __init__(
        self,
        cooling_mode: str = "bremsstrahlung",
        enable_viscous_heating: bool = True,
        cooling_timescale_limit: float = 0.1,
        radiation_constant: Optional[float] = None,
        stefan_boltzmann: Optional[float] = None
    ):
        """
        Initialize cooling model.

        Parameters
        ----------
        cooling_mode : str
            "bremsstrahlung", "diffusion", "blackbody", or "none".
        enable_viscous_heating : bool
            Track viscous heating (default True).
        cooling_timescale_limit : float
            Minimum t_cool in code units (default 0.1).
        radiation_constant : float, optional
            Override default a_rad.
        stefan_boltzmann : float, optional
            Override default σ_SB.
        """
        valid_modes = ["bremsstrahlung", "diffusion", "blackbody", "none"]
        if cooling_mode not in valid_modes:
            raise ValueError(f"Invalid cooling_mode: {cooling_mode}. Must be one of {valid_modes}")

        self.cooling_mode = cooling_mode
        self.enable_viscous_heating = enable_viscous_heating
        self.t_cool_min = cooling_timescale_limit

        self.a_rad = radiation_constant if radiation_constant is not None else self.a_rad_default
        self.sigma_sb = stefan_boltzmann if stefan_boltzmann is not None else self.sigma_sb_default

        # Cumulative energy tracking
        self.cumulative_radiated_energy = 0.0

    def compute_bremsstrahlung_cooling(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute optically thin bremsstrahlung (free-free) cooling rate.

        du/dt = -Λ_ff(T) ρ / m

        where Λ_ff ≈ 1.4 × 10^-27 T^(1/2) erg cm³ s⁻¹ (fully ionized hydrogen).

        Parameters
        ----------
        density : NDArrayFloat
            Mass density ρ [code units or g/cm³].
        temperature : NDArrayFloat
            Temperature T [Kelvin].
        internal_energy : NDArrayFloat
            Specific internal energy u.

        Returns
        -------
        du_dt : NDArrayFloat
            Cooling rate du/dt ≤ 0 [energy/mass/time].

        Notes
        -----
        Bremsstrahlung cooling is appropriate for optically thin, hot gas (T > 10^6 K).
        Cooling function: Λ_ff ≈ 1.4e-27 √T erg cm³ s⁻¹ (approximate).

        For more accurate cooling, use tabulated cooling functions (e.g., Sutherland & Dopita).
        """
        # Cooling function: Λ(T) ∝ √T (simplified bremsstrahlung)
        Lambda_ff = 1.4e-27 * np.sqrt(temperature)  # erg cm³ s⁻¹

        # Cooling rate: du/dt = -Λ ρ / m (per unit mass)
        # Assuming density in code units ~ g/cm³ for this estimate
        du_dt_cooling = -Lambda_ff * density

        # Apply cooling timescale limit: |du/dt| ≤ u / t_cool_min
        u_safe = np.maximum(internal_energy, 1e-30)
        max_cooling_rate = -u_safe / self.t_cool_min
        du_dt_cooling = np.maximum(du_dt_cooling, max_cooling_rate)

        return du_dt_cooling.astype(np.float32)

    def compute_diffusion_cooling(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        background_temperature: float = 1e4
    ) -> NDArrayFloat:
        """
        Compute optically thick radiative diffusion cooling (approximate).

        du/dt ≈ -(T - T_bg) / t_diff

        where t_diff ≈ (3 κ ρ h²) / (16 σ T³) (approximate diffusion timescale).

        Parameters
        ----------
        density : NDArrayFloat
            Mass density ρ.
        temperature : NDArrayFloat
            Temperature T [K].
        internal_energy : NDArrayFloat
            Specific internal energy u.
        smoothing_lengths : NDArrayFloat
            SPH smoothing lengths h (proxy for local scale).
        background_temperature : float
            Background temperature T_bg [K] (default 10^4 K).

        Returns
        -------
        du_dt : NDArrayFloat
            Cooling rate du/dt.

        Notes
        -----
        This is a simplified diffusion approximation suitable for optically thick regions.
        For full radiative diffusion, implement FLD (flux-limited diffusion).

        Opacity κ is assumed constant (~0.4 cm²/g, electron scattering).
        """
        # Electron scattering opacity (approximate)
        kappa = 0.4  # cm²/g

        # Diffusion timescale: t_diff ~ (3 κ ρ h²) / (16 σ T³)
        T_cubed = temperature**3
        h_squared = smoothing_lengths**2
        t_diff = (3.0 * kappa * density * h_squared) / (16.0 * self.sigma_sb * T_cubed)
        t_diff = np.maximum(t_diff, self.t_cool_min)  # Floor at minimum timescale

        # Cooling rate: du/dt ≈ -(T - T_bg) / t_diff
        # Convert temperature difference to energy difference (approximate)
        # Δu ≈ c_V ΔT, where c_V ~ k_B / ((γ-1) μ m_p)
        # For simplicity, use proportionality: du/dt ∝ (T - T_bg) / t_diff
        du_dt_cooling = -(temperature - background_temperature) / t_diff

        # Prevent heating (cooling only)
        du_dt_cooling = np.minimum(du_dt_cooling, 0.0)

        # Apply cooling timescale limit
        u_safe = np.maximum(internal_energy, 1e-30)
        max_cooling_rate = -u_safe / self.t_cool_min
        du_dt_cooling = np.maximum(du_dt_cooling, max_cooling_rate)

        return du_dt_cooling.astype(np.float32)

    def compute_blackbody_cooling(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute blackbody photosphere cooling (approximate).

        L_i ≈ 4π R_i² σ T_i⁴

        where R_i ~ h_i (SPH smoothing length as effective radius).

        du/dt = -L_i / m_i

        Parameters
        ----------
        density : NDArrayFloat
            Mass density ρ.
        temperature : NDArrayFloat
            Temperature T [K].
        internal_energy : NDArrayFloat
            Specific internal energy u.
        masses : NDArrayFloat
            Particle masses m.
        smoothing_lengths : NDArrayFloat
            SPH smoothing lengths h.

        Returns
        -------
        du_dt : NDArrayFloat
            Cooling rate du/dt.

        Notes
        -----
        This is a rough approximation using h as effective photosphere radius.
        For realistic photospheres, need optical depth calculation.
        """
        # Effective radius: R_eff ~ h
        R_eff = smoothing_lengths

        # Blackbody luminosity: L = 4π R² σ T⁴
        # Use float64 for T⁴ to avoid overflow
        T_64 = temperature.astype(np.float64)
        R_64 = R_eff.astype(np.float64)
        m_64 = masses.astype(np.float64)
        L_particle_64 = 4.0 * np.pi * R_64**2 * np.float64(self.sigma_sb) * T_64**4

        # Cooling rate: du/dt = -L / m
        du_dt_cooling_64 = -L_particle_64 / m_64
        du_dt_cooling = du_dt_cooling_64.astype(np.float32)

        # Apply cooling timescale limit
        u_safe = np.maximum(internal_energy, 1e-30)
        max_cooling_rate = -u_safe / self.t_cool_min
        du_dt_cooling = np.maximum(du_dt_cooling, max_cooling_rate)

        return du_dt_cooling.astype(np.float32)

    def compute_cooling_rate(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        masses: Optional[NDArrayFloat] = None,
        smoothing_lengths: Optional[NDArrayFloat] = None,
        **kwargs
    ) -> NDArrayFloat:
        """
        Compute cooling rate du/dt using active cooling mode.

        Parameters
        ----------
        density : NDArrayFloat
            Mass density ρ.
        temperature : NDArrayFloat
            Temperature T [K].
        internal_energy : NDArrayFloat
            Specific internal energy u.
        masses : NDArrayFloat, optional
            Particle masses (required for blackbody mode).
        smoothing_lengths : NDArrayFloat, optional
            SPH smoothing lengths (required for diffusion/blackbody).
        **kwargs
            Additional parameters (e.g., background_temperature for diffusion).

        Returns
        -------
        du_dt_cooling : NDArrayFloat
            Cooling rate du/dt ≤ 0.

        Notes
        -----
        Dispatches to appropriate cooling function based on self.cooling_mode.
        """
        if self.cooling_mode == "none":
            return np.zeros_like(internal_energy, dtype=np.float32)

        elif self.cooling_mode == "bremsstrahlung":
            return self.compute_bremsstrahlung_cooling(density, temperature, internal_energy)

        elif self.cooling_mode == "diffusion":
            if smoothing_lengths is None:
                raise ValueError("Diffusion cooling requires smoothing_lengths")
            background_T = kwargs.get('background_temperature', 1e4)
            return self.compute_diffusion_cooling(
                density, temperature, internal_energy, smoothing_lengths, background_T
            )

        elif self.cooling_mode == "blackbody":
            if masses is None or smoothing_lengths is None:
                raise ValueError("Blackbody cooling requires masses and smoothing_lengths")
            return self.compute_blackbody_cooling(
                density, temperature, internal_energy, masses, smoothing_lengths
            )

        else:
            raise ValueError(f"Unknown cooling_mode: {self.cooling_mode}")

    def compute_viscous_heating_rate(
        self,
        viscous_dissipation: NDArrayFloat,
        masses: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute viscous heating rate from artificial viscosity dissipation.

        du/dt = dE_visc/dt / m

        Parameters
        ----------
        viscous_dissipation : NDArrayFloat
            Viscous dissipation rate dE/dt per particle [energy/time].
        masses : NDArrayFloat
            Particle masses.

        Returns
        -------
        du_dt_heating : NDArrayFloat
            Heating rate du/dt ≥ 0.

        Notes
        -----
        Viscous dissipation typically comes from SPH artificial viscosity:
        dE_visc/dt = (1/2) Σ_j m_j Π_ij v_ij
        where Π_ij is the artificial viscosity term.

        This module expects pre-computed dissipation from SPH solver.
        """
        du_dt_heating = viscous_dissipation / masses
        return du_dt_heating.astype(np.float32)

    def compute_net_rates(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        masses: NDArrayFloat,
        smoothing_lengths: Optional[NDArrayFloat] = None,
        viscous_dissipation: Optional[NDArrayFloat] = None,
        **kwargs
    ) -> CoolingRates:
        """
        Compute net cooling/heating rates and luminosity.

        Parameters
        ----------
        density : NDArrayFloat
            Mass density ρ.
        temperature : NDArrayFloat
            Temperature T [K].
        internal_energy : NDArrayFloat
            Specific internal energy u.
        masses : NDArrayFloat
            Particle masses m.
        smoothing_lengths : NDArrayFloat, optional
            SPH smoothing lengths (required for some cooling modes).
        viscous_dissipation : NDArrayFloat, optional
            Viscous dissipation dE/dt per particle (if heating enabled).
        **kwargs
            Additional parameters for cooling functions.

        Returns
        -------
        rates : CoolingRates
            Cooling/heating rates and total luminosity.

        Notes
        -----
        Net rate: du/dt_net = du/dt_heating - |du/dt_cooling|
        Luminosity: L_bol = Σ m_i |du/dt_i|_cooling
        """
        # Compute cooling rate
        du_dt_cooling = self.compute_cooling_rate(
            density, temperature, internal_energy, masses, smoothing_lengths, **kwargs
        )

        # Compute heating rate (if enabled)
        if self.enable_viscous_heating and viscous_dissipation is not None:
            du_dt_heating = self.compute_viscous_heating_rate(viscous_dissipation, masses)
        else:
            du_dt_heating = np.zeros_like(du_dt_cooling, dtype=np.float32)

        # Net rate
        du_dt_net = du_dt_heating + du_dt_cooling  # cooling is negative

        # Total luminosity: L = Σ m |du/dt_cooling|
        # Only count cooling (not heating)
        luminosity_total = float(np.sum(masses * np.abs(du_dt_cooling)))

        rates = CoolingRates(
            du_dt_cooling=du_dt_cooling,
            du_dt_heating=du_dt_heating,
            du_dt_net=du_dt_net,
            luminosity_total=luminosity_total
        )

        return rates

    def compute_luminosity(
        self,
        density: NDArrayFloat,
        temperature: NDArrayFloat,
        internal_energy: NDArrayFloat,
        masses: NDArrayFloat,
        positions: Optional[NDArrayFloat] = None,
        smoothing_lengths: Optional[NDArrayFloat] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute luminosity diagnostics.

        Parameters
        ----------
        density : NDArrayFloat
            Mass density ρ.
        temperature : NDArrayFloat
            Temperature T [K].
        internal_energy : NDArrayFloat
            Specific internal energy u.
        masses : NDArrayFloat
            Particle masses m.
        positions : NDArrayFloat, optional
            Particle positions (for radial binning).
        smoothing_lengths : NDArrayFloat, optional
            SPH smoothing lengths.
        **kwargs
            Additional parameters.

        Returns
        -------
        luminosity_dict : dict
            Dictionary with keys:
            - 'L_bol': Total bolometric luminosity
            - 'L_mean': Mean luminosity per particle
            - 'L_max': Maximum particle luminosity
            (Future: radial bins, spectral bins)

        Notes
        -----
        Current implementation: simple total luminosity.
        Future enhancements:
        - L(r): luminosity vs radius
        - L(E): spectral energy distribution
        - Fallback rate correlation
        """
        # Compute cooling rates
        rates = self.compute_net_rates(
            density, temperature, internal_energy, masses, smoothing_lengths, **kwargs
        )

        # Per-particle luminosity: L_i = m_i |du/dt_i|_cooling
        L_particle = masses * np.abs(rates.du_dt_cooling)

        luminosity_dict = {
            'L_bol': rates.luminosity_total,
            'L_mean': float(np.mean(L_particle)),
            'L_max': float(np.max(L_particle))
        }

        return luminosity_dict

    def update_cumulative_radiated_energy(self, dt: float, luminosity: float):
        """
        Update cumulative radiated energy.

        Parameters
        ----------
        dt : float
            Timestep duration.
        luminosity : float
            Current total luminosity L.

        Notes
        -----
        E_radiated += L × dt
        """
        self.cumulative_radiated_energy += luminosity * dt

    def __repr__(self) -> str:
        """String representation."""
        return (f"SimpleCoolingModel(mode={self.cooling_mode}, "
                f"heating={self.enable_viscous_heating}, "
                f"E_rad_cumulative={self.cumulative_radiated_energy:.2e})")
