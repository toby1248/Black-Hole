#!/usr/bin/env python3
"""
Unit Conversion Utilities for TDE-SPH GUI (TASK3).

Provides conversion between dimensionless code units (geometrized G=c=1)
and physical units (CGS + astronomical conventions).

Code Unit System:
- Geometrized units: G = c = 1
- Lengths in units of gravitational radius Rg = GM/c^2
- Time in units of GM/c^3
- Mass in units of M (black hole or normalizing mass)

Physical Units:
- Lengths: solar radii (R_sun) or gravitational radii (R_g)
- Masses: solar masses (M_sun)
- Times: seconds (s)
- Energies: ergs
- Temperatures: Kelvin (K)
- Velocities: km/s or c
- Densities: g/cm^3

Author: TDE-SPH Development Team
Date: 2025-11-21
"""

from typing import Dict, Optional, Tuple
import numpy as np


# =============================================================================
# Physical Constants (CGS)
# =============================================================================

# Fundamental constants
G_CGS = 6.67430e-8  # Gravitational constant [cm^3 g^-1 s^-2]
C_CGS = 2.99792458e10  # Speed of light [cm/s]

# Solar units
M_SUN_CGS = 1.98892e33  # Solar mass [g]
R_SUN_CGS = 6.9634e10  # Solar radius [cm]
L_SUN_CGS = 3.828e33  # Solar luminosity [erg/s]

# Gravitational radius per solar mass
R_G_PER_M_SUN = G_CGS * M_SUN_CGS / (C_CGS * C_CGS)  # ~1.477e5 cm

# Boltzmann constant
K_B_CGS = 1.380649e-16  # Boltzmann constant [erg/K]

# Proton mass
M_PROTON_CGS = 1.6726219e-24  # Proton mass [g]


# =============================================================================
# Unit Type Definitions
# =============================================================================

UNIT_TYPES = {
    'length': {
        'code_label': 'code',
        'physical_label': 'R_g',
        'physical_label_alt': 'R_sun',
    },
    'mass': {
        'code_label': 'code',
        'physical_label': 'M_sun',
    },
    'time': {
        'code_label': 'code',
        'physical_label': 's',
    },
    'energy': {
        'code_label': 'code',
        'physical_label': 'erg',
    },
    'temperature': {
        'code_label': 'code',
        'physical_label': 'K',
    },
    'velocity': {
        'code_label': 'code',
        'physical_label': 'km/s',
        'physical_label_alt': 'c',
    },
    'density': {
        'code_label': 'code',
        'physical_label': 'g/cm^3',
    },
    'pressure': {
        'code_label': 'code',
        'physical_label': 'dyn/cm^2',
    },
}


# =============================================================================
# Conversion Factors Class
# =============================================================================

class ConversionFactors:
    """
    Holds precomputed conversion factors derived from simulation config.

    Attributes:
        m_bh: Black hole mass in solar masses
        m_star: Star mass in solar masses
        r_star: Star radius in solar radii
        length_to_cm: Conversion from code length to cm
        mass_to_g: Conversion from code mass to g
        time_to_s: Conversion from code time to seconds
        energy_to_erg: Conversion from code energy to ergs
        temperature_scale: Temperature scaling factor
        velocity_to_cm_s: Conversion from code velocity to cm/s
        density_to_g_cm3: Conversion from code density to g/cm^3
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize conversion factors from config.

        Parameters:
            config: Simulation configuration dictionary
        """
        self.config = config or {}

        # Extract parameters from config
        self._extract_parameters()

        # Compute conversion factors
        self._compute_factors()

    def _extract_parameters(self):
        """Extract physical parameters from config."""
        bh_config = self.config.get('black_hole', {})
        star_config = self.config.get('star', {})

        # Black hole mass [M_sun]
        self.m_bh = bh_config.get('mass', 1.0)

        # Star parameters
        self.m_star = star_config.get('mass', 1.0)  # [M_sun]
        self.r_star = star_config.get('radius', 1.0)  # [R_sun]

        # Mean molecular weight (for temperature conversion)
        eos_config = self.config.get('physics', {})
        self.mu = eos_config.get('mean_molecular_weight', 0.6)  # Ionized H/He

    def _compute_factors(self):
        """Compute all conversion factors."""
        # Normalizing mass (use BH mass if present, otherwise star mass)
        M_norm = self.m_bh if self.m_bh > 0 else self.m_star
        M_norm_cgs = M_norm * M_SUN_CGS

        # Gravitational radius (length unit)
        self.r_g_cm = G_CGS * M_norm_cgs / (C_CGS * C_CGS)

        # Length: code → cm
        self.length_to_cm = self.r_g_cm

        # Length: code → R_sun
        self.length_to_r_sun = self.r_g_cm / R_SUN_CGS

        # Mass: code → g (assuming code mass unit = M_norm)
        self.mass_to_g = M_norm_cgs

        # Time: code → s (t_g = GM/c^3)
        self.time_to_s = G_CGS * M_norm_cgs / (C_CGS * C_CGS * C_CGS)

        # Velocity: code → cm/s (v = c in geometrized units)
        self.velocity_to_cm_s = C_CGS

        # Energy: code → erg (E = Mc^2)
        self.energy_to_erg = M_norm_cgs * C_CGS * C_CGS

        # Density: code → g/cm^3
        # rho_code = M / L^3 → rho_cgs = (M_norm / r_g^3)
        self.density_to_g_cm3 = M_norm_cgs / (self.r_g_cm ** 3)

        # Pressure: code → dyn/cm^2
        # P = rho * c^2 (in geometrized units)
        self.pressure_to_dyn_cm2 = self.density_to_g_cm3 * C_CGS * C_CGS

        # Temperature scaling
        # T_code is typically dimensionless (T = k_B T_physical / (mu m_p c^2))
        # T_physical = T_code * (mu * m_p * c^2) / k_B
        self.temperature_scale = (self.mu * M_PROTON_CGS * C_CGS * C_CGS) / K_B_CGS


# =============================================================================
# Conversion Functions
# =============================================================================

def code_to_physical(value: float, unit_type: str, factors: ConversionFactors) -> float:
    """
    Convert code units to physical units.

    Parameters:
        value: Value in code units
        unit_type: Type of quantity ('length', 'mass', 'time', etc.)
        factors: ConversionFactors instance with precomputed factors

    Returns:
        Value in physical units
    """
    conversions = {
        'length': lambda v: v * factors.length_to_cm / R_SUN_CGS,  # → R_sun
        'length_rg': lambda v: v,  # Already in R_g
        'mass': lambda v: v * factors.mass_to_g / M_SUN_CGS,  # → M_sun
        'time': lambda v: v * factors.time_to_s,  # → seconds
        'velocity': lambda v: v * factors.velocity_to_cm_s / 1e5,  # → km/s
        'velocity_c': lambda v: v,  # → fraction of c
        'energy': lambda v: v * factors.energy_to_erg,  # → erg
        'density': lambda v: v * factors.density_to_g_cm3,  # → g/cm^3
        'pressure': lambda v: v * factors.pressure_to_dyn_cm2,  # → dyn/cm^2
        'temperature': lambda v: v * factors.temperature_scale,  # → K
    }

    if unit_type in conversions:
        return conversions[unit_type](value)
    else:
        # Unknown unit type, return unchanged
        return value


def physical_to_code(value: float, unit_type: str, factors: ConversionFactors) -> float:
    """
    Convert physical units to code units.

    Parameters:
        value: Value in physical units
        unit_type: Type of quantity ('length', 'mass', 'time', etc.)
        factors: ConversionFactors instance with precomputed factors

    Returns:
        Value in code units
    """
    conversions = {
        'length': lambda v: v * R_SUN_CGS / factors.length_to_cm,  # R_sun →
        'length_rg': lambda v: v,  # R_g → R_g
        'mass': lambda v: v * M_SUN_CGS / factors.mass_to_g,  # M_sun →
        'time': lambda v: v / factors.time_to_s,  # seconds →
        'velocity': lambda v: v * 1e5 / factors.velocity_to_cm_s,  # km/s →
        'velocity_c': lambda v: v,  # c → c
        'energy': lambda v: v / factors.energy_to_erg,  # erg →
        'density': lambda v: v / factors.density_to_g_cm3,  # g/cm^3 →
        'pressure': lambda v: v / factors.pressure_to_dyn_cm2,  # dyn/cm^2 →
        'temperature': lambda v: v / factors.temperature_scale,  # K →
    }

    if unit_type in conversions:
        return conversions[unit_type](value)
    else:
        return value


def get_unit_label(unit_type: str, use_physical: bool, alt: bool = False) -> str:
    """
    Get unit label for display.

    Parameters:
        unit_type: Type of quantity ('length', 'mass', etc.)
        use_physical: True for physical units, False for code units
        alt: Use alternative unit label if available

    Returns:
        Unit label string (e.g., 'R_sun', 'K', 'erg')
    """
    # Unicode symbols for nice display
    symbols = {
        'R_sun': 'R\u2609',  # R + sun symbol
        'M_sun': 'M\u2609',  # M + sun symbol
        'R_g': 'R\u2099',  # R_g (subscript g)
    }

    if not use_physical:
        return ''  # Dimensionless, no label

    labels = {
        'length': symbols.get('R_g', 'R_g') if not alt else symbols.get('R_sun', 'R_sun'),
        'length_rg': symbols.get('R_g', 'R_g'),
        'mass': symbols.get('M_sun', 'M_sun'),
        'time': 's',
        'velocity': 'km/s' if not alt else 'c',
        'velocity_c': 'c',
        'energy': 'erg',
        'density': 'g/cm\u00b3',  # superscript 3
        'pressure': 'dyn/cm\u00b2',  # superscript 2
        'temperature': 'K',
    }

    return labels.get(unit_type, '')


def format_value_with_units(value: float, unit_type: str, factors: ConversionFactors,
                            use_physical: bool, precision: int = 3) -> str:
    """
    Format value with optional unit conversion and label.

    Parameters:
        value: Value in code units
        unit_type: Type of quantity
        factors: ConversionFactors instance
        use_physical: Whether to convert to physical units
        precision: Number of significant figures

    Returns:
        Formatted string like "1.23e6 K" or "1.23e-3"
    """
    if use_physical:
        converted = code_to_physical(value, unit_type, factors)
        label = get_unit_label(unit_type, True)
        return f"{converted:.{precision}e} {label}".strip()
    else:
        return f"{value:.{precision}e}"


# =============================================================================
# Utility Functions
# =============================================================================

def get_conversion_info(config: Dict) -> str:
    """
    Get human-readable conversion information.

    Parameters:
        config: Simulation configuration

    Returns:
        Multi-line string describing unit conversions
    """
    factors = ConversionFactors(config)

    lines = [
        "Unit Conversion Factors:",
        f"  Black hole mass: {factors.m_bh:.2e} M_sun",
        f"  Gravitational radius: {factors.r_g_cm:.2e} cm = {factors.length_to_r_sun:.4e} R_sun",
        f"  Time unit: {factors.time_to_s:.2e} s",
        f"  Velocity unit: {factors.velocity_to_cm_s/1e5:.2e} km/s",
        f"  Energy unit: {factors.energy_to_erg:.2e} erg",
        f"  Density unit: {factors.density_to_g_cm3:.2e} g/cm^3",
        f"  Temperature scale: {factors.temperature_scale:.2e} K",
    ]

    return '\n'.join(lines)


def validate_config_for_conversion(config: Dict) -> Tuple[bool, str]:
    """
    Validate config has required parameters for unit conversion.

    Parameters:
        config: Simulation configuration

    Returns:
        Tuple of (is_valid, message)
    """
    warnings = []

    # Check for black hole mass
    bh_config = config.get('black_hole', {})
    if 'mass' not in bh_config:
        warnings.append("Black hole mass not specified, using default M_BH = 1.0 M_sun")

    # Check for star parameters
    star_config = config.get('star', {})
    if 'mass' not in star_config:
        warnings.append("Star mass not specified, using default M_star = 1.0 M_sun")

    is_valid = len(warnings) == 0
    message = '\n'.join(warnings) if warnings else "All conversion parameters available"

    return is_valid, message
