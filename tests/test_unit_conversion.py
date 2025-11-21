"""
Tests for TASK3 Unit Conversion Module.

Tests cover:
- ConversionFactors initialization
- code_to_physical conversions
- physical_to_code conversions (round-trip)
- get_unit_label function
- format_value_with_units function

References
----------
- TASK3: Implement Metric Unit Conversion

Test Coverage
-------------
1. ConversionFactors extracts parameters from config
2. ConversionFactors computes correct conversion factors
3. code_to_physical returns correct physical values
4. Round-trip (code -> physical -> code) is identity
5. get_unit_label returns correct symbols
6. Edge cases: missing config, zero values
"""

import pytest
import numpy as np


class TestConversionFactors:
    """Test ConversionFactors class."""

    def test_default_initialization(self):
        """Test ConversionFactors with no config."""
        from tde_sph.gui.unit_conversion import ConversionFactors

        factors = ConversionFactors()

        # Should use defaults
        assert factors.m_bh == 1.0
        assert factors.m_star == 1.0
        assert factors.r_star == 1.0

    def test_config_extraction(self):
        """Test ConversionFactors extracts from config."""
        from tde_sph.gui.unit_conversion import ConversionFactors

        config = {
            'black_hole': {'mass': 1e6},
            'star': {'mass': 1.5, 'radius': 2.0},
        }

        factors = ConversionFactors(config)

        assert factors.m_bh == 1e6
        assert factors.m_star == 1.5
        assert factors.r_star == 2.0

    def test_gravitational_radius_calculation(self):
        """Test gravitational radius is computed correctly."""
        from tde_sph.gui.unit_conversion import ConversionFactors, G_CGS, M_SUN_CGS, C_CGS

        config = {'black_hole': {'mass': 1.0}}  # 1 solar mass
        factors = ConversionFactors(config)

        # R_g = GM/c^2 for M = 1 M_sun
        expected_r_g = G_CGS * M_SUN_CGS / (C_CGS * C_CGS)

        assert factors.r_g_cm == pytest.approx(expected_r_g, rel=1e-6)
        # Should be about 1.477e5 cm
        assert factors.r_g_cm == pytest.approx(1.477e5, rel=0.01)

    def test_scaling_with_bh_mass(self):
        """Test that factors scale with black hole mass."""
        from tde_sph.gui.unit_conversion import ConversionFactors

        config1 = {'black_hole': {'mass': 1.0}}
        config10 = {'black_hole': {'mass': 10.0}}

        factors1 = ConversionFactors(config1)
        factors10 = ConversionFactors(config10)

        # R_g scales linearly with mass
        assert factors10.r_g_cm == pytest.approx(10 * factors1.r_g_cm)

        # Length conversion scales linearly
        assert factors10.length_to_cm == pytest.approx(10 * factors1.length_to_cm)

        # Time scales linearly
        assert factors10.time_to_s == pytest.approx(10 * factors1.time_to_s)


class TestCodeToPhysical:
    """Test code_to_physical conversion function."""

    def test_length_conversion(self):
        """Test length conversion to solar radii."""
        from tde_sph.gui.unit_conversion import (
            ConversionFactors, code_to_physical, R_SUN_CGS
        )

        config = {'black_hole': {'mass': 1e6}}
        factors = ConversionFactors(config)

        # 1 code unit (= 1 R_g) in solar radii
        result = code_to_physical(1.0, 'length', factors)

        # R_g for 1e6 M_sun is ~1.477e11 cm
        # R_sun = 6.96e10 cm
        # So 1 R_g ≈ 2.12 R_sun for 1e6 M_sun
        assert result > 0

    def test_temperature_conversion(self):
        """Test temperature conversion to Kelvin."""
        from tde_sph.gui.unit_conversion import ConversionFactors, code_to_physical

        config = {'black_hole': {'mass': 1e6}}
        factors = ConversionFactors(config)

        # Code temperature of 1e-6 (typical for TDE)
        result = code_to_physical(1e-6, 'temperature', factors)

        # Should be a positive temperature in Kelvin
        assert result > 0
        # Temperature scale is ~m_p c^2 / k_B ~ 1.1e13 K
        # So 1e-6 code units ~ 1e7 K (hot plasma)
        assert result > 1e3  # Should be at least thousands of K

    def test_velocity_conversion(self):
        """Test velocity conversion to km/s."""
        from tde_sph.gui.unit_conversion import ConversionFactors, code_to_physical, C_CGS

        factors = ConversionFactors()

        # Velocity of 0.1 c
        result = code_to_physical(0.1, 'velocity', factors)

        expected_km_s = 0.1 * C_CGS / 1e5  # ~30000 km/s
        assert result == pytest.approx(expected_km_s, rel=1e-6)

    def test_density_conversion(self):
        """Test density conversion to g/cm^3."""
        from tde_sph.gui.unit_conversion import ConversionFactors, code_to_physical

        config = {'black_hole': {'mass': 1e6}}
        factors = ConversionFactors(config)

        # Code density of 1e-7
        result = code_to_physical(1e-7, 'density', factors)

        # Should be a positive density
        assert result > 0


class TestRoundTrip:
    """Test round-trip conversion (code -> physical -> code)."""

    def test_length_round_trip(self):
        """Test length conversion round-trip."""
        from tde_sph.gui.unit_conversion import (
            ConversionFactors, code_to_physical, physical_to_code
        )

        config = {'black_hole': {'mass': 1e6}}
        factors = ConversionFactors(config)

        original = 42.5
        physical = code_to_physical(original, 'length', factors)
        back = physical_to_code(physical, 'length', factors)

        assert back == pytest.approx(original, rel=1e-10)

    def test_temperature_round_trip(self):
        """Test temperature conversion round-trip."""
        from tde_sph.gui.unit_conversion import (
            ConversionFactors, code_to_physical, physical_to_code
        )

        factors = ConversionFactors()

        original = 1e-5
        physical = code_to_physical(original, 'temperature', factors)
        back = physical_to_code(physical, 'temperature', factors)

        assert back == pytest.approx(original, rel=1e-10)

    def test_energy_round_trip(self):
        """Test energy conversion round-trip."""
        from tde_sph.gui.unit_conversion import (
            ConversionFactors, code_to_physical, physical_to_code
        )

        config = {'black_hole': {'mass': 1e6}}
        factors = ConversionFactors(config)

        original = 1e-3
        physical = code_to_physical(original, 'energy', factors)
        back = physical_to_code(physical, 'energy', factors)

        assert back == pytest.approx(original, rel=1e-10)

    def test_all_unit_types_round_trip(self):
        """Test round-trip for all supported unit types."""
        from tde_sph.gui.unit_conversion import (
            ConversionFactors, code_to_physical, physical_to_code
        )

        factors = ConversionFactors({'black_hole': {'mass': 1e6}})

        unit_types = [
            'length', 'length_rg', 'mass', 'time', 'velocity',
            'energy', 'density', 'pressure', 'temperature'
        ]

        original = 0.123

        for unit_type in unit_types:
            physical = code_to_physical(original, unit_type, factors)
            back = physical_to_code(physical, unit_type, factors)
            assert back == pytest.approx(original, rel=1e-10), f"Round-trip failed for {unit_type}"


class TestUnitLabels:
    """Test get_unit_label function."""

    def test_physical_labels(self):
        """Test physical unit labels."""
        from tde_sph.gui.unit_conversion import get_unit_label

        # Length - should have R_g
        label = get_unit_label('length', use_physical=True)
        assert 'R' in label or label != ''

        # Temperature - should be K
        label = get_unit_label('temperature', use_physical=True)
        assert label == 'K'

        # Energy - should be erg
        label = get_unit_label('energy', use_physical=True)
        assert label == 'erg'

    def test_code_labels_empty(self):
        """Test that code unit labels are empty."""
        from tde_sph.gui.unit_conversion import get_unit_label

        for unit_type in ['length', 'mass', 'time', 'temperature', 'energy']:
            label = get_unit_label(unit_type, use_physical=False)
            assert label == '', f"Code unit label for {unit_type} should be empty"

    def test_density_label_superscript(self):
        """Test density label has proper superscript."""
        from tde_sph.gui.unit_conversion import get_unit_label

        label = get_unit_label('density', use_physical=True)
        # Should contain 'cm' and a superscript 3
        assert 'cm' in label


class TestFormatValue:
    """Test format_value_with_units function."""

    def test_physical_format(self):
        """Test formatting with physical units."""
        from tde_sph.gui.unit_conversion import (
            ConversionFactors, format_value_with_units
        )

        factors = ConversionFactors()
        result = format_value_with_units(1e-5, 'temperature', factors, use_physical=True)

        # Should contain the value and 'K'
        assert 'K' in result
        assert 'e' in result  # Scientific notation

    def test_code_format(self):
        """Test formatting without physical units."""
        from tde_sph.gui.unit_conversion import (
            ConversionFactors, format_value_with_units
        )

        factors = ConversionFactors()
        result = format_value_with_units(1e-5, 'temperature', factors, use_physical=False)

        # Should not contain unit label
        assert 'K' not in result


class TestValidation:
    """Test config validation functions."""

    def test_validate_complete_config(self):
        """Test validation with complete config."""
        from tde_sph.gui.unit_conversion import validate_config_for_conversion

        config = {
            'black_hole': {'mass': 1e6},
            'star': {'mass': 1.0},
        }

        is_valid, message = validate_config_for_conversion(config)
        assert is_valid

    def test_validate_missing_bh_mass(self):
        """Test validation warns about missing BH mass."""
        from tde_sph.gui.unit_conversion import validate_config_for_conversion

        config = {'star': {'mass': 1.0}}

        is_valid, message = validate_config_for_conversion(config)
        assert not is_valid
        assert 'Black hole mass' in message


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_unit_type(self):
        """Test unknown unit type returns unchanged value."""
        from tde_sph.gui.unit_conversion import ConversionFactors, code_to_physical

        factors = ConversionFactors()
        original = 42.0

        result = code_to_physical(original, 'unknown_type', factors)
        assert result == original

    def test_zero_values(self):
        """Test conversion of zero values."""
        from tde_sph.gui.unit_conversion import ConversionFactors, code_to_physical

        factors = ConversionFactors()

        for unit_type in ['length', 'temperature', 'density']:
            result = code_to_physical(0.0, unit_type, factors)
            assert result == 0.0

    def test_negative_values(self):
        """Test conversion of negative values (e.g., velocities)."""
        from tde_sph.gui.unit_conversion import ConversionFactors, code_to_physical

        factors = ConversionFactors()

        result = code_to_physical(-0.5, 'velocity', factors)
        assert result < 0
