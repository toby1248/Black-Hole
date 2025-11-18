"""
Tests for SimpleCoolingModel (TASK-023).

Tests validate:
1. Initialization with different cooling modes
2. Bremsstrahlung cooling physics (optically thin)
3. Diffusion cooling physics (optically thick)
4. Blackbody cooling physics
5. Viscous heating integration
6. Net rates calculation
7. Luminosity diagnostics
8. Cumulative energy tracking
9. Cooling timescale limits
10. Mode switching
"""

import numpy as np
import pytest
from src.tde_sph.radiation import SimpleCoolingModel, CoolingRates


class TestSimpleCoolingModel:
    """Test suite for radiative cooling model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rho = np.array([1e-6, 1e-3, 1.0, 1e3], dtype=np.float32)
        self.T = np.array([1e6, 1e7, 1e8, 1e9], dtype=np.float32)
        self.u = np.array([1e14, 1e15, 1e16, 1e17], dtype=np.float32)
        self.m = np.ones(4, dtype=np.float32) * 1e30  # g
        self.h = np.array([1e10, 5e9, 1e9, 5e8], dtype=np.float32)  # cm

    def test_initialization_bremsstrahlung(self):
        """Test initialization with bremsstrahlung cooling."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung")
        assert model.cooling_mode == "bremsstrahlung"
        assert model.enable_viscous_heating is True
        assert model.t_cool_min == pytest.approx(0.1)
        assert model.cumulative_radiated_energy == 0.0

    def test_initialization_diffusion(self):
        """Test initialization with diffusion cooling."""
        model = SimpleCoolingModel(cooling_mode="diffusion")
        assert model.cooling_mode == "diffusion"

    def test_initialization_blackbody(self):
        """Test initialization with blackbody cooling."""
        model = SimpleCoolingModel(cooling_mode="blackbody")
        assert model.cooling_mode == "blackbody"

    def test_initialization_none(self):
        """Test initialization with no cooling."""
        model = SimpleCoolingModel(cooling_mode="none")
        assert model.cooling_mode == "none"

    def test_initialization_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid cooling_mode"):
            SimpleCoolingModel(cooling_mode="invalid_mode")

    def test_bremsstrahlung_cooling_physics(self):
        """Test bremsstrahlung cooling rate scales as √T."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung")

        # Test at different temperatures
        rho = np.array([1.0, 1.0], dtype=np.float32)
        T_low = np.array([1e6, 4e6], dtype=np.float32)  # T2 = 4 × T1
        u = np.array([1e15, 1e15], dtype=np.float32)

        du_dt = model.compute_bremsstrahlung_cooling(rho, T_low, u)

        # Cooling should scale as √T: Λ(4T) / Λ(T) = 2
        ratio = du_dt[1] / du_dt[0]
        assert ratio == pytest.approx(2.0, rel=0.01), f"Expected 2.0, got {ratio}"

        # Cooling rate should be negative
        assert np.all(du_dt <= 0), "Cooling rate should be non-positive"

    def test_bremsstrahlung_timescale_limit(self):
        """Test that bremsstrahlung cooling respects timescale limit."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung", cooling_timescale_limit=1.0)

        # Very low internal energy (should trigger limit)
        rho = np.array([1.0], dtype=np.float32)
        T = np.array([1e8], dtype=np.float32)
        u_low = np.array([1e10], dtype=np.float32)  # Very small

        du_dt = model.compute_bremsstrahlung_cooling(rho, T, u_low)

        # Check that |du/dt| ≤ u / t_cool_min
        expected_max = -u_low[0] / model.t_cool_min
        assert du_dt[0] >= expected_max, f"Cooling exceeded limit: {du_dt[0]} < {expected_max}"

    def test_diffusion_cooling_temperature_dependence(self):
        """Test diffusion cooling depends on temperature difference."""
        model = SimpleCoolingModel(cooling_mode="diffusion")

        rho = np.array([1.0, 1.0], dtype=np.float32)
        T_high = np.array([1e7, 2e7], dtype=np.float32)
        u = np.array([1e15, 1e15], dtype=np.float32)
        h = np.array([1e10, 1e10], dtype=np.float32)
        T_bg = 1e4

        du_dt = model.compute_diffusion_cooling(rho, T_high, u, h, T_bg)

        # Higher temperature should give stronger cooling (larger T - T_bg)
        assert du_dt[1] < du_dt[0], "Higher T should give stronger cooling"
        assert np.all(du_dt <= 0), "Diffusion cooling should be non-positive"

    def test_diffusion_no_heating(self):
        """Test that diffusion cooling doesn't heat gas below background T."""
        model = SimpleCoolingModel(cooling_mode="diffusion")

        # Gas below background temperature
        rho = np.array([1.0], dtype=np.float32)
        T_low = np.array([1e3], dtype=np.float32)  # Below T_bg = 1e4
        u = np.array([1e15], dtype=np.float32)
        h = np.array([1e10], dtype=np.float32)

        du_dt = model.compute_diffusion_cooling(rho, T_low, u, h, background_temperature=1e4)

        # Should not heat (du/dt <= 0 always)
        assert du_dt[0] <= 0, f"Should not heat: du/dt = {du_dt[0]}"

    def test_blackbody_cooling_stefan_boltzmann(self):
        """Test blackbody cooling scales as T⁴."""
        # Use larger timescale limit to avoid clamping
        model = SimpleCoolingModel(cooling_mode="blackbody", cooling_timescale_limit=1e5)

        # Test T⁴ scaling
        rho = np.array([1.0, 1.0], dtype=np.float32)
        T = np.array([1e6, 2e6], dtype=np.float32)  # T2 = 2 × T1
        u = np.array([1e18, 1e18], dtype=np.float32)  # Higher u to avoid limit
        m = np.array([1e30, 1e30], dtype=np.float32)
        h = np.array([1e10, 1e10], dtype=np.float32)

        du_dt = model.compute_blackbody_cooling(rho, T, u, m, h)

        # L ∝ T⁴, so du/dt ∝ T⁴ (for fixed m, h)
        # Expect du_dt[1] / du_dt[0] ≈ 2⁴ = 16
        ratio = du_dt[1] / du_dt[0]
        assert ratio == pytest.approx(16.0, rel=0.01), f"Expected 16.0, got {ratio}"

    def test_blackbody_cooling_area_dependence(self):
        """Test blackbody cooling depends on surface area (h²)."""
        # Use larger timescale limit to avoid clamping
        model = SimpleCoolingModel(cooling_mode="blackbody", cooling_timescale_limit=1e6)

        rho = np.array([1.0, 1.0], dtype=np.float32)
        T = np.array([1e7, 1e7], dtype=np.float32)
        u = np.array([1e22, 1e22], dtype=np.float32)  # Very high u to avoid limit
        m = np.array([1e30, 1e30], dtype=np.float32)
        h = np.array([1e10, 2e10], dtype=np.float32)  # h2 = 2 × h1

        du_dt = model.compute_blackbody_cooling(rho, T, u, m, h)

        # L ∝ R² ∝ h², so du/dt ∝ h² / m ∝ h² (for fixed m)
        # Expect du_dt[1] / du_dt[0] ≈ 4
        ratio = du_dt[1] / du_dt[0]
        assert ratio == pytest.approx(4.0, rel=0.01), f"Expected 4.0, got {ratio}"

    def test_viscous_heating_rate(self):
        """Test viscous heating computation."""
        model = SimpleCoolingModel(cooling_mode="none", enable_viscous_heating=True)

        # Viscous dissipation: dE/dt per particle
        visc_diss = np.array([1e40, 2e40, 3e40], dtype=np.float32)  # erg/s
        masses = np.array([1e30, 1e30, 1e30], dtype=np.float32)  # g

        du_dt_heating = model.compute_viscous_heating_rate(visc_diss, masses)

        # Should be: du/dt = dE/dt / m
        expected = visc_diss / masses
        np.testing.assert_array_almost_equal(du_dt_heating, expected, decimal=5)

        # Heating rate should be positive
        assert np.all(du_dt_heating >= 0), "Heating rate should be non-negative"

    def test_net_rates_cooling_only(self):
        """Test net rates with cooling only (no heating)."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung", enable_viscous_heating=False)

        rho = np.array([1.0], dtype=np.float32)
        T = np.array([1e7], dtype=np.float32)
        u = np.array([1e15], dtype=np.float32)
        m = np.array([1e30], dtype=np.float32)

        rates = model.compute_net_rates(rho, T, u, m)

        # Check structure
        assert isinstance(rates, CoolingRates)
        assert rates.du_dt_cooling[0] < 0, "Cooling should be negative"
        assert rates.du_dt_heating[0] == 0, "No heating enabled"
        assert rates.du_dt_net[0] == rates.du_dt_cooling[0], "Net = cooling only"
        assert rates.luminosity_total > 0, "Luminosity should be positive"

    def test_net_rates_cooling_and_heating(self):
        """Test net rates with both cooling and heating."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung", enable_viscous_heating=True)

        rho = np.array([1.0], dtype=np.float32)
        T = np.array([1e7], dtype=np.float32)
        u = np.array([1e15], dtype=np.float32)
        m = np.array([1e30], dtype=np.float32)
        visc_diss = np.array([1e40], dtype=np.float32)

        rates = model.compute_net_rates(rho, T, u, m, viscous_dissipation=visc_diss)

        # Check that heating is positive
        assert rates.du_dt_heating[0] > 0, "Heating should be positive"

        # Net should be sum (cooling is negative, heating is positive)
        expected_net = rates.du_dt_heating[0] + rates.du_dt_cooling[0]
        assert rates.du_dt_net[0] == pytest.approx(expected_net), "Net rate incorrect"

    def test_luminosity_computation(self):
        """Test luminosity diagnostics."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung")

        rho = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        T = np.array([1e7, 1e7, 1e7], dtype=np.float32)
        u = np.array([1e15, 1e15, 1e15], dtype=np.float32)
        m = np.array([1e30, 2e30, 3e30], dtype=np.float32)

        lum_dict = model.compute_luminosity(rho, T, u, m)

        # Check keys
        assert 'L_bol' in lum_dict
        assert 'L_mean' in lum_dict
        assert 'L_max' in lum_dict

        # Luminosity should be positive
        assert lum_dict['L_bol'] > 0
        assert lum_dict['L_mean'] > 0
        assert lum_dict['L_max'] >= lum_dict['L_mean']

    def test_cumulative_energy_tracking(self):
        """Test cumulative radiated energy tracking."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung")

        # Initial energy should be zero
        assert model.cumulative_radiated_energy == 0.0

        # Simulate timesteps
        dt = 1.0
        L1 = 1e45  # erg/s
        L2 = 2e45

        model.update_cumulative_radiated_energy(dt, L1)
        assert model.cumulative_radiated_energy == pytest.approx(L1 * dt)

        model.update_cumulative_radiated_energy(dt, L2)
        assert model.cumulative_radiated_energy == pytest.approx((L1 + L2) * dt)

    def test_cooling_mode_none(self):
        """Test that 'none' mode produces zero cooling."""
        model = SimpleCoolingModel(cooling_mode="none")

        rho = np.array([1.0], dtype=np.float32)
        T = np.array([1e7], dtype=np.float32)
        u = np.array([1e15], dtype=np.float32)

        du_dt = model.compute_cooling_rate(rho, T, u)

        # Should be zero
        assert du_dt[0] == 0.0

    def test_diffusion_requires_smoothing_lengths(self):
        """Test that diffusion mode raises error without smoothing lengths."""
        model = SimpleCoolingModel(cooling_mode="diffusion")

        rho = np.array([1.0], dtype=np.float32)
        T = np.array([1e7], dtype=np.float32)
        u = np.array([1e15], dtype=np.float32)

        with pytest.raises(ValueError, match="requires smoothing_lengths"):
            model.compute_cooling_rate(rho, T, u, smoothing_lengths=None)

    def test_blackbody_requires_masses_and_smoothing(self):
        """Test that blackbody mode requires masses and smoothing lengths."""
        model = SimpleCoolingModel(cooling_mode="blackbody")

        rho = np.array([1.0], dtype=np.float32)
        T = np.array([1e7], dtype=np.float32)
        u = np.array([1e15], dtype=np.float32)

        # Missing both
        with pytest.raises(ValueError, match="requires masses and smoothing_lengths"):
            model.compute_cooling_rate(rho, T, u)

        # Missing smoothing_lengths
        m = np.array([1e30], dtype=np.float32)
        with pytest.raises(ValueError, match="requires masses and smoothing_lengths"):
            model.compute_cooling_rate(rho, T, u, masses=m)

    def test_array_consistency(self):
        """Test that array operations work for different sizes."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung")

        for n in [1, 10, 100]:
            rho = np.ones(n, dtype=np.float32)
            T = 1e7 * np.ones(n, dtype=np.float32)
            u = 1e15 * np.ones(n, dtype=np.float32)
            m = 1e30 * np.ones(n, dtype=np.float32)

            du_dt = model.compute_cooling_rate(rho, T, u)
            lum_dict = model.compute_luminosity(rho, T, u, m)

            assert du_dt.shape == (n,)
            assert np.all(np.isfinite(du_dt))
            assert lum_dict['L_bol'] > 0

    def test_constants(self):
        """Test that physical constants are correctly set."""
        model = SimpleCoolingModel()

        # Check constants (CGS)
        assert model.sigma_sb == pytest.approx(5.670374e-5)  # Stefan-Boltzmann
        assert model.a_rad == pytest.approx(7.5657e-16)  # Radiation constant
        assert model.k_B == pytest.approx(1.380649e-16)  # Boltzmann
        assert model.m_p == pytest.approx(1.672621e-24)  # Proton mass
        assert model.c_light == pytest.approx(2.99792458e10)  # Speed of light

    def test_custom_constants(self):
        """Test initialization with custom constants."""
        custom_a = 1e-15
        custom_sigma = 1e-4

        model = SimpleCoolingModel(
            radiation_constant=custom_a,
            stefan_boltzmann=custom_sigma
        )

        assert model.a_rad == pytest.approx(custom_a)
        assert model.sigma_sb == pytest.approx(custom_sigma)

    def test_repr(self):
        """Test string representation."""
        model = SimpleCoolingModel(cooling_mode="bremsstrahlung")
        repr_str = repr(model)

        assert "SimpleCoolingModel" in repr_str
        assert "bremsstrahlung" in repr_str
        assert "E_rad_cumulative" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
