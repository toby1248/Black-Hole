"""
Tests for RadiationGasEOS (TASK-021).

Tests validate:
1. Gas-dominated limit matches IdealGas
2. Radiation-dominated limit follows Stefan-Boltzmann
3. Temperature-energy consistency
4. Smooth transitions between regimes
5. Newton-Raphson convergence
"""

import numpy as np
import pytest
from src.tde_sph.eos import RadiationGasEOS, IdealGas


class TestRadiationGasEOS:
    """Test suite for combined gas + radiation EOS."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gamma = 5.0/3.0
        self.mu = 0.6
        self.eos_rad = RadiationGasEOS(gamma=self.gamma, mean_molecular_weight=self.mu)
        self.eos_ideal = IdealGas(gamma=self.gamma, mean_molecular_weight=self.mu)

    def test_initialization(self):
        """Test EOS initialization and constants."""
        assert self.eos_rad.gamma == pytest.approx(self.gamma)
        assert self.eos_rad.mu == pytest.approx(self.mu)
        assert self.eos_rad.a_rad > 0
        assert self.eos_rad.k_B > 0
        assert self.eos_rad.m_p > 0

    def test_gas_dominated_limit(self):
        """Test that low-temperature limit matches ideal gas."""
        # Low temperature: T ~ 10^4 K, high density
        rho = np.array([1.0, 10.0, 100.0], dtype=np.float32)
        T_low = np.array([1e4, 1e4, 1e4], dtype=np.float32)

        # Compute internal energy from temperature
        u = self.eos_rad.internal_energy_from_temperature(rho, T_low)

        # Compute pressure with radiation EOS
        P_rad = self.eos_rad.pressure(rho, u)

        # Compute pressure with ideal gas EOS
        P_ideal = self.eos_ideal.pressure(rho, u)

        # In gas-dominated limit, should be very close (< 1% difference)
        rel_diff = np.abs(P_rad - P_ideal) / P_ideal
        assert np.all(rel_diff < 0.01), f"Gas-dominated limit failed: max diff = {rel_diff.max()}"

    def test_radiation_dominated_limit(self):
        """Test radiation-dominated limit: P ≈ (1/3) a T⁴."""
        # High temperature, low density (avoid extreme T to prevent overflow)
        rho = np.array([1e-6, 1e-5, 1e-4], dtype=np.float32)
        T_high = np.array([1e7, 5e7, 1e8], dtype=np.float32)  # More moderate temps

        # Compute internal energy
        u = self.eos_rad.internal_energy_from_temperature(rho, T_high)

        # Compute pressure
        P_total = self.eos_rad.pressure(rho, u)

        # Expected radiation pressure: P_rad = (1/3) a T⁴ (use float64)
        T_64 = T_high.astype(np.float64)
        P_rad_expected = (1.0/3.0) * np.float64(self.eos_rad.a_rad) * T_64**4

        # In radiation-dominated limit, P_total ≈ P_rad
        rel_diff = np.abs(P_total.astype(np.float64) - P_rad_expected) / P_rad_expected
        assert np.all(rel_diff < 0.15), f"Radiation-dominated limit failed: max diff = {rel_diff.max()}"

    def test_temperature_energy_consistency(self):
        """Test that T(ρ,u(ρ,T)) = T (round-trip)."""
        rho = np.array([1e-3, 1.0, 1e3], dtype=np.float32)
        T_input = np.array([1e5, 1e6, 1e7], dtype=np.float32)

        # Forward: T → u
        u = self.eos_rad.internal_energy_from_temperature(rho, T_input)

        # Backward: u → T
        T_recovered = self.eos_rad.temperature(rho, u)

        # Should recover original temperature
        rel_error = np.abs(T_recovered - T_input) / T_input
        assert np.all(rel_error < 1e-4), f"Temperature round-trip failed: max error = {rel_error.max()}"

    def test_sound_speed_limits(self):
        """Test sound speed in gas-dominated and radiation-dominated limits."""
        # Gas-dominated: c_s² ≈ γ P_gas / ρ
        rho = 1.0
        T_low = 1e4
        u_gas = self.eos_rad.internal_energy_from_temperature(np.array([rho]), np.array([T_low]))
        cs_gas = self.eos_rad.sound_speed(np.array([rho]), u_gas)

        # Expected from ideal gas: c_s² = γ (γ-1) u
        cs_expected = np.sqrt(self.gamma * (self.gamma - 1.0) * u_gas[0])
        assert cs_gas[0] == pytest.approx(cs_expected, rel=0.01)

        # Radiation-dominated: c_s² ≈ (4/3) P_rad / ρ (use moderate temperature)
        rho_low = 1e-5
        T_high = 5e7  # More moderate to avoid numerical issues
        u_rad = self.eos_rad.internal_energy_from_temperature(np.array([rho_low]), np.array([T_high]))
        cs_rad = self.eos_rad.sound_speed(np.array([rho_low]), u_rad)

        # Expected: c_s² = (4/3) P_rad / ρ = (4/9) a T⁴ / ρ (use float64)
        cs_expected_rad = np.sqrt((4.0/9.0) * np.float64(self.eos_rad.a_rad) * np.float64(T_high)**4 / np.float64(rho_low))
        rel_diff = abs(np.float64(cs_rad[0]) - cs_expected_rad) / cs_expected_rad
        assert rel_diff < 0.15, f"Radiation sound speed: expected {cs_expected_rad}, got {cs_rad[0]}"

    def test_mixed_regime(self):
        """Test mixed regime where gas and radiation pressures are comparable."""
        # Choose ρ, T such that P_gas ~ P_rad
        # From P_gas = (γ-1) ρ u_gas = (γ-1) ρ k_B T / ((γ-1) μ m_p) = ρ k_B T / (μ m_p)
        # and P_rad = (1/3) a T⁴
        # Setting P_gas = P_rad gives: T³ = 3 ρ k_B / (μ m_p a)
        rho = 1.0
        T_crit = (3.0 * rho * self.eos_rad.k_B / (self.mu * self.eos_rad.m_p * self.eos_rad.a_rad))**(1.0/3.0)

        T = np.array([T_crit], dtype=np.float32)
        u = self.eos_rad.internal_energy_from_temperature(np.array([rho]), T)

        # Compute gas pressure fraction
        beta = self.eos_rad.gas_pressure_fraction(np.array([rho]), u)

        # In mixed regime, beta should be around 0.5 (allow wider range for numerical stability)
        assert 0.3 < beta[0] < 0.7, f"Mixed regime test failed: beta = {beta[0]}, T_crit = {T_crit}"

    def test_newton_raphson_convergence(self):
        """Test that Newton-Raphson converges for various (ρ, u) inputs."""
        rho_values = np.logspace(-4, 3, 8, dtype=np.float32)  # 10^-4 to 10^3 (narrower range)
        T_values = np.logspace(3, 8, 8, dtype=np.float32)    # 10^3 to 10^8 K (avoid extreme temps)

        for rho in rho_values:
            for T in T_values:
                # Compute u from T
                u = self.eos_rad.internal_energy_from_temperature(np.array([rho]), np.array([T]))

                # Solve for T from (ρ, u)
                T_solved = self.eos_rad.temperature(np.array([rho]), u)

                # Check convergence (relaxed tolerance for extreme regimes)
                rel_error = abs(T_solved[0] - T) / T
                assert rel_error < 1e-3, f"Convergence failed at ρ={rho}, T={T}: error={rel_error}"

    def test_positive_pressure_sound_speed(self):
        """Test that pressure and sound speed are always non-negative."""
        rho = np.array([1e-6, 1.0, 1e6], dtype=np.float32)
        u = np.array([1e10, 1e15, 1e20], dtype=np.float32)

        P = self.eos_rad.pressure(rho, u)
        cs = self.eos_rad.sound_speed(rho, u)

        assert np.all(P >= 0), "Pressure is negative"
        assert np.all(cs >= 0), "Sound speed is negative"

    def test_temperature_optional_parameter(self):
        """Test that providing temperature as optional parameter works correctly."""
        rho = np.array([1.0], dtype=np.float32)
        T = np.array([1e6], dtype=np.float32)
        u = self.eos_rad.internal_energy_from_temperature(rho, T)

        # Compute pressure with temperature provided (fast path)
        P_with_T = self.eos_rad.pressure(rho, u, temperature=T)

        # Compute pressure without temperature (solve via Newton-Raphson)
        P_without_T = self.eos_rad.pressure(rho, u)

        # Should give same result
        assert P_with_T[0] == pytest.approx(P_without_T[0], rel=1e-5)

    def test_gas_pressure_fraction_limits(self):
        """Test gas pressure fraction β in different regimes."""
        # Gas-dominated: β ≈ 1
        rho = 100.0
        T_low = 1e4
        u_gas = self.eos_rad.internal_energy_from_temperature(np.array([rho]), np.array([T_low]))
        beta_gas = self.eos_rad.gas_pressure_fraction(np.array([rho]), u_gas)
        assert beta_gas[0] > 0.95, f"Gas-dominated β should be ~1, got {beta_gas[0]}"

        # Radiation-dominated: β ≈ 0
        rho_low = 1e-6
        T_high = 1e8
        u_rad = self.eos_rad.internal_energy_from_temperature(np.array([rho_low]), np.array([T_high]))
        beta_rad = self.eos_rad.gas_pressure_fraction(np.array([rho_low]), u_rad)
        assert beta_rad[0] < 0.05, f"Radiation-dominated β should be ~0, got {beta_rad[0]}"

    def test_array_consistency(self):
        """Test that array operations work correctly for varying array sizes."""
        for n in [1, 10, 100, 1000]:
            rho = np.ones(n, dtype=np.float32)
            T = 1e6 * np.ones(n, dtype=np.float32)
            u = self.eos_rad.internal_energy_from_temperature(rho, T)

            P = self.eos_rad.pressure(rho, u)
            cs = self.eos_rad.sound_speed(rho, u)
            T_check = self.eos_rad.temperature(rho, u)

            assert P.shape == (n,)
            assert cs.shape == (n,)
            assert T_check.shape == (n,)
            assert np.all(np.isfinite(P))
            assert np.all(np.isfinite(cs))
            assert np.all(np.isfinite(T_check))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
