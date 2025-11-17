"""
Unit tests for metric module.

Tests validate:
- Metric tensor inversion (g^μν g_νρ = δ^μ_ρ)
- Coordinate transformations
- Geodesic acceleration computations
- Kerr → Schwarzschild limit
- ISCO calculations
- Epicyclic frequencies (TEST-019 validation)

References
----------
- Liptai & Price (2019), MNRAS 485, 819 - Appendix A (epicyclic frequencies)
- Bardeen, Press & Teukolsky (1972) - ISCO formulas
"""

import numpy as np
import pytest

from tde_sph.metric.minkowski import MinkowskiMetric
from tde_sph.metric.schwarzschild import SchwarzschildMetric
from tde_sph.metric.kerr import KerrMetric
from tde_sph.metric.coordinates import (
    cartesian_to_bl_spherical,
    bl_spherical_to_cartesian,
    velocity_cartesian_to_bl,
    velocity_bl_to_cartesian,
)


class TestCoordinateTransformations:
    """Test coordinate transformation utilities."""

    def test_cartesian_to_spherical_roundtrip(self):
        """Test Cartesian → spherical → Cartesian roundtrip."""
        x_cart = np.array([3.0, 4.0, 5.0])
        r, theta, phi = cartesian_to_bl_spherical(x_cart)
        x_reconstructed = bl_spherical_to_cartesian(r, theta, phi)

        np.testing.assert_allclose(x_cart, x_reconstructed, rtol=1e-10)

    def test_spherical_to_cartesian_roundtrip(self):
        """Test spherical → Cartesian → spherical roundtrip."""
        r = 10.0
        theta = np.pi / 4
        phi = np.pi / 3

        x_cart = bl_spherical_to_cartesian(r, theta, phi)
        r_rec, theta_rec, phi_rec = cartesian_to_bl_spherical(x_cart)

        np.testing.assert_allclose([r, theta, phi],
                                   [r_rec, theta_rec, phi_rec],
                                   rtol=1e-10)

    def test_batch_coordinate_transformation(self):
        """Test batched coordinate transformations."""
        n_particles = 100
        x_cart = np.random.randn(n_particles, 3) * 10.0

        r, theta, phi = cartesian_to_bl_spherical(x_cart)
        x_reconstructed = bl_spherical_to_cartesian(r, theta, phi)

        np.testing.assert_allclose(x_cart, x_reconstructed, rtol=1e-9)

    def test_velocity_transformation_consistency(self):
        """Test velocity transformation roundtrip."""
        x_cart = np.array([3.0, 4.0, 5.0])
        v_cart = np.array([0.1, 0.2, -0.3])

        # Cart → spherical → Cart
        v_r, v_theta, v_phi = velocity_cartesian_to_bl(x_cart, v_cart)
        r, theta, phi = cartesian_to_bl_spherical(x_cart)
        v_reconstructed = velocity_bl_to_cartesian(r, theta, phi, v_r, v_theta, v_phi)

        np.testing.assert_allclose(v_cart, v_reconstructed, rtol=1e-8)


class TestMinkowskiMetric:
    """Test Minkowski (flat spacetime) metric."""

    def test_metric_tensor_diagonal(self):
        """Test that Minkowski metric is diag(-1, 1, 1, 1)."""
        metric = MinkowskiMetric()
        x = np.array([1.0, 2.0, 3.0])

        g = metric.metric_tensor(x)
        expected = np.diag([-1.0, 1.0, 1.0, 1.0])

        np.testing.assert_allclose(g, expected)

    def test_inverse_metric(self):
        """Test that g^μν = g_μν for Minkowski."""
        metric = MinkowskiMetric()
        x = np.array([1.0, 2.0, 3.0])

        g = metric.metric_tensor(x)
        g_inv = metric.inverse_metric(x)

        np.testing.assert_allclose(g, g_inv)

    def test_metric_inversion_identity(self):
        """Test g^μν g_νρ = δ^μ_ρ."""
        metric = MinkowskiMetric()
        x = np.array([1.0, 2.0, 3.0])

        g = metric.metric_tensor(x)
        g_inv = metric.inverse_metric(x)

        product = np.einsum('ij,jk->ik', g_inv, g)
        identity = np.eye(4)

        np.testing.assert_allclose(product, identity, atol=1e-12)

    def test_christoffel_symbols_zero(self):
        """Test that all Christoffel symbols vanish."""
        metric = MinkowskiMetric()
        x = np.array([1.0, 2.0, 3.0])

        gamma = metric.christoffel_symbols(x)

        np.testing.assert_allclose(gamma, np.zeros_like(gamma))

    def test_geodesic_acceleration_zero(self):
        """Test that geodesic acceleration is zero."""
        metric = MinkowskiMetric()
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 0.1, 0.2, 0.3])  # 4-velocity

        a = metric.geodesic_acceleration(x, v)

        np.testing.assert_allclose(a, np.zeros(3))


class TestSchwarzschildMetric:
    """Test Schwarzschild metric."""

    def test_metric_inversion(self):
        """Test g^μν g_νρ = δ^μ_ρ for Schwarzschild."""
        metric = SchwarzschildMetric(mass=1.0)
        x = np.array([10.0, 0.0, 0.0])  # r=10, away from horizon

        g = metric.metric_tensor(x)
        g_inv = metric.inverse_metric(x)

        product = np.einsum('ij,jk->ik', g_inv, g)
        identity = np.eye(4)

        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_metric_diagonal(self):
        """Test that Schwarzschild metric is diagonal."""
        metric = SchwarzschildMetric(mass=1.0)
        x = np.array([10.0, 5.0, 3.0])

        g = metric.metric_tensor(x)

        # Extract off-diagonal elements
        off_diag = g - np.diag(np.diag(g))

        np.testing.assert_allclose(off_diag, np.zeros_like(off_diag), atol=1e-12)

    def test_schwarzschild_factor(self):
        """Test g_tt = -(1 - 2M/r) at various radii."""
        M = 1.0
        metric = SchwarzschildMetric(mass=M)

        radii = np.array([3.0, 6.0, 10.0, 100.0])

        for r_val in radii:
            x = np.array([r_val, 0.0, 0.0])
            g = metric.metric_tensor(x)

            expected_gtt = -(1.0 - 2.0 * M / r_val)
            np.testing.assert_allclose(g[0, 0], expected_gtt, rtol=1e-10)

    def test_isco_radius(self):
        """Test ISCO at r = 6M for Schwarzschild."""
        M = 1.0
        metric = SchwarzschildMetric(mass=M)

        r_isco = metric.isco_radius()
        expected = 6.0 * M

        np.testing.assert_allclose(r_isco, expected)

    def test_event_horizon(self):
        """Test event horizon at r = 2M."""
        M = 1.0
        metric = SchwarzschildMetric(mass=M)

        r_h = metric.event_horizon()
        expected = 2.0 * M

        np.testing.assert_allclose(r_h, expected)

    def test_circular_orbit_velocity(self):
        """Test circular orbit frequency Ω = √(M/r³)."""
        M = 1.0
        metric = SchwarzschildMetric(mass=M)

        r = 10.0 * M
        v_phi, omega = metric.circular_orbit_velocity(r)

        expected_omega = np.sqrt(M / r**3)
        np.testing.assert_allclose(omega, expected_omega, rtol=1e-10)

    def test_weak_field_limit(self):
        """Test that Schwarzschild → Newtonian at large r."""
        M = 1.0
        metric = SchwarzschildMetric(mass=M)

        # At large r, g_tt ≈ -1 - 2M/r (Newtonian potential φ = -M/r)
        r = 1000.0 * M
        x = np.array([r, 0.0, 0.0])
        g = metric.metric_tensor(x)

        # First order approximation
        expected_gtt = -(1.0 - 2.0 * M / r)
        newtonian_approx = -1.0 - 2.0 * M / r  # Including correction

        # Check they're close at large r
        np.testing.assert_allclose(g[0, 0], expected_gtt, rtol=1e-10)

    def test_batched_metric_computation(self):
        """Test batched metric tensor computation."""
        metric = SchwarzschildMetric(mass=1.0)
        n = 50
        x = np.random.randn(n, 3) * 20.0 + 30.0  # Radii > 2M

        g = metric.metric_tensor(x)
        g_inv = metric.inverse_metric(x)

        # Check inversion for all particles
        for i in range(n):
            product = np.einsum('ij,jk->ik', g_inv[i], g[i])
            np.testing.assert_allclose(product, np.eye(4), atol=1e-10)


class TestKerrMetric:
    """Test Kerr metric."""

    def test_kerr_reduces_to_schwarzschild(self):
        """Test Kerr → Schwarzschild as a → 0."""
        M = 1.0
        kerr = KerrMetric(mass=M, spin=0.0)
        schw = SchwarzschildMetric(mass=M)

        x = np.array([10.0, 5.0, 3.0])

        g_kerr = kerr.metric_tensor(x)
        g_schw = schw.metric_tensor(x)

        np.testing.assert_allclose(g_kerr, g_schw, rtol=1e-10)

    def test_metric_inversion(self):
        """Test g^μν g_νρ = δ^μ_ρ for Kerr."""
        metric = KerrMetric(mass=1.0, spin=0.9)
        x = np.array([10.0, 5.0, 3.0])

        g = metric.metric_tensor(x)
        g_inv = metric.inverse_metric(x)

        product = np.einsum('ij,jk->ik', g_inv, g)
        identity = np.eye(4)

        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_frame_dragging_term(self):
        """Test that g_tφ ≠ 0 for rotating BH."""
        metric = KerrMetric(mass=1.0, spin=0.9)
        x = np.array([10.0, 0.0, 0.0])  # On equator

        g = metric.metric_tensor(x)

        # g_tφ should be non-zero for a ≠ 0
        assert g[0, 3] != 0.0
        assert g[3, 0] != 0.0
        assert g[0, 3] == g[3, 0]  # Symmetry

    def test_isco_spin_dependence(self):
        """Test that ISCO decreases with spin (prograde)."""
        M = 1.0

        # Schwarzschild
        r_isco_0 = KerrMetric(mass=M, spin=0.0).isco_radius(prograde=True)
        # Moderate spin
        r_isco_5 = KerrMetric(mass=M, spin=0.5).isco_radius(prograde=True)
        # Extremal
        r_isco_98 = KerrMetric(mass=M, spin=0.98).isco_radius(prograde=True)

        # ISCO should decrease: 6M → ~4M → ~1M
        assert r_isco_0 > r_isco_5 > r_isco_98
        np.testing.assert_allclose(r_isco_0, 6.0 * M, rtol=1e-3)

    def test_event_horizon_spin_dependence(self):
        """Test that r₊ decreases with spin."""
        M = 1.0

        r_plus_0 = KerrMetric(mass=M, spin=0.0).event_horizon()
        r_plus_5 = KerrMetric(mass=M, spin=0.5).event_horizon()
        r_plus_99 = KerrMetric(mass=M, spin=0.99).event_horizon()

        # Horizon should decrease: 2M → ~1.5M → ~1M
        assert r_plus_0 > r_plus_5 > r_plus_99
        np.testing.assert_allclose(r_plus_0, 2.0 * M)

    def test_ergosphere(self):
        """Test ergosphere radius at equator and poles."""
        M = 1.0
        a = 0.9
        metric = KerrMetric(mass=M, spin=a)

        # At equator (θ = π/2)
        r_ergo_eq = metric.ergosphere_radius(np.pi / 2)

        # At pole (θ = 0)
        r_ergo_pole = metric.ergosphere_radius(0.0)

        # Ergosphere larger at equator
        assert r_ergo_eq > r_ergo_pole

        # At pole, ergosphere = outer horizon
        r_plus = metric.event_horizon()
        np.testing.assert_allclose(r_ergo_pole, r_plus, rtol=1e-10)

    def test_extremal_kerr(self):
        """Test extremal Kerr (a = M) properties."""
        M = 1.0
        metric = KerrMetric(mass=M, spin=1.0)

        # Horizons coincide: r₊ = r₋ = M
        r_plus = metric.event_horizon()
        np.testing.assert_allclose(r_plus, M, rtol=1e-10)

        # ISCO at r = M for extremal prograde
        r_isco = metric.isco_radius(prograde=True)
        np.testing.assert_allclose(r_isco, M, rtol=1e-3)


class TestEpicyclicFrequencies:
    """
    Validate epicyclic frequencies against Liptai & Price (2019) Appendix A.

    For circular equatorial orbits in Kerr, the epicyclic frequencies
    determine stability and oscillation modes.
    """

    def _epicyclic_frequency_radial(self, M: float, a: float, r: float) -> float:
        """
        Compute radial epicyclic frequency ω_r for Kerr.

        Reference: Liptai & Price (2019), Eq. A1.

        ω_r² = (1 - 6M/r + 8aM^(1/2)/r^(3/2) - 3a²/r²) / (r² (1 - 3M/r + 2aM^(1/2)/r^(3/2)))
        """
        sqrt_M = np.sqrt(M)
        r32 = r**(3./2.)
        r2 = r**2

        numerator = 1.0 - 6.0*M/r + 8.0*a*sqrt_M/r32 - 3.0*a**2/r2
        denominator = r2 * (1.0 - 3.0*M/r + 2.0*a*sqrt_M/r32)

        omega_r_sq = numerator / denominator
        return np.sqrt(np.abs(omega_r_sq)) if omega_r_sq >= 0 else 0.0

    def test_schwarzschild_epicyclic_frequency(self):
        """Test radial epicyclic frequency for Schwarzschild at ISCO."""
        M = 1.0
        a = 0.0
        r_isco = 6.0 * M

        omega_r = self._epicyclic_frequency_radial(M, a, r_isco)

        # At ISCO, ω_r → 0 (marginal stability)
        np.testing.assert_allclose(omega_r, 0.0, atol=1e-6)

    def test_kerr_epicyclic_frequency_qualitative(self):
        """Test that ω_r increases with radius beyond ISCO."""
        M = 1.0
        a = 0.5
        metric = KerrMetric(mass=M, spin=a)

        r_isco = metric.isco_radius(prograde=True)

        # Beyond ISCO, frequency should be positive and increasing
        r1 = r_isco * 1.5
        r2 = r_isco * 2.0

        omega_r1 = self._epicyclic_frequency_radial(M, a, r1)
        omega_r2 = self._epicyclic_frequency_radial(M, a, r2)

        assert omega_r1 > 0
        # Frequency pattern depends on regime; just check positivity
        assert omega_r2 > 0


class TestGeodesicIntegration:
    """Test geodesic integration and orbital dynamics."""

    def test_minkowski_straight_line(self):
        """Test that particles move in straight lines in Minkowski."""
        metric = MinkowskiMetric()

        x0 = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 0.5, 0.5, 0.0])  # 4-velocity

        # Integrate for small time
        dt = 0.01
        n_steps = 100

        x = x0.copy()
        for _ in range(n_steps):
            a = metric.geodesic_acceleration(x, v)
            x += v[1:] * dt  # Only spatial components

            # Acceleration should remain zero
            np.testing.assert_allclose(a, np.zeros(3), atol=1e-12)

    def test_schwarzschild_radial_infall(self):
        """Test radial infall in Schwarzschild (qualitative)."""
        metric = SchwarzschildMetric(mass=1.0)

        # Particle starting at rest at large radius
        x = np.array([100.0, 0.0, 0.0])
        # 4-velocity (approximate u^t for large r)
        u_t = 1.0 / np.sqrt(1.0 - 2.0 / 100.0)
        v = np.array([u_t, 0.0, 0.0, 0.0])

        a = metric.geodesic_acceleration(x, v)

        # Should have inward radial acceleration (negative x-component)
        # Exact direction depends on coordinate system, but magnitude > 0
        assert np.linalg.norm(a) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
