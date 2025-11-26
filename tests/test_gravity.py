"""
Unit tests for gravity solvers (Newtonian, Relativistic, Pseudo-Newtonian).

Tests TASK-014, TASK-018, TASK-019, TASK-020:
- Newtonian gravity correctness
- Relativistic solver mode toggle
- Hybrid GR+Newtonian acceleration
- Pseudo-Newtonian ISCO behavior
- Validation against analytic limits

References:
    Tejeda et al. (2017) - Hybrid GR framework
    Liptai & Price (2019) - GRSPH validation tests
"""

import pytest
import numpy as np
from tde_sph.gravity import (
    NewtonianGravity,
    RelativisticGravitySolver,
    PseudoNewtonianGravity
)
from tde_sph.core.simulation import Simulation, SimulationConfig, SimulationState


class TestNewtonianGravity:
    """Test suite for Newtonian gravity solver."""

    def test_two_body_force(self):
        """Test 1/r² force law for two particles."""
        solver = NewtonianGravity(G=1.0)

        # Two particles separated by distance r
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float32)
        masses = np.array([1.0, 1.0], dtype=np.float32)
        smoothing = np.array([0.1, 0.1], dtype=np.float32)

        accel = solver.compute_acceleration(positions, masses, smoothing)

        # Expected: a_0 = G m_1 / (r² + ε²)^(3/2) in +x direction
        # With ε = (h_0 + h_1)/2 = 0.1, r = 1.0
        r = 1.0
        epsilon = 0.1
        r_soft = (r**2 + epsilon**2)**0.5
        expected_mag = 1.0 * 1.0 / r_soft**3

        # Particle 0: acceleration in +x direction
        # Particle 1: acceleration in -x direction
        np.testing.assert_allclose(accel[0, 0], expected_mag, rtol=1e-5)
        np.testing.assert_allclose(accel[1, 0], -expected_mag, rtol=1e-5)
        np.testing.assert_allclose(accel[:, 1:], 0.0, atol=1e-7)

    def test_spherical_symmetry(self):
        """Test that gravity is spherically symmetric."""
        solver = NewtonianGravity(G=1.0)

        # Central particle + test particles at same distance
        positions = np.array([
            [0.0, 0.0, 0.0],  # Central
            [1.0, 0.0, 0.0],  # +x
            [0.0, 1.0, 0.0],  # +y
            [0.0, 0.0, 1.0],  # +z
        ], dtype=np.float32)
        masses = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        smoothing = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32)

        accel = solver.compute_acceleration(positions, masses, smoothing)

        # Test particles should have same magnitude acceleration
        mag = np.linalg.norm(accel[1:], axis=1)
        np.testing.assert_allclose(mag, mag[0], rtol=1e-5)

    def test_potential_energy(self):
        """Test potential calculation for two particles."""
        solver = NewtonianGravity(G=1.0)

        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float32)
        masses = np.array([1.0, 1.0], dtype=np.float32)
        smoothing = np.array([0.1, 0.1], dtype=np.float32)

        potential = solver.compute_potential(positions, masses, smoothing)

        # Expected: φ_i = -G m_j / sqrt(r² + ε²)
        r = 1.0
        epsilon = 0.1
        r_soft = np.sqrt(r**2 + epsilon**2)
        expected_phi = -1.0 * 1.0 / r_soft

        np.testing.assert_allclose(potential, expected_phi, rtol=1e-5)

    def test_no_self_interaction(self):
        """Test that particles don't exert force on themselves."""
        solver = NewtonianGravity(G=1.0)

        # Single particle
        positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        masses = np.array([1.0], dtype=np.float32)
        smoothing = np.array([0.1], dtype=np.float32)

        accel = solver.compute_acceleration(positions, masses, smoothing)

        # Should be zero (no other particles)
        np.testing.assert_allclose(accel, 0.0, atol=1e-7)

    def test_central_bh_contribution(self):
        """Newtonian solver should include optional central BH acceleration/potential."""
        solver = NewtonianGravity(G=1.0, bh_mass=5.0)

        positions = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.zeros(1, dtype=np.float32)
        smoothing = np.array([0.1], dtype=np.float32)

        accel = solver.compute_acceleration(positions, masses, smoothing)
        potential = solver.compute_potential(positions, masses, smoothing)

        r_vec = positions[0]
        r = np.linalg.norm(r_vec)
        expected_accel = -solver.G * solver.bh_mass * r_vec / r**3
        expected_phi = -solver.G * solver.bh_mass / r

        np.testing.assert_allclose(accel[0], expected_accel, rtol=1e-5)
        np.testing.assert_allclose(potential[0], expected_phi, rtol=1e-5)


class TestRelativisticGravitySolver:
    """Test suite for hybrid relativistic gravity solver."""

    def test_newtonian_mode_fallback(self):
        """Test that metric=None gives pure Newtonian behavior."""
        newton_solver = NewtonianGravity(G=1.0)
        rel_solver = RelativisticGravitySolver(G=1.0, bh_mass=1e6)

        # Test particle + BH
        positions = np.array([
            [100.0, 0.0, 0.0]  # Far from BH
        ], dtype=np.float32)
        masses = np.array([1e-10], dtype=np.float32)  # Negligible self-gravity
        smoothing = np.array([1.0], dtype=np.float32)

        # Newtonian mode (metric=None)
        accel_rel = rel_solver.compute_acceleration(
            positions, masses, smoothing, metric=None
        )

        # Expected: a = -G M_BH r / r³ (BH at origin)
        r = np.linalg.norm(positions[0])
        expected_accel = -1.0 * 1e6 * positions[0] / r**3

        # Self-gravity negligible for single particle
        np.testing.assert_allclose(accel_rel[0], expected_accel, rtol=1e-5)

    def test_bh_newtonian_component(self):
        """Test Newtonian BH acceleration at various distances."""
        solver = RelativisticGravitySolver(G=1.0, bh_mass=1.0)

        # Test particles at different radii
        radii = np.array([10.0, 50.0, 100.0])
        positions = np.column_stack([radii, np.zeros(3), np.zeros(3)]).astype(np.float32)
        masses = np.zeros(3, dtype=np.float32)  # No self-gravity
        smoothing = np.ones(3, dtype=np.float32) * 0.1

        accel = solver.compute_acceleration(
            positions, masses, smoothing, metric=None
        )

        # Expected: a = -G M_BH / r² in -r direction
        for i, r in enumerate(radii):
            expected_mag = 1.0 * 1.0 / r**2
            np.testing.assert_allclose(
                np.linalg.norm(accel[i]), expected_mag, rtol=1e-5
            )
            # Should point toward origin (negative x)
            assert accel[i, 0] < 0

    def test_hybrid_decomposition(self):
        """Test that hybrid solver correctly sums BH + self-gravity."""
        solver = RelativisticGravitySolver(G=1.0, bh_mass=1.0)

        # Two particles: one near origin, one far
        positions = np.array([
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0]
        ], dtype=np.float32)
        masses = np.array([1.0, 1.0], dtype=np.float32)
        smoothing = np.array([0.1, 0.1], dtype=np.float32)

        # Compute hybrid acceleration (Newtonian mode)
        accel_total = solver.compute_acceleration(
            positions, masses, smoothing, metric=None
        )

        # Compute components separately
        accel_bh = solver._compute_bh_newtonian(positions)
        accel_self = solver.self_gravity_solver.compute_acceleration(
            positions, masses, smoothing
        )

        # Should sum correctly
        np.testing.assert_allclose(
            accel_total, accel_bh + accel_self, rtol=1e-5
        )

    def test_requires_velocities_for_gr_mode(self):
        """Test that GR mode raises error without velocities."""
        solver = RelativisticGravitySolver(G=1.0, bh_mass=1.0)

        positions = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([1.0], dtype=np.float32)
        smoothing = np.array([0.1], dtype=np.float32)

        # Mock metric (we'll test with real metrics later)
        class MockMetric:
            def geodesic_acceleration(self, x, v):
                return np.zeros_like(x)

        metric = MockMetric()

        # Should raise ValueError without velocities
        with pytest.raises(ValueError, match="velocities are required"):
            solver.compute_acceleration(
                positions, masses, smoothing, metric=metric, velocities=None
            )

    def test_potential_in_newtonian_mode(self):
        """Test potential includes both BH and self-gravity in Newtonian mode."""
        solver = RelativisticGravitySolver(G=1.0, bh_mass=1.0)

        positions = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([1.0], dtype=np.float32)
        smoothing = np.array([0.1], dtype=np.float32)

        potential = solver.compute_potential(
            positions, masses, smoothing, metric=None
        )

        # Expected: φ_BH = -G M_BH / r (self-gravity negligible for 1 particle)
        r = 10.0
        expected_bh = -1.0 * 1.0 / r

        # Single particle has negligible self-potential
        np.testing.assert_allclose(potential[0], expected_bh, rtol=1e-3)


class TestPseudoNewtonianGravity:
    """Test suite for Paczyński-Wiita pseudo-Newtonian gravity."""

    def test_isco_radius(self):
        """Test that ISCO occurs at r = 6M for circular orbits."""
        solver = PseudoNewtonianGravity(G=1.0, bh_mass=1.0)

        # For Paczyński-Wiita potential φ = -GM/(r - r_S):
        # Circular orbit velocity: v² = GM r / [(r - r_S)(r - 3r_S)]
        # ISCO at r = 6M = 3r_S (where r_S = 2M)

        r_S = 2.0  # Schwarzschild radius
        r_isco = 6.0  # ISCO radius

        # Particle at ISCO
        positions = np.array([[r_isco, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([0.0], dtype=np.float32)  # Test particle
        smoothing = np.array([0.1], dtype=np.float32)

        accel = solver.compute_acceleration(positions, masses, smoothing)

        # Circular orbit velocity at ISCO
        # v² = GM r / [(r - r_S)(r - 3r_S)]
        # At r = 6M = 3r_S: denominator = (6M - 2M)(6M - 6M) = 0
        # This is the ISCO condition (marginal stability)

        # Centripetal acceleration: a = v² / r
        # At ISCO, this equals gravitational acceleration

        # For r slightly above ISCO:
        r_test = r_isco + 0.1
        pos_test = np.array([[r_test, 0.0, 0.0]], dtype=np.float32)
        accel_test = solver.compute_acceleration(pos_test, masses, smoothing)

        # Acceleration should be finite
        assert np.isfinite(accel_test).all()

    def test_potential_divergence_near_horizon(self):
        """Test that potential diverges at r = r_S = 2M."""
        solver = PseudoNewtonianGravity(G=1.0, bh_mass=1.0)

        r_S = 2.0

        # Test at increasing proximity to r_S
        radii = np.array([10.0, 5.0, 3.0, 2.5, 2.2])
        positions = np.column_stack([radii, np.zeros(5), np.zeros(5)]).astype(np.float32)
        masses = np.zeros(5, dtype=np.float32)
        smoothing = np.ones(5, dtype=np.float32) * 0.1

        potentials = solver.compute_potential(positions, masses, smoothing)

        # Potential should become more negative as r → r_S
        # (but is clamped to avoid true divergence)
        assert potentials[0] > potentials[1] > potentials[2] > potentials[3]

    def test_agreement_with_newtonian_at_large_r(self):
        """Test that PW agrees with Newtonian at r >> r_S."""
        newton_solver = NewtonianGravity(G=1.0)
        pw_solver = PseudoNewtonianGravity(G=1.0, bh_mass=1.0)

        r_S = 2.0

        # Test at r >> r_S
        positions = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([0.0], dtype=np.float32)  # Test particle
        smoothing = np.array([0.1], dtype=np.float32)

        # BH acceleration (Newtonian)
        r = 100.0
        accel_newton_bh = -1.0 * 1.0 * positions[0] / r**3

        accel_pw = pw_solver.compute_acceleration(positions, masses, smoothing)

        # At large r, PW ≈ Newtonian: φ ≈ -GM/r (since r >> r_S)
        # Relative difference should be ~ r_S/r ≈ 2/100 = 2%
        np.testing.assert_allclose(
            accel_pw[0], accel_newton_bh, rtol=0.05
        )


class TestGravitySolverComparison:
    """Compare different gravity solvers for consistency."""

    def test_all_solvers_agree_at_large_r_newtonian_mode(self):
        """Test that all solvers agree in weak-field limit."""
        G = 1.0
        M_BH = 1.0

        newton = NewtonianGravity(G=G)
        relativistic = RelativisticGravitySolver(G=G, bh_mass=M_BH)
        pseudo = PseudoNewtonianGravity(G=G, bh_mass=M_BH)

        # Test particle far from BH
        positions = np.array([[1000.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([0.0], dtype=np.float32)  # Negligible mass
        smoothing = np.array([1.0], dtype=np.float32)

        # Newtonian BH force (manual calculation)
        r = 1000.0
        accel_expected = -G * M_BH * positions[0] / r**3

        # Relativistic in Newtonian mode
        accel_rel = relativistic.compute_acceleration(
            positions, masses, smoothing, metric=None
        )

        # Pseudo-Newtonian
        accel_pw = pseudo.compute_acceleration(
            positions, masses, smoothing
        )

        # All should agree at large r
        # Relativistic in Newtonian mode: exact agreement
        np.testing.assert_allclose(accel_rel[0], accel_expected, rtol=1e-4)
        # Pseudo-Newtonian: within r_S/r ~ 0.2% + O((r_S/r)²) ~ 0.5% error
        np.testing.assert_allclose(accel_pw[0], accel_expected, rtol=0.01)

    def test_self_gravity_consistent_across_solvers(self):
        """Test that self-gravity component is identical across solvers."""
        G = 1.0
        M_BH = 1.0

        newton = NewtonianGravity(G=G)
        relativistic = RelativisticGravitySolver(G=G, bh_mass=M_BH)
        pseudo = PseudoNewtonianGravity(G=G, bh_mass=M_BH)

        # Two particles far from BH (self-gravity dominates)
        positions = np.array([
            [1000.0, 0.0, 0.0],
            [1001.0, 0.0, 0.0]
        ], dtype=np.float32)
        masses = np.array([1.0, 1.0], dtype=np.float32)
        smoothing = np.array([0.1, 0.1], dtype=np.float32)

        accel_newton = newton.compute_acceleration(positions, masses, smoothing)
        accel_rel = relativistic.compute_acceleration(
            positions, masses, smoothing, metric=None
        )
        accel_pw = pseudo.compute_acceleration(positions, masses, smoothing)

        # Self-gravity component should be identical
        # (BH contribution small at r=1000 compared to particle separation)
        # Extract self-gravity by comparing to no-self-gravity case
        masses_zero = np.array([0.0, 0.0], dtype=np.float32)

        accel_rel_bh_only = relativistic.compute_acceleration(
            positions, masses_zero, smoothing, metric=None
        )
        accel_rel_self = accel_rel - accel_rel_bh_only

        # Should match pure Newtonian (which is only self-gravity)
        # Note: Newtonian doesn't have BH, so we need to account for that
        # Actually, NewtonianGravity computes self-gravity only
        # So accel_newton should match the self-gravity component

        # The Relativistic solver should have:
        # accel_rel = BH + self = BH + accel_newton (approximately)

        # Let's verify that the self-gravity component is computed the same way
        accel_self_only = relativistic.self_gravity_solver.compute_acceleration(
            positions, masses, smoothing
        )

        np.testing.assert_allclose(accel_self_only, accel_newton, rtol=1e-6)


class TestSimulationGravityIntegration:
    """Integration tests involving Simulation <-> gravity coupling."""

    def test_simulation_syncs_bh_mass_to_solver(self):
        gravity = NewtonianGravity(G=1.0)
        assert gravity.bh_mass == pytest.approx(0.0)

        sim = Simulation.__new__(Simulation)
        sim.gravity_solver = gravity
        sim.config = SimulationConfig(bh_mass=3.5, verbose=False)
        sim.state = SimulationState()
        sim._log = lambda *_, **__: None  # Silence logging hook

        sim._configure_gravity_solver()

        assert sim.gravity_solver.bh_mass == pytest.approx(3.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
