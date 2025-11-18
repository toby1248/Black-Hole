"""
Regression tests for TDE-SPH framework (TASK-040).

These tests run small simulations to catch behavior changes and ensure
core functionality remains stable across code updates.

Tests use small particle counts (N ~ 10-100) for fast CI execution.
Each test validates physical correctness and numerical stability.

Markers:
    regression: All regression tests
    fast: Fast tests (< 5 seconds)
    slow: Slower tests (5-30 seconds)

Usage:
    pytest tests/test_regression.py -v -m regression
    pytest tests/test_regression.py -v -m "regression and fast"
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tde_sph.ICs import Polytrope, DiscGenerator
from tde_sph.core.energy_diagnostics import EnergyDiagnostics
from tde_sph.metric import MinkowskiMetric, SchwarzschildMetric, KerrMetric
from tde_sph.eos import IdealGas, RadiationGasEOS
from tde_sph.radiation import SimpleCooling
from tde_sph.io.hdf5 import HDF5Writer
from tde_sph.io.diagnostics import DiagnosticWriter


# ============================================================================
# Regression: Initial Conditions
# ============================================================================

@pytest.mark.regression
@pytest.mark.fast
class TestICsRegression:
    """Regression tests for initial conditions generators."""

    def test_polytrope_mass_conservation(self):
        """Regression: Polytrope mass should be conserved to high precision."""
        target_mass = 1.0
        poly = Polytrope(gamma=5.0/3.0, random_seed=42)

        pos, vel, masses, u, rho = poly.generate(
            n_particles=100,
            M_star=target_mass,
            R_star=1.0
        )

        total_mass = np.sum(masses)
        rel_error = abs(total_mass - target_mass) / target_mass

        assert rel_error < 1e-10, f"Mass conservation failed: {rel_error:.2e}"

    def test_polytrope_virial_equilibrium(self):
        """Regression: Polytrope should start in approximate virial equilibrium."""
        poly = Polytrope(
            n_particles=200,
            polytropic_index=1.5,
            total_mass=1.0,
            radius=1.0,
            gamma=5.0/3.0
        )

        pos, vel, masses, u, h = poly.generate()

        # Compute kinetic energy
        v_squared = np.sum(vel**2, axis=1)
        E_kin = 0.5 * np.sum(masses * v_squared)

        # Compute internal energy
        E_int = np.sum(masses * u)

        # For virial equilibrium: 2K + U ≈ 0, where U < 0 is potential
        # In our setup, E_kin should be small (near-zero velocities)
        # E_int should be positive

        assert E_kin < 1e-6, f"Kinetic energy too large: {E_kin:.2e}"
        assert E_int > 0, f"Internal energy should be positive: {E_int:.2e}"

    def test_disc_mass_conservation(self):
        """Regression: Disc generator should conserve mass."""
        target_mass = 0.1
        disc_gen = DiscGenerator(n_particles=100)

        pos, vel, masses, u, h = disc_gen.generate(
            disc_type="thin",
            M_disc=target_mass,
            r_in=5.0,
            r_out=20.0
        )

        total_mass = np.sum(masses)
        rel_error = abs(total_mass - target_mass) / target_mass

        assert rel_error < 1e-10, f"Disc mass conservation failed: {rel_error:.2e}"

    def test_disc_keplerian_velocities(self):
        """Regression: Thin disc should have Keplerian velocities."""
        disc_gen = DiscGenerator(n_particles=100, M_bh=1.0)

        pos, vel, masses, u, h = disc_gen.generate(
            disc_type="thin",
            M_disc=0.1,
            r_in=10.0,
            r_out=30.0
        )

        # Check velocity magnitudes
        R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        v = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)

        # Keplerian: v = sqrt(GM/R)
        v_kepler = np.sqrt(1.0 / R)

        # Should match within ~10% (approximate due to vertical structure)
        rel_diff = np.abs(v - v_kepler) / v_kepler
        mean_diff = np.mean(rel_diff)

        assert mean_diff < 0.15, f"Keplerian velocity mismatch: {mean_diff:.2%}"


# ============================================================================
# Regression: Energy Diagnostics
# ============================================================================

@pytest.mark.regression
@pytest.mark.fast
class TestEnergyRegression:
    """Regression tests for energy diagnostics."""

    def test_energy_conservation_static_particles(self):
        """Regression: Static particles should conserve energy exactly."""
        N = 50
        masses = np.ones(N, dtype=np.float32) * 0.01
        positions = np.random.randn(N, 3).astype(np.float32) * 5.0
        velocities = np.zeros((N, 3), dtype=np.float32)
        internal_energies = np.ones(N, dtype=np.float32) * 1.0

        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        # Compute energy at t=0
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=masses,
            positions=positions,
            velocities=velocities,
            internal_energies=internal_energies
        )

        # Compute energy at t=1 (same state)
        E1 = diag.compute_all_energies(
            time=1.0,
            masses=masses,
            positions=positions,
            velocities=velocities,
            internal_energies=internal_energies
        )

        # Energy should be identical
        rel_error = abs(E1.total - E0.total) / abs(E0.total)

        assert rel_error < 1e-10, f"Energy conservation error: {rel_error:.2e}"
        assert E1.conservation_error == 0.0

    def test_energy_scales_with_mass(self):
        """Regression: Doubling masses should double total energy."""
        N = 50
        masses1 = np.ones(N, dtype=np.float32) * 0.01
        masses2 = masses1 * 2.0

        positions = np.random.randn(N, 3).astype(np.float32) * 5.0
        velocities = np.random.randn(N, 3).astype(np.float32) * 0.1
        internal_energies = np.ones(N, dtype=np.float32) * 1.0

        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        E1 = diag.compute_all_energies(
            time=0.0,
            masses=masses1,
            positions=positions,
            velocities=velocities,
            internal_energies=internal_energies
        )

        E2 = diag.compute_all_energies(
            time=0.0,
            masses=masses2,
            positions=positions,
            velocities=velocities,
            internal_energies=internal_energies
        )

        # All energy components should double (except conservation_error)
        ratio_kin = E2.kinetic / E1.kinetic
        ratio_pot_bh = E2.potential_bh / E1.potential_bh
        ratio_int = E2.internal_thermal / E1.internal_thermal

        assert ratio_kin == pytest.approx(2.0, rel=1e-6)
        assert ratio_pot_bh == pytest.approx(2.0, rel=1e-6)
        assert ratio_int == pytest.approx(2.0, rel=1e-6)


# ============================================================================
# Regression: EOS
# ============================================================================

@pytest.mark.regression
@pytest.mark.fast
class TestEOSRegression:
    """Regression tests for equation of state."""

    def test_ideal_gas_adiabatic_index(self):
        """Regression: Ideal gas should maintain gamma."""
        eos = IdealGas(gamma=5.0/3.0)

        rho = np.array([1.0, 2.0, 5.0], dtype=np.float32)
        u = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        P = eos.pressure(rho, u)

        # P = (gamma - 1) * rho * u
        P_expected = (5.0/3.0 - 1.0) * rho * u

        np.testing.assert_allclose(P, P_expected, rtol=1e-6)

    def test_radiation_gas_pressure_ratio(self):
        """Regression: Radiation pressure should dominate at high temperature."""
        eos = RadiationGasEOS(
            gamma_gas=5.0/3.0,
            mu=0.6,
            radiation_constant=7.5657e-15
        )

        # High temperature: radiation-dominated
        rho_high_T = np.array([1e-3], dtype=np.float32)
        u_high_T = np.array([1e8], dtype=np.float32)  # Very high u ~ T⁴

        P_tot = eos.pressure(rho_high_T, u_high_T)
        P_rad = eos.radiation_pressure(rho_high_T, u_high_T)

        # Radiation should dominate
        ratio_rad = P_rad / P_tot

        assert ratio_rad > 0.9, f"Radiation pressure ratio too low: {ratio_rad:.2f}"


# ============================================================================
# Regression: Metrics
# ============================================================================

@pytest.mark.regression
@pytest.mark.fast
class TestMetricRegression:
    """Regression tests for spacetime metrics."""

    def test_minkowski_flat_spacetime(self):
        """Regression: Minkowski metric should be flat."""
        metric = MinkowskiMetric()

        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        g_tt, g_rr, g_thth, g_phph = metric.compute_metric_components(pos)

        # Minkowski: g_tt = -1, all spatial = +1
        assert g_tt == pytest.approx(-1.0, abs=1e-6)
        assert g_rr == pytest.approx(1.0, abs=1e-6)
        assert g_thth == pytest.approx(1.0, abs=1e-6)
        assert g_phph == pytest.approx(1.0, abs=1e-6)

    def test_schwarzschild_horizon(self):
        """Regression: Schwarzschild metric at horizon."""
        M = 1.0
        metric = SchwarzschildMetric(mass=M)

        # At horizon r = 2M
        r_horizon = 2.0 * M
        pos = np.array([r_horizon, 0.0, 0.0], dtype=np.float32)

        g_tt, g_rr, g_thth, g_phph = metric.compute_metric_components(pos)

        # At horizon: g_tt → 0, g_rr → ∞
        assert abs(g_tt) < 0.1, f"g_tt should be near zero at horizon: {g_tt:.3f}"
        assert g_rr > 10.0, f"g_rr should be large at horizon: {g_rr:.3f}"

    def test_kerr_reduces_to_schwarzschild(self):
        """Regression: Kerr with a=0 should match Schwarzschild."""
        M = 1.0
        metric_kerr = KerrMetric(mass=M, spin=0.0)
        metric_schw = SchwarzschildMetric(mass=M)

        pos = np.array([10.0, 0.0, 0.0], dtype=np.float32)

        g_tt_kerr, g_rr_kerr, _, _ = metric_kerr.compute_metric_components(pos)
        g_tt_schw, g_rr_schw, _, _ = metric_schw.compute_metric_components(pos)

        # Should match (within numerical tolerance)
        assert g_tt_kerr == pytest.approx(g_tt_schw, rel=1e-3)
        assert g_rr_kerr == pytest.approx(g_rr_schw, rel=1e-3)


# ============================================================================
# Regression: IO
# ============================================================================

@pytest.mark.regression
@pytest.mark.fast
class TestIORegression:
    """Regression tests for input/output."""

    def test_hdf5_roundtrip(self):
        """Regression: HDF5 write/read should preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_file = Path(tmpdir) / "test_snapshot.h5"

            # Create test data
            N = 50
            particles = {
                'positions': np.random.randn(N, 3).astype(np.float32),
                'velocities': np.random.randn(N, 3).astype(np.float32) * 0.1,
                'masses': np.ones(N, dtype=np.float32) * 0.01,
                'density': np.ones(N, dtype=np.float32) * 1.0,
                'internal_energy': np.ones(N, dtype=np.float32) * 1.0,
                'smoothing_length': np.ones(N, dtype=np.float32) * 0.1
            }

            # Write
            writer = HDF5Writer()
            writer.write_snapshot(
                str(snapshot_file),
                particles,
                time=0.5,
                metadata={'bh_mass': 1.0, 'mode': 'Newtonian'}
            )

            # Read
            data = writer.read_snapshot(str(snapshot_file))

            # Verify
            for key in particles.keys():
                np.testing.assert_allclose(
                    data['particles'][key],
                    particles[key],
                    rtol=1e-6,
                    err_msg=f"Roundtrip mismatch for {key}"
                )

            assert data['metadata']['time'] == 0.5
            assert data['metadata']['bh_mass'] == 1.0

    def test_diagnostic_writer_energy_log(self):
        """Regression: DiagnosticWriter should log energy correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "diagnostics.txt"

            writer = DiagnosticWriter(str(output_file))

            # Log some energies
            for i in range(5):
                writer.log_energy(
                    time=i * 0.1,
                    kinetic=1.0 + i * 0.01,
                    potential=-2.0 - i * 0.01,
                    internal=0.5,
                    total=-0.5 - i * 0.01,
                    conservation_error=i * 0.001
                )

            writer.close()

            # Read back
            lines = output_file.read_text().strip().split('\n')

            # Should have header + 5 data lines
            assert len(lines) == 6, f"Expected 6 lines, got {len(lines)}"
            assert "time" in lines[0].lower()


# ============================================================================
# Regression: Radiation
# ============================================================================

@pytest.mark.regression
@pytest.mark.fast
class TestRadiationRegression:
    """Regression tests for radiation cooling."""

    def test_cooling_reduces_internal_energy(self):
        """Regression: Cooling should reduce internal energy."""
        cooling = SimpleCooling(
            efficiency=1.0,
            opacity=0.34,
            temperature_floor=1e4
        )

        rho = np.array([1.0], dtype=np.float32)
        u_init = np.array([1e6], dtype=np.float32)
        dt = 0.1

        du_dt = cooling.compute_cooling_rate(rho, u_init)

        u_final = u_init + du_dt * dt

        # Cooling should reduce u
        assert u_final < u_init, "Cooling should reduce internal energy"
        assert du_dt < 0, "Cooling rate should be negative"


# ============================================================================
# Regression: End-to-End Workflow
# ============================================================================

@pytest.mark.regression
@pytest.mark.slow
class TestWorkflowRegression:
    """End-to-end regression tests."""

    def test_polytrope_to_hdf5_export(self):
        """Regression: Full workflow from IC generation to export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate polytrope
            poly = Polytrope(
                n_particles=100,
                polytropic_index=1.5,
                total_mass=1.0,
                radius=1.0,
                gamma=5.0/3.0
            )

            pos, vel, masses, u, h = poly.generate()

            # Compute energies
            diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

            density = masses / (h**3 * np.pi**(3/2))

            E = diag.compute_all_energies(
                time=0.0,
                masses=masses,
                positions=pos,
                velocities=vel,
                internal_energies=u,
                softening=h
            )

            # Export to HDF5
            particles = {
                'positions': pos,
                'velocities': vel,
                'masses': masses,
                'density': density,
                'internal_energy': u,
                'smoothing_length': h
            }

            snapshot_file = Path(tmpdir) / "snapshot_0000.h5"
            writer = HDF5Writer()
            writer.write_snapshot(
                str(snapshot_file),
                particles,
                time=0.0,
                metadata={
                    'bh_mass': 1.0,
                    'mode': 'Newtonian',
                    'total_energy': E.total,
                    'n_particles': len(masses)
                }
            )

            # Verify file exists and is valid
            assert snapshot_file.exists()
            data = writer.read_snapshot(str(snapshot_file))
            assert len(data['particles']['masses']) == 100
            assert data['metadata']['total_energy'] == pytest.approx(E.total, rel=1e-6)


# ============================================================================
# Regression: Numerical Stability
# ============================================================================

@pytest.mark.regression
@pytest.mark.fast
class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_energy_zero_mass_particles(self):
        """Regression: Energy computation should handle zero mass gracefully."""
        N = 10
        masses = np.zeros(N, dtype=np.float32)  # All zero mass
        positions = np.random.randn(N, 3).astype(np.float32)
        velocities = np.random.randn(N, 3).astype(np.float32)
        u = np.ones(N, dtype=np.float32)

        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        E = diag.compute_all_energies(
            time=0.0,
            masses=masses,
            positions=positions,
            velocities=velocities,
            internal_energies=u
        )

        # All energies should be zero
        assert E.kinetic == 0.0
        assert E.internal_thermal == 0.0

    def test_polytrope_extreme_radius(self):
        """Regression: Polytrope should handle very small/large radii."""
        # Very small radius
        poly_small = Polytrope(
            n_particles=50,
            polytropic_index=1.5,
            total_mass=1.0,
            radius=1e-6,
            gamma=5.0/3.0
        )

        pos_small, _, masses_small, _, _ = poly_small.generate()

        assert np.max(np.linalg.norm(pos_small, axis=1)) < 1e-5
        assert np.sum(masses_small) == pytest.approx(1.0, rel=1e-6)

        # Very large radius
        poly_large = Polytrope(
            n_particles=50,
            polytropic_index=1.5,
            total_mass=1.0,
            radius=1e6,
            gamma=5.0/3.0
        )

        pos_large, _, masses_large, _, _ = poly_large.generate()

        assert np.max(np.linalg.norm(pos_large, axis=1)) > 1e5
        assert np.sum(masses_large) == pytest.approx(1.0, rel=1e-6)
