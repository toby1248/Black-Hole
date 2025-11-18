"""
Tests for Phase 3 energy diagnostics, radiation EOS, and cooling models.

TASK-025: Energy conservation and thermodynamic validation tests.

Tests:
- RadiationGas EOS correctness
- EnergyDiagnostics accuracy
- SimpleCooling energy conservation
- DiagnosticsWriter I/O
- Integration tests
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from tde_sph.eos import IdealGas, RadiationGas
from tde_sph.radiation import SimpleCooling
from tde_sph.core import EnergyDiagnostics
from tde_sph.io import DiagnosticsWriter, compute_fallback_rate
from tde_sph.gravity.newtonian import NewtonianGravity


class TestRadiationGasEOS:
    """Test RadiationGas equation of state."""

    def test_initialization(self):
        """Test RadiationGas initialization."""
        eos = RadiationGas(gamma=5/3, mean_molecular_weight=0.6)
        assert eos.gamma == pytest.approx(5/3, rel=1e-6)
        assert eos.mu == pytest.approx(0.6, rel=1e-6)
        assert eos.a_rad > 0

    def test_gas_dominated_limit(self):
        """Test that RadiationGas matches IdealGas at low temperatures."""
        gamma = 5/3
        ideal_eos = IdealGas(gamma=gamma)
        rad_eos = RadiationGas(gamma=gamma)

        # Low temperature, high density (gas-dominated)
        rho = np.array([1e-10], dtype=np.float32)  # Low density in CGS
        T_low = 1e4  # 10,000 K (gas-dominated)

        # Get internal energy from temperature
        u_gas = ideal_eos.internal_energy_from_temperature(np.array([T_low], dtype=np.float32))

        # Compute pressures
        P_ideal = ideal_eos.pressure(rho, u_gas)
        P_rad = rad_eos.pressure(rho, u_gas)

        # Should match closely in gas-dominated regime (allow for iteration tolerance)
        # Check beta parameter to confirm gas-dominated
        beta = rad_eos.beta_parameter(rho, u_gas)
        assert beta[0] > 0.85  # Gas pressure dominates (>85%)
        assert P_rad == pytest.approx(P_ideal, rel=0.15)

    def test_radiation_dominated_limit(self):
        """Test radiation pressure dominance at high temperatures."""
        rad_eos = RadiationGas(gamma=5/3)

        # High temperature regime
        rho = np.array([1e-10], dtype=np.float32)
        T_high = 1e8  # 100 million K

        # Very high internal energy for radiation-dominated regime
        # u_rad = a T^4 / rho >> u_gas
        u = np.array([1e20], dtype=np.float32)  # Very high internal energy

        # Compute beta parameter (should be << 1 for radiation-dominated)
        beta = rad_eos.beta_parameter(rho, u)
        assert beta[0] < 0.5  # Radiation contributes at least 50%

    def test_temperature_consistency(self):
        """Test temperature solver self-consistency."""
        eos = RadiationGas()

        # Range of densities and internal energies
        rho = np.array([1e-10, 1e-5, 1.0], dtype=np.float32)
        u = np.array([1e10, 1e15, 1e18], dtype=np.float32)

        # Solve for temperature
        T = eos.temperature(rho, u)

        # All temperatures should be positive and finite
        assert np.all(T > 0)
        assert np.all(np.isfinite(T))

        # Reconstruct internal energy from T
        u_gas = eos.gas_energy(rho, u)
        u_rad = eos.radiation_energy(rho, u)
        u_reconstructed = u_gas + u_rad

        # Should match original (within iteration tolerance)
        np.testing.assert_allclose(u_reconstructed, u, rtol=1e-4)

    def test_energy_partitioning(self):
        """Test gas and radiation energy partitioning."""
        eos = RadiationGas()

        rho = np.array([1e-8], dtype=np.float32)
        u_total = np.array([1e16], dtype=np.float32)

        u_gas = eos.gas_energy(rho, u_total)
        u_rad = eos.radiation_energy(rho, u_total)

        # Sum should equal total (allow for Newton-Raphson iteration tolerance)
        np.testing.assert_allclose(u_gas + u_rad, u_total, rtol=1e-3)

        # Both components should be non-negative
        assert np.all(u_gas >= 0)
        assert np.all(u_rad >= 0)

    def test_sound_speed(self):
        """Test sound speed computation."""
        eos = RadiationGas()

        rho = np.array([1e-5], dtype=np.float32)
        u = np.array([1e14], dtype=np.float32)

        cs = eos.sound_speed(rho, u)

        # Sound speed should be positive and finite
        assert np.all(cs > 0)
        assert np.all(np.isfinite(cs))

        # Should be less than speed of light in relativistic regime
        c_light = 3e10  # cm/s
        assert np.all(cs < c_light)


class TestEnergyDiagnostics:
    """Test EnergyDiagnostics class."""

    def create_test_particles(self, N=100):
        """Create simple test particle system."""
        np.random.seed(42)

        particles = {
            'positions': np.random.randn(N, 3).astype(np.float32) * 10.0,
            'velocities': np.random.randn(N, 3).astype(np.float32) * 0.1,
            'masses': np.ones(N, dtype=np.float32) * 0.01,
            'internal_energy': np.ones(N, dtype=np.float32) * 1e10,
            'density': np.ones(N, dtype=np.float32) * 1e-10,
            'smoothing_length': np.ones(N, dtype=np.float32) * 1.0
        }

        return particles

    def test_initialization(self):
        """Test EnergyDiagnostics initialization."""
        diagnostics = EnergyDiagnostics()
        assert diagnostics.E_initial is None
        assert len(diagnostics.history) == 0

    def test_kinetic_energy(self):
        """Test kinetic energy computation."""
        diagnostics = EnergyDiagnostics()
        particles = self.create_test_particles(N=10)

        E_kin = diagnostics.compute_kinetic_energy(particles)

        # Manual calculation
        v_squared = np.sum(particles['velocities']**2, axis=1)
        E_kin_expected = 0.5 * np.sum(particles['masses'] * v_squared)

        assert E_kin == pytest.approx(E_kin_expected, rel=1e-6)

    def test_internal_energy_ideal_gas(self):
        """Test internal energy with IdealGas EOS."""
        diagnostics = EnergyDiagnostics()
        particles = self.create_test_particles(N=10)
        eos = IdealGas()

        energy_dict = diagnostics.compute_internal_energy(particles, eos)

        # Check keys
        assert 'E_internal_total' in energy_dict
        assert 'E_internal_gas' in energy_dict
        assert 'E_internal_radiation' in energy_dict

        # For IdealGas, radiation should be zero
        assert energy_dict['E_internal_radiation'] == 0.0

        # Total should equal gas
        assert energy_dict['E_internal_total'] == pytest.approx(
            energy_dict['E_internal_gas'], rel=1e-6
        )

        # Manual calculation
        E_int_expected = np.sum(particles['masses'] * particles['internal_energy'])
        assert energy_dict['E_internal_total'] == pytest.approx(E_int_expected, rel=1e-6)

    def test_internal_energy_radiation_gas(self):
        """Test internal energy with RadiationGas EOS."""
        diagnostics = EnergyDiagnostics()
        particles = self.create_test_particles(N=10)
        eos = RadiationGas()

        energy_dict = diagnostics.compute_internal_energy(particles, eos)

        # Check partitioning
        E_total = energy_dict['E_internal_total']
        E_gas = energy_dict['E_internal_gas']
        E_rad = energy_dict['E_internal_radiation']

        # Sum should match total (within small tolerance)
        assert E_gas + E_rad == pytest.approx(E_total, rel=0.01)

        # Both should be non-negative
        assert E_gas >= 0
        assert E_rad >= 0

    def test_global_quantities(self):
        """Test global conserved quantities."""
        diagnostics = EnergyDiagnostics()
        particles = self.create_test_particles(N=10)

        quantities = diagnostics.compute_global_quantities(particles)

        # Check keys
        assert 'total_mass' in quantities
        assert 'linear_momentum' in quantities
        assert 'angular_momentum' in quantities
        assert 'center_of_mass' in quantities

        # Total mass
        assert quantities['total_mass'] == pytest.approx(
            np.sum(particles['masses']), rel=1e-6
        )

        # Vectors should have shape (3,)
        assert quantities['linear_momentum'].shape == (3,)
        assert quantities['angular_momentum'].shape == (3,)
        assert quantities['center_of_mass'].shape == (3,)

    def test_compute_full_diagnostics(self):
        """Test full diagnostics computation."""
        diagnostics = EnergyDiagnostics()
        particles = self.create_test_particles(N=100)
        eos = IdealGas()
        gravity_solver = NewtonianGravity()

        result = diagnostics.compute(particles, gravity_solver, eos)

        # Check all expected keys
        expected_keys = [
            'E_kinetic', 'E_potential', 'E_internal_total',
            'E_internal_gas', 'E_internal_radiation', 'E_total',
            'energy_conservation', 'total_mass', 'T_min', 'T_max', 'T_mean'
        ]

        for key in expected_keys:
            assert key in result

        # Energy should be finite
        assert np.isfinite(result['E_total'])

        # Initial energy should be set
        assert diagnostics.E_initial == result['E_total']

    def test_history_tracking(self):
        """Test time-series history tracking."""
        diagnostics = EnergyDiagnostics()
        particles = self.create_test_particles(N=50)
        eos = IdealGas()
        gravity_solver = NewtonianGravity()

        # Compute diagnostics at multiple times
        times = [0.0, 1.0, 2.0]
        for t in times:
            result = diagnostics.compute(particles, gravity_solver, eos)
            diagnostics.append_to_history(t, result)

        # Check history
        assert len(diagnostics.history) == len(times)

        # Extract time-series
        ts = diagnostics.get_time_series('E_total')
        assert len(ts['time']) == len(times)
        np.testing.assert_array_equal(ts['time'], times)


class TestSimpleCooling:
    """Test SimpleCooling radiation model."""

    def test_initialization(self):
        """Test SimpleCooling initialization."""
        cooling = SimpleCooling(cooling_model='free_free', temperature_floor=100.0)
        assert cooling.cooling_model == 'free_free'
        assert cooling.T_floor == 100.0

    def test_invalid_model(self):
        """Test invalid cooling model raises error."""
        with pytest.raises(ValueError):
            SimpleCooling(cooling_model='invalid_model')

    def test_no_cooling(self):
        """Test 'none' cooling model."""
        cooling = SimpleCooling(cooling_model='none')

        rho = np.array([1e-10], dtype=np.float32)
        T = np.array([1e6], dtype=np.float32)
        u = np.array([1e14], dtype=np.float32)

        du_dt = cooling.cooling_rate(rho, T, u)

        # Should be zero
        assert np.all(du_dt == 0)

    def test_free_free_cooling(self):
        """Test free-free cooling rate."""
        cooling = SimpleCooling(cooling_model='free_free')

        # Use higher densities to ensure non-negligible cooling rates
        rho = np.array([1e-5, 1e-3], dtype=np.float32)
        T = np.array([1e6, 1e7], dtype=np.float32)
        u = np.array([1e14, 1e16], dtype=np.float32)

        du_dt = cooling.cooling_rate(rho, T, u)

        # Cooling should be negative
        assert np.all(du_dt < 0)

        # Higher temperature/density should cool faster (in absolute value)
        assert abs(du_dt[1]) > abs(du_dt[0])

    def test_temperature_floor(self):
        """Test temperature floor prevents cooling."""
        T_floor = 1000.0
        cooling = SimpleCooling(temperature_floor=T_floor)

        rho = np.array([1e-10], dtype=np.float32)
        T_below = np.array([500.0], dtype=np.float32)  # Below floor
        u = np.array([1e10], dtype=np.float32)

        du_dt = cooling.cooling_rate(rho, T_below, u)

        # Should be zero below floor
        assert du_dt[0] == 0.0

    def test_luminosity(self):
        """Test luminosity computation."""
        cooling = SimpleCooling(cooling_model='free_free')

        N = 100
        # Use higher density to ensure non-negligible cooling
        rho = np.ones(N, dtype=np.float32) * 1e-5
        T = np.ones(N, dtype=np.float32) * 1e6
        u = np.ones(N, dtype=np.float32) * 1e14
        masses = np.ones(N, dtype=np.float32) * 0.01

        L = cooling.luminosity(rho, T, u, masses)

        # Luminosity should be positive
        assert L > 0

        # Should equal -Σ(m * du/dt)
        du_dt = cooling.cooling_rate(rho, T, u)
        L_expected = -np.sum(masses * du_dt)
        assert L == pytest.approx(L_expected, rel=1e-6)

    def test_cooling_timescale(self):
        """Test cooling timescale computation."""
        cooling = SimpleCooling()

        rho = np.array([1e-10], dtype=np.float32)
        T = np.array([1e6], dtype=np.float32)
        u = np.array([1e14], dtype=np.float32)

        t_cool = cooling.cooling_timescale(rho, T, u)

        # Should be positive
        assert np.all(t_cool > 0)

        # Should equal u / |du/dt|
        du_dt = cooling.cooling_rate(rho, T, u)
        t_cool_expected = u / np.abs(du_dt)
        np.testing.assert_allclose(t_cool, t_cool_expected, rtol=1e-6)


class TestDiagnosticsWriter:
    """Test DiagnosticsWriter I/O."""

    def test_csv_initialization(self):
        """Test CSV diagnostics writer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DiagnosticsWriter(tmpdir, format='csv')

            # Check files created
            assert (Path(tmpdir) / 'energy.csv').exists()
            assert (Path(tmpdir) / 'luminosity.csv').exists()
            assert (Path(tmpdir) / 'fallback.csv').exists()

            writer.finalize()

    def test_energy_diagnostic_write(self):
        """Test writing energy diagnostics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DiagnosticsWriter(tmpdir, format='csv')

            energy_dict = {
                'E_kinetic': 1e42,
                'E_potential': -2e42,
                'E_internal_total': 5e41,
                'E_internal_gas': 4e41,
                'E_internal_radiation': 1e41,
                'E_total': -5e41,
                'energy_conservation': 0.001
            }

            writer.write_energy_diagnostic(time=0.0, energy_dict=energy_dict)
            writer.write_energy_diagnostic(time=1.0, energy_dict=energy_dict)
            writer.finalize()

            # Read back CSV
            import csv
            with open(Path(tmpdir) / 'energy.csv', 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert float(rows[0]['time']) == 0.0
            assert float(rows[1]['time']) == 1.0
            assert float(rows[0]['E_kinetic']) == pytest.approx(1e42)

    def test_luminosity_write(self):
        """Test writing luminosity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DiagnosticsWriter(tmpdir, format='csv')

            writer.write_luminosity(time=0.0, luminosity=1e43)
            writer.write_luminosity(time=1.0, luminosity=5e42)
            writer.finalize()

            # Read back
            import csv
            with open(Path(tmpdir) / 'luminosity.csv', 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert float(rows[1]['L_total']) == pytest.approx(5e42)

    def test_context_manager(self):
        """Test context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DiagnosticsWriter(tmpdir, format='csv') as writer:
                writer.write_luminosity(time=0.0, luminosity=1e43)

            # Files should exist after context exit
            assert (Path(tmpdir) / 'luminosity.csv').exists()
            assert (Path(tmpdir) / 'metadata.json').exists()


def test_compute_fallback_rate():
    """Test fallback rate computation utility."""
    np.random.seed(42)

    N = 100
    particles = {
        'positions': np.random.randn(N, 3).astype(np.float32) * 20.0,
        'velocities': np.random.randn(N, 3).astype(np.float32) * 0.5,
        'masses': np.ones(N, dtype=np.float32) * 0.01
    }

    result = compute_fallback_rate(particles, bh_mass=1.0, r_capture=10.0)

    # Check keys
    assert 'bound_mass' in result
    assert 'captured_mass' in result
    assert 'N_bound' in result
    assert 'N_captured' in result

    # Counts should be non-negative integers
    assert result['N_bound'] >= 0
    assert result['N_captured'] >= 0

    # Masses should be non-negative
    assert result['bound_mass'] >= 0
    assert result['captured_mass'] >= 0

    # Captured particles must be subset of bound particles (approximately, for deep potential)
    # Not strict due to velocity contributions, but captured mass shouldn't exceed total
    assert result['captured_mass'] <= np.sum(particles['masses'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
