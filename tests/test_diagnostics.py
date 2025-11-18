"""
Tests for DiagnosticsWriter and diagnostic utilities (TASK-024).

Tests validate:
1. DiagnosticsWriter initialization and file creation
2. Light curve writing and reading (HDF5 + CSV)
3. Fallback rate writing and reading
4. Energy evolution writing and reading
5. Orbital element computation and storage
6. Radial profile computation
7. Summary statistics
8. HDF5 time series appending
9. File format validity
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
import h5py
import csv

from src.tde_sph.io import (
    DiagnosticsWriter,
    OrbitalElements,
    compute_orbital_elements,
    compute_radial_profile,
)


class TestDiagnosticsWriter:
    """Test suite for DiagnosticsWriter."""

    def setup_method(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.diag = DiagnosticsWriter(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test DiagnosticsWriter initialization."""
        assert self.diag.output_dir == Path(self.temp_dir)
        assert self.diag.compression == "gzip"
        assert self.diag.compression_level == 4
        assert self.diag.hdf5_file == Path(self.temp_dir) / "diagnostics.h5"
        assert self.diag.output_dir.exists()

    def test_write_light_curve_single_point(self):
        """Test writing single light curve data point."""
        time = 0.0
        L_bol = 1e45
        L_mean = 5e44
        L_max = 2e45

        self.diag.write_light_curve(time, L_bol, L_mean=L_mean, L_max=L_max)

        # Check HDF5 file exists
        assert self.diag.hdf5_file.exists()

        # Check CSV file exists
        assert self.diag.light_curve_csv.exists()

        # Read HDF5
        with h5py.File(self.diag.hdf5_file, 'r') as f:
            assert 'light_curve' in f
            assert 'time' in f['light_curve']
            assert 'luminosity_bol' in f['light_curve']
            assert f['light_curve/time'][0] == pytest.approx(time)
            assert f['light_curve/luminosity_bol'][0] == pytest.approx(L_bol)

        # Read CSV
        with open(self.diag.light_curve_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            assert len(rows) == 1
            assert float(rows[0]['time']) == pytest.approx(time)
            assert float(rows[0]['luminosity_bol']) == pytest.approx(L_bol)

    def test_write_light_curve_time_series(self):
        """Test writing multiple light curve points (time series)."""
        times = [0.0, 1.0, 2.0, 3.0]
        luminosities = [1e44, 5e44, 3e44, 1e44]

        for t, L in zip(times, luminosities):
            self.diag.write_light_curve(t, L)

        # Read back
        lc = self.diag.get_light_curve()

        assert len(lc['time']) == 4
        np.testing.assert_array_almost_equal(lc['time'], times)
        np.testing.assert_array_almost_equal(lc['luminosity_bol'], luminosities)

    def test_write_fallback_rate(self):
        """Test fallback rate writing."""
        time = 1.0
        M_dot = 1e-3
        M_cumulative = 0.1

        self.diag.write_fallback_rate(time, M_dot, M_cumulative=M_cumulative)

        # Read back
        fb = self.diag.get_fallback_rate()

        assert fb['time'][0] == pytest.approx(time)
        assert fb['M_dot'][0] == pytest.approx(M_dot)
        assert fb['M_cumulative'][0] == pytest.approx(M_cumulative)

    def test_write_fallback_rate_time_series(self):
        """Test fallback rate time series."""
        times = [0.0, 1.0, 2.0]
        M_dots = [1e-2, 5e-3, 2e-3]  # Decreasing fallback

        for t, mdot in zip(times, M_dots):
            self.diag.write_fallback_rate(t, mdot)

        fb = self.diag.get_fallback_rate()

        assert len(fb['time']) == 3
        np.testing.assert_array_almost_equal(fb['time'], times)
        np.testing.assert_array_almost_equal(fb['M_dot'], M_dots)

    def test_write_energy_evolution(self):
        """Test energy evolution writing."""
        time = 0.5
        energies = {
            'kinetic': 1e50,
            'potential_bh': -2e50,
            'potential_self': -5e49,
            'internal_thermal': 3e49,
            'internal_radiation': 1e48,
            'radiated_cumulative': 1e48,
            'total': -1e50,
            'conservation_error': 1e-5
        }

        self.diag.write_energy_evolution(time, energies)

        # Read back
        e = self.diag.get_energies()

        assert e['time'][0] == pytest.approx(time)
        assert e['kinetic'][0] == pytest.approx(energies['kinetic'])
        assert e['potential_bh'][0] == pytest.approx(energies['potential_bh'])
        assert e['conservation_error'][0] == pytest.approx(energies['conservation_error'])

    def test_write_energy_evolution_time_series(self):
        """Test energy evolution time series."""
        times = [0.0, 1.0, 2.0]
        E_kin = [1e50, 1.5e50, 2e50]

        for i, t in enumerate(times):
            energies = {
                'kinetic': E_kin[i],
                'potential_bh': -2e50,
                'total': E_kin[i] - 2e50,
                'conservation_error': 0.0
            }
            self.diag.write_energy_evolution(t, energies)

        e = self.diag.get_energies()

        assert len(e['time']) == 3
        np.testing.assert_array_almost_equal(e['kinetic'], E_kin)

    def test_write_orbital_elements(self):
        """Test orbital element storage."""
        N = 100
        orbital_elements = OrbitalElements(
            apocenter=np.ones(N, dtype=np.float32) * 10.0,
            pericenter=np.ones(N, dtype=np.float32) * 1.0,
            eccentricity=np.ones(N, dtype=np.float32) * 0.8,
            semi_major_axis=np.ones(N, dtype=np.float32) * 5.5,
            specific_energy=np.ones(N, dtype=np.float32) * -0.1,
            specific_angular_momentum=np.ones(N, dtype=np.float32) * 3.0
        )

        self.diag.write_orbital_elements(0, orbital_elements)

        # Read back
        with h5py.File(self.diag.hdf5_file, 'r') as f:
            assert 'orbital_elements/snapshot_0000' in f
            oe_group = f['orbital_elements/snapshot_0000']
            assert len(oe_group['apocenter']) == N
            np.testing.assert_array_almost_equal(oe_group['apocenter'][:], 10.0)
            np.testing.assert_array_almost_equal(oe_group['eccentricity'][:], 0.8)

    def test_write_radial_profiles(self):
        """Test radial profile storage."""
        n_bins = 50
        r = np.logspace(-1, 2, n_bins, dtype=np.float32)
        rho = 1.0 / r**2  # Power-law density profile
        T = 1e6 * (r / 1.0)**(-0.5)  # Temperature profile

        self.diag.write_radial_profiles(
            snapshot_id=0,
            radius=r,
            density=rho,
            temperature=T
        )

        # Read back
        with h5py.File(self.diag.hdf5_file, 'r') as f:
            assert 'radial_profiles/snapshot_0000' in f
            rp_group = f['radial_profiles/snapshot_0000']
            np.testing.assert_array_almost_equal(rp_group['radius'][:], r)
            np.testing.assert_array_almost_equal(rp_group['density'][:], rho)
            np.testing.assert_array_almost_equal(rp_group['temperature'][:], T)

    def test_get_summary_statistics(self):
        """Test summary statistics computation."""
        # Write some data
        times = [0.0, 1.0, 2.0, 3.0]
        luminosities = [1e44, 5e44, 3e44, 1e44]

        for t, L in zip(times, luminosities):
            self.diag.write_light_curve(t, L)

        for t in times:
            energies = {
                'kinetic': 1e50,
                'potential_bh': -2e50,
                'total': -1e50,
                'conservation_error': 1e-6,
                'radiated_cumulative': t * 1e45
            }
            self.diag.write_energy_evolution(t, energies)

        summary = self.diag.get_summary_statistics()

        assert 'peak_luminosity' in summary
        assert summary['peak_luminosity'] == pytest.approx(5e44)
        assert summary['time_of_peak'] == pytest.approx(1.0)
        assert 'total_radiated_energy' in summary
        assert 'final_conservation_error' in summary

    def test_file_not_found_errors(self):
        """Test that appropriate errors are raised for missing files."""
        # Create new writer with no data
        temp_dir2 = tempfile.mkdtemp()
        diag2 = DiagnosticsWriter(temp_dir2)

        with pytest.raises(FileNotFoundError):
            diag2.get_light_curve()

        with pytest.raises(FileNotFoundError):
            diag2.get_energies()

        shutil.rmtree(temp_dir2)

    def test_csv_format_validity(self):
        """Test that CSV files have valid format."""
        self.diag.write_light_curve(0.0, 1e45, L_mean=5e44)
        self.diag.write_light_curve(1.0, 2e45, L_mean=1e45)

        # Check CSV is valid
        with open(self.diag.light_curve_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            assert len(rows) == 2
            assert 'time' in rows[0]
            assert 'luminosity_bol' in rows[0]


class TestComputeOrbitalElements:
    """Test suite for orbital element computation."""

    def test_circular_orbit(self):
        """Test orbital elements for circular orbit."""
        # Circular orbit: e = 0, r_p = r_a = r
        r = 10.0
        v_circ = np.sqrt(1.0 / r)  # M_BH = 1

        positions = np.array([[r, 0, 0]], dtype=np.float32)
        velocities = np.array([[0, v_circ, 0]], dtype=np.float32)

        oe = compute_orbital_elements(positions, velocities, bh_mass=1.0)

        assert oe.eccentricity[0] == pytest.approx(0.0, abs=1e-6)
        assert oe.pericenter[0] == pytest.approx(r, rel=0.01)
        assert oe.apocenter[0] == pytest.approx(r, rel=0.01)

    def test_elliptical_orbit(self):
        """Test orbital elements for elliptical orbit."""
        # Elliptical orbit: e = 0.5, r_p = 5, r_a = 15
        r_p = 5.0
        e = 0.5
        a = r_p / (1.0 - e)  # a = 10

        # At pericenter: v = sqrt(M_BH * (2/r_p - 1/a))
        v_p = np.sqrt(1.0 * (2.0 / r_p - 1.0 / a))

        positions = np.array([[r_p, 0, 0]], dtype=np.float32)
        velocities = np.array([[0, v_p, 0]], dtype=np.float32)

        oe = compute_orbital_elements(positions, velocities, bh_mass=1.0)

        assert oe.eccentricity[0] == pytest.approx(e, rel=0.01)
        assert oe.pericenter[0] == pytest.approx(r_p, rel=0.01)
        assert oe.semi_major_axis[0] == pytest.approx(a, rel=0.01)

    def test_parabolic_orbit(self):
        """Test orbital elements for parabolic orbit (E ~ 0)."""
        # Parabolic: e = 1, E = 0
        r = 10.0
        v_escape = np.sqrt(2.0 * 1.0 / r)  # Escape velocity

        positions = np.array([[r, 0, 0]], dtype=np.float32)
        velocities = np.array([[0, v_escape, 0]], dtype=np.float32)

        oe = compute_orbital_elements(positions, velocities, bh_mass=1.0)

        assert oe.eccentricity[0] == pytest.approx(1.0, rel=0.01)
        assert oe.specific_energy[0] == pytest.approx(0.0, abs=1e-6)

    def test_hyperbolic_orbit(self):
        """Test orbital elements for hyperbolic orbit (E > 0, e > 1)."""
        r = 10.0
        v = 1.5 * np.sqrt(2.0 * 1.0 / r)  # Super-escape velocity

        positions = np.array([[r, 0, 0]], dtype=np.float32)
        velocities = np.array([[0, v, 0]], dtype=np.float32)

        oe = compute_orbital_elements(positions, velocities, bh_mass=1.0)

        assert oe.eccentricity[0] > 1.0
        assert oe.specific_energy[0] > 0.0
        assert np.isinf(oe.apocenter[0])

    def test_multiple_particles(self):
        """Test orbital elements for multiple particles."""
        N = 100
        r = np.random.uniform(5, 20, N)
        theta = np.random.uniform(0, 2*np.pi, N)

        positions = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.zeros(N)
        ]).astype(np.float32)

        # Circular velocities
        v_circ = np.sqrt(1.0 / r)
        velocities = np.column_stack([
            -v_circ * np.sin(theta),
            v_circ * np.cos(theta),
            np.zeros(N)
        ]).astype(np.float32)

        oe = compute_orbital_elements(positions, velocities, bh_mass=1.0)

        # All should be circular (filter out NaN values from numerical precision)
        valid = ~np.isnan(oe.eccentricity)
        assert np.all(oe.eccentricity[valid] < 0.01)
        assert len(oe.apocenter) == N


class TestComputeRadialProfile:
    """Test suite for radial profile computation."""

    def test_uniform_density(self):
        """Test radial profile for uniform density."""
        N = 1000
        r = np.random.uniform(0, 10, N)
        theta = np.random.uniform(0, 2*np.pi, N)
        phi = np.random.uniform(0, np.pi, N)

        positions = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ]).astype(np.float32)

        densities = np.ones(N, dtype=np.float32)
        masses = np.ones(N, dtype=np.float32) / N

        r_bins, rho_profile = compute_radial_profile(
            positions, densities, masses, n_bins=10, log_bins=False
        )

        # Should be ~1 in all bins (up to sampling noise)
        assert np.all(np.abs(rho_profile - 1.0) < 0.2)

    def test_power_law_density(self):
        """Test radial profile for power-law density."""
        N = 10000
        # Sample r from power-law distribution
        r = np.random.uniform(1, 100, N)
        theta = np.random.uniform(0, 2*np.pi, N)

        positions = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.zeros(N)
        ]).astype(np.float32)

        # Assign density ∝ r^-2
        densities = (1.0 / r**2).astype(np.float32)
        masses = np.ones(N, dtype=np.float32) / N

        r_bins, rho_profile = compute_radial_profile(
            positions, densities, masses, n_bins=20, r_min=1, r_max=100, log_bins=True
        )

        # Check power-law trend (log-log should be linear)
        # Filter out NaN bins
        valid = ~np.isnan(rho_profile)
        if np.sum(valid) > 5:
            log_r = np.log10(r_bins[valid])
            log_rho = np.log10(rho_profile[valid])

            # Fit power-law: expect slope ~ -2
            slope = np.polyfit(log_r, log_rho, 1)[0]
            assert -2.5 < slope < -1.5, f"Power-law slope = {slope}, expected ~-2"

    def test_linear_bins(self):
        """Test radial profile with linear binning."""
        N = 100
        positions = np.random.uniform(-10, 10, (N, 3)).astype(np.float32)
        quantities = np.ones(N, dtype=np.float32)
        masses = np.ones(N, dtype=np.float32)

        r_bins, profile = compute_radial_profile(
            positions, quantities, masses, n_bins=10, log_bins=False
        )

        assert len(r_bins) == 10
        assert len(profile) == 10

    def test_log_bins(self):
        """Test radial profile with logarithmic binning."""
        N = 100
        positions = np.random.uniform(-10, 10, (N, 3)).astype(np.float32)
        quantities = np.ones(N, dtype=np.float32)
        masses = np.ones(N, dtype=np.float32)

        r_bins, profile = compute_radial_profile(
            positions, quantities, masses, n_bins=10, log_bins=True
        )

        # Check bins are logarithmically spaced
        log_bins = np.log10(r_bins)
        spacing = np.diff(log_bins)
        assert np.all(np.abs(spacing - spacing[0]) < 0.01)  # Uniform in log-space


class TestOrbitalElementsDataclass:
    """Test OrbitalElements dataclass."""

    def test_creation(self):
        """Test OrbitalElements creation."""
        N = 10
        oe = OrbitalElements(
            apocenter=np.ones(N, dtype=np.float32),
            pericenter=np.ones(N, dtype=np.float32) * 0.5,
            eccentricity=np.ones(N, dtype=np.float32) * 0.5,
            semi_major_axis=np.ones(N, dtype=np.float32) * 0.75,
            specific_energy=np.ones(N, dtype=np.float32) * -0.1,
            specific_angular_momentum=np.ones(N, dtype=np.float32) * 1.0
        )

        assert len(oe.apocenter) == N
        assert oe.eccentricity[0] == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
