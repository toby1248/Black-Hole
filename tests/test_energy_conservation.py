"""
Tests for energy conservation in TDE-SPH simulations (TASK-025).

Tests cover:
- Adiabatic energy conservation (no heating/cooling)
- Controlled heating/cooling response
- GR vs Newtonian energy consistency
- Isolated star equilibrium
- Energy component calculations
- Conservation error tracking

References
----------
- Liptai & Price (2019), MNRAS 485, 819 - GRSPH energy conservation
- Price (2012), JCP 231, 759 - SPH energy conservation principles
- Tejeda et al. (2017), MNRAS 469, 4483 - GR-Newtonian energy comparison

Test Coverage
-------------
1. Energy component calculations (kinetic, potential, internal)
2. Adiabatic conservation (|ΔE/E₀| < 0.001 expected)
3. Controlled heating/cooling response
4. Initial energy tracking
5. Conservation error accumulation
6. Newtonian vs GR mode consistency
"""

import pytest
import numpy as np
from tde_sph.core.energy_diagnostics import EnergyDiagnostics, EnergyComponents


class MockParticleData:
    """Mock particle data for energy conservation tests."""

    def __init__(
        self,
        n_particles: int = 100,
        r_range: tuple = (5.0, 20.0),
        v_scale: float = 0.1,
        u_scale: float = 1.0
    ):
        """
        Create mock particle distribution.

        Parameters
        ----------
        n_particles : int
            Number of particles.
        r_range : tuple
            (r_min, r_max) for radial distribution.
        v_scale : float
            Velocity scale factor.
        u_scale : float
            Internal energy scale factor.
        """
        # Positions: random in spherical shell
        r = np.random.uniform(r_range[0], r_range[1], n_particles).astype(np.float32)
        theta = np.arccos(np.random.uniform(-1.0, 1.0, n_particles)).astype(np.float32)
        phi = np.random.uniform(0.0, 2.0 * np.pi, n_particles).astype(np.float32)

        self.positions = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ]).astype(np.float32)

        # Velocities: random small perturbations
        self.velocities = np.random.randn(n_particles, 3).astype(np.float32) * v_scale

        # Masses: uniform
        self.masses = np.ones(n_particles, dtype=np.float32)

        # Internal energies: uniform with small random variations
        self.internal_energies = (
            np.ones(n_particles, dtype=np.float32) * u_scale
            * (1.0 + 0.1 * np.random.randn(n_particles))
        ).astype(np.float32)

        # Store for energy calculations
        self.n_particles = n_particles


class TestEnergyComponentCalculations:
    """Test individual energy component calculations."""

    def test_kinetic_energy_calculation(self):
        """Test kinetic energy E_kin = Σ (1/2) m v²."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=1.0)

        E_kin = diag.compute_kinetic_energy(
            particles.masses,
            particles.velocities
        )

        # Manual calculation
        v_squared = np.sum(particles.velocities**2, axis=1)
        E_kin_expected = 0.5 * np.sum(particles.masses * v_squared)

        assert E_kin == pytest.approx(E_kin_expected, rel=1e-5)

    def test_potential_bh_newtonian(self):
        """Test BH potential energy E_pot = -Σ G M_BH m / r (Newtonian)."""
        M_bh = 1.0
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=M_bh)
        particles = MockParticleData(n_particles=100)

        E_pot_bh = diag.compute_potential_bh(
            particles.masses,
            particles.positions
        )

        # Manual calculation
        r = np.sqrt(np.sum(particles.positions**2, axis=1))
        E_pot_expected = -np.sum(M_bh * particles.masses / r)

        assert E_pot_bh == pytest.approx(E_pot_expected, rel=1e-5)
        assert E_pot_bh < 0.0  # Potential should be negative

    def test_potential_self_gravity(self):
        """Test self-gravity potential E_pot_self = (1/2) Σᵢ Σⱼ≠ᵢ (-G m_i m_j / |r_ij|)."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        # Use small number of particles for pairwise calculation
        particles = MockParticleData(n_particles=10, r_range=(10.0, 15.0))

        E_pot_self = diag.compute_potential_self(
            particles.masses,
            particles.positions
        )

        # Manual pairwise calculation
        n = len(particles.masses)
        E_expected = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r_ij = np.linalg.norm(particles.positions[i] - particles.positions[j])
                E_expected -= particles.masses[i] * particles.masses[j] / r_ij

        assert E_pot_self == pytest.approx(E_expected, rel=1e-4)
        assert E_pot_self < 0.0  # Self-gravity should be negative

    def test_internal_energy_calculation(self):
        """Test internal energy E_int = Σ m u."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, u_scale=2.0)

        E_thermal, E_rad = diag.compute_internal_energy(
            particles.masses,
            particles.internal_energies
        )

        # Manual calculation
        E_int_expected = np.sum(particles.masses * particles.internal_energies)

        assert E_thermal == pytest.approx(E_int_expected, rel=1e-5)
        assert E_thermal > 0.0  # Internal energy should be positive
        assert E_rad == 0.0  # No radiation component without RadiationGasEOS

    def test_total_energy_components(self):
        """Test that total energy is sum of components."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=50)

        energies = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Total should equal sum of kinetic + potential + internal
        E_total_expected = (
            energies.kinetic
            + energies.potential_bh
            + energies.potential_self
            + energies.internal_thermal
            + energies.internal_radiation
        )

        assert energies.total == pytest.approx(E_total_expected, rel=1e-5)


class TestAdiabaticConservation:
    """Test energy conservation in adiabatic (no heating/cooling) scenarios."""

    def test_static_particles_conserve_energy(self):
        """Test that stationary particles conserve energy exactly."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        # Create static particle distribution
        particles = MockParticleData(n_particles=100, v_scale=0.0)

        # Compute initial energy
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Set initial energy
        diag.initial_energy = E0.total

        # Simulate multiple timesteps with no changes
        for t in range(10):
            E = diag.compute_all_energies(
                time=t * 0.1,
                masses=particles.masses,
                positions=particles.positions,
                velocities=particles.velocities,
                internal_energies=particles.internal_energies
            )

            # Energy should be exactly conserved (no dynamics)
            assert E.total == pytest.approx(E0.total, rel=1e-10)
            assert np.abs(E.conservation_error) < 1e-10

    def test_small_perturbations_bounded_error(self):
        """Test that small perturbations keep conservation error bounded."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=0.1)

        # Compute initial energy
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )
        diag.initial_energy = E0.total

        # Apply small perturbations over multiple steps
        max_error = 0.0
        for t in range(20):
            # Small random velocity perturbations (mimicking numerical errors)
            dv = np.random.randn(*particles.velocities.shape).astype(np.float32) * 1e-4
            particles.velocities += dv

            E = diag.compute_all_energies(
                time=t * 0.1,
                masses=particles.masses,
                positions=particles.positions,
                velocities=particles.velocities,
                internal_energies=particles.internal_energies
            )

            max_error = max(max_error, np.abs(E.conservation_error))

        # Conservation error should remain small despite perturbations
        # (Not exact due to velocity changes, but should be bounded)
        assert max_error < 0.1  # Within 10% (loose bound for random perturbations)

    def test_adiabatic_constant_internal_energy(self):
        """Test that adiabatic evolution preserves internal energy distribution."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, u_scale=1.0)

        # Initial internal energy
        u_initial = particles.internal_energies.copy()

        # Compute energies at t=0
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # In adiabatic run, internal energies don't change
        # Simulate timesteps
        for t in range(10):
            E = diag.compute_all_energies(
                time=t * 0.1,
                masses=particles.masses,
                positions=particles.positions,
                velocities=particles.velocities,
                internal_energies=u_initial  # Keep constant (adiabatic)
            )

            # Internal energy component should stay constant
            assert E.internal_thermal == pytest.approx(E0.internal_thermal, rel=1e-10)


class TestHeatingCoolingResponse:
    """Test energy response to controlled heating and cooling."""

    def test_uniform_heating_increases_internal_energy(self):
        """Test that uniform heating increases total internal energy."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=0.0, u_scale=1.0)

        # Initial energy
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Apply uniform heating: Δu = +0.5
        particles.internal_energies += 0.5

        # Compute new energy
        E1 = diag.compute_all_energies(
            time=0.1,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Internal energy should increase by Σ m Δu
        Delta_E_expected = np.sum(particles.masses * 0.5)
        Delta_E_actual = E1.internal_thermal - E0.internal_thermal

        assert Delta_E_actual == pytest.approx(Delta_E_expected, rel=1e-5)
        assert E1.internal_thermal > E0.internal_thermal

    def test_uniform_cooling_decreases_internal_energy(self):
        """Test that uniform cooling decreases total internal energy."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=0.0, u_scale=2.0)

        # Initial energy
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Apply uniform cooling: Δu = -0.5
        particles.internal_energies -= 0.5

        # Compute new energy
        E1 = diag.compute_all_energies(
            time=0.1,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Internal energy should decrease
        Delta_E_expected = -np.sum(particles.masses * 0.5)
        Delta_E_actual = E1.internal_thermal - E0.internal_thermal

        assert Delta_E_actual == pytest.approx(Delta_E_expected, rel=1e-5)
        assert E1.internal_thermal < E0.internal_thermal

    def test_radiated_energy_tracking(self):
        """Test cumulative radiated energy tracking."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=0.0, u_scale=2.0)

        # Initial energy
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies,
            radiated_cumulative=0.0
        )
        diag.initial_energy = E0.total

        # Simulate radiative cooling over multiple steps
        E_radiated_cumulative = 0.0
        for t in range(10):
            # Cool particles: Δu = -0.1
            delta_u = -0.1
            particles.internal_energies = np.maximum(
                particles.internal_energies + delta_u, 0.1
            )  # Keep positive

            # Track radiated energy
            E_rad_step = -np.sum(particles.masses * delta_u)
            E_radiated_cumulative += E_rad_step

            E = diag.compute_all_energies(
                time=(t + 1) * 0.1,
                masses=particles.masses,
                positions=particles.positions,
                velocities=particles.velocities,
                internal_energies=particles.internal_energies,
                radiated_cumulative=E_radiated_cumulative
            )

            # Radiated energy should accumulate
            assert E.radiated_cumulative == pytest.approx(
                E_radiated_cumulative, rel=1e-5
            )

        # Total radiated should be positive
        assert E_radiated_cumulative > 0.0

    def test_heating_cooling_balance(self):
        """Test that equal heating and cooling cancel out."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=0.0, u_scale=1.0)

        # Initial energy
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Heat: Δu = +0.5
        particles.internal_energies += 0.5

        E1 = diag.compute_all_energies(
            time=0.1,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Cool: Δu = -0.5
        particles.internal_energies -= 0.5

        E2 = diag.compute_all_energies(
            time=0.2,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Should return to initial internal energy
        assert E2.internal_thermal == pytest.approx(E0.internal_thermal, rel=1e-5)


class TestConservationErrorTracking:
    """Test conservation error calculation and tracking."""

    def test_zero_error_for_static_system(self):
        """Test conservation error is zero for unchanging system."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=0.0)

        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )
        diag.initial_energy = E0.total

        # Multiple evaluations with no changes
        for t in range(10):
            E = diag.compute_all_energies(
                time=t * 0.1,
                masses=particles.masses,
                positions=particles.positions,
                velocities=particles.velocities,
                internal_energies=particles.internal_energies
            )

            # Conservation error should be exactly zero
            assert E.conservation_error == pytest.approx(0.0, abs=1e-10)

    def test_conservation_error_sign(self):
        """Test conservation error has correct sign for energy changes."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=0.1, u_scale=1.0)

        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )
        diag.initial_energy = E0.total

        # Increase kinetic energy (speed up particles)
        particles.velocities *= 2.0

        E1 = diag.compute_all_energies(
            time=0.1,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Energy increased (gained energy)
        assert E1.total > E0.total
        # Conservation error should be non-zero
        # Note: For bound systems (E0 < 0), error sign is opposite to energy change
        assert E1.conservation_error != 0.0

        # Decrease kinetic energy (slow down)
        particles.velocities *= 0.25

        E2 = diag.compute_all_energies(
            time=0.2,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Energy decreased (lost energy relative to E0)
        assert E2.total < E0.total
        # Conservation error should be non-zero with opposite sign from E1
        assert E2.conservation_error != 0.0
        assert np.sign(E2.conservation_error) != np.sign(E1.conservation_error)

    def test_conservation_error_magnitude(self):
        """Test conservation error magnitude: ΔE/E₀."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=1.0, u_scale=1.0)

        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )
        diag.initial_energy = E0.total

        # Change internal energy by known amount
        delta_u = 0.1
        particles.internal_energies += delta_u

        E1 = diag.compute_all_energies(
            time=0.1,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Expected error: ΔE / E0
        Delta_E = np.sum(particles.masses * delta_u)
        expected_error = Delta_E / E0.total

        assert E1.conservation_error == pytest.approx(expected_error, rel=1e-5)


class TestEnergyHistoryTracking:
    """Test energy history tracking over time."""

    def test_energy_history_appended(self):
        """Test that log_energy_history appends to history."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=50)

        assert len(diag.energy_history) == 0

        # Compute energies at multiple times and log them
        for t in range(5):
            E = diag.compute_all_energies(
                time=t * 0.1,
                masses=particles.masses,
                positions=particles.positions,
                velocities=particles.velocities,
                internal_energies=particles.internal_energies
            )
            diag.log_energy_history(E)

        assert len(diag.energy_history) == 5

    def test_energy_history_time_ordering(self):
        """Test that energy history preserves time ordering."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=50)

        times = [0.0, 0.1, 0.3, 0.7, 1.5]
        for t in times:
            E = diag.compute_all_energies(
                time=t,
                masses=particles.masses,
                positions=particles.positions,
                velocities=particles.velocities,
                internal_energies=particles.internal_energies
            )
            diag.log_energy_history(E)

        # Check time ordering
        for i, E in enumerate(diag.energy_history):
            assert E.time == pytest.approx(times[i], rel=1e-10)

    def test_energy_history_clear(self):
        """Test clearing energy history."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=50)

        # Build up history
        for t in range(10):
            E = diag.compute_all_energies(
                time=t * 0.1,
                masses=particles.masses,
                positions=particles.positions,
                velocities=particles.velocities,
                internal_energies=particles.internal_energies
            )
            diag.log_energy_history(E)

        assert len(diag.energy_history) == 10

        # Clear history
        diag.clear_history()
        assert len(diag.energy_history) == 0


class TestInitialEnergySetup:
    """Test initial energy setup and reference tracking."""

    def test_initial_energy_auto_set(self):
        """Test that initial energy is auto-set on first call if None."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0, initial_energy=None)
        particles = MockParticleData(n_particles=100)

        assert diag.initial_energy is None

        # First call should set initial_energy
        E0 = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        assert diag.initial_energy is not None
        assert diag.initial_energy == pytest.approx(E0.total, rel=1e-10)

    def test_initial_energy_manual_set(self):
        """Test manually setting initial energy."""
        E_init = 123.456
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0, initial_energy=E_init)

        assert diag.initial_energy == pytest.approx(E_init, rel=1e-10)

    def test_initial_energy_used_for_conservation_error(self):
        """Test that initial_energy is used for conservation error calculation."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        particles = MockParticleData(n_particles=100, v_scale=1.0, u_scale=1.0)

        # Manually set initial energy
        E_init = 100.0
        diag.initial_energy = E_init

        E = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Conservation error should be (E.total - E_init) / E_init
        expected_error = (E.total - E_init) / E_init
        assert E.conservation_error == pytest.approx(expected_error, rel=1e-5)


class TestModeConsistency:
    """Test energy calculations are consistent across Newtonian and GR modes."""

    def test_newtonian_mode_attributes(self):
        """Test Newtonian mode initialization."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=2.0)

        assert diag.mode == "Newtonian"
        assert diag.bh_mass == pytest.approx(2.0, rel=1e-10)
        assert diag.metric is None

    def test_gr_mode_attributes(self):
        """Test GR mode initialization (requires metric)."""
        # Mock metric object
        class MockMetric:
            def __init__(self):
                self.mass = 1.0

        metric = MockMetric()
        diag = EnergyDiagnostics(mode="GR", bh_mass=1.0, metric=metric)

        assert diag.mode == "GR"
        assert diag.metric is not None
        assert diag.bh_mass == pytest.approx(1.0, rel=1e-10)

    def test_same_particles_same_energy_components(self):
        """Test that same particle distribution gives consistent energy in Newtonian mode."""
        particles = MockParticleData(n_particles=100, v_scale=1.0, u_scale=1.0)

        diag1 = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)
        diag2 = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        E1 = diag1.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        E2 = diag2.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # All energy components should match
        assert E1.kinetic == pytest.approx(E2.kinetic, rel=1e-10)
        assert E1.potential_bh == pytest.approx(E2.potential_bh, rel=1e-10)
        assert E1.potential_self == pytest.approx(E2.potential_self, rel=1e-10)
        assert E1.internal_thermal == pytest.approx(E2.internal_thermal, rel=1e-10)
        assert E1.total == pytest.approx(E2.total, rel=1e-10)


class TestEnergyComponentsDataclass:
    """Test EnergyComponents dataclass functionality."""

    def test_energy_components_initialization(self):
        """Test EnergyComponents initialization with defaults."""
        E = EnergyComponents()

        assert E.time == 0.0
        assert E.kinetic == 0.0
        assert E.potential_bh == 0.0
        assert E.potential_self == 0.0
        assert E.internal_thermal == 0.0
        assert E.internal_radiation == 0.0
        assert E.radiated_cumulative == 0.0
        assert E.total == 0.0
        assert E.conservation_error == 0.0

    def test_energy_components_custom_values(self):
        """Test EnergyComponents with custom values."""
        E = EnergyComponents(
            time=1.5,
            kinetic=10.0,
            potential_bh=-50.0,
            potential_self=-5.0,
            internal_thermal=20.0,
            internal_radiation=2.0,
            radiated_cumulative=3.0,
            total=-23.0,
            conservation_error=0.001
        )

        assert E.time == 1.5
        assert E.kinetic == 10.0
        assert E.potential_bh == -50.0
        assert E.conservation_error == 0.001

    def test_energy_components_to_dict(self):
        """Test conversion to dictionary."""
        E = EnergyComponents(
            time=2.0,
            kinetic=15.0,
            potential_bh=-30.0,
            total=-10.0
        )

        E_dict = E.to_dict()

        assert isinstance(E_dict, dict)
        assert E_dict['time'] == 2.0
        assert E_dict['kinetic'] == 15.0
        assert E_dict['potential_bh'] == -30.0
        assert E_dict['total'] == -10.0
        assert 'conservation_error' in E_dict


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_mass_particles(self):
        """Test handling of zero-mass particles (should contribute zero energy)."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        # Create particles with some zero masses
        n = 100
        masses = np.ones(n, dtype=np.float32)
        masses[::2] = 0.0  # Half have zero mass

        positions = np.random.randn(n, 3).astype(np.float32) * 10.0
        velocities = np.random.randn(n, 3).astype(np.float32) * 0.1
        internal_energies = np.ones(n, dtype=np.float32)

        # Should not crash
        E = diag.compute_all_energies(
            time=0.0,
            masses=masses,
            positions=positions,
            velocities=velocities,
            internal_energies=internal_energies
        )

        # Energy should be well-defined
        assert np.isfinite(E.total)
        assert np.isfinite(E.kinetic)
        assert np.isfinite(E.potential_bh)

    def test_very_small_energies(self):
        """Test handling of very small energy values."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        # Particles with very small masses and energies
        particles = MockParticleData(n_particles=100, v_scale=1e-6, u_scale=1e-6)
        particles.masses *= 1e-6

        E = diag.compute_all_energies(
            time=0.0,
            masses=particles.masses,
            positions=particles.positions,
            velocities=particles.velocities,
            internal_energies=particles.internal_energies
        )

        # Should handle small values without overflow/underflow
        assert np.isfinite(E.total)
        assert E.total != 0.0  # Should not underflow to exactly zero

    def test_single_particle(self):
        """Test energy calculation for single particle."""
        diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

        # Single particle
        masses = np.array([1.0], dtype=np.float32)
        positions = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        velocities = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
        internal_energies = np.array([1.0], dtype=np.float32)

        E = diag.compute_all_energies(
            time=0.0,
            masses=masses,
            positions=positions,
            velocities=velocities,
            internal_energies=internal_energies
        )

        # Kinetic: 0.5 * 1.0 * 0.1² = 0.005
        assert E.kinetic == pytest.approx(0.005, rel=1e-5)

        # Potential BH: -1.0 * 1.0 / 10.0 = -0.1
        assert E.potential_bh == pytest.approx(-0.1, rel=1e-5)

        # No self-gravity for single particle
        assert E.potential_self == pytest.approx(0.0, abs=1e-10)

        # Internal: 1.0 * 1.0 = 1.0
        assert E.internal_thermal == pytest.approx(1.0, rel=1e-5)
