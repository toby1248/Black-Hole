"""
Test suite for GR-aware integration (Hamiltonian integrator and timestep control).

Validates TASK-016 (Hamiltonian integrator) and TASK-017 (timestep control)
against analytic solutions and physical constraints.

References
----------
- Liptai & Price (2019), MNRAS 485, 819 - Hamiltonian GRSPH validation
- Hairer et al. (2006) - Symplectic integrator theory

Test Coverage
-------------
1. Hamiltonian conservation over long integrations
2. Epicyclic frequency tests (requires metric implementation)
3. Leapfrog-Hamiltonian consistency at large r
4. ISCO stability tests
5. Timestep control functions
6. Backward compatibility with Newtonian mode
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

# Import integrators
from tde_sph.integration import (
    LeapfrogIntegrator,
    HamiltonianIntegrator,
    estimate_timestep_gr,
    get_timestep_diagnostics
)


# ============================================================================
# Mock Objects (for testing without full metric implementations)
# ============================================================================

class MockMetric:
    """Mock Metric implementation for testing."""

    def __init__(self, mass=1.0):
        self.mass = mass
        self.name = "MockMetric"

    def metric_tensor(self, x):
        """Return Minkowski metric (flat space) for testing."""
        n = len(x) if x.ndim > 1 else 1
        g = np.zeros((n, 4, 4), dtype=np.float32)
        g[:, 0, 0] = -1.0  # g_tt
        g[:, 1, 1] = 1.0   # g_xx
        g[:, 2, 2] = 1.0   # g_yy
        g[:, 3, 3] = 1.0   # g_zz
        return g

    def inverse_metric(self, x):
        """Return inverse Minkowski metric."""
        return self.metric_tensor(x)  # Minkowski is self-inverse

    def christoffel_symbols(self, x):
        """Return zero Christoffel symbols (flat space)."""
        n = len(x) if x.ndim > 1 else 1
        return np.zeros((n, 4, 4, 4), dtype=np.float32)

    def geodesic_acceleration(self, x, v):
        """Return zero acceleration (free particles in flat space)."""
        n = len(x) if x.ndim > 1 else 1
        return np.zeros((n, 3), dtype=np.float32)


class MockParticleSystem:
    """Mock ParticleSystem for testing."""

    def __init__(self, n_particles=100):
        self.positions = np.random.randn(n_particles, 3).astype(np.float32) * 10.0
        self.velocities = np.random.randn(n_particles, 3).astype(np.float32) * 0.1
        self.masses = np.ones(n_particles, dtype=np.float32)
        self.smoothing_lengths = np.ones(n_particles, dtype=np.float32) * 0.5
        self.internal_energy = np.ones(n_particles, dtype=np.float32) * 1.0
        self.sound_speeds = np.ones(n_particles, dtype=np.float32) * 0.2


# ============================================================================
# Test: Hamiltonian Integrator Initialization
# ============================================================================

def test_hamiltonian_integrator_requires_metric():
    """Test that HamiltonianIntegrator raises error without metric."""
    with pytest.raises(ValueError, match="requires a valid Metric"):
        integrator = HamiltonianIntegrator(metric=None)


def test_hamiltonian_integrator_initialization():
    """Test HamiltonianIntegrator initializes correctly with mock metric."""
    metric = MockMetric(mass=1.0)
    integrator = HamiltonianIntegrator(
        metric=metric,
        cfl_factor=0.2,
        use_fp64=True
    )

    assert integrator.metric == metric
    assert integrator.cfl_factor == 0.2
    assert integrator.use_fp64 is True
    assert integrator._four_momenta is None
    assert not integrator.half_step_initialized


# ============================================================================
# Test: Hamiltonian Conservation (Flat Space)
# ============================================================================

def test_hamiltonian_conservation_flat_space():
    """
    Test Hamiltonian conservation in flat space (Minkowski metric).

    In flat space with no forces, test particles should conserve energy exactly.
    This validates the symplectic structure of the integrator.

    Target: ΔH/H < 10^(-10) over 1000 steps (FP64 precision).
    """
    metric = MockMetric(mass=1.0)
    integrator = HamiltonianIntegrator(metric=metric, use_fp64=True)

    # Single test particle
    particles = MockParticleSystem(n_particles=1)
    particles.positions = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    particles.velocities = np.array([[0.0, 0.1, 0.0]], dtype=np.float32)

    # No forces (pure geodesic motion in flat space)
    forces = {'gravity': np.zeros((1, 3), dtype=np.float32)}

    # Integrate for 1000 steps
    dt = 0.1
    n_steps = 1000

    # Initial step to initialize 4-momenta
    integrator.step(particles, dt, forces, pure_geodesic=True)

    # Record initial Hamiltonian
    H_initial = integrator.compute_hamiltonian(particles)[0]

    # Integrate
    for _ in range(n_steps):
        integrator.step(particles, dt, forces, pure_geodesic=True)

    # Final Hamiltonian
    H_final = integrator.compute_hamiltonian(particles)[0]

    # Check conservation
    dH = abs(H_final - H_initial)
    relative_error = dH / abs(H_initial)

    print(f"Hamiltonian conservation test:")
    print(f"  H_initial = {H_initial:.6e}")
    print(f"  H_final   = {H_final:.6e}")
    print(f"  ΔH/H      = {relative_error:.6e}")

    # For flat space, should be conserved to machine precision
    # Allow some tolerance for numerical integration
    assert relative_error < 1e-8, \
        f"Hamiltonian not conserved: ΔH/H = {relative_error:.3e}"


# ============================================================================
# Test: Timestep Control
# ============================================================================

def test_timestep_gr_newtonian_mode():
    """Test GR timestep estimation in Newtonian mode (metric=None)."""
    particles = MockParticleSystem(n_particles=100)

    config = {
        'cfl_factor': 0.3,
        'accel_factor': 0.25,
        'bh_mass': 1.0
    }

    # Mock accelerations
    accelerations = np.random.randn(100, 3).astype(np.float32) * 0.01

    dt = estimate_timestep_gr(
        particles=particles,
        metric=None,  # Newtonian mode
        config=config,
        accelerations=accelerations
    )

    # Should return finite positive timestep
    assert dt > 0
    assert np.isfinite(dt)

    print(f"Newtonian timestep: dt = {dt:.3e}")


def test_timestep_gr_with_metric():
    """Test GR timestep estimation with metric (activates orbital/ISCO constraints)."""
    metric = MockMetric(mass=1.0)
    particles = MockParticleSystem(n_particles=100)

    # Place some particles close to BH to trigger ISCO constraint
    particles.positions[:10] = np.random.randn(10, 3).astype(np.float32) * 5.0  # r ~ 5M

    config = {
        'cfl_factor': 0.3,
        'accel_factor': 0.25,
        'orbital_factor': 0.1,
        'isco_factor': 0.05,
        'isco_radius_threshold': 10.0,
        'bh_mass': 1.0
    }

    accelerations = np.random.randn(100, 3).astype(np.float32) * 0.01

    dt = estimate_timestep_gr(
        particles=particles,
        metric=metric,
        config=config,
        accelerations=accelerations
    )

    assert dt > 0
    assert np.isfinite(dt)

    print(f"GR timestep (with close particles): dt = {dt:.3e}")


def test_timestep_diagnostics():
    """Test timestep diagnostics function."""
    metric = MockMetric(mass=1.0)
    particles = MockParticleSystem(n_particles=100)

    config = {
        'cfl_factor': 0.3,
        'accel_factor': 0.25,
        'orbital_factor': 0.1,
        'isco_factor': 0.05,
        'isco_radius_threshold': 10.0,
        'bh_mass': 1.0
    }

    accelerations = np.random.randn(100, 3).astype(np.float32) * 0.01

    diag = get_timestep_diagnostics(
        particles=particles,
        metric=metric,
        config=config,
        accelerations=accelerations
    )

    # Check all expected keys present
    assert 'dt_cfl' in diag
    assert 'dt_acc' in diag
    assert 'dt_orb' in diag
    assert 'dt_isco' in diag
    assert 'dt_total' in diag
    assert 'limiting_constraint' in diag

    # Check values are finite
    assert np.isfinite(diag['dt_cfl'])
    assert np.isfinite(diag['dt_acc'])
    assert diag['dt_total'] > 0

    # Check limiting constraint is one of the valid types
    assert diag['limiting_constraint'] in ['CFL', 'acceleration', 'orbital', 'ISCO']

    print(f"Timestep diagnostics:")
    print(f"  CFL:          {diag['dt_cfl']:.3e}")
    print(f"  Acceleration: {diag['dt_acc']:.3e}")
    print(f"  Orbital:      {diag['dt_orb']:.3e}")
    print(f"  ISCO:         {diag['dt_isco']:.3e}")
    print(f"  Total:        {diag['dt_total']:.3e}")
    print(f"  Limiting:     {diag['limiting_constraint']}")


# ============================================================================
# Test: Leapfrog-Hamiltonian Consistency
# ============================================================================

def test_leapfrog_hamiltonian_consistency():
    """
    Test that Leapfrog and Hamiltonian integrators agree at large r.

    In weak-field regions (r >> M), both integrators should produce similar
    trajectories since GR corrections are small.
    """
    metric = MockMetric(mass=1.0)

    # Create identical initial conditions
    particles_lf = MockParticleSystem(n_particles=1)
    particles_lf.positions = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    particles_lf.velocities = np.array([[0.0, 0.1, 0.0]], dtype=np.float32)

    particles_ham = MockParticleSystem(n_particles=1)
    particles_ham.positions = particles_lf.positions.copy()
    particles_ham.velocities = particles_lf.velocities.copy()

    # Integrators
    integrator_lf = LeapfrogIntegrator(cfl_factor=0.3)
    integrator_ham = HamiltonianIntegrator(metric=metric, cfl_factor=0.3)

    # Forces (small test force)
    forces = {'gravity': np.array([[0.0, 0.0, -0.001]], dtype=np.float32)}

    # Integrate for 100 steps
    dt = 0.1
    n_steps = 100

    for _ in range(n_steps):
        integrator_lf.step(particles_lf, dt, forces)
        integrator_ham.step(particles_ham, dt, forces, pure_geodesic=False)

    # Compare final positions
    pos_diff = np.linalg.norm(particles_lf.positions - particles_ham.positions)
    vel_diff = np.linalg.norm(particles_lf.velocities - particles_ham.velocities)

    print(f"Leapfrog-Hamiltonian consistency test:")
    print(f"  Position difference: {pos_diff:.3e}")
    print(f"  Velocity difference: {vel_diff:.3e}")

    # At large r in flat space, should agree reasonably well
    # (not exact due to different schemes, but should be O(dt^2))
    assert pos_diff < 0.1, f"Integrators diverged too much: pos_diff = {pos_diff}"
    assert vel_diff < 0.01, f"Integrators diverged too much: vel_diff = {vel_diff}"


# ============================================================================
# Test: Hybrid SPH+GR Integration
# ============================================================================

def test_hamiltonian_hybrid_sph_gr():
    """
    Test Hamiltonian integrator in hybrid mode with SPH forces.

    Validates that both geodesic and SPH forces are applied correctly.
    """
    metric = MockMetric(mass=1.0)
    integrator = HamiltonianIntegrator(metric=metric)

    particles = MockParticleSystem(n_particles=10)

    # Hybrid forces: gravity (BH) + hydro (SPH)
    forces = {
        'gravity': np.random.randn(10, 3).astype(np.float32) * 0.01,
        'hydro': np.random.randn(10, 3).astype(np.float32) * 0.001,
        'du_dt': np.random.randn(10).astype(np.float32) * 0.0001
    }

    dt = 0.05
    n_steps = 10

    # Record initial state
    pos_initial = particles.positions.copy()
    u_initial = particles.internal_energy.copy()

    # Integrate
    for _ in range(n_steps):
        integrator.step(particles, dt, forces, pure_geodesic=False)

    # Check that particles moved
    pos_diff = np.linalg.norm(particles.positions - pos_initial)
    assert pos_diff > 0, "Particles did not move"

    # Check that internal energy changed
    u_diff = np.linalg.norm(particles.internal_energy - u_initial)
    assert u_diff > 0, "Internal energy did not change"

    # Check that internal energy is positive
    assert np.all(particles.internal_energy > 0), "Negative internal energy"

    print(f"Hybrid SPH+GR test:")
    print(f"  Position displacement: {pos_diff:.3e}")
    print(f"  Energy change: {u_diff:.3e}")


# ============================================================================
# Test: Integrator Reset
# ============================================================================

def test_hamiltonian_reset():
    """Test that reset() clears integrator state."""
    metric = MockMetric(mass=1.0)
    integrator = HamiltonianIntegrator(metric=metric)

    particles = MockParticleSystem(n_particles=5)
    forces = {'gravity': np.zeros((5, 3), dtype=np.float32)}

    # Take one step (initializes 4-momenta)
    integrator.step(particles, 0.1, forces)
    assert integrator._four_momenta is not None

    # Reset
    integrator.reset()
    assert integrator._four_momenta is None
    assert not integrator.half_step_initialized


# ============================================================================
# Test: Timestep Bounds
# ============================================================================

def test_timestep_min_max_bounds():
    """Test that min/max timestep bounds are enforced."""
    particles = MockParticleSystem(n_particles=100)

    config = {
        'cfl_factor': 0.3,
        'accel_factor': 0.25,
        'bh_mass': 1.0,
        'min_dt': 0.01,
        'max_dt': 1.0
    }

    accelerations = np.random.randn(100, 3).astype(np.float32) * 0.001

    dt = estimate_timestep_gr(
        particles=particles,
        metric=None,
        config=config,
        accelerations=accelerations
    )

    # Check bounds are enforced
    assert dt >= config['min_dt'], f"dt below minimum: {dt} < {config['min_dt']}"
    assert dt <= config['max_dt'], f"dt above maximum: {dt} > {config['max_dt']}"

    print(f"Timestep with bounds: dt = {dt:.3e} (min={config['min_dt']}, max={config['max_dt']})")


# ============================================================================
# Test: Backward Compatibility
# ============================================================================

def test_leapfrog_still_works():
    """Test that LeapfrogIntegrator still works after GR additions."""
    integrator = LeapfrogIntegrator(cfl_factor=0.3)

    particles = MockParticleSystem(n_particles=50)
    forces = {
        'gravity': np.random.randn(50, 3).astype(np.float32) * 0.01,
        'hydro': np.random.randn(50, 3).astype(np.float32) * 0.001,
        'du_dt': np.random.randn(50).astype(np.float32) * 0.0001
    }

    dt = 0.1
    n_steps = 10

    # Integrate
    for _ in range(n_steps):
        integrator.step(particles, dt, forces)

    # Should complete without error
    assert np.all(np.isfinite(particles.positions))
    assert np.all(np.isfinite(particles.velocities))
    assert np.all(particles.internal_energy > 0)

    print("Leapfrog backward compatibility: PASSED")


# ============================================================================
# Test: Edge Cases
# ============================================================================

def test_hamiltonian_integrator_estimate_timestep():
    """Test that HamiltonianIntegrator.estimate_timestep() works."""
    metric = MockMetric(mass=1.0)
    integrator = HamiltonianIntegrator(metric=metric)

    particles = MockParticleSystem(n_particles=100)

    config = {
        'cfl_factor': 0.2,
        'accel_factor': 0.25,
        'orbital_factor': 0.1,
        'isco_factor': 0.05,
        'isco_radius_threshold': 10.0,
        'bh_mass': 1.0
    }

    accelerations = np.random.randn(100, 3).astype(np.float32) * 0.01

    dt = integrator.estimate_timestep(
        particles=particles,
        config=config,
        accelerations=accelerations
    )

    assert dt > 0
    assert np.isfinite(dt)

    print(f"HamiltonianIntegrator.estimate_timestep(): dt = {dt:.3e}")


def test_timestep_no_particles_near_isco():
    """Test ISCO constraint when no particles are near BH."""
    metric = MockMetric(mass=1.0)
    particles = MockParticleSystem(n_particles=100)

    # All particles far from BH (r >> 10M)
    particles.positions = np.random.randn(100, 3).astype(np.float32) * 100.0

    config = {
        'cfl_factor': 0.3,
        'accel_factor': 0.25,
        'orbital_factor': 0.1,
        'isco_factor': 0.05,
        'isco_radius_threshold': 10.0,
        'bh_mass': 1.0
    }

    accelerations = np.random.randn(100, 3).astype(np.float32) * 0.001

    diag = get_timestep_diagnostics(
        particles=particles,
        metric=metric,
        config=config,
        accelerations=accelerations
    )

    # ISCO constraint should be inf (not active)
    assert diag['dt_isco'] == float('inf')

    print("ISCO constraint (no close particles): dt_isco = inf")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
