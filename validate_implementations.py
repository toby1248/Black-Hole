#!/usr/bin/env python3
"""
Simple validation script for NewtonianGravity and IdealGas implementations.

Tests basic functionality and edge cases without requiring full test framework.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/Black-Hole/src')

from tde_sph.gravity import NewtonianGravity
from tde_sph.eos import IdealGas


def test_newtonian_gravity():
    """Test Newtonian gravity solver."""
    print("=" * 70)
    print("Testing NewtonianGravity")
    print("=" * 70)

    # Create simple two-particle system
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float32)

    masses = np.array([1.0, 1.0], dtype=np.float32)
    smoothing = np.array([0.1, 0.1], dtype=np.float32)

    solver = NewtonianGravity(G=1.0)
    print(f"Solver: {solver}")

    # Test acceleration
    accel = solver.compute_acceleration(positions, masses, smoothing)
    print(f"\nAcceleration shape: {accel.shape}")
    print(f"Acceleration dtype: {accel.dtype}")
    print(f"Particle 0 acceleration: {accel[0]}")
    print(f"Particle 1 acceleration: {accel[1]}")

    # Check symmetry (equal and opposite)
    print(f"\nAcceleration sum (should be ~0): {np.sum(accel, axis=0)}")
    assert np.allclose(accel[0], -accel[1], atol=1e-6), "Accelerations not symmetric!"

    # Test potential
    potential = solver.compute_potential(positions, masses, smoothing)
    print(f"\nPotential shape: {potential.shape}")
    print(f"Potential dtype: {potential.dtype}")
    print(f"Particle 0 potential: {potential[0]:.6f}")
    print(f"Particle 1 potential: {potential[1]:.6f}")

    # Check symmetry (should be equal for equal masses)
    assert np.allclose(potential[0], potential[1], atol=1e-6), "Potentials not symmetric!"

    # Test many-particle system
    print("\n" + "-" * 70)
    print("Testing with 100 particles...")
    np.random.seed(42)
    N = 100
    positions_many = np.random.randn(N, 3).astype(np.float32)
    masses_many = np.ones(N, dtype=np.float32)
    smoothing_many = 0.1 * np.ones(N, dtype=np.float32)

    accel_many = solver.compute_acceleration(positions_many, masses_many, smoothing_many)
    potential_many = solver.compute_potential(positions_many, masses_many, smoothing_many)

    print(f"Acceleration range: [{accel_many.min():.3f}, {accel_many.max():.3f}]")
    print(f"Potential range: [{potential_many.min():.3f}, {potential_many.max():.3f}]")

    print("\n✓ NewtonianGravity tests passed!")


def test_ideal_gas():
    """Test ideal gas EOS."""
    print("\n" + "=" * 70)
    print("Testing IdealGas EOS")
    print("=" * 70)

    # Test gamma = 5/3 (monatomic)
    eos_monatomic = IdealGas(gamma=5.0/3.0, mean_molecular_weight=0.6)
    print(f"EOS (monatomic): {eos_monatomic}")

    # Test single particle
    density = np.array([1.0], dtype=np.float32)
    internal_energy = np.array([1.0e12], dtype=np.float32)  # CGS-like units

    pressure = eos_monatomic.pressure(density, internal_energy)
    sound_speed = eos_monatomic.sound_speed(density, internal_energy)
    temperature = eos_monatomic.temperature(density, internal_energy)

    print(f"\nSingle particle test:")
    print(f"  Density: {density[0]:.2e}")
    print(f"  Internal energy: {internal_energy[0]:.2e}")
    print(f"  Pressure: {pressure[0]:.2e}")
    print(f"  Sound speed: {sound_speed[0]:.2e}")
    print(f"  Temperature: {temperature[0]:.2e} K")

    # Verify P = (gamma - 1) * rho * u
    expected_pressure = (5.0/3.0 - 1.0) * density[0] * internal_energy[0]
    assert np.allclose(pressure[0], expected_pressure, rtol=1e-5), "Pressure formula incorrect!"

    # Test gamma = 4/3 (relativistic)
    print("\n" + "-" * 70)
    eos_relativistic = IdealGas(gamma=4.0/3.0, mean_molecular_weight=0.6)
    print(f"EOS (relativistic): {eos_relativistic}")

    pressure_rel = eos_relativistic.pressure(density, internal_energy)
    sound_speed_rel = eos_relativistic.sound_speed(density, internal_energy)

    print(f"\nRelativistic gas:")
    print(f"  Pressure: {pressure_rel[0]:.2e}")
    print(f"  Sound speed: {sound_speed_rel[0]:.2e}")

    # Test with arrays
    print("\n" + "-" * 70)
    print("Testing with 100 particles...")
    np.random.seed(42)
    N = 100
    densities = np.random.uniform(0.1, 10.0, N).astype(np.float32)
    energies = np.random.uniform(1e11, 1e13, N).astype(np.float32)

    pressures = eos_monatomic.pressure(densities, energies)
    sound_speeds = eos_monatomic.sound_speed(densities, energies)
    temperatures = eos_monatomic.temperature(densities, energies)

    print(f"Pressure range: [{pressures.min():.2e}, {pressures.max():.2e}]")
    print(f"Sound speed range: [{sound_speeds.min():.2e}, {sound_speeds.max():.2e}]")
    print(f"Temperature range: [{temperatures.min():.2e}, {temperatures.max():.2e}] K")

    # Test round-trip: T -> u -> T
    u_from_T = eos_monatomic.internal_energy_from_temperature(temperatures)
    T_reconstructed = eos_monatomic.temperature(densities, u_from_T)
    assert np.allclose(temperatures, T_reconstructed, rtol=1e-4), "T->u->T round-trip failed!"
    print("\n✓ T -> u -> T round-trip successful")

    # Test edge cases
    print("\n" + "-" * 70)
    print("Testing edge cases...")

    # Zero density and energy
    zero_density = np.array([0.0], dtype=np.float32)
    zero_energy = np.array([0.0], dtype=np.float32)

    p_zero = eos_monatomic.pressure(zero_density, zero_energy)
    cs_zero = eos_monatomic.sound_speed(zero_density, zero_energy)
    T_zero = eos_monatomic.temperature(zero_density, zero_energy)

    print(f"Zero density/energy: P={p_zero[0]}, cs={cs_zero[0]}, T={T_zero[0]}")
    assert p_zero[0] == 0.0, "Zero pressure expected!"
    assert T_zero[0] == 0.0, "Zero temperature expected!"

    # Negative values (should be handled gracefully)
    negative_energy = np.array([-1.0], dtype=np.float32)
    p_neg = eos_monatomic.pressure(density, negative_energy)
    print(f"Negative energy: P={p_neg[0]} (should be 0)")
    assert p_neg[0] == 0.0, "Should handle negative energy gracefully!"

    print("\n✓ IdealGas EOS tests passed!")


def test_interface_compliance():
    """Test that implementations comply with abstract interfaces."""
    print("\n" + "=" * 70)
    print("Testing Interface Compliance")
    print("=" * 70)

    from tde_sph.core.interfaces import GravitySolver, EOS

    # Check inheritance
    gravity = NewtonianGravity()
    eos = IdealGas()

    print(f"NewtonianGravity is GravitySolver: {isinstance(gravity, GravitySolver)}")
    print(f"IdealGas is EOS: {isinstance(eos, EOS)}")

    assert isinstance(gravity, GravitySolver), "NewtonianGravity must inherit from GravitySolver!"
    assert isinstance(eos, EOS), "IdealGas must inherit from EOS!"

    print("\n✓ Interface compliance verified!")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("TDE-SPH Module Validation")
    print("=" * 70)

    try:
        test_newtonian_gravity()
        test_ideal_gas()
        test_interface_compliance()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
