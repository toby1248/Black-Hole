"""
Test BH gravity direction for a STATIC particle (v=0).
For a particle at rest, gravity should point toward BH (negative x).
"""
import numpy as np
from tde_sph.metric import SchwarzschildMetric
from tde_sph.gravity import RelativisticGravitySolver

print("="*70)
print("TEST: BH Gravity Direction (Static Particle)")
print("="*70)

# Create metric
M_bh = 1.0
metric = SchwarzschildMetric(mass=M_bh)

# Create gravity solver
gravity_solver = RelativisticGravitySolver(
    G=1.0,
    bh_mass=M_bh
)

# Test particle at (10, 0, 0) with ZERO velocity
positions = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)
velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)  # AT REST
masses = np.array([1.0], dtype=np.float32)
smoothing_lengths = np.array([0.1], dtype=np.float32)

print(f"\nParticle state:")
print(f"  Position: {positions[0]}")
print(f"  Velocity: {velocities[0]} (AT REST)")
print(f"  Distance: {np.linalg.norm(positions[0]):.3f} M")

# Compute acceleration
accel = gravity_solver.compute_acceleration(
    positions,
    masses,
    smoothing_lengths,
    metric=metric,
    velocities=velocities
)

print(f"\nBH Gravitational acceleration:")
print(f"  a = {accel[0]}")
print(f"  |a| = {np.linalg.norm(accel[0]):.6f}")

# Check direction
r_hat = positions[0] / np.linalg.norm(positions[0])
a_hat = accel[0] / np.linalg.norm(accel[0])
dot_product = np.dot(r_hat, a_hat)

print(f"\nDirection check:")
print(f"  r_hat (radial outward): {r_hat}")
print(f"  a_hat (accel direction): {a_hat}")
print(f"  dot(r_hat, a_hat) = {dot_product:.3f}")

if dot_product < -0.5:
    print("  [OK] Acceleration points toward BH (attractive)")
elif dot_product > 0.5:
    print("  [ERROR] Acceleration points away from BH (repulsive!)")
else:
    print("  [WARNING] Acceleration is tangential")

# Also test direct metric call
print("\n" + "="*70)
print("Direct metric.geodesic_acceleration() call:")
print("="*70)
accel_direct = metric.geodesic_acceleration(positions, velocities)
print(f"  a = {accel_direct[0]}")
print(f"  |a| = {np.linalg.norm(accel_direct[0]):.6f}")
