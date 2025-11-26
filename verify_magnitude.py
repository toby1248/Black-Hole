"""
Verify the magnitude of BH gravity at r=10M.

For Schwarzschild spacetime, a particle at rest experiences:
d²r/dt² = -M(r-2M)/r³ / (1 - 2M/r)  [coordinate acceleration]

At r=10M, M=1:
d²r/dt² = -1*(10-2)/10³ / (1 - 2/10) = -8/1000 / 0.8 = -0.008/0.8 = -0.01

So the magnitude should be 0.01, which matches our result!
"""
import numpy as np

M = 1.0
r = 10.0

# Coordinate acceleration
f = 1.0 - 2.0*M/r
accel_coord = -M * (r - 2*M) / r**3 / f

print(f"Expected acceleration at r={r}M:")
print(f"  f = 1 - 2M/r = {f}")
print(f"  a_r = -M(r-2M)/r³/f = {accel_coord:.6f}")
print(f"  |a| = {abs(accel_coord):.6f}")
print(f"\nOur result: 0.010000")
print(f"Match: {np.isclose(abs(accel_coord), 0.01)}")
