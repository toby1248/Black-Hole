# API Reference: Integration & Initial Conditions Modules

## Quick Start

```python
from tde_sph.integration import LeapfrogIntegrator
from tde_sph.ICs import Polytrope
import numpy as np

# Generate polytropic star
poly = Polytrope(gamma=5.0/3.0, eta=1.2, random_seed=42)
positions, velocities, masses, internal_energies, densities = poly.generate(
    n_particles=100000,
    M_star=1.0,
    R_star=1.0
)

# Create integrator
integrator = LeapfrogIntegrator(cfl_factor=0.3)

# Estimate timestep
dt = integrator.estimate_timestep(particles, accelerations=total_accel)

# Take a step
forces = {'gravity': grav_accel, 'hydro': hydro_accel, 'du_dt': heating_rate}
integrator.step(particles, dt, forces)
```

---

## LeapfrogIntegrator

### Constructor

```python
LeapfrogIntegrator(
    cfl_factor: float = 0.3,
    accel_factor: float = 0.25,
    internal_energy_evolution: bool = True
)
```

**Parameters:**
- `cfl_factor`: CFL safety factor for timestep (0.2-0.5 typical)
- `accel_factor`: Acceleration-based timestep safety factor (0.25-0.5)
- `internal_energy_evolution`: Whether to evolve internal energy

### Methods

#### step()
```python
def step(
    self,
    particles: ParticleSystem,
    dt: float,
    forces: Dict[str, NDArrayFloat],
    **kwargs
) -> None
```

Advance particle system by one timestep using kick-drift-kick leapfrog.

**Parameters:**
- `particles`: Particle system with attributes:
  - `positions`: NDArrayFloat, shape (N, 3)
  - `velocities`: NDArrayFloat, shape (N, 3)
  - `internal_energy`: NDArrayFloat, shape (N,)
- `dt`: Timestep duration (float)
- `forces`: Dictionary with keys:
  - `'gravity'`: NDArrayFloat, shape (N, 3) - gravitational acceleration
  - `'hydro'`: NDArrayFloat, shape (N, 3) - hydrodynamic acceleration
  - `'du_dt'`: NDArrayFloat, shape (N,) - internal energy change rate

**Modifies:** `particles.positions`, `particles.velocities`, `particles.internal_energy` in-place

**Notes:**
- First call performs half-step initialization
- Subsequent calls perform full kick-drift-kick
- Enforces minimum internal energy (1e-10)

#### estimate_timestep()
```python
def estimate_timestep(
    self,
    particles: ParticleSystem,
    cfl_factor: Optional[float] = None,
    **kwargs
) -> float
```

Estimate safe timestep based on CFL condition and acceleration.

**Parameters:**
- `particles`: Particle system with attributes:
  - `positions`: NDArrayFloat, shape (N, 3)
  - `velocities`: NDArrayFloat, shape (N, 3)
  - `smoothing_lengths`: NDArrayFloat, shape (N,)
  - `sound_speeds`: NDArrayFloat, shape (N,) OR `internal_energy` for fallback
- `cfl_factor`: Override default CFL factor (optional)
- `**kwargs`:
  - `accelerations`: NDArrayFloat, shape (N, 3) - current acceleration (optional)
  - `sound_speeds`: NDArrayFloat, shape (N,) - if not in particles
  - `min_dt`: float - absolute minimum timestep
  - `max_dt`: float - absolute maximum timestep

**Returns:** Recommended timestep (float)

**Constraints:**
- CFL: `dt_cfl = C * min(h / (c_s + |v|))`
- Acceleration: `dt_acc = C * min(sqrt(h / |a|))`
- Returns: `min(dt_cfl, dt_acc, max_dt)`

#### reset()
```python
def reset(self) -> None
```

Reset integrator state (for restarting simulation).

---

## Polytrope

### Constructor

```python
Polytrope(
    gamma: float = 5.0/3.0,
    eta: float = 1.2,
    random_seed: Optional[int] = 42
)
```

**Parameters:**
- `gamma`: Adiabatic index
  - 5/3: Ideal monatomic gas (convective envelope)
  - 4/3: Radiation-dominated (massive stars, white dwarfs)
- `eta`: SPH smoothing length factor (1.2-1.5 typical)
  - `h_i = eta * (m_i / rho_i)^(1/3)`
- `random_seed`: Random seed for reproducible particle placement (None for random)

### Methods

#### generate()
```python
def generate(
    self,
    n_particles: int,
    **kwargs
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]
```

Generate polytropic star initial conditions.

**Parameters:**
- `n_particles`: Number of SPH particles (10³ to 10⁶)
- `**kwargs`:
  - `M_star`: float, default 1.0 - Total stellar mass (code units)
  - `R_star`: float, default 1.0 - Stellar radius (code units)
  - `position`: NDArrayFloat, shape (3,), default [0,0,0] - Star center
  - `velocity`: NDArrayFloat, shape (3,), default [0,0,0] - Star bulk velocity

**Returns:** Tuple of:
1. `positions`: NDArrayFloat, shape (n_particles, 3) - Particle positions
2. `velocities`: NDArrayFloat, shape (n_particles, 3) - Particle velocities
3. `masses`: NDArrayFloat, shape (n_particles,) - Particle masses (uniform)
4. `internal_energies`: NDArrayFloat, shape (n_particles,) - Specific internal energies
5. `densities`: NDArrayFloat, shape (n_particles,) - Mass densities

**Guarantees:**
- Total mass: `sum(masses) = M_star` (exact)
- Particle masses: Uniform, `m_i = M_star / n_particles`
- Spatial extent: `max(|r|) ≤ R_star`
- Initial velocities: Zero in star rest frame (plus bulk velocity)
- Density profile: Follows Lane-Emden solution for polytrope index n=1/(γ-1)

#### compute_smoothing_lengths()
```python
def compute_smoothing_lengths(
    self,
    masses: NDArrayFloat,
    densities: NDArrayFloat
) -> NDArrayFloat
```

Compute SPH smoothing lengths.

**Parameters:**
- `masses`: NDArrayFloat, shape (N,) - Particle masses
- `densities`: NDArrayFloat, shape (N,) - Particle densities

**Returns:** `smoothing_lengths`: NDArrayFloat, shape (N,)

**Formula:** `h_i = eta * (m_i / rho_i)^(1/3)`

---

## Data Types

All arrays use **float32** for GPU performance:

```python
from numpy.typing import NDArrayFloat
NDArrayFloat = np.ndarray[np.float32]
```

### Particle System Requirements

Your `ParticleSystem` class should have these attributes for full compatibility:

```python
class ParticleSystem:
    positions: NDArrayFloat          # shape (N, 3)
    velocities: NDArrayFloat         # shape (N, 3)
    masses: NDArrayFloat             # shape (N,)
    internal_energy: NDArrayFloat    # shape (N,)
    densities: NDArrayFloat          # shape (N,)
    smoothing_lengths: NDArrayFloat  # shape (N,)
    sound_speeds: NDArrayFloat       # shape (N,) - optional, computed from u if missing
    accelerations: NDArrayFloat      # shape (N, 3) - optional, for timestep estimation
```

---

## Units & Conventions

### Dimensionless Units
Both modules work internally in dimensionless units:
- G = 1 (gravitational constant)
- M = 1 (reference mass)
- R = 1 (reference radius)

### Physical Units
Convert to physical units via `M_star` and `R_star`:
- Length: `x_physical = x_code * R_star`
- Mass: `m_physical = m_code * M_star`
- Density: `rho_physical = rho_code * (M_star / R_star^3)`
- Velocity: `v_physical = v_code * sqrt(G * M_star / R_star)`
- Energy: `E_physical = E_code * (G * M_star^2 / R_star)`
- Time: `t_physical = t_code * sqrt(R_star^3 / (G * M_star))`

### Example Unit Conversion
```python
# Code units: G=1, M_star=1 solar mass, R_star=1 solar radius
# To physical (SI):
G_SI = 6.674e-11       # m^3 kg^-1 s^-2
M_sun = 1.989e30       # kg
R_sun = 6.957e8        # m

# Velocity scale
v_scale = np.sqrt(G_SI * M_sun / R_sun)  # ~618 km/s

# Time scale
t_scale = np.sqrt(R_sun**3 / (G_SI * M_sun))  # ~1682 s ~ 28 min
```

---

## Common Patterns

### Pattern 1: Generate Star and Evolve
```python
# Create star
poly = Polytrope(gamma=5.0/3.0, random_seed=42)
pos, vel, mass, u, rho = poly.generate(
    n_particles=50000,
    M_star=1.0,
    R_star=1.0,
    position=np.array([10.0, 0.0, 0.0]),  # Offset from BH
    velocity=np.array([0.0, 0.5, 0.0])    # Orbital velocity
)

# Compute smoothing lengths
h = poly.compute_smoothing_lengths(mass, rho)

# Create particle system
particles = ParticleSystem(pos, vel, mass, u, rho, h)

# Initialize integrator
integrator = LeapfrogIntegrator(cfl_factor=0.3)

# Main loop
t = 0.0
while t < t_max:
    # Compute forces (from gravity, SPH modules)
    accel_grav = gravity_solver.compute_acceleration(...)
    accel_hydro = sph_solver.compute_hydro_forces(...)
    du_dt = sph_solver.compute_heating_rate(...)

    forces = {
        'gravity': accel_grav,
        'hydro': accel_hydro,
        'du_dt': du_dt
    }

    # Estimate timestep
    dt = integrator.estimate_timestep(
        particles,
        accelerations=accel_grav + accel_hydro
    )

    # Advance
    integrator.step(particles, dt, forces)
    t += dt
```

### Pattern 2: Multiple Polytropes with Different γ
```python
# Convective star (γ=5/3)
poly_conv = Polytrope(gamma=5.0/3.0, random_seed=1)
pos1, vel1, m1, u1, rho1 = poly_conv.generate(10000, M_star=1.0, R_star=1.0)

# Radiation-dominated star (γ=4/3)
poly_rad = Polytrope(gamma=4.0/3.0, random_seed=2)
pos2, vel2, m2, u2, rho2 = poly_rad.generate(10000, M_star=1.0, R_star=1.5)

# Combine (e.g., for binary)
positions = np.vstack([pos1, pos2])
velocities = np.vstack([vel1, vel2])
masses = np.concatenate([m1, m2])
```

### Pattern 3: Adaptive Timestep with Bounds
```python
integrator = LeapfrogIntegrator(cfl_factor=0.3)

# Set absolute bounds
dt_min = 1e-6  # Minimum timestep (prevent stalling)
dt_max = 0.1   # Maximum timestep (for output cadence)

dt = integrator.estimate_timestep(
    particles,
    accelerations=total_accel,
    min_dt=dt_min,
    max_dt=dt_max
)
```

---

## Performance Tips

### Polytrope Generation
- **Cache Lane-Emden solutions:** Solutions are cached automatically per instance
- **Reuse instances:** Create one `Polytrope` object and call `generate()` multiple times
- **Parallel generation:** Create multiple stars in parallel threads (thread-safe if different random seeds)

```python
# Good: Reuse instance
poly = Polytrope(gamma=5.0/3.0)
stars = [poly.generate(10000, M_star=1.0, R_star=1.0) for _ in range(10)]

# Avoid: Recreating instance (re-solves Lane-Emden each time)
stars = [Polytrope(gamma=5.0/3.0).generate(10000) for _ in range(10)]
```

### Leapfrog Integration
- **Vectorization:** All operations are vectorized NumPy - avoid Python loops
- **In-place updates:** Particle arrays updated in-place, no memory copies
- **Timestep caching:** If forces don't change much, cache timestep for multiple steps

```python
# Moderate: Re-estimate dt every step
for step in range(n_steps):
    dt = integrator.estimate_timestep(particles, accelerations=accel)
    integrator.step(particles, dt, forces)

# Faster: Re-estimate dt every N steps
dt = integrator.estimate_timestep(particles, accelerations=accel)
for step in range(n_steps):
    integrator.step(particles, dt, forces)
    if step % 10 == 0:  # Update every 10 steps
        dt = integrator.estimate_timestep(particles, accelerations=accel)
```

---

## Error Handling

### Common Issues

**Issue:** "Particles outside stellar radius"
- **Cause:** Bug in radial sampling (fixed in current version)
- **Check:** `max(|r|) <= R_star` should always hold

**Issue:** "Timestep too small / simulation stalls"
- **Cause:** Very high accelerations or small smoothing lengths
- **Solution:** Set `min_dt` parameter or check for unphysical forces

**Issue:** "Negative internal energy"
- **Cause:** Excessive cooling or numerical errors
- **Solution:** LeapfrogIntegrator enforces `u >= 1e-10` floor automatically

**Issue:** "Energy not conserved"
- **Cause:** Timestep too large, or non-conservative forces
- **Solution:** Reduce `cfl_factor`, check force implementations

### Validation Checks

```python
# After IC generation
assert np.abs(np.sum(masses) - M_star) < 1e-6, "Mass not conserved"
assert np.max(np.linalg.norm(positions, axis=1)) <= R_star * 1.01, "Particles outside star"
assert np.all(masses > 0), "Negative masses"
assert np.all(densities > 0), "Negative densities"
assert np.all(internal_energies > 0), "Negative internal energy"

# During integration
E_total = compute_total_energy(particles)
assert np.abs((E_total - E_initial) / E_initial) < 0.01, "Energy drift > 1%"
```

---

## References

### Leapfrog Integration
- Price, D. J. (2012), JCP, 231, 759 - SPH review
- Springel, V. (2005), MNRAS, 364, 1105 - GADGET-2
- Rosswog, S. (2009), New Astron. Rev., 53, 78 - SPH TDEs

### Polytropic Stars
- Guillochon & Ramirez-Ruiz (2013), ApJ, 767, 25 - TDE stellar models
- Chandrasekhar (1939), "Introduction to Stellar Structure" - Lane-Emden theory
- Rosswog et al. (2009) - SPH TDE simulations

### SPH Timesteps
- Price (2012), Eq. 60-62 - CFL and acceleration constraints
- Monaghan (2005), Rep. Prog. Phys., 68, 1703 - SPH review

---

## Support & Extensions

For questions or feature requests, see:
- Main documentation: `/home/user/Black-Hole/CLAUDE.md`
- Implementation summary: `/home/user/Black-Hole/IMPLEMENTATION_SUMMARY.md`
- Example code: `/home/user/Black-Hole/examples/test_integration_and_ics.py`
