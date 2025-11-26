# Orbit Setup Quick Reference

## Parameter Guide

### Eccentricity (e)
Controls the shape of the orbit:

| Value Range | Orbit Type | Typical Use Case |
|-------------|------------|------------------|
| e = 0       | Circular   | Stable orbiting star (rare for TDEs) |
| 0 < e < 1   | Elliptical | Bound orbits, multiple passages |
| e = 1       | Parabolic  | Marginally bound, single passage |
| e > 1       | Hyperbolic | Unbound, high-velocity encounter |

**Typical TDE**: e ≈ 0.95-0.99 (highly eccentric)

### Pericentre (r_p)
Closest approach distance in units of **tidal radius** (R_t):

```
R_t = R_star * (M_BH / M_star)^(1/3)
```

| r_p Value | Physical Meaning |
|-----------|------------------|
| r_p < 1   | Deep penetration, strong disruption |
| r_p ≈ 1   | Classical TDE at tidal radius |
| r_p > 2   | Weak/partial disruption |
| r_p > 5   | Minimal tidal effects |

**Typical range**: 0.5 - 5.0

### Starting Distance (r_init)
Initial position as multiple of periapsis:

```
r_init = starting_distance * r_p
```

| Value | Usage |
|-------|-------|
| 1.0   | Start at periapsis (already disrupting) |
| 2-3   | Start approaching, watch disruption develop |
| 5-10  | Start far away, see full approach |

**Recommended**: 3.0 (compromise between speed and completeness)

## Configuration Examples

### Deep Plunge TDE
```yaml
orbit:
  pericentre: 0.5        # Half tidal radius
  eccentricity: 0.98     # Highly eccentric
  starting_distance: 3.0
```
**Result**: Strong tidal forces, complete disruption, high fallback rate.

### Grazing Encounter
```yaml
orbit:
  pericentre: 2.0        # 2× tidal radius
  eccentricity: 0.95
  starting_distance: 4.0
```
**Result**: Partial disruption, some material remains bound to remnant.

### Parabolic Orbit
```yaml
orbit:
  pericentre: 1.0        # At tidal radius
  eccentricity: 1.0      # Exactly parabolic
  starting_distance: 5.0
```
**Result**: Classic TDE, star on marginally bound orbit from infinity.

### Circular Orbit (Non-TDE)
```yaml
orbit:
  pericentre: 10.0       # Well outside tidal radius
  eccentricity: 0.0      # Circular
  starting_distance: 1.0 # Start at orbital radius
```
**Result**: Stable orbit, no disruption (good for testing SPH without tides).

## Physics Formulas

### Initial Velocity Magnitude
For elliptical orbits (e < 1):
```
v = sqrt(2 * (E_orb + G*M_BH/r))
where E_orb = -G*M_BH/(2*a), a = r_p/(1-e)
```

For parabolic orbits (e = 1):
```
v = sqrt(2*G*M_BH/r)
```

### Velocity Components
Given velocity magnitude `v` and angular momentum `L`:
```
v_tangential = L / r
v_radial = sqrt(v^2 - v_tangential^2)
```

### Angular Momentum
At periapsis:
```
L = r_p * sqrt(G*M_BH*(1+e)/r_p) = sqrt(G*M_BH*r_p*(1+e))
```

## Coordinate Convention

- **Position**: Star starts at (-r_init, 0, 0)
- **Velocity**: 
  - x-component: +v_radial (toward BH)
  - y-component: -v_tangential (prograde orbit)
  - z-component: 0
- **Angular Momentum**: Points in +z direction

## Validation Checklist

Use these checks to verify orbit setup:

### Energy Conservation
```python
E_check = 0.5 * v_mag**2 - G*M_BH/r_init
E_expected = -G*M_BH/(2*a)  # For elliptical
assert np.isclose(E_check, E_expected)
```

### Angular Momentum Magnitude
```python
L_vec = np.cross(position, velocity)
L_magnitude = np.linalg.norm(L_vec)
L_expected = sqrt(G*M_BH*r_p*(1+e))
assert np.isclose(L_magnitude, L_expected)
```

### Angular Momentum Direction
```python
assert L_vec[2] > 0  # Positive z-component for prograde orbit
```

### Velocity at Periapsis
```python
v_periapsis = sqrt(G*M_BH*(1+e)/r_p)
```

## Common Issues

### Wrong Angular Momentum Sign
**Symptom**: Orbit goes clockwise instead of counter-clockwise  
**Cause**: Tangential velocity in +y instead of -y  
**Fix**: Use `vel[:, 1] += -v_tangential`

### Energy Not Conserved
**Symptom**: Energy check fails, orbit doesn't close  
**Cause**: Incorrect semi-major axis or velocity calculation  
**Fix**: Verify `a = r_p / (1 - e)` for e < 1

### Particle Escapes Immediately
**Symptom**: All particles fly away on first step  
**Cause**: Velocity too high (possibly hyperbolic orbit)  
**Fix**: Check eccentricity < 1, verify velocity formula

### No Disruption Occurs
**Symptom**: Star passes through periapsis intact  
**Cause**: pericentre too large or timestep too coarse  
**Fix**: Reduce pericentre to < 2 R_t, decrease dt_max

## Testing Your Setup

Run the orbit validation tests:
```bash
uv run pytest tests/test_orbit_setup.py -v
```

Expected output:
```
test_orbit_setup_circular PASSED
test_orbit_setup_elliptical PASSED
test_orbit_setup_parabolic PASSED
test_orbit_approaching_periapsis PASSED
```

## Quick Start

1. **Choose your scenario**:
   - Deep TDE: `pericentre: 0.5, eccentricity: 0.98`
   - Classical TDE: `pericentre: 1.0, eccentricity: 0.95`
   - Grazing: `pericentre: 2.5, eccentricity: 0.90`

2. **Set starting distance**: 3.0 (recommended default)

3. **Run simulation** and watch in GUI:
   - Statistics: Median distance shrinks to periapsis, then grows
   - Percentiles: Distance spreads as tidal tails form
   - Energy: Some particles become unbound (E > 0)

4. **Verify physics**:
   - Angular momentum conserved (check logs)
   - Energy evolution makes sense (tidal heating)
   - Disruption occurs near periapsis

## References

- Test suite: `tests/test_orbit_setup.py`
- Implementation: `gui/simulation_thread.py` lines 180-235
- Physics: Murray & Dermott (1999) "Solar System Dynamics"
- TDE context: Rees (1988) Nature 333, 523

---
**Quick Tip**: For a dramatic visualization, try `pericentre: 0.7, eccentricity: 0.98, starting_distance: 4.0`. This gives a classic "spaghettification" TDE with clear tidal tails.
