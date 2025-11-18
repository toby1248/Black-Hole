# TDE-SPH Example Demonstrations (TASK-039)

Comprehensive examples showcasing the TDE-SPH framework capabilities for Newtonian and General Relativistic tidal disruption event simulations.

## Quick Start

All examples are standalone Python scripts that can be run directly:

```bash
# Example 1: Newtonian TDE
python examples/01_newtonian_tde_demo.py

# Example 2: Schwarzschild GR comparison
python examples/02_schwarzschild_gr_comparison.py

# Example 3: Kerr black hole with inclined orbit
python examples/03_kerr_inclined_orbit.py
```

## Examples Overview

### 01_newtonian_tde_demo.py

**Newtonian Tidal Disruption Event Simulation**

Demonstrates a complete Newtonian TDE workflow:
- Generates a polytropic star (n=1.5, solar-type)
- Places star on parabolic orbit heading toward periapsis
- Computes energy components (kinetic, potential, internal)
- Tracks energy conservation
- Exports snapshots to HDF5, PLY, and VTK formats
- Creates visualization plots

**Physics:**
- Star: M_star = 0.001 M_BH, R_star = 0.01 (code units)
- Orbit: Parabolic (e ≈ 1), periapsis r_p = 0.05
- Tidal radius: r_t = R_star × (M_BH / M_star)^(1/3)
- Penetration factor: β = r_t / r_p

**Outputs:**
- `output_newtonian_tde/snapshot_*.h5`: HDF5 snapshots
- `output_newtonian_tde/snapshot_*.ply`: Blender point clouds
- `output_newtonian_tde/snapshot_*.vtk`: ParaView data
- `output_newtonian_tde/energy_evolution.png`: Energy diagnostic plots
- `output_newtonian_tde/particles_3d.html`: Interactive 3D visualization (if Plotly available)

**Expected Results:**
- Tidal disruption at periapsis
- Debris stream formation
- Energy conservation ΔE/E₀ < 0.01
- Visualization of particle distribution

**References:**
- Guillochon & Ramirez-Ruiz (2013) - TDE simulations
- Tejeda et al. (2017) - Relativistic TDEs
- Price (2012) - SPH methodology

---

### 02_schwarzschild_gr_comparison.py

**Schwarzschild GR vs Newtonian Orbit Comparison**

Compares a test particle on an eccentric orbit in Schwarzschild spacetime versus Newtonian gravity.

**Physics:**
- Eccentric orbit: semi-major axis a = 15M, eccentricity e = 0.4
- Schwarzschild metric (non-rotating BH, spin a = 0)
- ISCO radius: r_ISCO = 6M (Schwarzschild)
- Periapsis precession in GR: Δϕ ≈ 6πGM / (c²a(1-e²)) per orbit

**Features:**
- Integrates same orbit in both Newtonian and GR modes
- Measures periapsis precession (GR effect)
- Compares with theoretical prediction
- Visualizes trajectory differences

**Outputs:**
- `output_schwarzschild_comparison/schwarzschild_vs_newtonian.png`: Multi-panel comparison plots
  - XY trajectory overlay
  - Radial distance vs time
  - Orbital velocity evolution
  - Precession detail zoom

**Expected Results:**
- GR orbit shows periapsis precession (~15-30° over 5 orbits)
- Newtonian orbit has minimal precession (~0°)
- Agreement with theoretical GR prediction within ~10-20%
- Orbital period differs slightly (GR vs Newtonian)

**Key Findings:**
- Periapsis precession is a hallmark of GR
- Effect increases closer to ISCO
- Demonstrates hybrid GR-SPH approach validity

**References:**
- Tejeda et al. (2017) - Hybrid GR approach
- Liptai & Price (2019) - GRSPH validation
- Misner, Thorne, Wheeler (1973) - Gravitation

---

### 03_kerr_inclined_orbit.py

**Kerr Black Hole with Inclined Orbit**

Demonstrates orbital dynamics around a spinning (Kerr) black hole, including orbits inclined to the equatorial plane.

**Physics:**
- Kerr black hole: spin parameter a = 0.7 (moderate spin)
- ISCO (prograde): r_ISCO ≈ 2.5M (depends on a)
- ISCO (retrograde): r_ISCO ≈ 8M
- ISCO (Schwarzschild, a=0): r_ISCO = 6M
- Lense-Thirring precession: Ω_LT ∝ 2GMa / (c²r³) (frame dragging)

**Features:**
- Computes Kerr ISCO for prograde/retrograde orbits
- Integrates equatorial orbit (i = 0°)
- Integrates inclined orbit (i = 45°)
- Demonstrates frame dragging effects
- Compares orbital plane precession

**Outputs:**
- `output_kerr_inclined/kerr_inclined_orbits.png`: Multi-panel orbital analysis
  - 3D trajectories (equatorial vs inclined)
  - XY projection (equatorial plane)
  - XZ projection (meridional plane)
  - Radial distance evolution
  - Vertical motion (z vs time)
  - Angular momentum alignment with spin axis

**Expected Results:**
- Prograde ISCO ~60% smaller than Schwarzschild
- Equatorial orbit remains in z=0 plane
- Inclined orbit shows vertical oscillations
- Frame dragging causes orbital plane precession (Lense-Thirring effect)
- Angular momentum vector precesses around BH spin axis

**Key Findings:**
- BH spin dramatically affects ISCO radius
- Prograde orbits more stable than retrograde
- Inclined orbits exhibit nodal precession
- Frame dragging visible in orbital dynamics

**References:**
- Liptai et al. (2019) - GRSPH Kerr disc formation
- Bardeen et al. (1972) - Kerr metric properties
- Lu & Bonnerot (2019) - Spin-induced TDE stream self-crossing

---

## Physics Summary

### Units
All examples use **geometrized units**: G = c = M_BH = 1
- Distances in units of M (Schwarzschild radius M ≈ 1.48 km for M_☉)
- Times in units of M/c ≈ 4.93 μs for M_☉
- Velocities in units of c

### Coordinate Systems
- **Cartesian (x, y, z)**: Used for particle positions
- **Boyer-Lindquist (r, θ, φ)**: Natural for Kerr metric (conversion handled internally)
- **Spin axis**: Always along +z direction

### Key Concepts

**Tidal Disruption:**
- Tidal radius: r_t = R_star × (M_BH / M_star)^(1/3)
- Penetration factor: β = r_t / r_p
- β < 0.7: Weak encounter
- β ~ 1: Partial disruption
- β > 1.5: Full disruption

**GR Effects:**
- Periapsis precession (Schwarzschild & Kerr)
- ISCO: Innermost stable circular orbit
- Frame dragging (Kerr only, Lense-Thirring effect)
- Orbital plane precession (inclined Kerr orbits)

**Energy Conservation:**
- E_total = E_kin + E_pot_BH + E_pot_self + E_int - E_radiated
- Conservation error: ΔE/E₀ = (E(t) - E(0)) / E(0)
- Good integrators: |ΔE/E₀| < 0.001

---

## Technical Notes

### Simplified Demonstrations
These examples use **simplified integrators** for demonstration purposes:
- Simple Euler or leapfrog time stepping
- Approximate GR accelerations (not full geodesic equation)
- Neglected self-gravity (test particle approximation)

For **production simulations**, use:
- Full `Simulation` class with proper time integration (RK4, leapfrog with adaptive timestep)
- Hamiltonian integrator for accurate GR orbits near ISCO
- SPH neighbor finding and hydro forces
- Self-gravity via tree code or direct summation

### Dependencies
Required:
- NumPy
- Matplotlib
- h5py

Optional:
- Plotly (for 3D interactive visualizations)
- TDE-SPH framework modules

### Output Formats

**HDF5 (.h5):**
- Snapshot format: `/particles/{positions, velocities, masses, ...}`
- Metadata: `/metadata/{time, bh_mass, mode, ...}`
- Read with `tde_sph.io.hdf5.HDF5Writer`

**PLY (.ply):**
- Point cloud format for Blender
- Import: File > Import > Stanford (.ply)
- Includes RGB colors (mapped from density or other field)

**VTK (.vtk):**
- Unstructured grid for ParaView
- Open directly with ParaView: File > Open
- Supports scalar fields (density, energy) and vector fields (velocity)

**PNG (.png):**
- Diagnostic plots (energy evolution, trajectories)
- High resolution (150 dpi)

---

## Extending the Examples

### Adding Your Own Simulations

1. **Modify Initial Conditions:**
   - Change star mass, radius, polytropic index
   - Adjust orbital parameters (eccentricity, periapsis, inclination)
   - Try different BH spins (a = 0 to 0.998)

2. **Run Full SPH Evolution:**
   - Replace simplified integrator with `Simulation` class
   - Enable neighbor finding and hydro forces
   - Add radiation cooling and luminosity tracking

3. **Visualize Results:**
   - Export snapshots to PLY/VTK for Blender/ParaView rendering
   - Create light curves from luminosity diagnostic
   - Plot energy evolution over full simulation

### Example Modifications

**Higher eccentricity orbit:**
```python
position, velocity, r_peri, r_apo = setup_eccentric_orbit(
    semi_major_axis=20.0,
    eccentricity=0.7,  # More elongated
    bh_mass=1.0
)
```

**Extreme Kerr spin:**
```python
spin = 0.998  # Nearly maximal (a = 0.998 M)
r_isco_prograde = compute_kerr_isco(spin, prograde=True)
# r_isco ≈ 1.24M (very close to horizon at 2M)
```

**Polar orbit:**
```python
position, velocity, params = setup_inclined_orbit(
    semi_major_axis=15.0,
    eccentricity=0.2,
    inclination=90.0,  # Polar orbit (perpendicular to equator)
    bh_mass=1.0
)
```

---

## Known Limitations

1. **Simplified GR integration:** These demos use approximate GR accelerations, not full geodesic integration. For quantitative accuracy near ISCO, use full Hamiltonian formulation.

2. **Single time step / short runs:** Examples 1 runs only 1-2 time steps for speed. For full TDE evolution, extend to hundreds of orbital periods.

3. **No SPH evolution:** Hydro forces, neighbor finding, and density updates are not included. These are conceptual demos of the framework workflow.

4. **Test particle approximation:** Self-gravity between particles is neglected or simplified. For realistic TDEs, enable full self-gravity.

5. **No radiation transport:** Radiation cooling and luminosity are not computed in these examples (see TASK-023 for radiation module).

---

## Future Examples (Planned)

- **Full TDE simulation:** Polytrope star with complete SPH evolution through disruption and circularization
- **Stream-disc collision:** Using tilted disc IC generator (TASK-034)
- **Radiation-dominated TDE:** With gas+radiation EOS and cooling (TASK-021, TASK-023)
- **Long-term accretion:** Disc formation and accretion flow onto BH
- **Comparison: Newtonian vs Schwarzschild vs Kerr:** Side-by-side TDE simulations

---

## References

### Papers
- Guillochon & Ramirez-Ruiz (2013), ApJ, 767, 25 - "Hydrodynamical simulations to determine the feeding rate of black holes by the tidal disruption of stars"
- Tejeda et al. (2017), MNRAS, 469, 4483 - "Relativistic hydrodynamics with Newtonian codes"
- Liptai & Price (2019), MNRAS, 485, 819 - "General relativistic smoothed particle hydrodynamics"
- Liptai et al. (2019), MNRAS, 487, 4790 - "Disc formation from tidal disruption events in GRSPH"
- Lu & Bonnerot (2019), MNRAS, 484, 1506 - "Spin-induced offset stream self-crossing shocks"
- Bardeen et al. (1972), ApJ, 178, 347 - "Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation"
- Price (2012), JCP, 231, 759 - "Smoothed particle hydrodynamics and magnetohydrodynamics"

### Textbooks
- Misner, Thorne & Wheeler (1973), "Gravitation" - Classic GR textbook
- Shapiro & Teukolsky (1983), "Black Holes, White Dwarfs and Neutron Stars" - Compact object astrophysics

---

## Contact & Support

For questions, issues, or contributions to the TDE-SPH framework:
- GitHub: [TDE-SPH Repository]
- Documentation: See `/docs` directory
- Bug reports: GitHub Issues

---

**Author:** TDE-SPH Development Team
**Date:** 2025-11-18
**Version:** Phase 3 Implementation
