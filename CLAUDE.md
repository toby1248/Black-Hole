---
goal: Relativistic SPH framework for stellar tidal disruption events around SMBHs (Python/NumPy/CUDA)
version: 1.0
date_created: 2025-11-15
last_updated: 2025-11-15
owner: TDE-SPH Dev Team
status: Planned
tags: [feature, architecture, GR, SPH, tidal-disruption, CUDA, Python]
---

## Feature Implementation System Guidelines

- this is a debugging step. Ignore any further references to agents you might find and focus on the high level fixes

### Context Optimisation Rules

- When analysing code, keep **comments intact** unless they are clearly stale or misleading.
- Each agent should limit changes to its **designated module**, except when explicitly instructed.
- Small, cross-cutting updates (docs, config defaults) should be coordinated through the **core** or **config** modules.

### Feature Implementation Guidelines

- **CRITICAL:** Make minimal, well-motivated changes; keep diffs focused.
- **CRITICAL:** Preserve existing naming conventions and file organisation.
- Adhere to the established architecture: `Simulation` orchestrates pluggable components.
- Prefer reusing existing utilities over duplicating functionality; if a new utility is required, place it in a shared, well-named location.

## Implementation Phase 3 — Thermodynamics, energies & luminosity plus a visuals upgrade

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

## Software and computational constraints

- **CON-001**: Primary implementation language is Python 3.11+; heavy numerics offloaded to:
  - CUDA kernels via Numba, CuPy, or PyTorch/JAX (choice must support custom kernels).  
  - NumPy for CPU fallback.

- **CON-002**: Target hardware: Single RTX 4090 with 24 GB VRAM, 64 GB system RAM, Ryzen 7800X3D.  

- **CON-003**: Avoid dependence on heavy, non-Python ecosystems (e.g. no mandatory FORTRAN/C++ build system in v1). C/C++ extensions allowed for performance-critical hot spots later.

- **CON-004**: Plotting via Plotly 3D volume/point cloud; data export to HDF5/Parquet for external rendering (e.g. Blender, ParaView).

- **CON-005**: Code must be unit-testable (pytest) and support automated regression tests with fixed random seeds.
  
- **CON-005**: FP32 by default, precision agnostic wherever possible, use of or support for fast CUDA tensor operations wherever possible


## Guidelines and patterns

- **GUD-001**: All physics modules (gravity, metric, EOS, radiation, viscosity, transport) must implement clearly specified Python interfaces so they can be swapped with alternative implementations.

- **GUD-002**: Separate *configuration* (YAML/JSON) from *code* to enable reproducible runs with different BH mass/spin, stellar models, and orbits.

- **GUD-003**: Use dimensionless units internally (e.g. \( G = c = M_\mathrm{BH} = 1 \)) with conversion utilities for physical units.

- **PAT-001**: Adopt a “system + component” architecture: `Simulation` orchestrates components (`Metric`, `GravitySolver`, `SPHSolver`, `EOS`, `RadiationModel`, `TimeStepper`, `ICGenerator`, `IOManager`, `Visualizer`).


---

## Risks & Assumptions

- **RISK-001 (Complexity of GR integration)**: Implementing Kerr metric correctly (especially Christoffel symbols and coordinate singularities) is error-prone. Mitigation: rely on well-tested expressions from Tejeda et al. (2017) and Liptai & Price (2019), unit-test against analytic geodesics.

- **RISK-002 (Performance bottlenecks in Python)**: If GPU kernels and I/O are not carefully designed, Python overhead could limit performance at large N. Mitigation: profile early, use Numba/CUDA or CuPy for heavy loops, and minimise Python-level per-particle operations.

- **RISK-003 (Self-gravity accuracy)**: Approximating self-gravity Newtonianly while BH gravity is relativistic may introduce inaccuracies for very deep encounters. Mitigation: monitor parameter regimes where hybrid approximation is valid (guided by Tejeda et al. 2017 and later TDE work).

- **RISK-004 (Radiation transport oversimplification)**: Simple cooling/luminosity models may not capture full observable signatures. Mitigation: treat radiation module as explicitly pluggable; document limitations clearly.

- **RISK-005 (Validation data)**: Exact benchmarks for full 3D GRSPH TDEs are rare; most comparisons will be qualitative. Mitigation: compare against published GRSPH results (Liptai et al. 2019; recent nozzle shock and SPH-EXA papers) for trends and morphology.

- **ASSUMPTION-001**: Fixed background metric (non-evolving BH spacetime) is sufficient; we do not couple to full numerical relativity.

- **ASSUMPTION-002**: Self-gravity of the star and debris can be modelled Newtonianly without severely compromising the key TDE dynamics for the parameter space of interest.

- **ASSUMPTION-003**: RTX 4090 VRAM and system RAM are sufficient for the target particle counts (10⁵–10⁷) given efficient data structures.

---

## Related Specifications / Further Reading

- Tejeda, E., Gafton, E., Rosswog, S. & Korobkin, O. (2017), MNRAS, 469, 4483 – “Tidal disruptions by rotating black holes: relativistic hydrodynamics with Newtonian codes” [arXiv:1701.00303].
- Liptai, D. & Price, D. J. (2019), MNRAS, 485, 819 – “General relativistic smoothed particle hydrodynamics” [arXiv:1901.08064].
- Liptai, D., Price, D. J. & Lodato, G. (2019), MNRAS, 487, 4790 – “Disc formation from tidal disruption of stars on eccentric orbits by Kerr black holes using GRSPH” [arXiv:1910.10154].
- Price, D. J. (2012), JCP, 231, 759 – SPH review.
- Price, D. J. et al. (2018), PASA, 35, e031 – PHANTOM SPH code.
- “Converged simulations of the nozzle shock in tidal disruption events” (2025, GRSPH + APR).
- Cabezón et al. (2025+), “Tidal disruption events with SPH-EXA: resolving the return of the stream” [arXiv:2510.26663].
- Mahapatra et al. (2024–2025), “Partial tidal disruptions of spinning eccentric white dwarfs …” [arXiv:2401.17031, 2410.12727].
- Lupi (2025), “A general relativistic magnetohydrodynamics extension to mesh-less schemes in the code GIZMO” [arXiv:2506.15775].
- Lu & Bonnerot (2019); Bonnerot et al. (2023), “Spin-induced offset stream self-crossing shocks in tidal disruption events” [arXiv:2303.16230].
- Guillochon, J. & Ramirez-Ruiz, E. (2013), ApJ, 767, 25 – stellar disruption simulations and stellar structure fits.
- “Relativistic effects on tidal disruption kicks of solitary stars” (MNRAS 449, 771, 2015).
