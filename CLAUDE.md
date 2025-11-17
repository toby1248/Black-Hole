---
goal: Relativistic SPH framework for stellar tidal disruption events around SMBHs (Python/NumPy/CUDA)
version: 1.0
date_created: 2025-11-15
last_updated: 2025-11-17
owner: TDE-SPH Dev Team
status: Planned
tags: [feature, architecture, GR, SPH, tidal-disruption, CUDA, Python]
---

## Feature Implementation System Guidelines

### Parallel Feature Implementation Priorities

- Prefer **small, well-scoped tasks** over monolithic changes.
- Use **parallel agents** when different modules can be changed independently (e.g. `sph/`, `gravity/`, `metric/`, `eos/`, `io/`).
- Avoid parallel edits to the **same file or tightly coupled module** to minimise merge conflicts and integration bugs.

### Parallel Feature Implementation Workflow

1. **Design & Ownership**

- Identify affected modules (core, sph, gravity, metric, eos, radiation, integration, ICs, io, visualization, config).
- Assign at most one agent per module for the duration of the feature.

2. **Module-local Implementation**

- Each agent implements changes **only inside its module**, respecting interfaces from `tde_sph/core`.
- New APIs must be documented in docstrings and, if cross-module, briefly nd clearly summarised in `core`.
- Do not expose or call any variables or functions cross-module without complete understanding of the behavior

3. **Tests & Diagnostics**

- Each agent adds or updates unit tests in `tests/` for their module.
- Prefer small, fast tests suitable for CI.

4. **Integration & Configuration**

- Core agent wires new functionality via configuration (`tde_sph/config`) and `Simulation` orchestration.
- Ensure defaults keep existing examples working.

5. **Review, Run & Validate**

- Run test suite and at least one example simulation (Newtonian + one GR) before considering the feature complete.
- Resolve any interface mismatches or performance regressions.

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

- **GOAL-003**: Extend EOS and energy accounting to include radiation pressure, energy tracking, and simple luminosity proxies.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-021 | Implement combined gas + radiation pressure EOS in `tde_sph/eos/radiation_gas.py`, with consistent internal energy and temperature handling. |  |  |
| TASK-022 | Add global energy bookkeeping in `tde_sph/core/energy_diagnostics.py` to compute kinetic, potential (BH + self-gravity), internal (thermal + radiation) and total energies per snapshot. |  |  |
| TASK-023 | Implement simple radiative cooling / luminosity model (e.g. local cooling function or FLD-lite) in `tde_sph/radiation/simple_cooling.py`. |  |  |
| TASK-024 | Provide diagnostic outputs for light curves (fallback rate approximation, luminosity vs time) in `tde_sph/io/diagnostics.py`. |  |  |
| TASK-025 | Add tests ensuring energy conservation in adiabatic runs and correct response to controlled heating/cooling scenarios. |  |  |

| TASK-034 | Implement accretion disc IC generator in `tde_sph/ICs/disc.py`, for now without additional considerations |  |  |



 - **GOAL-006**: Create a GUI as a wrapper for yaml and an upgrade to PyQt

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-037 | Replace Plotly 3D visualiser with PyQtGraph, to support time-scrubbing animations and optional derived quantities (e.g. temperature, energy density) as colour maps. |  |  |
| TASK-100 | PyQT-based GUI to select and modify yaml presets and control the sim. Metric unit conversion. Large space to display PyQtGraph live data from the running sim, per-timestep snapshots when open |  |  |
| TASK-101 | Provide a library of PyQtGraph and Matplotlib 2D and 3D visualisation options to be exposed in a GUI menu, including transparency, smoothing, iso-surface, etc|  |  |
| TASK-038 | Provide a converter in `tools/export_to_blender.py` (or similar) that writes point clouds or volume grids suitable for Blender/ParaView.|  |  |
| TASK-102 | Spatial and temporal output interpolation and/or smoothing, for volume graph display and export to Blender. Not to replace/overwrite the 'clean' data |  |  |
| TASK-039 | Create example notebooks (`examples/`) showcasing a Newtonian run, a Schwarzschild GR run, and a Kerr inclined orbit run, with comparison plots. |  |  |
| TASK-040 | Add automated regression tests (using small N) and continuous integration scripts to validate core functionality. |  |  |

The code will be written in Python with NumPy/CUDA acceleration, support both full GR and Newtonian modes, and target an RTX 4090 with 64 GB RAM and a Ryzen 7800X3D.

The minimum deliverable is a physically robust, architecturally clean prototype that can:

- Evolve a self-gravitating star past pericentre in a fixed SMBH spacetime (Kerr/Schwarzschild).
- Toggle between general relativistic dynamics and a purely Newtonian model.
- Track thermodynamic and energetic quantities (gas + radiation pressure, kinetic, potential, thermal, luminosity proxies).
- Visualise the debris in 3D via Plotly and export data for external rendering.
  
The goal is to extend, not rewrite: add new classes and configs, tighten interfaces, and ensure IO/visualization can distinguish “mode” and units. This plan assumes future agents will edit only where needed, primarily per-submodule `CLAUDE.md` plus select code hotspots.

Refer to IMPLEMENTATION_PLAN.md and INSTRUCTIONS.md before making any changes to any code


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
