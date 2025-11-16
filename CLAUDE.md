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
- New APIs must be documented in docstrings and, if cross-module, briefly summarised in `core`.

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

### Special Folder Rules

- The `prompts/` directory is **off-limits** to all CLAUDE agents at runtime:
  - Do NOT open, read, or modify files under `prompts/`.
  - Treat your operating instructions as already loaded; configuration comes from code/config, not from `prompts/`.
- Subfolder `CLAUDE.md` files provide **local module instructions** and must be followed when working in that module.


## Plan: Phase 2 GR upgrade & agent instructions

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan specifies a modular, relativistic smoothed particle hydrodynamics (SPH) framework to simulate stellar tidal disruption events (TDEs) around supermassive black holes (SMBHs). The code will be written in Python with NumPy/CUDA acceleration, support both full GR and Newtonian modes, and target an RTX 4090 with 64 GB RAM and a Ryzen 7800X3D.

The minimum deliverable is a physically robust, architecturally clean prototype that can:

- Evolve a self-gravitating star past pericentre in a fixed SMBH spacetime (Kerr/Schwarzschild).
- Toggle between general relativistic dynamics and a purely Newtonian model.
- Track thermodynamic and energetic quantities (gas + radiation pressure, kinetic, potential, thermal, luminosity proxies).
- Visualise the debris in 3D via Plotly and export data for external rendering.
  
Phase 1 implements a clean, modular Newtonian SPH TDE framework. Phase 2 should introduce GR-capable components (metrics, GR gravity/orbits, Hamiltonian/GR-aware integration, GR–Newtonian toggle) without breaking the existing Newtonian path. The goal is to extend, not rewrite: add new classes and configs, tighten interfaces, and ensure IO/visualization can distinguish “mode” and units. This plan assumes future agents will edit only where needed, primarily per-submodule `CLAUDE.md` plus select code hotspots.

Refer to IMPLEMENTATION_PLAN.md and IMPLEMENTATION_NOTES.md before making any changes to any code

The architecture prioritises **replaceable modules** over peak performance: each major physical ingredient (metric, EOS, radiation, viscosity, transport, etc.) must be encapsulated behind well-defined interfaces so that more sophisticated implementations can be swapped in later.

## Software and computational constraints

- **CON-001**: Primary implementation language is Python 3.11+; heavy numerics offloaded to:
  - CUDA kernels via Numba, CuPy, or PyTorch/JAX (choice must support custom kernels).  
  - NumPy for CPU fallback.

- **CON-002**: Target hardware: Single RTX 4090 with 24 GB VRAM, 64 GB system RAM, Ryzen 7800X3D.  

- **CON-003**: Avoid dependence on heavy, non-Python ecosystems (e.g. no mandatory FORTRAN/C++ build system in v1). C/C++ extensions allowed for performance-critical hot spots later.

- **CON-004**: Plotting via Plotly 3D volume/point cloud; data export to HDF5/Parquet for external rendering (e.g. Blender, ParaView).

- **CON-005**: Code must be unit-testable (pytest) and support automated regression tests with fixed random seeds.

- **CON-006**: FP32 by default, precision agnostic wherever possible, use of or support for fast CUDA tensor operations wherever possible


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
