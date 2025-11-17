# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan specifies a modular, relativistic smoothed particle hydrodynamics (SPH) framework to simulate stellar tidal disruption events (TDEs) around supermassive black holes (SMBHs). The code will be written in Python with NumPy/CUDA acceleration, support both full GR and Newtonian modes, and target an RTX 4090 with 64 GB RAM and a Ryzen 7800X3D.

The minimum deliverable is a physically robust, architecturally clean prototype that can:

- Evolve a self-gravitating star past pericentre in a fixed SMBH spacetime (Kerr/Schwarzschild).
- Toggle between general relativistic dynamics and a purely Newtonian model.
- Track thermodynamic and energetic quantities (gas + radiation pressure, kinetic, potential, thermal, luminosity proxies).
- Visualise the debris in 3D via Plotly and export data for external rendering.

The architecture prioritises **replaceable modules** over peak performance: each major physical ingredient (metric, EOS, radiation, viscosity, transport, etc.) must be encapsulated behind well-defined interfaces so that more sophisticated implementations can be swapped in later.
- Phase 1 implements a clean, modular Newtonian SPH TDE framework. 
- Phase 2 introduced GR-capable components without breaking the existing Newtonian path.  
- Phase 3 should Extend EOS and energy accounting (radiation pressure, energy tracking, luminosity proxies) without removing computationally simpler algorithms.

The goal is to extend, not rewrite: add new classes and configs, tighten interfaces, and ensure IO/visualization can distinguish “mode” and units. This plan assumes future agents will edit only where needed, primarily per-submodule `CLAUDE.md` plus select code hotspots.



## 1. Requirements & Constraints

### 1.1 Physical modelling requirements

- **REQ-001 (SPH core)**: Implement 3D smoothed particle hydrodynamics (SPH) with adaptive smoothing lengths, neighbour finding, and kernel-based gradient estimates, following modern SPH practice (e.g. Price 2012 review; Price et al. 2018 PHANTOM code)  
  - *Refs*:  
    - Price, D. J. (2012), Journal of Computational Physics, 231, 759 – “Smoothed particle hydrodynamics and magnetohydrodynamics”  
    - Price, D. J. et al. (2018), PASA, 35, e031 – “PHANTOM: A Smoothed Particle Hydrodynamics and Magnetohydrodynamics Code for Astrophysics”.

- **REQ-002 (Relativistic dynamics)**: Adopt a *hybrid relativistic* formulation similar to Tejeda, Rosswog & co-authors and GRSPH codes:
  - Exact relativistic motion in a fixed Kerr/Schwarzschild spacetime for orbital dynamics (test-particle limit).
  - Newtonian or quasi-Newtonian treatment of stellar self-gravity.
  - *Refs*:  
    - Tejeda, E. et al. (2017), MNRAS, 469, 4483 – “Tidal disruptions by rotating black holes: relativistic hydrodynamics with Newtonian codes” [arXiv:1701.00303].  
    - Tejeda & Rosswog (2013), MNRAS, 433, 1930 – generalized pseudo-Newtonian potential.  
    - Liptai & Price (2019), MNRAS, 485, 819 – “General relativistic smoothed particle hydrodynamics” [arXiv:1901.08064].  
    - Liptai et al. (2019), MNRAS, 487, 4790 – “Disc formation from tidal disruption of stars on eccentric orbits by Kerr black holes using GRSPH” [arXiv:1910.10154].

- **REQ-003 (Parallel Newtonian model)**: Provide a Newtonian-only mode in the same framework:
  - Replace GR metric with Newtonian potential φ = −GM/r (or pseudo-Newtonian for comparison).
  - Keep SPH, self-gravity, thermodynamics identical for clean GR–Newtonian comparison.  
  - *Ref*: Newtonian TDE SPH implementations (e.g. Rosswog et al. 2009; Guillochon & Ramirez-Ruiz 2013).

- **REQ-004 (Dynamic timesteps)**: Implement individual or block timesteps per particle, with constraints:
  - Courant/Friedrichs–Lewy (CFL) condition based on local sound speed and smoothing length.  
  - Additional relativistic constraints near ISCO and pericentre to ensure orbital accuracy.  
  - *Refs*:  
    - Liptai & Price (2019) – Hamiltonian time integration for GR orbits.  
    - Rosswog (2009) – SPH timestep criteria in TDE contexts.

- **REQ-005 (Spacetime modelling)**: Support:
  - Schwarzschild and Kerr metrics via Boyer–Lindquist or Kerr–Schild coordinates.  
  - Functions to compute metric \( g_{\mu\nu} \), inverse, Christoffel symbols, and geodesic accelerations at arbitrary spatial positions.  
  - *Refs*:  
    - Tejeda et al. (2017) – explicit SPH-compatible accelerations in Kerr spacetime.  
    - Liptai & Price (2019) – GRSPH in fixed Kerr/Schwarzschild metrics.

- **REQ-006 (Self-gravity)**: Compute self-gravity of SPH particles via:
  - Leading option: tree-based Newtonian gravity (Barnes–Hut) or fast GPU pairwise approximation with softening comparable to smoothing length.  
  - Optionally augmented by a pseudo-relativistic correction, but full self-consistent GR for self-gravity is *not* required in v1.  
  - *Refs*:  
    - Tejeda et al. (2017); “relativistic hydrodynamics with Newtonian self-gravity”.  
    - Price & Monaghan (2007) for SPH self-gravity implementations.

- **REQ-007 (Stellar models)**: Include tabulated initial stellar models:
  - Polytropic spheres (γ = 5/3, γ = 4/3) matching Rosswog/Guillochon TDE benchmarks.  
  - At least one main-sequence solar-like model (ρ(r), P(r), T(r)) and optionally white-dwarf-like.  
  - *Refs*:  
    - Guillochon & Ramirez-Ruiz (2013), ApJ, 767, 25 – analytic fits for stellar structure.  
    - Rosswog et al. (2009) – SPH TDE of solar-type stars.

- **REQ-008 (Thermodynamics & EOS)**: Implement:
  - Ideal gas EOS with adiabatic index γ, plus radiation pressure for optically thick gas:  
    \( P = P_\mathrm{gas} + P_\mathrm{rad} = (\gamma - 1) u \rho + \tfrac{1}{3} a T^4 \).  
  - Track internal energy u, temperature T, entropy-like variable (as in GRSPH) if needed.  
  - *Refs*:  
    - Liptai & Price (2019) – entropy-conservative GRSPH.  
    - Standard stellar structure texts (Kippenhahn & Weigert).

- **REQ-009 (Energy accounting & luminosity)**: Track:
  - Kinetic, potential (BH + self-gravity), internal (thermal + radiation) energies per particle and globally.  
  - Approximate radiative luminosity via:  
    - (a) local dissipation proxies (e.g. artificial viscosity heating),  
    - (b) simple diffusion/escape approximations.  
  - *Refs*:  
    - TDE luminosity estimates from Lodato & Rossi (2011); Dai et al. (2015).  
    - GRSPH energy accounting from Liptai & Price (2019).

- **REQ-010 (ISCO vs large radii optimisations)**:  
  - Provide radius-dependent timestep and integration strategies:
    - High-accuracy integrator near/below ISCO (e.g. symplectic/Hamiltonian),  
    - Cheaper schemes at large radii (classical SPH leapfrog).  
  - *Refs*:  
    - Liptai & Price (2019) – Hamiltonian integrator tests for epicyclic frequencies.  
    - Tejeda et al. (2017) – accurate Kerr geodesic motion.

### 1.2 Advanced physics to be considered (not all mandatory in v1)

Below: physics requested + additional effects from literature, with complexity vs. value.

- **Energy transport mechanisms (REQ-011)**:
  - Radiative diffusion (flux-limited diffusion, FLD) vs. simple cooling law.  
  - Conduction along the stream and in any forming disc.  
  - *Refs*:  
    - Shiokawa et al. (2015) MRI/energy transport in TDE discs (GRMHD).  
    - Sądowski et al. (2016) – radiative GRMHD discs.

- **Gas viscosity, friction and diffusion (REQ-012)**:
  - Physical viscosity and turbulent transport via an α-prescription or explicit Navier–Stokes viscosity on SPH particles.  
  - Distinguish artificial shock viscosity vs. physical viscosity.  
  - *Refs*:  
    - Cullen & Dehnen (2010) – improved SPH viscosity switches.  
    - Liptai & Price (2019) – separation of shock viscosity and conductivity.

- **Black hole spin & frame dragging (REQ-013)**:
  - Implement full Kerr metric with spin parameter a.  
  - Capture Lense–Thirring precession, nodal precession of inclined orbits.  
  - *Refs*:  
    - Kesden (2012), PRD 85, 024037 – analytic Kerr TDE parameter study.  
    - Liptai et al. (2019); Tejeda et al. (2017).

- **Existing misaligned accretion disc (REQ-014)**:
  - Represent pre-existing disc as either:
    - SPH particles in a quasi-steady disc configuration, or  
    - Analytic background density/pressure field + drag forces.  
  - Study stream–disc interaction and warps.  
  - *Refs*:  
    - Franchini et al. (2016–2024) – tilted discs with GIZMO.  
    - Bonnerot & Rossi (2019) – stream-disk collisions in TDEs.

- **Magnetic fields and MHD (REQ-015)**:
  - Long-term goal: extend to (GR)MHD SPH (embedding induction equation and Lorentz forces).  
  - *Refs*:  
    - Liptai & Price (2019) discuss GRSPH extension to MHD.  
    - GIZMO GRMHD extension (Lupi 2025): “A general relativistic magnetohydrodynamics extension to mesh-less schemes in the code GIZMO”.

- **Adaptive particle refinement (REQ-016)**:
  - Use particle splitting/refinement near shocks/pericentre to resolve nozzle shocks and self-crossing.  
  - *Refs*:  
    - Nealon & Price (2025);  
    - “Converged simulations of the nozzle shock in tidal disruption events” (2025, GRSPH + adaptive particle refinement).  
    - SPH-EXA TDE simulations with up to 10¹⁰ particles [arXiv:2510.26663].

- **Stream self-crossing shocks & offset collisions (REQ-017)**:
  - Incorporate accurate shock capturing and resolve relativistic self-crossing shock geometry.  
  - *Refs*:  
    - Lu & Bonnerot (2019);  
    - “Spin-induced offset stream self-crossing shocks in tidal disruption events” [arXiv:2303.16230];  
    - SPH-EXA study of stream return [arXiv:2510.26663].

- **Partial disruptions, surviving cores, and kicks (REQ-018)**:
  - Model partially disrupted stars and surviving core trajectories.  
  - *Refs*:  
    - Mainetti et al. (2017);  
    - “Relativistic effects on tidal disruption kicks of solitary stars” (MNRAS 449, 771, 2015).

- **Off-equatorial and eccentric orbits, WD TDEs (REQ-019)**:
  - Support generic initial orbital parameters and stellar spins.  
  - *Refs*:  
    - “Partial tidal disruptions of spinning eccentric white dwarfs by spinning intermediate-mass black holes” [arXiv:2401.17031];  
    - “Partial tidal disruption of White Dwarfs in off-equatorial orbits around Kerr black holes” [arXiv:2410.12727].

## 2. Implementation Steps

### Implementation Phase 1 — Core architecture, Newtonian SPH baseline (complete)

- **GOAL-001**: Establish a clean Python package structure with an SPH engine, Newtonian gravity, and basic TDE initial conditions, plus Plotly visualisation and data export. No GR yet.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create Python package skeleton `tde_sph/` with subpackages: `core`, `sph`, `gravity`, `metric`, `eos`, `radiation`, `ICs`, `integration`, `io`, `visualization`, `config`, and top-level entrypoint `scripts/run_simulation.py`. Include `pyproject.toml` or `setup.cfg`. |  |  |
| TASK-002 | Define abstract base classes (ABCs) in `tde_sph/core/interfaces.py` for `Metric`, `GravitySolver`, `EOS`, `RadiationModel`, `TimeIntegrator`, `ICGenerator`, `Visualizer`. |  |  |
| TASK-003 | Implement a basic SPH particle container and neighbour search in `tde_sph/sph/particles.py` and `tde_sph/sph/kernels.py` (e.g. cubic spline kernel, adaptive smoothing lengths). |  |  |
| TASK-004 | Implement Newtonian-only `GravitySolver` in `tde_sph/gravity/newtonian.py` with tree-based (or O(N²) to start) gravity and softening = smoothing length. |  |  |
| TASK-005 | Implement SPH hydrodynamic forces (pressure gradients, artificial viscosity, optionally thermal conduction) in `tde_sph/sph/hydro_forces.py` using GPU kernels (Numba/CUDA or CuPy). |  |  |
| TASK-006 | Implement a simple ideal-gas EOS in `tde_sph/eos/ideal_gas.py` and energy update scheme (u, T) with gamma parameter. |  |  |
| TASK-007 | Create stellar initial conditions generator in `tde_sph/ICs/polytrope.py` for γ=5/3 and γ=4/3 polytropes (matching standard TDE tests). |  |  |
| TASK-008 | Implement a simple timestepper (global dt then per-particle dt later) in `tde_sph/integration/leapfrog.py` with CFL condition and energy checks. |  |  |
| TASK-009 | Implement basic I/O in `tde_sph/io/hdf5.py` for saving particle snapshots (positions, velocities, internal energy, density, etc.) and configuration metadata. |  |  |
| TASK-010 | Implement Plotly-based 3D scatter/volume visualisation in `tde_sph/visualization/plotly_3d.py` plus scripts to visualise a snapshot. |  |  |
| TASK-011 | Add `README.md` with quick-start instructions and example Newtonian TDE run (star on parabolic orbit around Newtonian BH). |  |  |
| TASK-012 | Add unit tests using `pytest` for kernels, neighbour search, gravity, simple hydro tests (e.g. Sod shock tube, static polytrope equilibrium). |  |  |

### Implementation Phase 2 — Relativistic framework & GR–Newtonian toggle

- **GOAL-002**: Introduce fixed background metrics (Schwarzschild, Kerr), GR-inspired potentials, and a clean switch between Newtonian and relativistic dynamics, still with Newtonian self-gravity.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-013 | Implement `Metric` subclasses in `tde_sph/metric/` for Minkowski, Schwarzschild, Kerr (Boyer–Lindquist and/or Kerr–Schild). Provide methods: `metric_tensor(x)`, `inverse_metric(x)`, `christoffel_symbols(x)`, and `geodesic_acceleration(x, v)`. |  |  |
| TASK-014 | Implement hybrid relativistic acceleration following Tejeda et al. (2017) in `tde_sph/gravity/relativistic_orbit.py`, combining BH relativistic acceleration with Newtonian self-gravity. |  |  |
| TASK-015 | Design configuration system in `tde_sph/config/` to specify BH mass, spin, metric type, initial orbit parameters, stellar model, and model mode (`"GR"` or `"Newtonian"`). |  |  |
| TASK-016 | Extend `TimeIntegrator` to support Hamiltonian-like integrator for test-particle motion in fixed metrics (per Liptai & Price 2019) near ISCO, falling back to leapfrog at large radii. |  |  |
| TASK-017 | Implement radius-dependent timestep strategy in `tde_sph/integration/timestep_control.py` (stricter thresholds near ISCO). |  |  |
| TASK-018 | Provide a `RelativisticGravitySolver` wrapper that uses `Metric` for BH gravity and the existing Newtonian solver for self-gravity, with a runtime toggle to select Newtonian-only mode. |  |  |
| TASK-019 | Implement unit tests/benchmarks to compare numerical epicyclic and vertical frequencies with analytic Kerr predictions (as in Liptai & Price 2019) and to verify geodesic periapsis precession. |  |  |
| TASK-020 | Validate the relativistic vs Newtonian trajectory for a test star (no hydrodynamics) and document differences. |  |  |

### Implementation Phase 3 — Thermodynamics, energies & luminosity

- **GOAL-003**: Extend EOS and energy accounting to include radiation pressure, energy tracking, and simple luminosity proxies.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-021 | Implement combined gas + radiation pressure EOS in `tde_sph/eos/radiation_gas.py`, with consistent internal energy and temperature handling. |  |  |
| TASK-022 | Add global energy bookkeeping in `tde_sph/core/energy_diagnostics.py` to compute kinetic, potential (BH + self-gravity), internal (thermal + radiation) and total energies per snapshot. |  |  |
| TASK-023 | Implement simple radiative cooling / luminosity model (e.g. local cooling function or FLD-lite) in `tde_sph/radiation/simple_cooling.py`. |  |  |
| TASK-024 | Provide diagnostic outputs for light curves (fallback rate approximation, luminosity vs time) in `tde_sph/io/diagnostics.py`. |  |  |
| TASK-025 | Add tests ensuring energy conservation in adiabatic runs and correct response to controlled heating/cooling scenarios. |  |  |

### Implementation Phase 4 — Dynamic timesteps, individual stepping & performance

- **GOAL-004**: Introduce individual / block timesteps, optimise GPU kernels, and refine neighbour search for large N.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-026 | Implement per-particle timestep estimation based on CFL, acceleration, and orbital timescale in `tde_sph/integration/individual_timesteps.py`. |  |  |
| TASK-027 | Implement block-timestep scheme (hierarchical timestep levels) compatible with GPU kernels and neighbour search. |  |  |
| TASK-028 | Optimise SPH and gravity CUDA kernels for coalesced memory access and minimise host–device transfers. |  |  |
| TASK-029 | Replace naive neighbour search with a GPU-accelerated uniform grid / hash or tree in `tde_sph/sph/neighbours_gpu.py`. |  |  |
| TASK-030 | Conduct scaling tests (10⁵–10⁷ particles) and document performance and memory usage on RTX 4090. |  |  |

### Implementation Phase 5 — Advanced physics options & misaligned discs

- **GOAL-005**: Layer in optional physics modules: viscosity, simple energy transport, BH spin effects, misaligned discs, stream–stream collisions.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-031 | Implement physical viscosity model (e.g. α-viscosity mapped to SPH) separate from artificial viscosity; add switches in `tde_sph/sph/viscosity.py`. |  |  |
| TASK-032 | Implement simple energy transport (diffusion-like) operator in `tde_sph/radiation/diffusion.py` with tunable coefficients. |  |  |
| TASK-033 | Enable arbitrary initial orbital inclinations and BH spin orientation; validate nodal precession against analytic expectations. |  |  |
| TASK-034 | Implement pre-existing disc IC generator in `tde_sph/ICs/disc.py` and optional analytic drag term for low-cost disc representation. |  |  |
| TASK-035 | Provide test suite for stream–disc and stream–stream collision setups, verifying expected shock formation and entropy generation (qualitative). |  |  |

### Implementation Phase 6 — Visualisation, export & workflow tooling

- **GOAL-006**: Finalise visualisation, data export, and high-level scripting for experiment design.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-036 | Implement snapshot export to HDF5/Parquet with schema including particle positions, velocities, internal energy, density, metric identifiers, and run metadata. |  |  |
| TASK-037 | Extend Plotly 3D visualiser to support time-scrubbing animations and optional derived quantities (e.g. temperature, energy density) as colour maps. |  |  |
| TASK-038 | Provide a converter in `tools/export_to_blender.py` (or similar) that writes point clouds or volume grids suitable for Blender/ParaView. |  |  |
| TASK-039 | Create example notebooks (`examples/`) showcasing a Newtonian run, a Schwarzschild GR run, and a Kerr inclined orbit run, with comparison plots. |  |  |
| TASK-040 | Add automated regression tests (using small N) and continuous integration scripts to validate core functionality. |  |  |

---


## 3. Alternatives

- **ALT-001 (Grid-based GRHD instead of SPH)**: Use a finite-volume GRHD code (e.g. CoCoNuT, HARM, or Einstein Toolkit) instead of SPH.  
  - **Reason not chosen**: Significantly higher complexity to integrate with Python/CUDA; less natural for Lagrangian particles and adaptive resolution along debris streams.

- **ALT-002 (Use existing GRSPH/GRHD code as a library)**: Wrap existing codes like PHANTOM, GRSPH (Liptai & Price), GIZMO-GRHD via Python bindings.  
  - **Reason not chosen**: These codes are large, with their own build and runtime ecosystems; full control and modifiability is reduced. Our design emphasises educational clarity and modularity even at the cost of initial performance.

- **ALT-003 (Pure pseudo-Newtonian potentials instead of proper GR)**: Use only generalized pseudo-Newtonian potentials (e.g. Tejeda & Rosswog 2013) without explicit metrics.  
  - **Reason not chosen**: Good for fast studies, but you explicitly require a “full Einsteinian spacetime model”; we keep pseudo-Newtonian as a *mode* for comparison and algorithm development, not as the primary model.

- **ALT-004 (Particle-mesh self-gravity instead of tree)**: Use PM/FFT-based gravity solves.  
  - **Reason not chosen**: Less natural for highly elongated debris streams and very non-uniform density distributions typical of TDEs; tree-based or direct SPH-like gravity is more flexible.

---

## 4. Dependencies

- **DEP-001**: Python packages:
  - `numpy`, `scipy` (basic numerics),  
  - `numba` or `cupy` (CUDA acceleration),  
  - `h5py` (HDF5 I/O),  
  - `plotly` (3D visualisation),  
  - `pydantic` or `omegaconf` (configuration management),  
  - `pytest` (testing).

- **DEP-002**: System requirements:
  - CUDA toolkit compatible with RTX 4090,  
  - GPU drivers properly configured,  
  - Optional: `ffmpeg` for movie creation.

- **DEP-003**: Optional physics data:
  - Tabulated stellar models (e.g. MESA outputs, or fits from Guillochon & Ramirez-Ruiz 2013).  
  - Opacities tables if advanced radiation transport is later added.

---

## 5. Files

(High-level file map; exact names may be refined.)

- **FILE-001**: `pyproject.toml` – project metadata, dependencies.
- **FILE-002**: `README.md` – usage overview and examples.
- **FILE-003**: `tde_sph/core/simulation.py` – main `Simulation` orchestrator class.
- **FILE-004**: `tde_sph/core/interfaces.py` – ABC definitions for pluggable modules.
- **FILE-005**: `tde_sph/sph/particles.py` – particle arrays and state management.
- **FILE-006**: `tde_sph/sph/kernels.py` – SPH kernel implementations.
- **FILE-007**: `tde_sph/sph/hydro_forces.py` – SPH hydrodynamic force computation (CUDA/CPU).
- **FILE-008**: `tde_sph/sph/neighbours_cpu.py` & `neighbours_gpu.py` – neighbour search.
- **FILE-009**: `tde_sph/gravity/newtonian.py` – Newtonian gravity solver.
- **FILE-010**: `tde_sph/gravity/relativistic_orbit.py` – Kerr/Schwarzschild BH accelerations and hybrid gravity.
- **FILE-011**: `tde_sph/metric/schwarzschild.py` – Schwarzschild metric and geodesic helpers.
- **FILE-012**: `tde_sph/metric/kerr.py` – Kerr metric implementation (BL and/or KS coordinates).
- **FILE-013**: `tde_sph/eos/ideal_gas.py` – ideal gas EOS.
- **FILE-014**: `tde_sph/eos/radiation_gas.py` – gas + radiation EOS.
- **FILE-015**: `tde_sph/radiation/simple_cooling.py` – basic cooling/luminosity model.
- **FILE-016**: `tde_sph/integration/leapfrog.py` – leapfrog/velocity-Verlet integrator.
- **FILE-017**: `tde_sph/integration/hamiltonian.py` – Hamiltonian-like integrator for GR orbits.
- **FILE-018**: `tde_sph/integration/timestep_control.py` – global & individual timestep logic.
- **FILE-019**: `tde_sph/ICs/polytrope.py` – stellar polytrope IC generator.
- **FILE-020**: `tde_sph/ICs/disc.py` – accretion disc ICs.
- **FILE-021**: `tde_sph/io/hdf5.py` – snapshot and diagnostic HDF5 I/O.
- **FILE-022**: `tde_sph/io/diagnostics.py` – energy and light-curve outputs.
- **FILE-023**: `tde_sph/visualization/plotly_3d.py` – 3D Plotly visualisation utilities.
- **FILE-024**: `scripts/run_simulation.py` – command-line entrypoint.
- **FILE-025**: `tests/` – unit and regression tests.

---


## 6. Testing

- **TEST-001 (SPH unit tests)**: Validate SPH kernels, neighbour search, and density estimates against analytic solutions for uniform density spheres and test problems (e.g. Sod shock tube, static polytrope).
- **TEST-002 (Gravity tests)**:
  - Newtonian gravity: compare forces against analytic 1/r² for small systems.  
  - Relativistic BH force: verify geodesic orbits for test particles in Schwarzschild/Kerr metrics reproduce known epicyclic and vertical oscillation frequencies (cf. Liptai & Price 2019).
- **TEST-003 (Energy conservation)**: In an adiabatic, closed system with no cooling, verify that total energy is conserved to within tolerance over many orbits.
- **TEST-004 (GR vs Newtonian comparison)**:
  - Run identical initial conditions with GR BH metric vs Newtonian potential and quantify differences in debris energy distribution, fallback rates, and pericentre shifts.
- **TEST-005 (ISCO & timestep control)**: Test near-ISCO orbits to ensure stability and accuracy of Hamiltonian integrator and timestep constraints.
- **TEST-006 (Stellar equilibrium)**: Evolve an isolated polytropic star in Newtonian gravity and verify equilibrium (no secular drift in structure).
- **TEST-007 (Performance scalability)**: Run scaling tests for increasing particle counts and record wall-clock times and memory usage.
