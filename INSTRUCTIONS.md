# Plan: Phase 3 agent instructions

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

The goal is to extend, not rewrite: add new classes and configs, tighten interfaces, and ensure IO/visualization can distinguish “mode” and units. Agents should edit only where needed, primarily per-submodule `CLAUDE.md` plus select code hotspots. Contextless reviewer agents will double-check the logic of each self contained section without bias and distraction 


## Steps
  1. Review `CLAUDE.md` and code in each subpackage (`core`, `sph`, `gravity`, `metric`, `integration`, `ICs`, `eos`, `radiation`, `io`, `visualization`, `config`) to align docs with actual Phase 2 state and Phase 3 goals. If there is no CLAUDE.md in any subpackage's top level folder create one.
  2. Create a NOTES.md in each folder that contains a CLAUDE.md and instruct the sub agents to use it.
  3. Identify and document major architectural or numerical risks in \subpackage\`CLAUDE.md` where appropriate (e.g., timestep control in strong fields, energy diagnostics in GR, unit conventions), so implementation agents can design tests and validation strategies.
  4. In each subfolder containing an agent's task add a short “Phase 3 tasks for agents” section to `CLAUDE.md` in that folder, listing concrete responsibilities and cross-module expectations (e.g., `core` config flags, GR metric hooks). Ensure the coder agents also have access to the global CLAUDE.md instructions context
  5. Once the coder agent has finished these sections modify \subpackage\`CLAUDE.md` again and add to it the reviewer sub-agent prompt found at the bottom of this file, and spawn the reviewer with no context outside the atomic task's directory

## Further Considerations
- If goals have changed or modules have been added update the `CLAUDE.md` files with fresh information
- Maintain full backward compatibility: Newtonian examples/tests must still pass, so new GR paths should be opt-in via config and cleanly tested.
- FP32 by default, FP64 only where necessary, precision agnostic preferred
- Consider compatibility with or implementation of GPU mixed precision tensor computation with linear algebra as a major benefit for current or future use
- Don't use absolute filepaths. Export raw data to the `output` folder and all visualisations and summary documents to `results`
- Catch and mitigate errors from extreme interactions. Identify and flag bad data and throw warnings for debug. Filter visualisation scaling for extreme outliers

# Reviewer Sub-Agent Prompt - Not for programming agents
**ROLE:** Unbiased AI-to-AI code reviewer agent. Succinct and blunt.

**SCOPE:** Review code in the directory containing this file. To you nothing else exists.

**GOAL:** Verify the high level functionality and logic of the module

**TASKS:**
  1. Identify correctness and robustness issues.
  2. Flag missing context or hidden assumptions.
  3. Suggest minimal, concrete improvements.
  4. Add to or create the local NOTES.md




