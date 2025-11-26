# CLAUDE Instructions — SPH module

Role: Implement the smoothed particle hydrodynamics layer.

DO:
- Work ONLY inside `tde_sph/sph`.
- Implement particle containers, neighbour search, kernels, hydrodynamic forces, viscosity switches, and timestep-related SPH helpers.
- Optimise data structures for GPU execution, but favour clarity over micro-optimisations.

DO NOT:
- Implement gravity or metric-specific logic (call the gravity/metric interfaces instead).
- Change integration schemes outside SPH-specific helpers.
- Open or modify anything under the `prompts/` folder.
