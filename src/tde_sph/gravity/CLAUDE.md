# CLAUDE Instructions — gravity module

Role: Provide gravity solvers (Newtonian and relativistic hybrid) behind a stable interface.

DO:
- Work ONLY inside `tde_sph/gravity`.
- Implement Newtonian self-gravity, tree or direct summation variants, and the hybrid BH+Newtonian self-gravity schemes.
- Respect the `Metric` and `GravitySolver` interfaces defined in `tde_sph/core`.

DO NOT:
- Implement SPH kernels or hydrodynamic forces.
- Redefine spacetime metrics (use `tde_sph/metric`).
- Open or modify anything under the `prompts/` folder.
