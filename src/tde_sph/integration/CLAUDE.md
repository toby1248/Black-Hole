# CLAUDE Instructions — integration module

Role: Time integration and timestep control.

DO:
- Work ONLY inside `tde_sph/integration`.
- Implement leapfrog/velocity-Verlet and Hamiltonian-like integrators.
- Implement global and individual timestep controllers, including ISCO-aware constraints.

DO NOT:
- Implement physics forces themselves (call SPH, gravity, radiation modules instead).
- Own configuration files — accept parameters from the core `Simulation`.
- Open or modify anything under the `prompts/` folder.
