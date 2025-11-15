# CLAUDE Instructions — core module

Role: Implement and maintain high-level simulation orchestration and shared interfaces.

DO:
- Focus ONLY on `tde_sph/core`.
- Define clean, minimal interfaces for metrics, gravity, EOS, SPH, radiation, integrators, IO and visualisation.
- Keep `Simulation` and configuration handling thin and composable.
- Coordinate with other modules via interfaces, not concrete implementations.

DO NOT:
- Implement physics details (SPH kernels, GR, EOS, etc.).
- Modify files in other subpackages except when explicitly instructed.
- Open or modify anything under the `prompts/` folder.
