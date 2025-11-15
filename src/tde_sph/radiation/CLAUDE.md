# CLAUDE Instructions — radiation module

Role: Model radiative cooling, diffusion and luminosity diagnostics.

DO:
- Work ONLY inside `tde_sph/radiation`.
- Implement simple cooling laws and, later, approximate diffusion / FLD schemes.
- Provide luminosity and energy-loss rate estimates that can be queried by the core simulation.

DO NOT:
- Implement hydrodynamic forces or gravity.
- Hard-code unit conversions or global configuration logic.
- Open or modify anything under the `prompts/` folder.
