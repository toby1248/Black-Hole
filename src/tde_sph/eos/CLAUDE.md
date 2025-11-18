# CLAUDE Instructions — EOS module

Role: Provide equation-of-state (EOS) and thermodynamic closures.

DO:
- Work ONLY inside `tde_sph/eos`.
- Implement ideal-gas and gas+radiation EOS variants.
- Provide functions/classes to map (density, internal energy, composition) to (pressure, temperature, sound speed)

DO NOT:
- Implement gravity, metric, or SPH kernels.
- Decide timestepping; expose only thermodynamic quantities.
- Open or modify anything under the `prompts/` folder.
