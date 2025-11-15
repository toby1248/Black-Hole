# CLAUDE Instructions — initial conditions (ICs) module

Role: Generate initial particle configurations for stars, discs and orbits.

DO:
- Work ONLY inside `tde_sph/ICs`.
- Implement polytropic stellar models, misaligned discs, orbital parameterisation and mapping to SPH particle sets.
- Keep IC generators deterministic (seeded RNG) for reproducibility.

DO NOT:
- Implement runtime integration logic or physics updates.
- Change gravity, metric or EOS implementations.
- Open or modify anything under the `prompts/` folder.
