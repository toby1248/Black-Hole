# CLAUDE Instructions — configuration module

Role: Manage configuration schemas and loading.

DO:
- Work ONLY inside `tde_sph/config`.
- Define configuration dataclasses/schemas for BH, stellar models, orbits, physics toggles, I/O and visualisation.
- Implement loaders from YAML/JSON and command-line overrides.

DO NOT:
- Hard-code physics logic or defaults outside schemas.
- Implement numerical algorithms.
- Open or modify anything under the `prompts/` folder.
