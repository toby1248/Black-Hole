# CLAUDE Instructions — IO module

Role: Handle simulation I/O, snapshot storage and diagnostics.

DO:
- Work ONLY inside `tde_sph/io`.
- Implement snapshot writing (HDF5/Parquet) and diagnostics output (energies, fallback rates, etc.).
- Maintain stable schemas to keep external tools working.

DO NOT:
- Implement physics calculations.
- Change visualisation logic (use visualisation module hooks instead).
- Open or modify anything under the `prompts/` folder.
