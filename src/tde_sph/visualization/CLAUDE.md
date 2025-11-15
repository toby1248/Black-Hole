# CLAUDE Instructions — visualisation module

Role: Provide Plotly-based 3D visualisation and export helpers.

DO:
- Work ONLY inside `tde_sph/visualization`.
- Create Plotly 3D scatter/volume plots of SPH snapshots.
- Provide utilities to export frames or grids for external rendering tools (Blender, ParaView).

DO NOT:
- Implement core physics or I/O formats.
- Change simulation stepping or configuration.
- Open or modify anything under the `prompts/` folder.
