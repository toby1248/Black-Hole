# TDE-SPH Tools

Command-line utilities and conversion scripts for TDE-SPH simulation data.

## Tools

### export_to_blender.py

Export HDF5 snapshots to Blender/ParaView visualization formats.

**Supported Formats:**
- **PLY** (Polygon File Format): Point clouds for Blender
- **VTK** (Visualization Toolkit): Unstructured grids for ParaView
- **OBJ** (Wavefront): Simple point clouds for Blender

**Features:**
- Particle positions with color mapping (density, temperature, velocity magnitude)
- Scalar field export (density, internal energy, etc.)
- Vector field export (velocities)
- Batch conversion of multiple snapshots
- Multiple color maps (viridis, plasma, hot, cool)
- Optional point size from smoothing length

**Usage Examples:**

```bash
# Export single snapshot to PLY
python tools/export_to_blender.py snapshot_0000.h5 -o output.ply -f ply --color density

# Export to VTK for ParaView
python tools/export_to_blender.py snapshot_0000.h5 -o output.vtk -f vtk

# Export to OBJ for Blender
python tools/export_to_blender.py snapshot_0000.h5 -o output.obj -f obj

# Batch export all snapshots
python tools/export_to_blender.py snapshots/*.h5 -o renders/ -f ply --batch --color velocity_magnitude

# Use different color map
python tools/export_to_blender.py snapshot_0000.h5 -o output.ply -f ply --color internal_energy --cmap plasma
```

### convert_hdf5_to_webjson.py

Convert HDF5 snapshots into the JSON schema used by the `web/` visualizer (`data_loader.js`).

```bash
# Single snapshot -> snapshot_0000.json (next to input)
python tools/convert_hdf5_to_webjson.py outputs/snapshot_0000.h5

# Batch convert and downsample for browser performance
python tools/convert_hdf5_to_webjson.py outputs/snapshot_*.h5 -o web/data/ --stride 4 --limit 50000

# Custom output name with flat positions array
python tools/convert_hdf5_to_webjson.py outputs/snapshot_0042.h5 -o web/data/tde_snapshot_0042.json --flatten-positions
```

**Notes:**
- Outputs fields: `time`, `step`, `n_particles`, `positions`, `density`, `temperature`, `internal_energy`, `velocity_magnitude`, `pressure`, `entropy`.
- Derives `velocity_magnitude` from velocities when missing; `pressure`/`entropy` fall back to ideal-gas relations using `--gamma` (default 5/3).
- Use `--stride`/`--limit` to shrink large particle sets before loading in the browser.

**Python API:**

```python
from tools.export_to_blender import SnapshotExporter

# Create exporter
exporter = SnapshotExporter(verbose=True)

# Export PLY with density coloring
exporter.export_ply(
    "snapshot_0000.h5",
    "output.ply",
    color_by="density",
    cmap="viridis"
)

# Export VTK with custom fields
exporter.export_vtk(
    "snapshot_0000.h5",
    "output.vtk",
    scalars=["density", "internal_energy"],
    vectors=["velocities"]
)

# Batch export
snapshot_files = ["snap_0000.h5", "snap_0001.h5", "snap_0002.h5"]
exporter.batch_export(
    snapshot_files,
    "renders/",
    format="ply",
    color_by="temperature"
)
```

**Blender Import:**
1. Open Blender
2. File > Import > Stanford (.ply) or Wavefront (.obj)
3. Select exported file
4. Points will appear with colors mapped from selected field

**ParaView Import:**
1. Open ParaView
2. File > Open
3. Select exported .vtk file
4. Click "Apply" in Properties panel
5. Use "Representation: Points" for point cloud view
6. Color by any exported scalar field

**Color Fields:**
- `density`: Particle density
- `internal_energy`: Specific internal energy
- `velocity_magnitude`: |v| = sqrt(vx² + vy² + vz²)
- `smoothing_length`: SPH smoothing length
- Any custom field in the HDF5 snapshot

**Color Maps:**
- `viridis`: Dark blue → green → yellow (default)
- `plasma`: Dark purple → orange → yellow
- `hot`: Black → red → yellow → white
- `cool`: Cyan → magenta

---

## Development Notes

### TASK-038: Export Tool Implementation

**Completed:** 2025-11-18

**Files:**
- `tools/export_to_blender.py` (561 lines)
- `tests/test_export_to_blender.py` (622 lines, 27 tests)

**Test Coverage:** 27/27 passing (100%)

**Formats Implemented:**
1. **PLY (ASCII)**: Point cloud with RGB colors and optional radius
2. **VTK (Legacy ASCII)**: Unstructured grid with scalar/vector fields
3. **OBJ**: Simple vertex-only point cloud

**Physics Integration:**
- Reads HDF5 snapshots from `tde_sph.io.hdf5.HDF5Writer` format
- Supports all particle fields (positions, velocities, masses, density, etc.)
- Color mapping from physical quantities
- Handles both Newtonian and GR snapshots

**Known Limitations:**
- Color maps are approximate (simple linear interpolation)
- No SPH kernel interpolation to grids (see TASK-102)
- Binary PLY/VTK not yet implemented (ASCII only for simplicity)
- No mesh generation (points only, no surfaces)

**Future Enhancements (not part of Phase 3):**
- Binary file formats for larger datasets
- SPH interpolation to volume grids
- Isosurface extraction
- Time-series animation export
- Integration with matplotlib colormaps

**References:**
- PLY format: http://paulbourke.net/dataformats/ply/
- VTK format: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
- OBJ format: https://en.wikipedia.org/wiki/Wavefront_.obj_file
