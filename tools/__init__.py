"""
Tools module for TDE-SPH simulations.

Provides command-line utilities and conversion scripts for working with
TDE-SPH simulation data.

Modules
-------
export_to_blender : SnapshotExporter
    Export HDF5 snapshots to Blender/ParaView formats (PLY, VTK, OBJ).

Examples
--------
From Python:
    >>> from tools.export_to_blender import SnapshotExporter
    >>> exporter = SnapshotExporter()
    >>> exporter.export_ply("snapshot.h5", "output.ply", color_by="density")

From command line:
    $ python tools/export_to_blender.py snapshot.h5 -o output.ply -f ply
"""

from tools.export_to_blender import SnapshotExporter

__all__ = ["SnapshotExporter"]
