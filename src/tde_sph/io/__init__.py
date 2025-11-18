"""
I/O module: HDF5 snapshots and diagnostics output.
"""

from tde_sph.io.hdf5 import (
    HDF5Writer,
    write_snapshot,
    read_snapshot,
)
from tde_sph.io.diagnostics import (
    DiagnosticsWriter,
    OrbitalElements,
    compute_orbital_elements,
    compute_radial_profile,
)

# Backward compatibility alias
DiagnosticWriter = DiagnosticsWriter

__all__ = [
    'HDF5Writer',
    'write_snapshot',
    'read_snapshot',
    'DiagnosticsWriter',
    'DiagnosticWriter',
    'OrbitalElements',
    'compute_orbital_elements',
    'compute_radial_profile',
]
