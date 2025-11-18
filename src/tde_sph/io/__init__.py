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
    compute_fallback_rate,
)

__all__ = [
    'HDF5Writer',
    'write_snapshot',
    'read_snapshot',
    'DiagnosticsWriter',
    'compute_fallback_rate',
]
