"""
HDF5 snapshot I/O for TDE-SPH simulations.

This module provides efficient snapshot storage and retrieval using HDF5 format
with compression. Implements REQ-009 (energy accounting & I/O) and CON-004 (data export).

Design:
- Organized group structure: /particles, /metadata
- Compression enabled (gzip level 4) for efficient storage
- Includes versioning and metadata for reproducibility
- Schema designed for compatibility with external analysis tools (ParaView, Blender, etc.)

Example usage:
    >>> writer = HDF5Writer()
    >>> writer.write_snapshot(
    ...     "snapshot_0000.h5",
    ...     particles,
    ...     time=0.0,
    ...     metadata={"bh_mass": 1e6}
    ... )
    >>> data = writer.read_snapshot("snapshot_0000.h5")
"""

import h5py
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy.typing as npt

# Type alias matching core interfaces
NDArrayFloat = npt.NDArray[np.float32]


class HDF5Writer:
    """
    HDF5-based snapshot writer for TDE-SPH particle data.

    Stores particle arrays and simulation metadata with compression.
    Designed for efficient storage and compatibility with analysis pipelines.

    Attributes
    ----------
    compression : str
        Compression algorithm (default: 'gzip')
    compression_level : int
        Compression level 0-9 (default: 4, balances speed vs size)
    code_version : str
        Version identifier for the TDE-SPH code
    """

    def __init__(
        self,
        compression: str = "gzip",
        compression_level: int = 4,
        code_version: str = "1.0.0"
    ):
        """
        Initialize HDF5Writer.

        Parameters
        ----------
        compression : str, optional
            Compression algorithm ('gzip', 'lzf', or None). Default: 'gzip'.
        compression_level : int, optional
            Compression level for gzip (0-9). Default: 4.
        code_version : str, optional
            Version string for the code. Default: '1.0.0'.
        """
        self.compression = compression
        self.compression_level = compression_level if compression == "gzip" else None
        self.code_version = code_version

    def write_snapshot(
        self,
        filename: str,
        particles: Dict[str, NDArrayFloat],
        time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write particle snapshot to HDF5 file.

        Creates organized HDF5 structure with particle data and metadata.
        Automatically includes timestamp and basic simulation info.

        Parameters
        ----------
        filename : str
            Path to output HDF5 file. Parent directory must exist.
        particles : Dict[str, NDArrayFloat]
            Dictionary containing particle arrays. Expected keys:
            - 'positions': shape (N, 3)
            - 'velocities': shape (N, 3)
            - 'masses': shape (N,)
            - 'density': shape (N,)
            - 'internal_energy': shape (N,)
            - 'smoothing_length': shape (N,)
            Optional keys:
            - 'ids': shape (N,) - particle IDs
            - 'acceleration': shape (N, 3) - current acceleration
            - 'temperature': shape (N,) - temperature
            - 'pressure': shape (N,) - pressure
        time : float
            Current simulation time (in code units).
        metadata : Dict[str, Any], optional
            Additional metadata to store (BH mass, spin, orbital params, etc.).

        Notes
        -----
        - All particle arrays are converted to float32 for efficiency
        - Compression is applied to all datasets
        - Metadata is stored as attributes on the file root and /metadata group

        Examples
        --------
        >>> particles = {
        ...     'positions': np.random.randn(10000, 3).astype(np.float32),
        ...     'velocities': np.random.randn(10000, 3).astype(np.float32),
        ...     'masses': np.ones(10000, dtype=np.float32) * 1e-4,
        ...     'density': np.ones(10000, dtype=np.float32),
        ...     'internal_energy': np.ones(10000, dtype=np.float32) * 1e5,
        ...     'smoothing_length': np.ones(10000, dtype=np.float32) * 0.1,
        ... }
        >>> writer = HDF5Writer()
        >>> writer.write_snapshot("test.h5", particles, time=0.0)
        """
        # Ensure parent directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Validate required particle fields
        required_fields = ['positions', 'velocities', 'masses', 'density',
                          'internal_energy', 'smoothing_length']
        missing = [f for f in required_fields if f not in particles]
        if missing:
            raise ValueError(f"Missing required particle fields: {missing}")

        # Get number of particles
        n_particles = len(particles['masses'])

        with h5py.File(filename, 'w') as f:
            # Create particle data group
            particle_group = f.create_group('particles')

            # Write particle arrays with compression
            for key, array in particles.items():
                # Convert to float32 if needed (CON-002: FP32 optimization)
                if array.dtype != np.float32:
                    array = array.astype(np.float32)

                particle_group.create_dataset(
                    key,
                    data=array,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                    chunks=True  # Enable chunking for better compression
                )

            # Add particle group attributes
            particle_group.attrs['n_particles'] = n_particles
            particle_group.attrs['time'] = time

            # Create metadata group
            meta_group = f.create_group('metadata')
            meta_group.attrs['code_version'] = self.code_version
            meta_group.attrs['creation_time'] = datetime.now().isoformat()
            meta_group.attrs['simulation_time'] = time
            meta_group.attrs['n_particles'] = n_particles

            # Store user-provided metadata
            if metadata is not None:
                for key, value in metadata.items():
                    # Handle different types appropriately
                    if isinstance(value, (int, float, str, bool)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        meta_group.create_dataset(
                            key,
                            data=value,
                            compression=self.compression,
                            compression_opts=self.compression_level
                        )
                    elif isinstance(value, (list, tuple)):
                        # Convert to numpy array for storage
                        meta_group.create_dataset(
                            key,
                            data=np.array(value),
                            compression=self.compression,
                            compression_opts=self.compression_level
                        )
                    else:
                        # Store string representation for complex types
                        meta_group.attrs[key] = str(value)

            # Add root-level attributes for quick access
            f.attrs['time'] = time
            f.attrs['n_particles'] = n_particles
            f.attrs['code_version'] = self.code_version

    def read_snapshot(
        self,
        filename: str,
        load_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Read particle snapshot from HDF5 file.

        Parameters
        ----------
        filename : str
            Path to HDF5 snapshot file.
        load_metadata : bool, optional
            If True, include metadata in returned dict. Default: True.

        Returns
        -------
        data : Dict[str, Any]
            Dictionary containing:
            - 'particles': Dict of particle arrays (positions, velocities, etc.)
            - 'time': float - simulation time
            - 'n_particles': int - number of particles
            - 'metadata': Dict - additional metadata (if load_metadata=True)

        Examples
        --------
        >>> data = writer.read_snapshot("snapshot_0000.h5")
        >>> positions = data['particles']['positions']
        >>> time = data['time']
        >>> print(f"Loaded {data['n_particles']} particles at t={time}")
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"Snapshot file not found: {filename}")

        with h5py.File(filename, 'r') as f:
            # Read particle data
            particle_group = f['particles']
            particles = {}

            for key in particle_group.keys():
                particles[key] = particle_group[key][:]

            # Get basic info
            time = float(f.attrs['time'])
            n_particles = int(f.attrs['n_particles'])

            # Prepare return dictionary
            result = {
                'particles': particles,
                'time': time,
                'n_particles': n_particles,
            }

            # Load metadata if requested
            if load_metadata and 'metadata' in f:
                meta_group = f['metadata']
                metadata = {}

                # Load attributes
                for key in meta_group.attrs.keys():
                    metadata[key] = meta_group.attrs[key]

                # Load datasets
                for key in meta_group.keys():
                    metadata[key] = meta_group[key][:]

                result['metadata'] = metadata

            return result

    def list_snapshots(self, directory: str, pattern: str = "snapshot_*.h5") -> list:
        """
        List all snapshot files in a directory.

        Parameters
        ----------
        directory : str
            Directory to search.
        pattern : str, optional
            Glob pattern for snapshot files. Default: "snapshot_*.h5".

        Returns
        -------
        files : list of Path
            Sorted list of snapshot files.

        Examples
        --------
        >>> snapshots = writer.list_snapshots("output/")
        >>> for snap in snapshots:
        ...     data = writer.read_snapshot(snap)
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        files = sorted(dir_path.glob(pattern))
        return files

    def get_snapshot_info(self, filename: str) -> Dict[str, Any]:
        """
        Get basic information about a snapshot without loading full data.

        Useful for quickly scanning through many snapshots.

        Parameters
        ----------
        filename : str
            Path to HDF5 snapshot file.

        Returns
        -------
        info : Dict[str, Any]
            Dictionary with time, n_particles, code_version, etc.

        Examples
        --------
        >>> info = writer.get_snapshot_info("snapshot_0042.h5")
        >>> print(f"Time: {info['time']}, N: {info['n_particles']}")
        """
        with h5py.File(filename, 'r') as f:
            info = {
                'filename': filename,
                'time': float(f.attrs['time']),
                'n_particles': int(f.attrs['n_particles']),
                'code_version': f.attrs['code_version'],
            }

            # Get list of available particle fields
            if 'particles' in f:
                info['particle_fields'] = list(f['particles'].keys())

            return info


def write_snapshot(
    filename: str,
    particles: Dict[str, NDArrayFloat],
    time: float,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Convenience function to write a snapshot with default settings.

    Parameters
    ----------
    filename : str
        Output HDF5 file path.
    particles : Dict[str, NDArrayFloat]
        Particle data arrays.
    time : float
        Simulation time.
    metadata : Dict[str, Any], optional
        Additional metadata.
    **kwargs
        Additional arguments passed to HDF5Writer constructor.

    Examples
    --------
    >>> write_snapshot("snap.h5", particles, time=1.0, metadata={"bh_mass": 1e6})
    """
    writer = HDF5Writer(**kwargs)
    writer.write_snapshot(filename, particles, time, metadata)


def read_snapshot(filename: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to read a snapshot with default settings.

    Parameters
    ----------
    filename : str
        Input HDF5 file path.
    **kwargs
        Additional arguments passed to HDF5Writer.read_snapshot().

    Returns
    -------
    data : Dict[str, Any]
        Snapshot data including particles and metadata.

    Examples
    --------
    >>> data = read_snapshot("snap.h5")
    >>> positions = data['particles']['positions']
    """
    writer = HDF5Writer()
    return writer.read_snapshot(filename, **kwargs)
