"""
Diagnostic outputs and light curve logging for TDE-SPH simulations.

Implements structured time-series outputs for:
- Energy evolution (kinetic, potential, internal, total)
- Luminosity light curves
- Fallback rates and bound mass
- Thermodynamic profiles

Supports multiple output formats: CSV (default), HDF5, JSON.

Usage:
    >>> diagnostics = DiagnosticsWriter("output/diagnostics", format='csv')
    >>> diagnostics.write_energy_diagnostic(time=0.0, energy_dict=energies)
    >>> diagnostics.write_luminosity(time=0.0, luminosity=1e42)
    >>> diagnostics.finalize()
"""

import numpy as np
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float32]


class DiagnosticsWriter:
    """
    Time-series diagnostic output writer for TDE-SPH simulations.

    Writes structured diagnostic data in CSV, HDF5, or JSON formats.
    Manages multiple output files for different diagnostic types.

    Parameters
    ----------
    output_dir : str or Path
        Directory for diagnostic outputs.
    format : str, optional
        Output format: 'csv', 'hdf5', or 'json' (default: 'csv').
    append : bool, optional
        If True, append to existing files (default: False).
    buffer_size : int, optional
        Number of entries to buffer before writing (default: 1, write immediately).

    Attributes
    ----------
    output_dir : Path
        Output directory path.
    format : str
        Active output format.
    files : Dict[str, Any]
        Open file handles and writers.
    buffers : Dict[str, List]
        Data buffers for each diagnostic type.

    Methods
    -------
    write_energy_diagnostic(time, energy_dict)
        Log energy components.
    write_luminosity(time, luminosity, **components)
        Log luminosity and components.
    write_fallback_rate(time, fallback_rate, bound_mass)
        Log fallback rate and bound mass.
    write_radial_profile(time, radii, quantities_dict)
        Log radial profiles.
    finalize()
        Close files and write metadata.

    Notes
    -----
    CSV format is pandas-compatible with headers.
    HDF5 format uses datasets for each time-series.
    JSON format writes metadata and summary statistics.
    """

    def __init__(
        self,
        output_dir: str,
        format: str = 'csv',
        append: bool = False,
        buffer_size: int = 1
    ):
        """
        Initialize diagnostics writer.

        Parameters
        ----------
        output_dir : str
            Output directory path.
        format : str, optional
            Format: 'csv', 'hdf5', 'json' (default: 'csv').
        append : bool, optional
            Append to existing files (default: False).
        buffer_size : int, optional
            Buffer size (default: 1, immediate write).
        """
        valid_formats = ['csv', 'hdf5', 'json']
        if format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}, got '{format}'")

        self.output_dir = Path(output_dir)
        self.format = format
        self.append = append
        self.buffer_size = buffer_size

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File handles and writers
        self.files: Dict[str, Any] = {}
        self.writers: Dict[str, Any] = {}
        self.buffers: Dict[str, List] = {}

        # Metadata
        self.metadata: Dict[str, Any] = {
            'creation_time': datetime.now().isoformat(),
            'format': self.format
        }

        # Initialize files
        self._initialize_files()

    def _initialize_files(self) -> None:
        """Initialize output files based on format."""
        if self.format == 'csv':
            self._initialize_csv_files()
        elif self.format == 'hdf5':
            self._initialize_hdf5_files()
        elif self.format == 'json':
            self._initialize_json_files()

    def _initialize_csv_files(self) -> None:
        """Initialize CSV files for each diagnostic type."""
        # Energy diagnostics CSV
        energy_file = self.output_dir / 'energy.csv'
        mode = 'a' if self.append and energy_file.exists() else 'w'
        self.files['energy'] = open(energy_file, mode, newline='')
        self.writers['energy'] = csv.writer(self.files['energy'])

        # Write header if new file
        if mode == 'w':
            energy_header = [
                'time', 'E_kinetic', 'E_potential', 'E_internal_total',
                'E_internal_gas', 'E_internal_radiation', 'E_total',
                'energy_conservation'
            ]
            self.writers['energy'].writerow(energy_header)
            self.files['energy'].flush()

        # Luminosity CSV
        lum_file = self.output_dir / 'luminosity.csv'
        mode = 'a' if self.append and lum_file.exists() else 'w'
        self.files['luminosity'] = open(lum_file, mode, newline='')
        self.writers['luminosity'] = csv.writer(self.files['luminosity'])

        if mode == 'w':
            lum_header = ['time', 'L_total']
            self.writers['luminosity'].writerow(lum_header)
            self.files['luminosity'].flush()

        # Fallback rate CSV
        fb_file = self.output_dir / 'fallback.csv'
        mode = 'a' if self.append and fb_file.exists() else 'w'
        self.files['fallback'] = open(fb_file, mode, newline='')
        self.writers['fallback'] = csv.writer(self.files['fallback'])

        if mode == 'w':
            fb_header = ['time', 'fallback_rate', 'bound_mass']
            self.writers['fallback'].writerow(fb_header)
            self.files['fallback'].flush()

    def _initialize_hdf5_files(self) -> None:
        """Initialize HDF5 file for time-series data."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 format. Install with: pip install h5py")

        hdf5_file = self.output_dir / 'diagnostics.h5'
        mode = 'a' if self.append else 'w'
        self.files['hdf5'] = h5py.File(hdf5_file, mode)

        # Create groups for different diagnostics
        if 'energy' not in self.files['hdf5']:
            self.files['hdf5'].create_group('energy')
        if 'luminosity' not in self.files['hdf5']:
            self.files['hdf5'].create_group('luminosity')
        if 'fallback' not in self.files['hdf5']:
            self.files['hdf5'].create_group('fallback')

        # Initialize buffers for HDF5
        self.buffers = {
            'energy': [],
            'luminosity': [],
            'fallback': []
        }

    def _initialize_json_files(self) -> None:
        """Initialize JSON buffers (written at finalize)."""
        self.buffers = {
            'energy': [],
            'luminosity': [],
            'fallback': []
        }

    def write_energy_diagnostic(
        self,
        time: float,
        energy_dict: Dict[str, float]
    ) -> None:
        """
        Write energy diagnostic entry.

        Parameters
        ----------
        time : float
            Simulation time.
        energy_dict : Dict[str, float]
            Energy components from EnergyDiagnostics.compute().
            Expected keys: E_kinetic, E_potential, E_internal_total,
            E_internal_gas, E_internal_radiation, E_total, energy_conservation.

        Notes
        -----
        Missing keys are filled with NaN.
        """
        if self.format == 'csv':
            row = [
                time,
                energy_dict.get('E_kinetic', np.nan),
                energy_dict.get('E_potential', np.nan),
                energy_dict.get('E_internal_total', np.nan),
                energy_dict.get('E_internal_gas', np.nan),
                energy_dict.get('E_internal_radiation', np.nan),
                energy_dict.get('E_total', np.nan),
                energy_dict.get('energy_conservation', np.nan)
            ]
            self.writers['energy'].writerow(row)
            self.files['energy'].flush()

        elif self.format == 'hdf5':
            entry = {'time': time, **energy_dict}
            self.buffers['energy'].append(entry)
            if len(self.buffers['energy']) >= self.buffer_size:
                self._flush_hdf5_buffer('energy')

        elif self.format == 'json':
            entry = {'time': time, **energy_dict}
            self.buffers['energy'].append(entry)

    def write_luminosity(
        self,
        time: float,
        luminosity: float,
        **components
    ) -> None:
        """
        Write luminosity entry.

        Parameters
        ----------
        time : float
            Simulation time.
        luminosity : float
            Total luminosity [erg/s].
        **components
            Additional luminosity components (e.g., L_gas, L_radiation).
        """
        if self.format == 'csv':
            row = [time, luminosity]
            # Add component columns if header needs updating
            self.writers['luminosity'].writerow(row)
            self.files['luminosity'].flush()

        elif self.format == 'hdf5':
            entry = {'time': time, 'L_total': luminosity, **components}
            self.buffers['luminosity'].append(entry)
            if len(self.buffers['luminosity']) >= self.buffer_size:
                self._flush_hdf5_buffer('luminosity')

        elif self.format == 'json':
            entry = {'time': time, 'L_total': luminosity, **components}
            self.buffers['luminosity'].append(entry)

    def write_fallback_rate(
        self,
        time: float,
        fallback_rate: float,
        bound_mass: float
    ) -> None:
        """
        Write fallback rate entry.

        Parameters
        ----------
        time : float
            Simulation time.
        fallback_rate : float
            Mass fallback rate dM/dt.
        bound_mass : float
            Total bound mass.
        """
        if self.format == 'csv':
            row = [time, fallback_rate, bound_mass]
            self.writers['fallback'].writerow(row)
            self.files['fallback'].flush()

        elif self.format == 'hdf5':
            entry = {'time': time, 'fallback_rate': fallback_rate, 'bound_mass': bound_mass}
            self.buffers['fallback'].append(entry)
            if len(self.buffers['fallback']) >= self.buffer_size:
                self._flush_hdf5_buffer('fallback')

        elif self.format == 'json':
            entry = {'time': time, 'fallback_rate': fallback_rate, 'bound_mass': bound_mass}
            self.buffers['fallback'].append(entry)

    def write_radial_profile(
        self,
        time: float,
        radii: NDArrayFloat,
        quantities_dict: Dict[str, NDArrayFloat]
    ) -> None:
        """
        Write radial profile snapshot.

        Parameters
        ----------
        time : float
            Simulation time.
        radii : NDArrayFloat, shape (N_bins,)
            Radial bins.
        quantities_dict : Dict[str, NDArrayFloat]
            Quantities vs radius (density, temperature, etc.).

        Notes
        -----
        Profiles are written to separate HDF5 files in profiles/ subdirectory.
        CSV format not supported for profiles (use HDF5).
        """
        if self.format == 'csv':
            # Skip profiles for CSV (too complex)
            return

        elif self.format in ['hdf5', 'json']:
            # Create profiles subdirectory
            profiles_dir = self.output_dir / 'profiles'
            profiles_dir.mkdir(exist_ok=True)

            # HDF5 profile snapshot
            if self.format == 'hdf5':
                import h5py
                profile_file = profiles_dir / f'profile_{int(time*1000):06d}.h5'

                with h5py.File(profile_file, 'w') as f:
                    f.attrs['time'] = time
                    f.create_dataset('radii', data=radii, compression='gzip')

                    for key, values in quantities_dict.items():
                        f.create_dataset(key, data=values, compression='gzip')

    def _flush_hdf5_buffer(self, diagnostic_type: str) -> None:
        """Flush buffered data to HDF5 file."""
        if not self.buffers.get(diagnostic_type):
            return

        import h5py

        group = self.files['hdf5'][diagnostic_type]
        buffer_data = self.buffers[diagnostic_type]

        # Extract all keys from first entry
        if not buffer_data:
            return

        keys = buffer_data[0].keys()

        # Append to each dataset
        for key in keys:
            values = np.array([entry[key] for entry in buffer_data])

            if key in group:
                # Extend existing dataset
                dset = group[key]
                old_size = dset.shape[0]
                new_size = old_size + len(values)
                dset.resize(new_size, axis=0)
                dset[old_size:new_size] = values
            else:
                # Create new dataset
                maxshape = (None,) + values.shape[1:] if values.ndim > 1 else (None,)
                group.create_dataset(
                    key,
                    data=values,
                    maxshape=maxshape,
                    compression='gzip'
                )

        # Clear buffer
        self.buffers[diagnostic_type] = []

    def finalize(self) -> None:
        """
        Close files and write metadata.

        Should be called at end of simulation to ensure all data is written.
        """
        # Flush HDF5 buffers
        if self.format == 'hdf5':
            for key in list(self.buffers.keys()):
                if self.buffers[key]:
                    self._flush_hdf5_buffer(key)

        # Write JSON files
        if self.format == 'json':
            for diagnostic_type, data in self.buffers.items():
                json_file = self.output_dir / f'{diagnostic_type}.json'
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2, default=_json_serializer)

        # Close file handles
        for file_handle in self.files.values():
            if hasattr(file_handle, 'close'):
                file_handle.close()

        # Write metadata
        metadata_file = self.output_dir / 'metadata.json'
        self.metadata['finalize_time'] = datetime.now().isoformat()

        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=_json_serializer)

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add custom metadata.

        Parameters
        ----------
        key : str
            Metadata key.
        value : Any
            Metadata value (must be JSON-serializable).
        """
        self.metadata[key] = value

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()

    def __repr__(self) -> str:
        """String representation."""
        return f"DiagnosticsWriter(output_dir='{self.output_dir}', format='{self.format}')"


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return str(obj)


def compute_fallback_rate(
    particles: Dict[str, NDArrayFloat],
    bh_mass: float = 1.0,
    r_capture: float = 10.0
) -> Dict[str, float]:
    """
    Compute fallback rate proxy from particle data.

    Parameters
    ----------
    particles : Dict[str, NDArrayFloat]
        Particle data with 'positions', 'velocities', 'masses'.
    bh_mass : float, optional
        Black hole mass in code units (default: 1.0).
    r_capture : float, optional
        Capture radius in code units (default: 10.0).

    Returns
    -------
    fallback_info : Dict[str, float]
        Dictionary with:
        - 'bound_mass': Total bound mass
        - 'captured_mass': Mass inside r_capture
        - 'N_bound': Number of bound particles
        - 'N_captured': Number of captured particles

    Notes
    -----
    A particle is bound if E = (1/2)v² - GM/r < 0.
    Fallback rate dM/dt is estimated by time-differencing bound mass.
    """
    positions = particles['positions']
    velocities = particles['velocities']
    masses = particles['masses']

    # Compute specific energies
    r = np.linalg.norm(positions, axis=1)
    v_squared = np.sum(velocities**2, axis=1)

    # Specific energy (kinetic + potential)
    E_specific = 0.5 * v_squared - bh_mass / np.maximum(r, 1e-10)

    # Bound particles
    bound = E_specific < 0
    bound_mass = np.sum(masses[bound])
    N_bound = np.sum(bound)

    # Captured particles
    captured = r < r_capture
    captured_mass = np.sum(masses[captured])
    N_captured = np.sum(captured)

    return {
        'bound_mass': float(bound_mass),
        'captured_mass': float(captured_mass),
        'N_bound': int(N_bound),
        'N_captured': int(N_captured)
    }
