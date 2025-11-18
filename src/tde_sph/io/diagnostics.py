"""
Diagnostic outputs for TDE-SPH simulations (TASK-024).

Implements comprehensive time-series diagnostics:
- Light curves (luminosity vs time)
- Fallback rates (mass return rate vs time)
- Energy evolution (all energy components vs time)
- Orbital elements (apocenter, pericenter, eccentricity, etc.)
- Radial profiles (density, temperature, pressure vs radius)

Supports REQ-009 (energy accounting & luminosity) and CON-004 (data export).

Output formats:
- HDF5 time series: efficient storage with compression
- CSV summaries: lightweight exports for plotting
- Snapshot metadata: embedded diagnostics in simulation snapshots

References:
    Lodato & Rossi (2011) - TDE light curves
    Guillochon & Ramirez-Ruiz (2013) - Fallback rates
    Dai et al. (2015) - Optical/UV light curves
"""

import h5py
import numpy as np
import csv
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float32]


@dataclass
class OrbitalElements:
    """
    Container for orbital elements of debris particles.

    Attributes
    ----------
    apocenter : NDArrayFloat
        Apocenter distance r_a for each particle.
    pericenter : NDArrayFloat
        Pericenter distance r_p for each particle.
    eccentricity : NDArrayFloat
        Orbital eccentricity e for each particle.
    semi_major_axis : NDArrayFloat
        Semi-major axis a for each particle.
    specific_energy : NDArrayFloat
        Specific orbital energy E/m for each particle.
    specific_angular_momentum : NDArrayFloat
        Specific angular momentum |L|/m for each particle.
    """
    apocenter: NDArrayFloat
    pericenter: NDArrayFloat
    eccentricity: NDArrayFloat
    semi_major_axis: NDArrayFloat
    specific_energy: NDArrayFloat
    specific_angular_momentum: NDArrayFloat


class DiagnosticsWriter:
    """
    Diagnostic outputs for TDE-SPH simulations.

    Handles time-series diagnostics (light curves, fallback rates, energies),
    radial profiles, and orbital element distributions.

    Parameters
    ----------
    output_dir : str or Path
        Directory for diagnostic outputs (created if doesn't exist).
    compression : str, optional
        HDF5 compression algorithm (default 'gzip').
    compression_level : int, optional
        HDF5 compression level 0-9 (default 4).

    Attributes
    ----------
    output_dir : Path
        Output directory path.
    compression : str
        HDF5 compression algorithm.
    compression_level : int
        HDF5 compression level.
    hdf5_file : Path
        Path to HDF5 diagnostics file.

    Notes
    -----
    **Output structure**:

    HDF5 file (`diagnostics.h5`):
    - `/light_curve/time`, `/light_curve/luminosity`, `/light_curve/L_mean`, etc.
    - `/fallback_rate/time`, `/fallback_rate/M_dot`
    - `/energies/time`, `/energies/kinetic`, `/energies/potential_bh`, etc.
    - `/radial_profiles/radius`, `/radial_profiles/density`, etc. (per snapshot)

    CSV files:
    - `light_curve.csv`: time, L_bol, L_mean, L_max
    - `fallback_rate.csv`: time, M_dot, M_cumulative
    - `energies.csv`: time, E_kin, E_pot_bh, E_pot_self, E_int, E_rad, E_tot, dE/E0

    **Usage**:
    ```python
    diag = DiagnosticsWriter("output/diagnostics")

    # Write light curve
    diag.write_light_curve(t, L_bol, L_mean=L_mean, L_max=L_max)

    # Write energy evolution
    diag.write_energy_evolution(t, energies)  # energies from EnergyDiagnostics

    # Write fallback rate
    diag.write_fallback_rate(t, M_dot, M_cumulative=M_cum)

    # Write radial profiles
    diag.write_radial_profiles(snapshot_id, r_bins, rho_profile, T_profile=T)
    ```
    """

    def __init__(
        self,
        output_dir: str,
        compression: str = "gzip",
        compression_level: int = 4
    ):
        """
        Initialize diagnostics writer.

        Parameters
        ----------
        output_dir : str
            Directory for diagnostic files.
        compression : str
            HDF5 compression ('gzip', 'lzf', or None).
        compression_level : int
            Compression level (0-9 for gzip).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.compression_level = compression_level if compression == "gzip" else None

        self.hdf5_file = self.output_dir / "diagnostics.h5"

        # Initialize CSV files
        self.light_curve_csv = self.output_dir / "light_curve.csv"
        self.fallback_csv = self.output_dir / "fallback_rate.csv"
        self.energies_csv = self.output_dir / "energies.csv"

    def write_light_curve(
        self,
        time: float,
        luminosity_bol: float,
        L_mean: Optional[float] = None,
        L_max: Optional[float] = None,
        **kwargs
    ):
        """
        Write light curve data point to HDF5 and CSV.

        Parameters
        ----------
        time : float
            Current simulation time.
        luminosity_bol : float
            Total bolometric luminosity L_bol.
        L_mean : float, optional
            Mean luminosity per particle.
        L_max : float, optional
            Maximum particle luminosity.
        **kwargs
            Additional light curve quantities (e.g., L_UV, L_optical).

        Notes
        -----
        Appends to `/light_curve/` group in HDF5 file and `light_curve.csv`.
        """
        # Write to HDF5
        with h5py.File(self.hdf5_file, 'a') as f:
            lc_group = f.require_group('light_curve')

            # Append time and luminosity
            self._append_dataset(lc_group, 'time', time)
            self._append_dataset(lc_group, 'luminosity_bol', luminosity_bol)

            if L_mean is not None:
                self._append_dataset(lc_group, 'L_mean', L_mean)
            if L_max is not None:
                self._append_dataset(lc_group, 'L_max', L_max)

            # Additional kwargs
            for key, value in kwargs.items():
                self._append_dataset(lc_group, key, value)

        # Write to CSV
        csv_exists = self.light_curve_csv.exists()
        with open(self.light_curve_csv, 'a', newline='') as csvfile:
            fieldnames = ['time', 'luminosity_bol', 'L_mean', 'L_max'] + list(kwargs.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not csv_exists:
                writer.writeheader()

            row = {
                'time': time,
                'luminosity_bol': luminosity_bol,
                'L_mean': L_mean if L_mean is not None else 0.0,
                'L_max': L_max if L_max is not None else 0.0,
                **kwargs
            }
            writer.writerow(row)

    def write_fallback_rate(
        self,
        time: float,
        M_dot: float,
        M_cumulative: Optional[float] = None,
        **kwargs
    ):
        """
        Write fallback rate data to HDF5 and CSV.

        Parameters
        ----------
        time : float
            Current simulation time.
        M_dot : float
            Fallback rate dM/dt (mass per time).
        M_cumulative : float, optional
            Cumulative mass returned to r_fallback.
        **kwargs
            Additional quantities (e.g., M_dot_unbound, mean_specific_energy).

        Notes
        -----
        Fallback rate is defined as mass crossing r_fallback inward per unit time.
        Typical scaling: Ṁ_fb ∝ t^(-5/3) (parabolic orbits).

        References:
            Guillochon & Ramirez-Ruiz (2013) - Analytical fallback rates
        """
        # Write to HDF5
        with h5py.File(self.hdf5_file, 'a') as f:
            fb_group = f.require_group('fallback_rate')

            self._append_dataset(fb_group, 'time', time)
            self._append_dataset(fb_group, 'M_dot', M_dot)

            if M_cumulative is not None:
                self._append_dataset(fb_group, 'M_cumulative', M_cumulative)

            for key, value in kwargs.items():
                self._append_dataset(fb_group, key, value)

        # Write to CSV
        csv_exists = self.fallback_csv.exists()
        with open(self.fallback_csv, 'a', newline='') as csvfile:
            fieldnames = ['time', 'M_dot', 'M_cumulative'] + list(kwargs.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not csv_exists:
                writer.writeheader()

            row = {
                'time': time,
                'M_dot': M_dot,
                'M_cumulative': M_cumulative if M_cumulative is not None else 0.0,
                **kwargs
            }
            writer.writerow(row)

    def write_energy_evolution(
        self,
        time: float,
        energies: Dict[str, float]
    ):
        """
        Write energy evolution data to HDF5 and CSV.

        Parameters
        ----------
        time : float
            Current simulation time.
        energies : dict
            Dictionary from EnergyComponents.to_dict() with keys:
            'kinetic', 'potential_bh', 'potential_self', 'internal_thermal',
            'internal_radiation', 'radiated_cumulative', 'total', 'conservation_error'.

        Notes
        -----
        Typical usage:
        ```python
        energy_diag = EnergyDiagnostics(...)
        energies_comp = energy_diag.compute_all_energies(...)
        diag.write_energy_evolution(t, energies_comp.to_dict())
        ```
        """
        # Write to HDF5
        with h5py.File(self.hdf5_file, 'a') as f:
            e_group = f.require_group('energies')

            self._append_dataset(e_group, 'time', time)

            for key, value in energies.items():
                if key != 'time':  # Skip time (already written)
                    self._append_dataset(e_group, key, value)

        # Write to CSV
        csv_exists = self.energies_csv.exists()
        with open(self.energies_csv, 'a', newline='') as csvfile:
            fieldnames = ['time'] + list(energies.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not csv_exists:
                writer.writeheader()

            row = {'time': time, **energies}
            writer.writerow(row)

    def write_orbital_elements(
        self,
        snapshot_id: int,
        orbital_elements: OrbitalElements
    ):
        """
        Write orbital element distributions to HDF5.

        Parameters
        ----------
        snapshot_id : int
            Snapshot identifier for organizing data.
        orbital_elements : OrbitalElements
            Orbital elements dataclass with per-particle arrays.

        Notes
        -----
        Stores full per-particle distributions in HDF5:
        `/orbital_elements/snapshot_{id:04d}/apocenter`, etc.

        For large N, consider binned distributions or summary statistics.
        """
        with h5py.File(self.hdf5_file, 'a') as f:
            oe_group = f.require_group(f'orbital_elements/snapshot_{snapshot_id:04d}')

            for key, value in asdict(orbital_elements).items():
                oe_group.create_dataset(
                    key,
                    data=value,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )

    def write_radial_profiles(
        self,
        snapshot_id: int,
        radius: NDArrayFloat,
        density: NDArrayFloat,
        temperature: Optional[NDArrayFloat] = None,
        pressure: Optional[NDArrayFloat] = None,
        internal_energy: Optional[NDArrayFloat] = None,
        **kwargs
    ):
        """
        Write radial profiles to HDF5.

        Parameters
        ----------
        snapshot_id : int
            Snapshot identifier.
        radius : NDArrayFloat
            Radial bin centers.
        density : NDArrayFloat
            Density profile ρ(r).
        temperature : NDArrayFloat, optional
            Temperature profile T(r).
        pressure : NDArrayFloat, optional
            Pressure profile P(r).
        internal_energy : NDArrayFloat, optional
            Internal energy profile u(r).
        **kwargs
            Additional profiles (e.g., entropy, velocity_radial).

        Notes
        -----
        Stores in `/radial_profiles/snapshot_{id:04d}/`.
        Profiles are typically mass-averaged or volume-averaged in radial bins.
        """
        with h5py.File(self.hdf5_file, 'a') as f:
            rp_group = f.require_group(f'radial_profiles/snapshot_{snapshot_id:04d}')

            rp_group.create_dataset(
                'radius',
                data=radius,
                compression=self.compression,
                compression_opts=self.compression_level
            )
            rp_group.create_dataset(
                'density',
                data=density,
                compression=self.compression,
                compression_opts=self.compression_level
            )

            if temperature is not None:
                rp_group.create_dataset(
                    'temperature',
                    data=temperature,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            if pressure is not None:
                rp_group.create_dataset(
                    'pressure',
                    data=pressure,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            if internal_energy is not None:
                rp_group.create_dataset(
                    'internal_energy',
                    data=internal_energy,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )

            for key, value in kwargs.items():
                rp_group.create_dataset(
                    key,
                    data=value,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )

    def get_light_curve(self) -> Dict[str, NDArrayFloat]:
        """
        Read light curve time series from HDF5.

        Returns
        -------
        light_curve : dict
            Dictionary with keys 'time', 'luminosity_bol', 'L_mean', 'L_max', etc.

        Raises
        ------
        FileNotFoundError
            If diagnostics.h5 does not exist.
        KeyError
            If light_curve group does not exist in HDF5.
        """
        if not self.hdf5_file.exists():
            raise FileNotFoundError(f"Diagnostics file not found: {self.hdf5_file}")

        with h5py.File(self.hdf5_file, 'r') as f:
            if 'light_curve' not in f:
                raise KeyError("No light_curve data found in diagnostics file")

            lc_group = f['light_curve']
            light_curve = {key: np.array(lc_group[key]) for key in lc_group.keys()}

        return light_curve

    def get_energies(self) -> Dict[str, NDArrayFloat]:
        """
        Read energy evolution time series from HDF5.

        Returns
        -------
        energies : dict
            Dictionary with keys 'time', 'kinetic', 'potential_bh', etc.
        """
        if not self.hdf5_file.exists():
            raise FileNotFoundError(f"Diagnostics file not found: {self.hdf5_file}")

        with h5py.File(self.hdf5_file, 'r') as f:
            if 'energies' not in f:
                raise KeyError("No energies data found in diagnostics file")

            e_group = f['energies']
            energies = {key: np.array(e_group[key]) for key in e_group.keys()}

        return energies

    def get_fallback_rate(self) -> Dict[str, NDArrayFloat]:
        """
        Read fallback rate time series from HDF5.

        Returns
        -------
        fallback_rate : dict
            Dictionary with keys 'time', 'M_dot', 'M_cumulative', etc.
        """
        if not self.hdf5_file.exists():
            raise FileNotFoundError(f"Diagnostics file not found: {self.hdf5_file}")

        with h5py.File(self.hdf5_file, 'r') as f:
            if 'fallback_rate' not in f:
                raise KeyError("No fallback_rate data found in diagnostics file")

            fb_group = f['fallback_rate']
            fallback_rate = {key: np.array(fb_group[key]) for key in fb_group.keys()}

        return fallback_rate

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics from diagnostic data.

        Returns
        -------
        summary : dict
            Dictionary with keys:
            - 'peak_luminosity': Maximum L_bol
            - 'time_of_peak': Time of peak luminosity
            - 'total_radiated_energy': Cumulative radiated energy
            - 'final_conservation_error': Final energy conservation error
            - 'peak_fallback_rate': Maximum Ṁ_fb
            - 'total_returned_mass': Final M_cumulative

        Notes
        -----
        Useful for quick summaries and regression testing.
        """
        summary = {}

        try:
            lc = self.get_light_curve()
            if 'luminosity_bol' in lc:
                idx_peak = np.argmax(lc['luminosity_bol'])
                summary['peak_luminosity'] = float(lc['luminosity_bol'][idx_peak])
                summary['time_of_peak'] = float(lc['time'][idx_peak])

            energies = self.get_energies()
            if 'radiated_cumulative' in energies:
                summary['total_radiated_energy'] = float(energies['radiated_cumulative'][-1])
            if 'conservation_error' in energies:
                summary['final_conservation_error'] = float(energies['conservation_error'][-1])

            fb = self.get_fallback_rate()
            if 'M_dot' in fb:
                summary['peak_fallback_rate'] = float(np.max(fb['M_dot']))
            if 'M_cumulative' in fb:
                summary['total_returned_mass'] = float(fb['M_cumulative'][-1])

        except (FileNotFoundError, KeyError):
            pass  # Return partial summary if some data missing

        return summary

    def _append_dataset(self, group: h5py.Group, name: str, value: float):
        """
        Append a single value to an HDF5 dataset (time series).

        If dataset doesn't exist, creates it with expandable maxshape.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to append to.
        name : str
            Dataset name.
        value : float
            Value to append.
        """
        if name in group:
            # Dataset exists - resize and append
            dset = group[name]
            current_size = dset.shape[0]
            dset.resize((current_size + 1,))
            dset[current_size] = value
        else:
            # Create new dataset with expandable size
            group.create_dataset(
                name,
                data=np.array([value]),
                maxshape=(None,),
                compression=self.compression,
                compression_opts=self.compression_level
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"DiagnosticsWriter(output_dir={self.output_dir})"


# Utility functions for computing orbital elements and profiles

def compute_orbital_elements(
    positions: NDArrayFloat,
    velocities: NDArrayFloat,
    bh_mass: float = 1.0
) -> OrbitalElements:
    """
    Compute orbital elements for debris particles.

    Parameters
    ----------
    positions : NDArrayFloat, shape (N, 3)
        Particle positions.
    velocities : NDArrayFloat, shape (N, 3)
        Particle velocities.
    bh_mass : float
        Black hole mass (default 1.0 in code units).

    Returns
    -------
    orbital_elements : OrbitalElements
        Orbital elements for each particle.

    Notes
    -----
    Assumes Keplerian orbits in Newtonian potential.
    For GR, these are approximate (coordinate-dependent).

    Formulas:
    - E = v²/2 - M_BH/r (specific energy)
    - L = r × v (specific angular momentum)
    - a = -M_BH / (2E) (semi-major axis)
    - e = sqrt(1 + (2 E L²) / M_BH²) (eccentricity)
    - r_p = a(1 - e), r_a = a(1 + e)
    """
    N = len(positions)

    # Compute radius and speed
    r = np.sqrt(np.sum(positions**2, axis=1))
    v_squared = np.sum(velocities**2, axis=1)

    # Specific energy: E = v²/2 - M/r
    E_spec = 0.5 * v_squared - bh_mass / r

    # Specific angular momentum: L = r × v
    L_vec = np.cross(positions, velocities)
    L_mag = np.sqrt(np.sum(L_vec**2, axis=1))

    # Semi-major axis: a = -M / (2E) for bound orbits (E < 0)
    a = np.where(E_spec < 0, -bh_mass / (2.0 * E_spec), np.inf)

    # Eccentricity: e = sqrt(1 + 2 E L² / M²)
    e = np.sqrt(1.0 + (2.0 * E_spec * L_mag**2) / bh_mass**2)
    e = np.clip(e, 0.0, None)  # Ensure e >= 0

    # Pericenter and apocenter
    r_p = np.where(E_spec < 0, a * (1.0 - e), r * (1.0 - e) / (1.0 + e))
    r_a = np.where(E_spec < 0, a * (1.0 + e), np.inf)

    orbital_elements = OrbitalElements(
        apocenter=r_a.astype(np.float32),
        pericenter=r_p.astype(np.float32),
        eccentricity=e.astype(np.float32),
        semi_major_axis=a.astype(np.float32),
        specific_energy=E_spec.astype(np.float32),
        specific_angular_momentum=L_mag.astype(np.float32)
    )

    return orbital_elements


def compute_radial_profile(
    positions: NDArrayFloat,
    quantities: NDArrayFloat,
    masses: NDArrayFloat,
    n_bins: int = 50,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    log_bins: bool = True
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Compute mass-weighted radial profile of a quantity.

    Parameters
    ----------
    positions : NDArrayFloat, shape (N, 3)
        Particle positions.
    quantities : NDArrayFloat, shape (N,)
        Quantity to profile (e.g., density, temperature).
    masses : NDArrayFloat, shape (N,)
        Particle masses.
    n_bins : int, optional
        Number of radial bins (default 50).
    r_min : float, optional
        Minimum radius (default: min(r)).
    r_max : float, optional
        Maximum radius (default: max(r)).
    log_bins : bool, optional
        Use logarithmic binning (default True).

    Returns
    -------
    r_bins : NDArrayFloat
        Radial bin centers.
    profile : NDArrayFloat
        Mass-weighted average of quantity in each bin.

    Notes
    -----
    profile[i] = Σ_j (m_j * q_j) / Σ_j m_j for particles in bin i.
    Empty bins are set to NaN.
    """
    r = np.sqrt(np.sum(positions**2, axis=1))

    if r_min is None:
        r_min = np.min(r)
    if r_max is None:
        r_max = np.max(r)

    # Create bins
    if log_bins:
        r_min_safe = max(r_min, 1e-10)  # Avoid log(0)
        bin_edges = np.logspace(np.log10(r_min_safe), np.log10(r_max), n_bins + 1)
    else:
        bin_edges = np.linspace(r_min, r_max, n_bins + 1)

    r_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute profile
    profile = np.full(n_bins, np.nan, dtype=np.float32)

    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if np.any(mask):
            # Mass-weighted average
            profile[i] = np.sum(masses[mask] * quantities[mask]) / np.sum(masses[mask])

    return r_bins.astype(np.float32), profile
