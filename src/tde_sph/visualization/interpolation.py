#!/usr/bin/env python3
"""
Spatial and Temporal Interpolation & Smoothing (TASK-102)

Provides utilities for:
1. SPH kernel-weighted spatial interpolation (particles → grid)
2. Temporal interpolation between snapshots
3. Smoothing filters for visualization

Use cases:
- Generate 2D/3D volumetric grids for visualization
- Create smooth animations between coarse time snapshots
- Apply filters to reduce numerical noise in outputs

Architecture:
- SPHInterpolator: Particle → grid using SPH kernel weights
- TemporalInterpolator: Linear/cubic interpolation between snapshots
- SmoothingFilters: Gaussian, median, bilateral filters

References:
- Price (2012) - SPH review, kernel interpolation
- Liptai & Price (2019) - GRSPH rendering techniques
- Springel (2010) - Gadget-2 visualization

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import numpy as np
from typing import Tuple, Optional, Callable, List, Dict
from pathlib import Path
import warnings

# Type aliases
NDArrayFloat = np.ndarray


class SPHInterpolator:
    """
    SPH kernel-weighted interpolation from particles to regular grids.

    Maps particle-based SPH data to uniform Cartesian grids using
    kernel-weighted averaging, suitable for volumetric visualization
    (e.g., volume rendering, slice plots, contours).
    """

    # SPH kernels
    KERNELS = ['cubic_spline', 'wendland_c2', 'gaussian']

    def __init__(self, kernel: str = 'cubic_spline'):
        """
        Initialize SPH interpolator.

        Parameters
        ----------
        kernel : str, optional
            SPH kernel function: 'cubic_spline', 'wendland_c2', 'gaussian'
            (default: 'cubic_spline').
        """
        if kernel not in self.KERNELS:
            raise ValueError(f"Unknown kernel '{kernel}'. Choose from: {self.KERNELS}")
        self.kernel_name = kernel
        self.kernel_func = self._get_kernel_function(kernel)

    def _get_kernel_function(self, kernel: str) -> Callable:
        """Get kernel function by name."""
        if kernel == 'cubic_spline':
            return self._cubic_spline_kernel
        elif kernel == 'wendland_c2':
            return self._wendland_c2_kernel
        elif kernel == 'gaussian':
            return self._gaussian_kernel
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    @staticmethod
    def _cubic_spline_kernel(q: NDArrayFloat) -> NDArrayFloat:
        """
        Cubic spline kernel (M4 spline, standard SPH kernel).

        W(q) = C × { 1 - 3/2 q² + 3/4 q³,    0 ≤ q < 1
                    1/4 (2 - q)³,             1 ≤ q < 2
                    0,                        q ≥ 2 }

        where q = r/h, C = 1/(πh³) in 3D.

        Parameters
        ----------
        q : np.ndarray
            Dimensionless distance q = r / h.

        Returns
        -------
        W : np.ndarray
            Kernel weights.
        """
        W = np.zeros_like(q)
        mask1 = q < 1.0
        mask2 = (q >= 1.0) & (q < 2.0)

        W[mask1] = 1.0 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3
        W[mask2] = 0.25 * (2.0 - q[mask2])**3

        return W

    @staticmethod
    def _wendland_c2_kernel(q: NDArrayFloat) -> NDArrayFloat:
        """
        Wendland C² kernel (compact support, smooth).

        W(q) = C × (1 - q)⁴ (1 + 4q),  0 ≤ q < 1
               0,                       q ≥ 1

        where C = 21/(16πh³) in 3D.

        Parameters
        ----------
        q : np.ndarray
            Dimensionless distance q = r / h.

        Returns
        -------
        W : np.ndarray
            Kernel weights.
        """
        W = np.zeros_like(q)
        mask = q < 1.0

        W[mask] = (1.0 - q[mask])**4 * (1.0 + 4.0 * q[mask])

        return W

    @staticmethod
    def _gaussian_kernel(q: NDArrayFloat) -> NDArrayFloat:
        """
        Gaussian kernel (infinite support, truncated at q=3).

        W(q) = C × exp(-q²),  q < 3
               0,             q ≥ 3

        where C = 1/(π^(3/2) h³) in 3D.

        Parameters
        ----------
        q : np.ndarray
            Dimensionless distance q = r / h.

        Returns
        -------
        W : np.ndarray
            Kernel weights.
        """
        W = np.zeros_like(q)
        mask = q < 3.0

        W[mask] = np.exp(-q[mask]**2)

        return W

    def interpolate_to_grid(
        self,
        positions: NDArrayFloat,
        field_values: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        grid_shape: Tuple[int, int, int] = (64, 64, 64),
        bounds: Optional[Tuple[Tuple[float, float], ...]] = None
    ) -> Tuple[NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]]:
        """
        Interpolate particle field to regular 3D grid using SPH kernel.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
            Particle positions.
        field_values : np.ndarray, shape (N,) or (N, M)
            Field values to interpolate (scalar or vector).
        smoothing_lengths : np.ndarray, shape (N,)
            SPH smoothing lengths for each particle.
        grid_shape : tuple of int, optional
            Grid dimensions (nx, ny, nz) (default: (64, 64, 64)).
        bounds : tuple of tuples, optional
            Grid bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
            If None, computed from particle positions ± 2 × max(h).

        Returns
        -------
        grid : np.ndarray, shape grid_shape or grid_shape + (M,)
            Interpolated field on regular grid.
        coordinates : tuple of np.ndarray
            (X, Y, Z) coordinate arrays for grid (for np.meshgrid).

        Notes
        -----
        Interpolation formula:
            F(r_grid) = Σᵢ Fᵢ W(|r_grid - rᵢ|, hᵢ)
        where W is the SPH kernel.

        For sparse grids or large N, consider using spatial hashing or KD-tree
        to avoid O(N × grid_size) cost.

        Examples
        --------
        >>> positions = np.random.randn(1000, 3)
        >>> density = np.ones(1000)
        >>> h = np.ones(1000) * 0.1
        >>> interp = SPHInterpolator(kernel='cubic_spline')
        >>> grid, (X, Y, Z) = interp.interpolate_to_grid(positions, density, h)
        >>> grid.shape
        (64, 64, 64)
        """
        N = len(positions)
        if len(field_values) != N:
            raise ValueError(f"Field values length ({len(field_values)}) != N ({N})")
        if len(smoothing_lengths) != N:
            raise ValueError(f"Smoothing lengths length ({len(smoothing_lengths)}) != N ({N})")

        # Determine grid bounds
        if bounds is None:
            margin = 2.0 * np.max(smoothing_lengths)
            pos_min = np.min(positions, axis=0) - margin
            pos_max = np.max(positions, axis=0) + margin
            bounds = (
                (pos_min[0], pos_max[0]),
                (pos_min[1], pos_max[1]),
                (pos_min[2], pos_max[2])
            )

        # Create grid
        x = np.linspace(bounds[0][0], bounds[0][1], grid_shape[0])
        y = np.linspace(bounds[1][0], bounds[1][1], grid_shape[1])
        z = np.linspace(bounds[2][0], bounds[2][1], grid_shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Initialize grid
        is_vector_field = field_values.ndim == 2
        if is_vector_field:
            grid = np.zeros(grid_shape + (field_values.shape[1],), dtype=np.float32)
        else:
            grid = np.zeros(grid_shape, dtype=np.float32)

        # Interpolate using SPH kernel
        # For each particle, add its contribution to nearby grid points
        for i in range(N):
            pos = positions[i]
            h = smoothing_lengths[i]
            field = field_values[i]

            # Compute distance from particle to all grid points
            dx = X - pos[0]
            dy = Y - pos[1]
            dz = Z - pos[2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            q = r / h

            # Apply kernel
            W = self.kernel_func(q)

            # Kernel normalization (approximate)
            # C_3D for cubic spline = 1 / (π h³)
            if self.kernel_name == 'cubic_spline':
                norm = 1.0 / (np.pi * h**3)
            elif self.kernel_name == 'wendland_c2':
                norm = 21.0 / (16.0 * np.pi * h**3)
            elif self.kernel_name == 'gaussian':
                norm = 1.0 / (np.pi**1.5 * h**3)
            else:
                norm = 1.0

            W *= norm

            # Add contribution
            if is_vector_field:
                for k in range(field_values.shape[1]):
                    grid[..., k] += W * field[k]
            else:
                grid += W * field

        return grid, (X, Y, Z)

    def interpolate_to_slice(
        self,
        positions: NDArrayFloat,
        field_values: NDArrayFloat,
        smoothing_lengths: NDArrayFloat,
        slice_axis: str = 'z',
        slice_position: float = 0.0,
        grid_shape: Tuple[int, int] = (128, 128),
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    ) -> Tuple[NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat]]:
        """
        Interpolate particle field to 2D slice using SPH kernel.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
            Particle positions.
        field_values : np.ndarray, shape (N,)
            Field values to interpolate (scalar only).
        smoothing_lengths : np.ndarray, shape (N,)
            SPH smoothing lengths.
        slice_axis : str, optional
            Axis perpendicular to slice: 'x', 'y', or 'z' (default: 'z').
        slice_position : float, optional
            Position of slice along slice_axis (default: 0.0).
        grid_shape : tuple of int, optional
            2D grid dimensions (default: (128, 128)).
        bounds : tuple of tuples, optional
            2D grid bounds for in-plane axes.
            If None, computed from particle positions.

        Returns
        -------
        slice_grid : np.ndarray, shape grid_shape
            Interpolated field on 2D slice.
        coordinates : tuple of np.ndarray
            (X, Y) coordinate arrays for slice.

        Examples
        --------
        >>> # XY slice at z=0
        >>> interp = SPHInterpolator()
        >>> slice_grid, (X, Y) = interp.interpolate_to_slice(
        ...     positions, density, h, slice_axis='z', slice_position=0.0
        ... )
        """
        # Determine slice axes
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if slice_axis not in axis_map:
            raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got '{slice_axis}'")

        slice_idx = axis_map[slice_axis]
        in_plane_indices = [i for i in range(3) if i != slice_idx]

        # Filter particles near slice (within 2h)
        distances_to_slice = np.abs(positions[:, slice_idx] - slice_position)
        max_influence = 2.0 * smoothing_lengths  # Kernel support radius
        near_slice = distances_to_slice <= max_influence

        if np.sum(near_slice) == 0:
            warnings.warn(f"No particles found near slice at {slice_axis}={slice_position}")
            return np.zeros(grid_shape, dtype=np.float32), (np.zeros(grid_shape), np.zeros(grid_shape))

        # Positions and fields of particles near slice
        pos_slice = positions[near_slice]
        field_slice = field_values[near_slice]
        h_slice = smoothing_lengths[near_slice]

        # Determine grid bounds
        if bounds is None:
            margin = 2.0 * np.max(h_slice)
            pos_min = np.min(pos_slice[:, in_plane_indices], axis=0) - margin
            pos_max = np.max(pos_slice[:, in_plane_indices], axis=0) + margin
            bounds = (
                (pos_min[0], pos_max[0]),
                (pos_min[1], pos_max[1])
            )

        # Create 2D grid
        x = np.linspace(bounds[0][0], bounds[0][1], grid_shape[0])
        y = np.linspace(bounds[1][0], bounds[1][1], grid_shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Initialize slice grid
        slice_grid = np.zeros(grid_shape, dtype=np.float32)

        # Interpolate
        for i in range(len(pos_slice)):
            pos_2d = pos_slice[i, in_plane_indices]
            h = h_slice[i]
            field = field_slice[i]

            # Distance in 2D
            dx = X - pos_2d[0]
            dy = Y - pos_2d[1]
            r_2d = np.sqrt(dx**2 + dy**2)

            # Include distance perpendicular to slice
            dz = np.abs(slice_position - pos_slice[i, slice_idx])
            r_3d = np.sqrt(r_2d**2 + dz**2)
            q = r_3d / h

            # Apply kernel
            W = self.kernel_func(q)

            # Normalization
            if self.kernel_name == 'cubic_spline':
                norm = 1.0 / (np.pi * h**3)
            elif self.kernel_name == 'wendland_c2':
                norm = 21.0 / (16.0 * np.pi * h**3)
            elif self.kernel_name == 'gaussian':
                norm = 1.0 / (np.pi**1.5 * h**3)
            else:
                norm = 1.0

            W *= norm

            # Add contribution
            slice_grid += W * field

        return slice_grid, (X, Y)


class TemporalInterpolator:
    """
    Temporal interpolation between snapshots for smooth animations.

    Interpolates particle positions, velocities, and fields between
    discrete time snapshots using linear or cubic spline interpolation.
    """

    METHODS = ['linear', 'cubic']

    def __init__(self, method: str = 'linear'):
        """
        Initialize temporal interpolator.

        Parameters
        ----------
        method : str, optional
            Interpolation method: 'linear' or 'cubic' (default: 'linear').
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {self.METHODS}")
        self.method = method

    def interpolate_snapshots(
        self,
        snapshot_before: Dict,
        snapshot_after: Dict,
        t_interp: float
    ) -> Dict:
        """
        Interpolate between two snapshots at intermediate time.

        Parameters
        ----------
        snapshot_before : dict
            Earlier snapshot with keys: 'time', 'positions', 'velocities', etc.
        snapshot_after : dict
            Later snapshot.
        t_interp : float
            Interpolation time (must be between snapshot times).

        Returns
        -------
        snapshot_interp : dict
            Interpolated snapshot at t_interp.

        Notes
        -----
        Assumes particle IDs are consistent between snapshots (same ordering).
        For simulations with particle creation/destruction, use particle tracking.

        Examples
        --------
        >>> t_before = 0.0
        >>> t_after = 1.0
        >>> t_interp = 0.5
        >>> snapshot_mid = interp.interpolate_snapshots(snap0, snap1, t_interp)
        """
        t0 = snapshot_before['time']
        t1 = snapshot_after['time']

        if not (t0 <= t_interp <= t1):
            raise ValueError(f"Interpolation time {t_interp} not in range [{t0}, {t1}]")

        if t1 == t0:
            # Same time, return copy of first snapshot
            return snapshot_before.copy()

        # Linear interpolation weight
        alpha = (t_interp - t0) / (t1 - t0)

        # Interpolate fields
        snapshot_interp = {'time': t_interp}

        for key in ['positions', 'velocities', 'internal_energy', 'density', 'smoothing_length']:
            if key in snapshot_before and key in snapshot_after:
                if self.method == 'linear':
                    snapshot_interp[key] = (
                        (1 - alpha) * snapshot_before[key] + alpha * snapshot_after[key]
                    )
                elif self.method == 'cubic':
                    # Cubic requires 4 points; fallback to linear for 2 points
                    snapshot_interp[key] = (
                        (1 - alpha) * snapshot_before[key] + alpha * snapshot_after[key]
                    )

        # Masses don't change (copy from before)
        if 'masses' in snapshot_before:
            snapshot_interp['masses'] = snapshot_before['masses'].copy()

        # Metadata
        if 'metadata' in snapshot_before:
            snapshot_interp['metadata'] = snapshot_before['metadata'].copy()

        return snapshot_interp

    def generate_interpolated_sequence(
        self,
        snapshots: List[Dict],
        frames_per_interval: int = 10
    ) -> List[Dict]:
        """
        Generate smooth sequence with interpolated frames between snapshots.

        Parameters
        ----------
        snapshots : list of dict
            List of snapshots in time order.
        frames_per_interval : int, optional
            Number of interpolated frames between each pair of snapshots
            (default: 10).

        Returns
        -------
        interpolated_sequence : list of dict
            Smooth sequence with original + interpolated frames.

        Examples
        --------
        >>> # 3 original snapshots → 3 + 2×10 = 23 frames
        >>> smooth_sequence = interp.generate_interpolated_sequence(
        ...     snapshots=[snap0, snap1, snap2], frames_per_interval=10
        ... )
        """
        if len(snapshots) < 2:
            return snapshots

        interpolated_sequence = []

        for i in range(len(snapshots) - 1):
            snap_before = snapshots[i]
            snap_after = snapshots[i + 1]

            t0 = snap_before['time']
            t1 = snap_after['time']

            # Add initial snapshot
            interpolated_sequence.append(snap_before)

            # Add interpolated frames
            for j in range(1, frames_per_interval + 1):
                alpha = j / (frames_per_interval + 1)
                t_interp = t0 + alpha * (t1 - t0)
                snap_interp = self.interpolate_snapshots(snap_before, snap_after, t_interp)
                interpolated_sequence.append(snap_interp)

        # Add final snapshot
        interpolated_sequence.append(snapshots[-1])

        return interpolated_sequence


class SmoothingFilters:
    """
    Smoothing filters for reducing noise in visualizations.

    Provides Gaussian, median, and bilateral filters for post-processing
    gridded data or particle fields.
    """

    @staticmethod
    def gaussian_filter_3d(
        data: NDArrayFloat,
        sigma: float = 1.0
    ) -> NDArrayFloat:
        """
        Apply 3D Gaussian smoothing filter.

        Parameters
        ----------
        data : np.ndarray, shape (nx, ny, nz)
            3D grid data.
        sigma : float, optional
            Standard deviation of Gaussian kernel (in grid units).

        Returns
        -------
        smoothed : np.ndarray, shape (nx, ny, nz)
            Smoothed data.

        Notes
        -----
        Simple implementation using separable convolution.
        For production, consider scipy.ndimage.gaussian_filter.
        """
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(data, sigma=sigma, mode='constant')
        except ImportError:
            warnings.warn("scipy not available, returning unfiltered data")
            return data.copy()

    @staticmethod
    def median_filter_3d(
        data: NDArrayFloat,
        size: int = 3
    ) -> NDArrayFloat:
        """
        Apply 3D median filter (noise reduction).

        Parameters
        ----------
        data : np.ndarray, shape (nx, ny, nz)
            3D grid data.
        size : int, optional
            Filter window size (default: 3).

        Returns
        -------
        filtered : np.ndarray
            Median-filtered data.
        """
        try:
            from scipy.ndimage import median_filter
            return median_filter(data, size=size, mode='constant')
        except ImportError:
            warnings.warn("scipy not available, returning unfiltered data")
            return data.copy()

    @staticmethod
    def bilateral_filter_2d(
        data: NDArrayFloat,
        spatial_sigma: float = 1.0,
        intensity_sigma: float = 0.1
    ) -> NDArrayFloat:
        """
        Apply 2D bilateral filter (edge-preserving smoothing).

        Parameters
        ----------
        data : np.ndarray, shape (nx, ny)
            2D grid data.
        spatial_sigma : float, optional
            Spatial kernel sigma (grid units).
        intensity_sigma : float, optional
            Intensity kernel sigma (data units).

        Returns
        -------
        filtered : np.ndarray
            Bilateral-filtered data.

        Notes
        -----
        Bilateral filter smooths while preserving edges (shocks, interfaces).
        Requires opencv-python or skimage.
        """
        try:
            import cv2
            # cv2 expects uint8 or float32
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
            data_8bit = (data_normalized * 255).astype(np.uint8)
            filtered_8bit = cv2.bilateralFilter(
                data_8bit,
                d=int(2 * spatial_sigma),
                sigmaColor=intensity_sigma * 255,
                sigmaSpace=spatial_sigma
            )
            # Convert back to original scale
            filtered = filtered_8bit.astype(np.float32) / 255.0 * (np.max(data) - np.min(data)) + np.min(data)
            return filtered
        except ImportError:
            try:
                from skimage.restoration import denoise_bilateral
                filtered = denoise_bilateral(
                    data,
                    sigma_spatial=spatial_sigma,
                    sigma_color=intensity_sigma
                )
                return filtered
            except ImportError:
                warnings.warn("opencv-python or scikit-image required for bilateral filter, returning unfiltered")
                return data.copy()


# Convenience functions

def particles_to_grid_3d(
    positions: NDArrayFloat,
    field_values: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    grid_shape: Tuple[int, int, int] = (64, 64, 64),
    kernel: str = 'cubic_spline'
) -> Tuple[NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]]:
    """
    Convenience function: interpolate particles to 3D grid.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Particle positions.
    field_values : np.ndarray, shape (N,)
        Field values.
    smoothing_lengths : np.ndarray, shape (N,)
        Smoothing lengths.
    grid_shape : tuple of int, optional
        Grid dimensions (default: (64, 64, 64)).
    kernel : str, optional
        SPH kernel (default: 'cubic_spline').

    Returns
    -------
    grid : np.ndarray, shape grid_shape
        Interpolated grid.
    coordinates : tuple of np.ndarray
        (X, Y, Z) meshgrid coordinates.

    Examples
    --------
    >>> grid, (X, Y, Z) = particles_to_grid_3d(pos, density, h, grid_shape=(128, 128, 128))
    """
    interp = SPHInterpolator(kernel=kernel)
    return interp.interpolate_to_grid(positions, field_values, smoothing_lengths, grid_shape)


def particles_to_slice_2d(
    positions: NDArrayFloat,
    field_values: NDArrayFloat,
    smoothing_lengths: NDArrayFloat,
    slice_axis: str = 'z',
    slice_position: float = 0.0,
    grid_shape: Tuple[int, int] = (128, 128),
    kernel: str = 'cubic_spline'
) -> Tuple[NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat]]:
    """
    Convenience function: interpolate particles to 2D slice.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Particle positions.
    field_values : np.ndarray, shape (N,)
        Field values.
    smoothing_lengths : np.ndarray, shape (N,)
        Smoothing lengths.
    slice_axis : str, optional
        Slice normal axis: 'x', 'y', 'z' (default: 'z').
    slice_position : float, optional
        Slice position along axis (default: 0.0).
    grid_shape : tuple of int, optional
        2D grid dimensions (default: (128, 128)).
    kernel : str, optional
        SPH kernel (default: 'cubic_spline').

    Returns
    -------
    slice_grid : np.ndarray, shape grid_shape
        Interpolated slice.
    coordinates : tuple of np.ndarray
        (X, Y) meshgrid coordinates.

    Examples
    --------
    >>> slice_grid, (X, Y) = particles_to_slice_2d(pos, density, h, 'z', 0.0)
    """
    interp = SPHInterpolator(kernel=kernel)
    return interp.interpolate_to_slice(
        positions, field_values, smoothing_lengths,
        slice_axis, slice_position, grid_shape
    )


if __name__ == "__main__":
    # Demo: SPH interpolation to grid
    print("=== SPH Interpolation Demo ===")
    print("Generating test particles...")

    # Generate test particles (Gaussian blob)
    N = 1000
    positions = np.random.randn(N, 3) * 0.5
    density = np.exp(-np.sum(positions**2, axis=1))
    smoothing_lengths = np.ones(N) * 0.1

    print(f"  N = {N} particles")
    print(f"  Density range: [{np.min(density):.3f}, {np.max(density):.3f}]")

    # Interpolate to grid
    print("\nInterpolating to 64³ grid...")
    interp = SPHInterpolator(kernel='cubic_spline')
    grid, (X, Y, Z) = interp.interpolate_to_grid(
        positions, density, smoothing_lengths, grid_shape=(64, 64, 64)
    )

    print(f"  Grid shape: {grid.shape}")
    print(f"  Grid range: [{np.min(grid):.3f}, {np.max(grid):.3f}]")

    # Interpolate to slice
    print("\nInterpolating to XY slice at z=0...")
    slice_grid, (X_slice, Y_slice) = interp.interpolate_to_slice(
        positions, density, smoothing_lengths,
        slice_axis='z', slice_position=0.0, grid_shape=(128, 128)
    )

    print(f"  Slice shape: {slice_grid.shape}")
    print(f"  Slice range: [{np.min(slice_grid):.3f}, {np.max(slice_grid):.3f}]")

    print("\nDemo complete!")
