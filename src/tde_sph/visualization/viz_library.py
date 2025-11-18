#!/usr/bin/env python3
"""
Visualization Library (TASK-101)

Pre-configured visualization functions for common TDE-SPH plots using:
- Matplotlib: Publication-quality 2D plots (slices, histograms, time series)
- PyQtGraph: Interactive real-time plots (optional)

Provides high-level API for standard visualizations:
- Density/temperature/velocity slices
- Energy evolution plots
- Particle histograms and phase diagrams
- Radial profiles
- Light curves and fallback rates
- Orbital trajectories

Architecture:
- MatplotlibVisualizer: Static publication plots
- PyQtGraphVisualizer: Interactive real-time plots (optional, requires PyQt)
- Convenience functions for one-line plotting

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# Type aliases
NDArrayFloat = np.ndarray

# Try to import PyQtGraph for real-time plotting
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class MatplotlibVisualizer:
    """
    Matplotlib-based visualization library for TDE-SPH.

    Provides pre-configured functions for common publication-quality plots.
    """

    # Color schemes
    COLORMAPS = {
        'density': 'viridis',
        'temperature': 'hot',
        'velocity': 'coolwarm',
        'energy': 'plasma',
        'pressure': 'magma'
    }

    @staticmethod
    def plot_density_slice(
        slice_data: NDArrayFloat,
        X: NDArrayFloat,
        Y: NDArrayFloat,
        title: str = "Density Slice",
        cmap: str = 'viridis',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        log_scale: bool = True,
        figsize: Tuple[float, float] = (8, 7),
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot 2D density slice with colorbar.

        Parameters
        ----------
        slice_data : np.ndarray, shape (nx, ny)
            2D slice data.
        X, Y : np.ndarray
            Meshgrid coordinates.
        title : str, optional
            Plot title.
        cmap : str, optional
            Colormap name (default: 'viridis').
        vmin, vmax : float, optional
            Color scale limits. If None, auto-scaled.
        log_scale : bool, optional
            Use logarithmic color scale (default: True).
        figsize : tuple, optional
            Figure size in inches (default: (8, 7)).
        save_path : str, optional
            Path to save figure. If None, not saved.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        ax : Axes
            Matplotlib axes.

        Examples
        --------
        >>> from tde_sph.visualization import particles_to_slice_2d, MatplotlibVisualizer
        >>> slice_grid, (X, Y) = particles_to_slice_2d(pos, density, h, 'z', 0.0)
        >>> fig, ax = MatplotlibVisualizer.plot_density_slice(slice_grid, X, Y)
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Apply log scale if requested
        if log_scale:
            plot_data = np.log10(slice_data + 1e-10)
            label = f'log₁₀(Density)'
        else:
            plot_data = slice_data
            label = 'Density'

        # Plot
        im = ax.imshow(
            plot_data.T,
            origin='lower',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='bilinear'
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label=label)

        # Labels
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_energy_evolution(
        times: NDArrayFloat,
        energies: Dict[str, NDArrayFloat],
        title: str = "Energy Evolution",
        figsize: Tuple[float, float] = (10, 8),
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
        Plot energy evolution with conservation error.

        Parameters
        ----------
        times : np.ndarray
            Time array.
        energies : dict
            Dictionary with keys: 'kinetic', 'potential_bh', 'potential_self',
            'internal_thermal', 'total', 'conservation_error'.
        title : str, optional
            Plot title.
        figsize : tuple, optional
            Figure size.
        save_path : str, optional
            Path to save figure.

        Returns
        -------
        fig : Figure
        axes : tuple of Axes
            (ax_energy, ax_error)

        Examples
        --------
        >>> from tde_sph.visualization import MatplotlibVisualizer
        >>> history = energy_diag.get_energy_history_arrays()
        >>> fig, (ax1, ax2) = MatplotlibVisualizer.plot_energy_evolution(
        ...     history['time'], history
        ... )
        >>> plt.show()
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Energy components
        ax1.plot(times, energies['kinetic'], 'r-', label='Kinetic', linewidth=2)
        ax1.plot(times, energies['potential_bh'], 'b-', label='Potential (BH)', linewidth=2)
        ax1.plot(times, energies['potential_self'], 'g-', label='Potential (self)', linewidth=2)
        ax1.plot(times, energies['internal_thermal'], 'm-', label='Internal', linewidth=2)
        ax1.plot(times, energies['total'], 'k--', label='Total', linewidth=2.5)

        ax1.set_ylabel('Energy', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Conservation error
        ax2.plot(times, energies['conservation_error'], 'k-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5)

        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('ΔE / E₀', fontsize=12)
        ax2.set_title('Energy Conservation Error', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, (ax1, ax2)

    @staticmethod
    def plot_phase_diagram(
        x_data: NDArrayFloat,
        y_data: NDArrayFloat,
        weights: Optional[NDArrayFloat] = None,
        x_label: str = 'Density',
        y_label: str = 'Temperature',
        title: str = "Phase Diagram",
        log_x: bool = True,
        log_y: bool = True,
        bins: int = 64,
        cmap: str = 'hot',
        figsize: Tuple[float, float] = (8, 7),
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot 2D phase diagram (e.g., density-temperature).

        Parameters
        ----------
        x_data, y_data : np.ndarray
            Data for x and y axes.
        weights : np.ndarray, optional
            Weights for histogram (e.g., particle masses).
        x_label, y_label : str
            Axis labels.
        title : str
            Plot title.
        log_x, log_y : bool
            Use logarithmic scales.
        bins : int
            Number of bins for 2D histogram.
        cmap : str
            Colormap.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Save path.

        Returns
        -------
        fig, ax : Figure, Axes

        Examples
        --------
        >>> # Density-temperature phase diagram
        >>> fig, ax = MatplotlibVisualizer.plot_phase_diagram(
        ...     density, temperature, weights=masses,
        ...     x_label='Density [code units]',
        ...     y_label='Temperature [code units]'
        ... )
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create 2D histogram
        if log_x:
            x_data = np.log10(x_data + 1e-10)
            x_label = f'log₁₀({x_label})'
        if log_y:
            y_data = np.log10(y_data + 1e-10)
            y_label = f'log₁₀({y_label})'

        H, xedges, yedges = np.histogram2d(
            x_data, y_data,
            bins=bins,
            weights=weights
        )

        # Plot
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            H.T,
            origin='lower',
            extent=extent,
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear',
            norm=mpl.colors.LogNorm() if weights is not None else None
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        if weights is not None:
            cbar.set_label('Mass', fontsize=10)
        else:
            cbar.set_label('Count', fontsize=10)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_radial_profile(
        radii: NDArrayFloat,
        quantity: NDArrayFloat,
        quantity_label: str = 'Density',
        title: str = "Radial Profile",
        log_x: bool = True,
        log_y: bool = True,
        bins: int = 50,
        method: str = 'mean',
        figsize: Tuple[float, float] = (8, 6),
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot radial profile (binned average vs radius).

        Parameters
        ----------
        radii : np.ndarray
            Particle radii.
        quantity : np.ndarray
            Quantity to bin (density, temperature, etc.).
        quantity_label : str
            Y-axis label.
        title : str
            Plot title.
        log_x, log_y : bool
            Logarithmic scales.
        bins : int
            Number of radial bins.
        method : str
            Binning method: 'mean' or 'median'.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Save path.

        Returns
        -------
        fig, ax : Figure, Axes

        Examples
        --------
        >>> # Density profile
        >>> r = np.linalg.norm(positions, axis=1)
        >>> fig, ax = MatplotlibVisualizer.plot_radial_profile(
        ...     r, density, quantity_label='Density'
        ... )
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create radial bins
        r_min = np.min(radii[radii > 0]) if np.any(radii > 0) else 1e-10
        r_max = np.max(radii)

        if log_x:
            r_bins = np.logspace(np.log10(r_min), np.log10(r_max), bins + 1)
        else:
            r_bins = np.linspace(r_min, r_max, bins + 1)

        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

        # Bin data
        bin_indices = np.digitize(radii, r_bins) - 1
        binned_quantity = []

        for i in range(bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                if method == 'mean':
                    binned_quantity.append(np.mean(quantity[mask]))
                elif method == 'median':
                    binned_quantity.append(np.median(quantity[mask]))
                else:
                    raise ValueError(f"Unknown method: {method}")
            else:
                binned_quantity.append(np.nan)

        binned_quantity = np.array(binned_quantity)

        # Plot
        ax.plot(r_centers, binned_quantity, 'b-', linewidth=2, marker='o', markersize=4)

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        ax.set_xlabel('Radius', fontsize=12)
        ax.set_ylabel(quantity_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_histogram(
        data: NDArrayFloat,
        label: str = 'Quantity',
        title: str = "Histogram",
        bins: int = 50,
        log_x: bool = False,
        log_y: bool = True,
        weights: Optional[NDArrayFloat] = None,
        figsize: Tuple[float, float] = (8, 6),
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot histogram of particle quantity.

        Parameters
        ----------
        data : np.ndarray
            Data to histogram.
        label : str
            X-axis label.
        title : str
            Plot title.
        bins : int
            Number of bins.
        log_x, log_y : bool
            Logarithmic scales.
        weights : np.ndarray, optional
            Weights (e.g., masses).
        figsize : tuple
            Figure size.
        save_path : str, optional
            Save path.

        Returns
        -------
        fig, ax : Figure, Axes
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Filter out NaN and inf
        valid = np.isfinite(data)
        data_clean = data[valid]
        weights_clean = weights[valid] if weights is not None else None

        if log_x:
            data_clean = np.log10(data_clean + 1e-10)
            label = f'log₁₀({label})'

        # Histogram
        counts, bin_edges, patches = ax.hist(
            data_clean,
            bins=bins,
            weights=weights_clean,
            color='steelblue',
            edgecolor='black',
            alpha=0.7
        )

        if log_y:
            ax.set_yscale('log')

        ax.set_xlabel(label, fontsize=12)
        ylabel = 'Mass' if weights is not None else 'Count'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_trajectory(
        positions: NDArrayFloat,
        title: str = "Orbital Trajectory",
        color: Optional[Union[str, NDArrayFloat]] = None,
        colorbar_label: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 8),
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot 2D orbital trajectory.

        Parameters
        ----------
        positions : np.ndarray, shape (N_times, 3) or (N_times, 2)
            Positions along trajectory.
        title : str
            Plot title.
        color : str or np.ndarray, optional
            Color specification. If array, color by quantity (e.g., time).
        colorbar_label : str, optional
            Colorbar label.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Save path.

        Returns
        -------
        fig, ax : Figure, Axes

        Examples
        --------
        >>> # Trajectory colored by time
        >>> positions = ...  # (N, 3) array
        >>> times = np.arange(len(positions))
        >>> fig, ax = MatplotlibVisualizer.plot_trajectory(
        ...     positions[:, :2], color=times, colorbar_label='Time'
        ... )
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Extract XY positions
        if positions.shape[1] == 3:
            x, y = positions[:, 0], positions[:, 1]
        else:
            x, y = positions[:, 0], positions[:, 1]

        # Plot
        if isinstance(color, np.ndarray):
            scatter = ax.scatter(x, y, c=color, cmap='viridis', s=10, alpha=0.7)
            if colorbar_label:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(colorbar_label, fontsize=10)
        else:
            ax.plot(x, y, color=color or 'blue', linewidth=2, alpha=0.7)
            ax.scatter([x[0]], [y[0]], color='green', s=100, marker='o', label='Start', zorder=10)
            ax.scatter([x[-1]], [y[-1]], color='red', s=100, marker='x', label='End', zorder=10)
            ax.legend()

        # Black hole at origin
        ax.scatter([0], [0], color='black', s=200, marker='*', label='BH', zorder=10)

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax


# Convenience functions

def quick_density_slice(
    slice_data: NDArrayFloat,
    X: NDArrayFloat,
    Y: NDArrayFloat,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Quick density slice plot (one-liner).

    Examples
    --------
    >>> from tde_sph.visualization import particles_to_slice_2d, quick_density_slice
    >>> slice_grid, (X, Y) = particles_to_slice_2d(pos, density, h, 'z', 0.0)
    >>> fig, ax = quick_density_slice(slice_grid, X, Y, 'density_slice.png')
    >>> plt.show()
    """
    return MatplotlibVisualizer.plot_density_slice(
        slice_data, X, Y, cmap='viridis', log_scale=True, save_path=save_path
    )


def quick_energy_plot(
    energy_history: Dict[str, NDArrayFloat],
    save_path: Optional[str] = None
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Quick energy evolution plot (one-liner).

    Examples
    --------
    >>> from tde_sph.visualization import quick_energy_plot
    >>> history = energy_diag.get_energy_history_arrays()
    >>> fig, axes = quick_energy_plot(history, 'energy_evolution.png')
    >>> plt.show()
    """
    times = energy_history['time']
    return MatplotlibVisualizer.plot_energy_evolution(times, energy_history, save_path=save_path)


def quick_phase_diagram(
    density: NDArrayFloat,
    temperature: NDArrayFloat,
    masses: Optional[NDArrayFloat] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Quick density-temperature phase diagram (one-liner).

    Examples
    --------
    >>> from tde_sph.visualization import quick_phase_diagram
    >>> fig, ax = quick_phase_diagram(density, temperature, masses, 'phase_diagram.png')
    >>> plt.show()
    """
    return MatplotlibVisualizer.plot_phase_diagram(
        density, temperature, weights=masses,
        x_label='Density', y_label='Temperature',
        save_path=save_path
    )


def quick_radial_profile(
    positions: NDArrayFloat,
    quantity: NDArrayFloat,
    quantity_label: str = 'Density',
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Quick radial profile plot (one-liner).

    Examples
    --------
    >>> from tde_sph.visualization import quick_radial_profile
    >>> fig, ax = quick_radial_profile(positions, density, 'Density', 'radial_profile.png')
    >>> plt.show()
    """
    radii = np.linalg.norm(positions, axis=1)
    return MatplotlibVisualizer.plot_radial_profile(
        radii, quantity, quantity_label=quantity_label, save_path=save_path
    )


if __name__ == "__main__":
    # Demo: create sample plots
    print("=== Visualization Library Demo ===")
    print("Generating sample data...")

    # Sample data
    np.random.seed(42)
    N = 1000

    # Density slice
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    density_slice = np.exp(-R**2)

    print("\n1. Density slice plot...")
    fig1, ax1 = quick_density_slice(density_slice, X, Y)
    print("   ✓ Created")

    # Energy evolution
    times = np.linspace(0, 10, 100)
    energies = {
        'time': times,
        'kinetic': np.ones(100) * 1.0 + np.random.randn(100) * 0.01,
        'potential_bh': np.ones(100) * -2.0 + np.random.randn(100) * 0.02,
        'potential_self': np.ones(100) * -0.5 + np.random.randn(100) * 0.01,
        'internal_thermal': np.ones(100) * 0.5 + np.random.randn(100) * 0.01,
        'total': np.ones(100) * -1.0 + np.random.randn(100) * 0.005,
        'conservation_error': np.random.randn(100) * 0.001
    }

    print("2. Energy evolution plot...")
    fig2, axes2 = quick_energy_plot(energies)
    print("   ✓ Created")

    # Phase diagram
    density = np.random.lognormal(0, 1, N)
    temperature = np.random.lognormal(0, 0.5, N)
    masses = np.ones(N) * 0.001

    print("3. Phase diagram...")
    fig3, ax3 = quick_phase_diagram(density, temperature, masses)
    print("   ✓ Created")

    print("\nDemo complete! Close plots to exit.")
    plt.show()
