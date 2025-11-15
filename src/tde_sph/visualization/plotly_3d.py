"""
Plotly-based 3D visualization for TDE-SPH simulations.

This module provides interactive 3D visualization of particle data using Plotly,
implementing the Visualizer interface. Supports CON-004 (visualization requirements).

Features:
- Interactive 3D scatter plots with rotation, zoom, pan
- Flexible color mapping (density, temperature, velocity, etc.)
- Logarithmic color scales for wide dynamic ranges
- Automatic downsampling for large particle counts (>10⁵)
- Animation from snapshot sequences
- Export to HTML and static images (PNG, PDF via kaleido)

Example usage:
    >>> viz = Plotly3DVisualizer()
    >>> fig = viz.plot_particles(
    ...     positions,
    ...     quantities={'density': density, 'temperature': temperature},
    ...     color_by='density',
    ...     log_scale=True
    ... )
    >>> fig.show()
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, Any, List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

from tde_sph.core.interfaces import Visualizer

# Type alias matching core interfaces
NDArrayFloat = npt.NDArray[np.float32]


class Plotly3DVisualizer(Visualizer):
    """
    Plotly-based 3D particle visualizer implementing the Visualizer interface.

    Provides interactive 3D scatter plots with flexible color mapping,
    automatic downsampling for performance, and animation capabilities.

    Attributes
    ----------
    max_particles_plot : int
        Maximum particles to plot without downsampling (default: 100,000)
    default_marker_size : float
        Default marker size in pixels (default: 2)
    colorscale : str
        Default Plotly colorscale (default: 'Viridis')
    """

    def __init__(
        self,
        max_particles_plot: int = 100_000,
        default_marker_size: float = 2.0,
        colorscale: str = 'Viridis'
    ):
        """
        Initialize Plotly3DVisualizer.

        Parameters
        ----------
        max_particles_plot : int, optional
            Maximum particles to display before downsampling. Default: 100,000.
            Set higher for powerful systems, lower for smoother interaction.
        default_marker_size : float, optional
            Default marker size in pixels. Default: 2.0.
        colorscale : str, optional
            Default Plotly colorscale name. Options: 'Viridis', 'Plasma',
            'Inferno', 'Magma', 'Jet', 'Hot', 'Cool', 'Rainbow', etc.
            Default: 'Viridis'.
        """
        self.max_particles_plot = max_particles_plot
        self.default_marker_size = default_marker_size
        self.colorscale = colorscale

    def plot_particles(
        self,
        positions: NDArrayFloat,
        quantities: Optional[Dict[str, NDArrayFloat]] = None,
        color_by: str = 'density',
        log_scale: bool = True,
        marker_size: Optional[Union[float, NDArrayFloat]] = None,
        title: str = "TDE-SPH Particle Distribution",
        downsample: bool = True,
        colorscale: Optional[str] = None,
        show_colorbar: bool = True,
        axis_labels: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create 3D visualization of particle distribution.

        Parameters
        ----------
        positions : NDArrayFloat, shape (N, 3)
            Particle positions (x, y, z).
        quantities : Optional[Dict[str, NDArrayFloat]]
            Dictionary of quantities to visualize. Common keys:
            - 'density': mass density
            - 'temperature': temperature
            - 'velocity_magnitude': |v|
            - 'internal_energy': specific internal energy
            - 'pressure': gas pressure
        color_by : str, optional
            Quantity name to use for coloring particles. Must be a key in
            'quantities' dict. Default: 'density'.
        log_scale : bool, optional
            Use logarithmic color scale. Default: True.
            Recommended for quantities with wide dynamic range (density, etc.).
        marker_size : float or NDArrayFloat, optional
            Marker size. Can be:
            - float: uniform size for all particles
            - array: size per particle (e.g., proportional to mass)
            Default: uses self.default_marker_size.
        title : str, optional
            Plot title. Default: "TDE-SPH Particle Distribution".
        downsample : bool, optional
            Automatically downsample if N > max_particles_plot. Default: True.
        colorscale : str, optional
            Plotly colorscale. If None, uses self.colorscale.
        show_colorbar : bool, optional
            Show colorbar legend. Default: True.
        axis_labels : Dict[str, str], optional
            Custom axis labels. Keys: 'x', 'y', 'z'.
            Default: {'x': 'x', 'y': 'y', 'z': 'z'}.
        **kwargs
            Additional arguments passed to go.Scatter3d().

        Returns
        -------
        fig : go.Figure
            Plotly Figure object. Use fig.show() to display interactively,
            fig.write_html() to save, or fig.write_image() for static export.

        Examples
        --------
        >>> # Basic density visualization
        >>> fig = viz.plot_particles(positions, quantities={'density': rho})
        >>> fig.show()

        >>> # Temperature with custom colorscale
        >>> fig = viz.plot_particles(
        ...     positions,
        ...     quantities={'temperature': T},
        ...     color_by='temperature',
        ...     colorscale='Hot',
        ...     log_scale=False
        ... )
        >>> fig.write_html("temperature_3d.html")

        >>> # Size particles by mass
        >>> sizes = masses / masses.max() * 10  # Scale to 0-10 pixels
        >>> fig = viz.plot_particles(positions, marker_size=sizes)
        """
        n_particles = len(positions)

        # Handle downsampling for large datasets
        if downsample and n_particles > self.max_particles_plot:
            # Random downsampling
            indices = np.random.choice(
                n_particles,
                self.max_particles_plot,
                replace=False
            )
            positions = positions[indices]

            # Also downsample quantities
            if quantities is not None:
                quantities = {
                    key: val[indices] for key, val in quantities.items()
                }

            warnings.warn(
                f"Downsampled from {n_particles:,} to {self.max_particles_plot:,} "
                f"particles for visualization. Set downsample=False or increase "
                f"max_particles_plot to override.",
                UserWarning
            )
            n_particles = self.max_particles_plot

        # Extract color data
        if quantities is not None and color_by in quantities:
            color_data = quantities[color_by]

            # Apply log scale if requested
            if log_scale:
                # Avoid log(0) issues
                color_data = np.where(
                    color_data > 0,
                    np.log10(color_data),
                    np.log10(np.abs(color_data).min() + 1e-30)
                )
                colorbar_title = f"log₁₀({color_by})"
            else:
                colorbar_title = color_by
        else:
            # No color data, use uniform color
            color_data = np.ones(n_particles)
            colorbar_title = None
            show_colorbar = False

        # Determine marker size
        if marker_size is None:
            marker_size = self.default_marker_size

        # Select colorscale
        if colorscale is None:
            colorscale = self.colorscale

        # Default axis labels
        if axis_labels is None:
            axis_labels = {'x': 'x', 'y': 'y', 'z': 'z'}

        # Create scatter plot
        scatter = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=color_data,
                colorscale=colorscale,
                colorbar=dict(
                    title=colorbar_title,
                    thickness=20,
                    len=0.7
                ) if show_colorbar else None,
                showscale=show_colorbar,
                line=dict(width=0)  # No marker outlines for performance
            ),
            text=self._create_hover_text(positions, quantities),
            hovertemplate=(
                "<b>Position</b><br>"
                "x: %{x:.3e}<br>"
                "y: %{y:.3e}<br>"
                "z: %{z:.3e}<br>"
                "%{text}"
                "<extra></extra>"
            ),
            **kwargs
        )

        # Create figure
        fig = go.Figure(data=[scatter])

        # Update layout for better 3D viewing
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title=axis_labels.get('x', 'x')),
                yaxis=dict(title=axis_labels.get('y', 'y')),
                zaxis=dict(title=axis_labels.get('z', 'z')),
                aspectmode='data',  # Equal aspect ratio
            ),
            showlegend=False,
            hovermode='closest',
            width=900,
            height=700,
        )

        return fig

    def animate(
        self,
        snapshots: List[Dict[str, Any]],
        color_by: str = 'density',
        log_scale: bool = True,
        title: str = "TDE-SPH Evolution",
        frame_duration: int = 100,
        transition_duration: int = 50,
        **kwargs
    ) -> go.Figure:
        """
        Create animation from sequence of snapshots.

        Parameters
        ----------
        snapshots : list of dict
            List of snapshot dictionaries, each containing:
            - 'positions': NDArrayFloat, shape (N, 3)
            - 'quantities': Dict[str, NDArrayFloat] (optional)
            - 'time': float (optional, for title)
        color_by : str, optional
            Quantity to color by. Default: 'density'.
        log_scale : bool, optional
            Use logarithmic color scale. Default: True.
        title : str, optional
            Base title for animation. Default: "TDE-SPH Evolution".
        frame_duration : int, optional
            Duration of each frame in milliseconds. Default: 100.
        transition_duration : int, optional
            Transition duration between frames in ms. Default: 50.
        **kwargs
            Additional arguments passed to plot_particles().

        Returns
        -------
        fig : go.Figure
            Animated Plotly Figure with time slider and play/pause controls.

        Notes
        -----
        For large numbers of snapshots or particles, consider:
        - Reducing frame count (skip frames)
        - Reducing particles per snapshot (downsampling)
        - Saving to HTML for web-based viewing

        Examples
        --------
        >>> snapshots = [
        ...     {'positions': pos0, 'quantities': {'density': rho0}, 'time': 0.0},
        ...     {'positions': pos1, 'quantities': {'density': rho1}, 'time': 1.0},
        ... ]
        >>> fig = viz.animate(snapshots, frame_duration=200)
        >>> fig.write_html("evolution.html")
        """
        if not snapshots:
            raise ValueError("snapshots list is empty")

        # Get global color scale limits across all frames
        color_min, color_max = self._get_global_color_limits(
            snapshots, color_by, log_scale
        )

        # Create frames
        frames = []
        for i, snapshot in enumerate(snapshots):
            positions = snapshot['positions']
            quantities = snapshot.get('quantities', None)
            time = snapshot.get('time', i)

            # Downsample if needed
            n_particles = len(positions)
            if n_particles > self.max_particles_plot:
                indices = np.random.choice(
                    n_particles, self.max_particles_plot, replace=False
                )
                positions = positions[indices]
                if quantities is not None:
                    quantities = {k: v[indices] for k, v in quantities.items()}

            # Get color data
            if quantities is not None and color_by in quantities:
                color_data = quantities[color_by]
                if log_scale:
                    color_data = np.where(
                        color_data > 0,
                        np.log10(color_data),
                        np.log10(np.abs(color_data).min() + 1e-30)
                    )
            else:
                color_data = np.ones(len(positions))

            # Create frame
            frame = go.Frame(
                data=[go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='markers',
                    marker=dict(
                        size=self.default_marker_size,
                        color=color_data,
                        colorscale=self.colorscale,
                        cmin=color_min,
                        cmax=color_max,
                        colorbar=dict(
                            title=f"log₁₀({color_by})" if log_scale else color_by,
                            thickness=20,
                            len=0.7
                        ),
                        showscale=True,
                        line=dict(width=0)
                    ),
                    text=self._create_hover_text(positions, quantities),
                    hovertemplate=(
                        "<b>Position</b><br>"
                        "x: %{x:.3e}<br>"
                        "y: %{y:.3e}<br>"
                        "z: %{z:.3e}<br>"
                        "%{text}"
                        "<extra></extra>"
                    ),
                )],
                name=f"frame_{i}",
                layout=go.Layout(
                    title=f"{title} (t = {time:.3f})"
                )
            )
            frames.append(frame)

        # Create initial frame
        initial_snapshot = snapshots[0]
        fig = self.plot_particles(
            initial_snapshot['positions'],
            quantities=initial_snapshot.get('quantities', None),
            color_by=color_by,
            log_scale=log_scale,
            title=f"{title} (t = {initial_snapshot.get('time', 0.0):.3f})",
            **kwargs
        )

        # Add frames to figure
        fig.frames = frames

        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [
                            None,
                            {
                                'frame': {'duration': frame_duration, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': transition_duration}
                            }
                        ]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [
                            [None],
                            {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }
                        ]
                    }
                ],
                'x': 0.1,
                'y': 0.0,
                'xanchor': 'right',
                'yanchor': 'top'
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [
                            [f"frame_{i}"],
                            {
                                'frame': {'duration': frame_duration, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': transition_duration}
                            }
                        ],
                        'label': f"{snapshots[i].get('time', i):.2f}",
                        'method': 'animate'
                    }
                    for i in range(len(snapshots))
                ],
                'active': 0,
                'y': 0.0,
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'yanchor': 'top',
                'transition': {'duration': transition_duration},
            }]
        )

        return fig

    def _create_hover_text(
        self,
        positions: NDArrayFloat,
        quantities: Optional[Dict[str, NDArrayFloat]]
    ) -> List[str]:
        """
        Create hover text showing particle properties.

        Parameters
        ----------
        positions : NDArrayFloat
            Particle positions.
        quantities : Optional[Dict[str, NDArrayFloat]]
            Additional quantities to display.

        Returns
        -------
        hover_text : List[str]
            Hover text for each particle.
        """
        n_particles = len(positions)
        hover_text = []

        for i in range(n_particles):
            text = ""
            if quantities is not None:
                for key, values in quantities.items():
                    text += f"<br>{key}: {values[i]:.3e}"
            hover_text.append(text)

        return hover_text

    def _get_global_color_limits(
        self,
        snapshots: List[Dict[str, Any]],
        color_by: str,
        log_scale: bool
    ) -> tuple:
        """
        Get global min/max for color scale across all snapshots.

        Ensures consistent coloring in animations.

        Parameters
        ----------
        snapshots : list
            List of snapshot dictionaries.
        color_by : str
            Quantity name to extract limits for.
        log_scale : bool
            Whether to compute limits in log space.

        Returns
        -------
        vmin, vmax : float
            Global minimum and maximum values.
        """
        all_values = []

        for snapshot in snapshots:
            quantities = snapshot.get('quantities', {})
            if color_by in quantities:
                values = quantities[color_by]
                if log_scale:
                    # Filter out zeros/negatives before log
                    values = values[values > 0]
                    if len(values) > 0:
                        all_values.append(np.log10(values))
                else:
                    all_values.append(values)

        if all_values:
            all_values = np.concatenate(all_values)
            return float(all_values.min()), float(all_values.max())
        else:
            return 0.0, 1.0


def quick_plot(
    positions: NDArrayFloat,
    color_by: Optional[NDArrayFloat] = None,
    log_scale: bool = True,
    **kwargs
) -> go.Figure:
    """
    Convenience function for quick 3D particle visualization.

    Parameters
    ----------
    positions : NDArrayFloat, shape (N, 3)
        Particle positions.
    color_by : NDArrayFloat, shape (N,), optional
        Values to color particles by. If None, uniform coloring.
    log_scale : bool, optional
        Use logarithmic color scale. Default: True.
    **kwargs
        Additional arguments passed to Plotly3DVisualizer.plot_particles().

    Returns
    -------
    fig : go.Figure
        Plotly Figure object.

    Examples
    --------
    >>> fig = quick_plot(positions, color_by=density)
    >>> fig.show()
    """
    viz = Plotly3DVisualizer()

    quantities = None
    color_key = 'value'
    if color_by is not None:
        quantities = {color_key: color_by}

    return viz.plot_particles(
        positions,
        quantities=quantities,
        color_by=color_key if color_by is not None else None,
        log_scale=log_scale,
        **kwargs
    )
