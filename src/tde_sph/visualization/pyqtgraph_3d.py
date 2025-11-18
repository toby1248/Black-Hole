#!/usr/bin/env python3
"""
PyQtGraph 3D Visualizer with Time-Scrubbing (TASK-037)

Hardware-accelerated OpenGL visualization for TDE-SPH simulations with:
- Interactive 3D particle rendering
- Time-scrubbing animation controls
- Colormaps for physical quantities
- Camera controls (rotate, zoom, pan)
- Snapshot sequence playback

Provides significantly better performance and interactivity than Plotly for
large particle counts, leveraging GPU acceleration via OpenGL.

Architecture:
- PyQtGraph GLViewWidget for OpenGL rendering
- GLScatterPlotItem for particle display
- QTimer for animation playback
- Custom UI controls for time-scrubbing

Dependencies:
- PyQt6 or PyQt5
- pyqtgraph >= 0.13.0
- OpenGL-capable GPU

References:
- PyQtGraph documentation: https://pyqtgraph.readthedocs.io/
- Qt OpenGL: https://doc.qt.io/qt-6/qtopengl-index.html

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import h5py

try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from PyQt6.QtCore import Qt, QTimer
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5 import QtWidgets, QtCore, QtGui
        from PyQt5.QtCore import Qt, QTimer
        PYQT_VERSION = 5
    except ImportError:
        raise ImportError("PyQt6 or PyQt5 required for pyqtgraph_3d visualizer")

import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Type aliases
NDArrayFloat = np.ndarray


class SnapshotLoader:
    """
    Load HDF5 snapshot sequences for visualization.

    Handles reading particle data from TDE-SPH HDF5 snapshots and provides
    efficient caching for time-scrubbing playback.
    """

    def __init__(self, snapshot_files: List[str], cache_size: int = 10):
        """
        Initialize snapshot loader.

        Parameters
        ----------
        snapshot_files : list of str
            Paths to HDF5 snapshot files, in time order.
        cache_size : int, optional
            Number of snapshots to keep in memory cache (default: 10).
        """
        self.snapshot_files = [Path(f) for f in snapshot_files]
        self.cache_size = cache_size
        self._cache: Dict[int, Dict] = {}
        self._cache_order: List[int] = []

        if not self.snapshot_files:
            raise ValueError("No snapshot files provided")

        # Validate all files exist
        for f in self.snapshot_files:
            if not f.exists():
                raise FileNotFoundError(f"Snapshot file not found: {f}")

    def __len__(self) -> int:
        """Number of snapshots."""
        return len(self.snapshot_files)

    def load_snapshot(self, index: int) -> Dict:
        """
        Load snapshot by index.

        Parameters
        ----------
        index : int
            Snapshot index (0 to len(self) - 1).

        Returns
        -------
        snapshot : dict
            Dictionary with keys:
            - 'time': simulation time
            - 'positions': (N, 3) array of particle positions
            - 'velocities': (N, 3) array of particle velocities
            - 'masses': (N,) array of particle masses
            - 'density': (N,) array of densities
            - 'internal_energy': (N,) array of internal energies
            - 'smoothing_length': (N,) array of smoothing lengths
            - 'metadata': dict of simulation metadata
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Snapshot index {index} out of range [0, {len(self) - 1}]")

        # Check cache
        if index in self._cache:
            return self._cache[index]

        # Load from file
        snapshot_path = self.snapshot_files[index]
        with h5py.File(snapshot_path, 'r') as f:
            snapshot = {
                'time': f['time'][()] if 'time' in f else 0.0,
                'positions': f['particles/positions'][:],
                'velocities': f['particles/velocities'][:],
                'masses': f['particles/masses'][:],
                'density': f['particles/density'][:],
                'internal_energy': f['particles/internal_energy'][:],
                'smoothing_length': f['particles/smoothing_length'][:],
                'metadata': dict(f.attrs) if hasattr(f, 'attrs') else {}
            }

        # Add to cache
        self._add_to_cache(index, snapshot)

        return snapshot

    def _add_to_cache(self, index: int, snapshot: Dict):
        """Add snapshot to cache, evicting oldest if cache is full."""
        if index in self._cache:
            # Move to end (most recent)
            self._cache_order.remove(index)
            self._cache_order.append(index)
            return

        # Evict oldest if cache is full
        if len(self._cache) >= self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]

        self._cache[index] = snapshot
        self._cache_order.append(index)

    def get_time_array(self) -> NDArrayFloat:
        """
        Get array of simulation times for all snapshots.

        Returns
        -------
        times : np.ndarray, shape (N_snapshots,)
            Simulation time for each snapshot.

        Notes
        -----
        This loads time metadata from all snapshots, but not full particle data.
        """
        times = []
        for snapshot_file in self.snapshot_files:
            with h5py.File(snapshot_file, 'r') as f:
                time = f['time'][()] if 'time' in f else 0.0
                times.append(time)
        return np.array(times)


class ParticleVisualizer3D(QtWidgets.QMainWindow):
    """
    PyQtGraph 3D particle visualizer with time-scrubbing controls.

    Provides interactive OpenGL visualization of SPH particle snapshots with
    animation playback, colormaps, and camera controls.
    """

    # Colormap definitions (field_name -> colormap_name)
    COLORMAPS = {
        'density': 'viridis',
        'internal_energy': 'plasma',
        'temperature': 'hot',
        'velocity_magnitude': 'cool',
        'masses': 'cividis',
        'smoothing_length': 'twilight'
    }

    def __init__(
        self,
        snapshot_files: List[str],
        color_by: str = 'density',
        point_size: float = 2.0,
        background: str = 'black',
        title: str = "TDE-SPH 3D Visualizer"
    ):
        """
        Initialize 3D visualizer.

        Parameters
        ----------
        snapshot_files : list of str
            Paths to HDF5 snapshot files in time order.
        color_by : str, optional
            Physical quantity to color particles by:
            'density', 'internal_energy', 'temperature',
            'velocity_magnitude', 'masses', 'smoothing_length'.
        point_size : float, optional
            Particle marker size in pixels (default: 2.0).
        background : str, optional
            Background color: 'black' or 'white' (default: 'black').
        title : str, optional
            Window title.
        """
        super().__init__()

        self.snapshot_loader = SnapshotLoader(snapshot_files)
        self.color_by = color_by
        self.point_size = point_size
        self.background = background

        # Animation state
        self.current_frame = 0
        self.is_playing = False
        self.fps = 10  # Frames per second
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._advance_frame)

        # Cached data
        self.current_snapshot: Optional[Dict] = None
        self.scatter_plot: Optional[gl.GLScatterPlotItem] = None

        # Build UI
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1200, 800)
        self._build_ui()

        # Load initial snapshot
        self._load_frame(0)

    def _build_ui(self):
        """Build user interface."""
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 3D view widget (OpenGL)
        self.view_widget = gl.GLViewWidget()
        if self.background == 'white':
            self.view_widget.setBackgroundColor('w')
        else:
            self.view_widget.setBackgroundColor('k')

        # Add coordinate axes
        axis = gl.GLAxisItem()
        axis.setSize(x=1, y=1, z=1)
        self.view_widget.addItem(axis)

        # Add grid
        grid = gl.GLGridItem()
        grid.scale(0.1, 0.1, 0.1)
        self.view_widget.addItem(grid)

        main_layout.addWidget(self.view_widget, stretch=10)

        # Info label
        self.info_label = QtWidgets.QLabel("Frame: 0 / 0 | Time: 0.0")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        font = self.info_label.font()
        font.setPointSize(12)
        self.info_label.setFont(font)
        main_layout.addWidget(self.info_label)

        # Timeline slider
        self.timeline_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(len(self.snapshot_loader) - 1)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow if PYQT_VERSION == 6 else QtWidgets.QSlider.TicksBelow)
        self.timeline_slider.setTickInterval(max(1, len(self.snapshot_loader) // 20))
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        main_layout.addWidget(self.timeline_slider)

        # Control buttons
        control_layout = QtWidgets.QHBoxLayout()

        # Play/Pause button
        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.play_button.clicked.connect(self._toggle_play)
        control_layout.addWidget(self.play_button)

        # Step backward
        step_back_button = QtWidgets.QPushButton("◀ Step Back")
        step_back_button.clicked.connect(self._step_backward)
        control_layout.addWidget(step_back_button)

        # Step forward
        step_forward_button = QtWidgets.QPushButton("Step Forward ▶")
        step_forward_button.clicked.connect(self._step_forward)
        control_layout.addWidget(step_forward_button)

        # Reset button
        reset_button = QtWidgets.QPushButton("⟲ Reset")
        reset_button.clicked.connect(self._reset_view)
        control_layout.addWidget(reset_button)

        # FPS control
        control_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_spinbox = QtWidgets.QSpinBox()
        self.fps_spinbox.setMinimum(1)
        self.fps_spinbox.setMaximum(60)
        self.fps_spinbox.setValue(self.fps)
        self.fps_spinbox.valueChanged.connect(self._on_fps_changed)
        control_layout.addWidget(self.fps_spinbox)

        # Color by dropdown
        control_layout.addWidget(QtWidgets.QLabel("Color by:"))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(list(self.COLORMAPS.keys()))
        self.color_combo.setCurrentText(self.color_by)
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        control_layout.addWidget(self.color_combo)

        # Point size control
        control_layout.addWidget(QtWidgets.QLabel("Point size:"))
        self.size_spinbox = QtWidgets.QDoubleSpinBox()
        self.size_spinbox.setMinimum(0.1)
        self.size_spinbox.setMaximum(20.0)
        self.size_spinbox.setSingleStep(0.5)
        self.size_spinbox.setValue(self.point_size)
        self.size_spinbox.valueChanged.connect(self._on_size_changed)
        control_layout.addWidget(self.size_spinbox)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

    def _load_frame(self, frame_index: int):
        """Load and display a specific frame."""
        if frame_index < 0 or frame_index >= len(self.snapshot_loader):
            return

        self.current_frame = frame_index
        self.current_snapshot = self.snapshot_loader.load_snapshot(frame_index)

        # Update UI
        self._update_display()
        self._update_info_label()

        # Update slider without triggering callback
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_index)
        self.timeline_slider.blockSignals(False)

    def _update_display(self):
        """Update 3D display with current snapshot."""
        if self.current_snapshot is None:
            return

        positions = self.current_snapshot['positions']

        # Compute color values
        colors = self._compute_colors()

        # Remove old scatter plot
        if self.scatter_plot is not None:
            self.view_widget.removeItem(self.scatter_plot)

        # Create new scatter plot
        self.scatter_plot = gl.GLScatterPlotItem(
            pos=positions,
            color=colors,
            size=self.point_size,
            pxMode=True  # Size in screen pixels
        )
        self.view_widget.addItem(self.scatter_plot)

    def _compute_colors(self) -> NDArrayFloat:
        """
        Compute RGBA colors for particles based on selected field.

        Returns
        -------
        colors : np.ndarray, shape (N, 4)
            RGBA colors for each particle, values in [0, 1].
        """
        if self.current_snapshot is None:
            return np.ones((1, 4), dtype=np.float32)

        # Get field values
        if self.color_by == 'density':
            field = self.current_snapshot['density']
        elif self.color_by == 'internal_energy':
            field = self.current_snapshot['internal_energy']
        elif self.color_by == 'temperature':
            # Approximate temperature from internal energy (ideal gas)
            # T ∝ u (assuming constant mean molecular weight)
            field = self.current_snapshot['internal_energy']
        elif self.color_by == 'velocity_magnitude':
            velocities = self.current_snapshot['velocities']
            field = np.linalg.norm(velocities, axis=1)
        elif self.color_by == 'masses':
            field = self.current_snapshot['masses']
        elif self.color_by == 'smoothing_length':
            field = self.current_snapshot['smoothing_length']
        else:
            # Default: density
            field = self.current_snapshot['density']

        # Normalize to [0, 1]
        field_min = np.min(field)
        field_max = np.max(field)
        if field_max - field_min < 1e-10:
            # Constant field
            normalized = np.ones_like(field) * 0.5
        else:
            normalized = (field - field_min) / (field_max - field_min)

        # Apply colormap
        cmap_name = self.COLORMAPS.get(self.color_by, 'viridis')
        colors = self._apply_colormap(normalized, cmap_name)

        return colors

    def _apply_colormap(self, values: NDArrayFloat, cmap_name: str) -> NDArrayFloat:
        """
        Apply colormap to normalized values.

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Normalized values in [0, 1].
        cmap_name : str
            Colormap name.

        Returns
        -------
        colors : np.ndarray, shape (N, 4)
            RGBA colors in [0, 1].
        """
        # Simple colormap implementations
        # For production, consider using matplotlib.cm colormaps
        N = len(values)
        colors = np.ones((N, 4), dtype=np.float32)

        if cmap_name == 'viridis':
            # Approximate viridis: purple -> blue -> green -> yellow
            colors[:, 0] = 0.267 * (1 - values) + 0.993 * values  # R
            colors[:, 1] = 0.005 * (1 - values) + 0.906 * values  # G
            colors[:, 2] = 0.329 * (1 - values) + 0.144 * values  # B
        elif cmap_name == 'plasma':
            # Approximate plasma: purple -> pink -> orange -> yellow
            colors[:, 0] = 0.050 + 0.950 * values  # R
            colors[:, 1] = 0.030 + 0.870 * values**1.5  # G
            colors[:, 2] = 0.527 * (1 - values)**2  # B
        elif cmap_name == 'hot':
            # Hot: black -> red -> yellow -> white
            colors[:, 0] = np.minimum(1.0, values * 3.0)
            colors[:, 1] = np.maximum(0.0, np.minimum(1.0, (values - 0.33) * 3.0))
            colors[:, 2] = np.maximum(0.0, (values - 0.67) * 3.0)
        elif cmap_name == 'cool':
            # Cool: cyan -> magenta
            colors[:, 0] = values
            colors[:, 1] = 1 - values
            colors[:, 2] = 1.0
        elif cmap_name == 'cividis':
            # Approximate cividis (colorblind-friendly)
            colors[:, 0] = 0.0 + 1.0 * values
            colors[:, 1] = 0.1 + 0.8 * values
            colors[:, 2] = 0.3 + 0.4 * values
        elif cmap_name == 'twilight':
            # Approximate twilight (cyclic)
            phase = values * 2 * np.pi
            colors[:, 0] = 0.5 + 0.5 * np.cos(phase)
            colors[:, 1] = 0.5 + 0.5 * np.cos(phase + 2 * np.pi / 3)
            colors[:, 2] = 0.5 + 0.5 * np.cos(phase + 4 * np.pi / 3)
        else:
            # Default: grayscale
            colors[:, :3] = values[:, np.newaxis]

        # Alpha channel (fully opaque)
        colors[:, 3] = 1.0

        return colors

    def _update_info_label(self):
        """Update information label."""
        if self.current_snapshot is None:
            return

        time = self.current_snapshot['time']
        n_frames = len(self.snapshot_loader)
        text = f"Frame: {self.current_frame + 1} / {n_frames} | Time: {time:.4f}"
        self.info_label.setText(text)

    def _toggle_play(self):
        """Toggle play/pause animation."""
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        """Start animation playback."""
        self.is_playing = True
        self.play_button.setText("⏸ Pause")
        interval_ms = int(1000 / self.fps)
        self.animation_timer.start(interval_ms)

    def _pause(self):
        """Pause animation playback."""
        self.is_playing = False
        self.play_button.setText("▶ Play")
        self.animation_timer.stop()

    def _advance_frame(self):
        """Advance to next frame (called by timer)."""
        next_frame = self.current_frame + 1
        if next_frame >= len(self.snapshot_loader):
            # Loop back to start
            next_frame = 0
        self._load_frame(next_frame)

    def _step_forward(self):
        """Step forward one frame."""
        self._pause()
        next_frame = min(self.current_frame + 1, len(self.snapshot_loader) - 1)
        self._load_frame(next_frame)

    def _step_backward(self):
        """Step backward one frame."""
        self._pause()
        prev_frame = max(self.current_frame - 1, 0)
        self._load_frame(prev_frame)

    def _reset_view(self):
        """Reset view to initial state."""
        self._pause()
        self._load_frame(0)
        self.view_widget.setCameraPosition(distance=2.0)

    def _on_slider_changed(self, value: int):
        """Handle timeline slider change."""
        self._pause()
        self._load_frame(value)

    def _on_fps_changed(self, value: int):
        """Handle FPS spinbox change."""
        self.fps = value
        if self.is_playing:
            # Restart timer with new interval
            self._pause()
            self._play()

    def _on_color_changed(self, field_name: str):
        """Handle color-by dropdown change."""
        self.color_by = field_name
        self._update_display()

    def _on_size_changed(self, value: float):
        """Handle point size spinbox change."""
        self.point_size = value
        self._update_display()


def visualize_snapshots(
    snapshot_files: List[str],
    color_by: str = 'density',
    point_size: float = 2.0,
    background: str = 'black',
    title: str = "TDE-SPH 3D Visualizer"
):
    """
    Launch interactive 3D visualizer for snapshot sequence.

    Parameters
    ----------
    snapshot_files : list of str
        Paths to HDF5 snapshot files in time order.
    color_by : str, optional
        Physical quantity to color particles by (default: 'density').
    point_size : float, optional
        Particle marker size in pixels (default: 2.0).
    background : str, optional
        Background color: 'black' or 'white' (default: 'black').
    title : str, optional
        Window title.

    Returns
    -------
    app : QtWidgets.QApplication
        Qt application instance (keep reference to prevent garbage collection).
    window : ParticleVisualizer3D
        Visualizer window instance.

    Examples
    --------
    >>> from glob import glob
    >>> from tde_sph.visualization.pyqtgraph_3d import visualize_snapshots
    >>> snapshot_files = sorted(glob("output/snapshot_*.h5"))
    >>> app, window = visualize_snapshots(snapshot_files, color_by='density')
    >>> app.exec()  # Start event loop

    Notes
    -----
    Requires PyQt6 or PyQt5 and pyqtgraph to be installed.
    OpenGL-capable GPU recommended for smooth performance.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = ParticleVisualizer3D(
        snapshot_files=snapshot_files,
        color_by=color_by,
        point_size=point_size,
        background=background,
        title=title
    )
    window.show()

    return app, window


if __name__ == "__main__":
    # Demo: visualize snapshots from command line
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive 3D visualization of TDE-SPH snapshots"
    )
    parser.add_argument(
        "snapshots",
        nargs="+",
        help="HDF5 snapshot files to visualize (in time order)"
    )
    parser.add_argument(
        "--color-by",
        default="density",
        choices=list(ParticleVisualizer3D.COLORMAPS.keys()),
        help="Physical quantity to color particles by"
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Particle marker size in pixels"
    )
    parser.add_argument(
        "--background",
        choices=["black", "white"],
        default="black",
        help="Background color"
    )

    args = parser.parse_args()

    app, window = visualize_snapshots(
        snapshot_files=args.snapshots,
        color_by=args.color_by,
        point_size=args.point_size,
        background=args.background
    )

    sys.exit(app.exec() if PYQT_VERSION == 6 else app.exec_())
