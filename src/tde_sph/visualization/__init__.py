"""
Visualization module: 3D plotting, animation, and interpolation.

Provides visualization backends and data processing tools:
1. Plotly3DVisualizer: Browser-based interactive 3D plots
2. PyQtGraph visualizer: Hardware-accelerated OpenGL rendering with time-scrubbing
3. Interpolation utilities: SPH kernel-weighted grid interpolation, temporal smoothing

Choose based on your needs:
- Plotly: Export to HTML, no PyQt dependency, good for single snapshots
- PyQtGraph: Real-time animation, better performance for large N, requires PyQt6/PyQt5
- Interpolation: Convert particles to volumetric grids for slice plots, volume rendering
"""

from tde_sph.visualization.plotly_3d import (
    Plotly3DVisualizer,
    quick_plot,
)

# PyQtGraph visualizer (optional, requires PyQt6 or PyQt5)
try:
    from tde_sph.visualization.pyqtgraph_3d import (
        ParticleVisualizer3D,
        SnapshotLoader,
        visualize_snapshots,
    )
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    # Provide dummy placeholders
    ParticleVisualizer3D = None
    SnapshotLoader = None
    visualize_snapshots = None

# Interpolation and smoothing utilities
from tde_sph.visualization.interpolation import (
    SPHInterpolator,
    TemporalInterpolator,
    SmoothingFilters,
    particles_to_grid_3d,
    particles_to_slice_2d,
)

# Matplotlib visualization library
from tde_sph.visualization.viz_library import (
    MatplotlibVisualizer,
    quick_density_slice,
    quick_energy_plot,
    quick_phase_diagram,
    quick_radial_profile,
)

__all__ = [
    # Plotly
    'Plotly3DVisualizer',
    'quick_plot',
    # PyQtGraph
    'ParticleVisualizer3D',
    'SnapshotLoader',
    'visualize_snapshots',
    'HAS_PYQTGRAPH',
    # Interpolation
    'SPHInterpolator',
    'TemporalInterpolator',
    'SmoothingFilters',
    'particles_to_grid_3d',
    'particles_to_slice_2d',
    # Matplotlib Library
    'MatplotlibVisualizer',
    'quick_density_slice',
    'quick_energy_plot',
    'quick_phase_diagram',
    'quick_radial_profile',
]
