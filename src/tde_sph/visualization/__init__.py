"""
Visualization module: 3D plotting and animation.

Provides two visualization backends:
1. Plotly3DVisualizer: Browser-based interactive 3D plots
2. PyQtGraph visualizer: Hardware-accelerated OpenGL rendering with time-scrubbing

Choose based on your needs:
- Plotly: Export to HTML, no PyQt dependency, good for single snapshots
- PyQtGraph: Real-time animation, better performance for large N, requires PyQt6/PyQt5
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

__all__ = [
    'Plotly3DVisualizer',
    'quick_plot',
    'ParticleVisualizer3D',
    'SnapshotLoader',
    'visualize_snapshots',
    'HAS_PYQTGRAPH',
]
