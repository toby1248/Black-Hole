"""GPU TreeSPH density regression tests."""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy", reason="CuPy is required for GPU density tests")

from tde_sph.gpu import HAS_CUDA
from tde_sph.gpu.treesph_kernels import compute_density_treesph
from tde_sph.sph.kernels import CubicSplineKernel


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA backend not available")
def test_treesph_density_includes_self_term():
    """Particles without neighbours must recover the self-contribution W(0)."""
    kernel = CubicSplineKernel(dim=3)

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    masses = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    smoothing_lengths = np.array([0.5, 0.5, 0.2], dtype=np.float32)

    # Particle 0 and 1 see each other; particle 2 is isolated (zero neighbours)
    neighbour_lists = np.array(
        [
            [1, -1],
            [0, -1],
            [-1, -1],
        ],
        dtype=np.int32,
    )
    neighbour_counts = np.array([1, 1, 0], dtype=np.int32)

    densities_gpu = compute_density_treesph(
        cupy.asarray(positions),
        cupy.asarray(masses),
        cupy.asarray(smoothing_lengths),
        cupy.asarray(neighbour_lists),
        cupy.asarray(neighbour_counts),
    ).get()

    # Reference density using the CPU cubic spline kernel definition
    expected = np.zeros_like(masses)
    for i in range(len(masses)):
        hi = smoothing_lengths[i]
        expected[i] += masses[i] * kernel.kernel(0.0, hi)
        for k in range(neighbour_counts[i]):
            j = neighbour_lists[i, k]
            if j < 0:
                continue
            r = np.linalg.norm(positions[i] - positions[j])
            expected[i] += masses[j] * kernel.kernel(r, hi)

    np.testing.assert_allclose(densities_gpu, expected, rtol=1e-6, atol=1e-7)
