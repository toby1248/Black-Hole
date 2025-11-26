"""Regression tests for neighbour search and smoothing-length updates."""

import numpy as np

from tde_sph.sph.neighbours_cpu import (
    find_neighbours_bruteforce,
    update_smoothing_lengths,
)


def test_neighbour_symmetrize_flag_changes_counts():
    """Ensure the symmetrize flag controls how neighbour counts are built."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.75, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    smoothing_lengths = np.array([0.01, 0.5, 0.5], dtype=np.float32)

    neighbours_sym, _ = find_neighbours_bruteforce(
        positions,
        smoothing_lengths,
        symmetrize=True,
    )
    neighbours_asym, _ = find_neighbours_bruteforce(
        positions,
        smoothing_lengths,
        symmetrize=False,
    )

    # Particle 0 should see both neighbours in the symmetric search
    assert len(neighbours_sym[0]) == 2
    # With per-particle support it should see none (r > 2 h_0)
    assert len(neighbours_asym[0]) == 0
    # Particles with large smoothing length still retain neighbours
    assert len(neighbours_asym[1]) >= 1


def test_update_smoothing_lengths_expands_underpopulated_particle():
    """Particles with too-few neighbours should grow their smoothing length."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    masses = np.full(4, 0.25, dtype=np.float32)
    smoothing_lengths = np.array([0.01, 2.0, 2.0, 2.0], dtype=np.float32)

    h_updated = update_smoothing_lengths(
        positions,
        masses,
        smoothing_lengths,
        target_neighbours=10,
        max_iterations=5,
        tolerance=0.2,
    )

    # The under-populated first particle should increase its smoothing length
    assert h_updated[0] > smoothing_lengths[0]
    # Ensure the update remains numerically well-behaved for the rest
    assert np.all(np.isfinite(h_updated[1:]))
