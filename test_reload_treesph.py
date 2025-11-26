"""
Force reload of TreeSPH CUDA kernels and test.
"""
import sys
import importlib

# Remove cached modules
if 'tde_sph.gpu.treesph_kernels' in sys.modules:
    del sys.modules['tde_sph.gpu.treesph_kernels']

import numpy as np
import cupy as cp

# Now import fresh
from tde_sph.gpu.octree_gpu import GPUOctree
from tde_sph.gpu.treesph_kernels import compute_density_treesph, compute_hydro_treesph

# Simple test
n = 2
positions = cp.array([[0, 0, 0], [1, 0, 0]], dtype=cp.float32)
velocities = cp.zeros((n, 3), dtype=cp.float32)
masses = cp.ones(n, dtype=cp.float32) * 1e-4
smoothing_lengths = cp.ones(n, dtype=cp.float32) * 0.6
densities = cp.ones(n, dtype=cp.float32)
pressures = cp.ones(n, dtype=cp.float32)
sound_speeds = cp.ones(n, dtype=cp.float32)

print("Force-Reloaded TreeSPH Kernel Test")
print("="*70)

# Build octree
octree = GPUOctree()
octree.build(positions, smoothing_lengths)

# Find neighbours
neighbour_lists, neighbour_counts = octree.find_neighbours(
    positions, smoothing_lengths, 2.0, 128
)

counts = neighbour_counts.get()
print(f"Neighbours found: {counts}")

# Test hydro
accelerations, du_dt = compute_hydro_treesph(
    positions, velocities, masses, smoothing_lengths,
    densities, pressures, sound_speeds,
    neighbour_lists, neighbour_counts
)

print(f"Accelerations:")
for i in range(n):
    acc = accelerations[i].get()
    print(f"  Particle {i}: [{acc[0]:.10f}, {acc[1]:.10f}, {acc[2]:.10f}]")

if cp.max(cp.abs(accelerations)) > 1e-10:
    print("\n[OK] TreeSPH kernels working after reload!")
else:
    print("\n[ERROR] Still zero after reload")
