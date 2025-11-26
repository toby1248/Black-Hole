"""
Debug TreeSPH kernel issues.
"""
import numpy as np
import cupy as cp
from tde_sph.gpu.octree import Octree

# Create simple test case
n = 10
positions = cp.random.randn(n, 3).astype(cp.float32) * 0.5 + 10.0
masses = cp.ones(n, dtype=cp.float32) * 1e-4
smoothing_lengths = cp.ones(n, dtype=cp.float32) * 0.5
velocities = cp.random.randn(n, 3).astype(cp.float32) * 0.1

print("Test data:")
print(f"  N particles: {n}")
print(f"  Positions range: {float(cp.min(cp.linalg.norm(positions, axis=1))):.2f} to {float(cp.max(cp.linalg.norm(positions, axis=1))):.2f}")
print(f"  Smoothing lengths: all {float(smoothing_lengths[0]):.2f}")

# Build octree
print("\nBuilding octree...")
octree = Octree()
octree.build(positions, smoothing_lengths)

# Compute neighbours
support_radius = 2.0
max_neighbours = 128
print(f"\nComputing neighbours (support_radius={support_radius}, max_neighbours={max_neighbours})...")
neighbour_lists, neighbour_counts = octree.find_neighbours(
    positions, smoothing_lengths, support_radius, max_neighbours
)

print(f"  neighbour_lists shape: {neighbour_lists.shape}")
print(f"  neighbour_counts shape: {neighbour_counts.shape}")
print(f"  Neighbour counts: min={int(cp.min(neighbour_counts))}, max={int(cp.max(neighbour_counts))}, mean={float(cp.mean(neighbour_counts)):.1f}")

# Show first few particles' neighbours
print("\nFirst 3 particles' neighbours:")
for i in range(min(3, n)):
    count = int(neighbour_counts[i])
    neighs = neighbour_lists[i, :count].get()
    print(f"  Particle {i}: {count} neighbours: {neighs}")

# Try density computation
print("\n" + "="*70)
print("Testing density kernel...")
print("="*70)

from tde_sph.gpu.treesph_kernels import compute_density_treesph

densities = compute_density_treesph(
    positions,
    masses,
    smoothing_lengths,
    neighbour_lists,
    neighbour_counts
)

print(f"Densities: min={float(cp.min(densities)):.6f}, max={float(cp.max(densities)):.6f}, mean={float(cp.mean(densities)):.6f}")
print(f"First 5 densities: {densities[:5].get()}")

# Try hydro computation
print("\n" + "="*70)
print("Testing hydro kernel...")
print("="*70)

pressures = cp.ones(n, dtype=cp.float32) * 1.0
sound_speeds = cp.ones(n, dtype=cp.float32) * 1.0

from tde_sph.gpu.treesph_kernels import compute_hydro_treesph

accelerations, du_dt = compute_hydro_treesph(
    positions,
    velocities,
    masses,
    smoothing_lengths,
    densities,
    pressures,
    sound_speeds,
    neighbour_lists,
    neighbour_counts
)

print(f"Accelerations shape: {accelerations.shape}")
print(f"Accelerations: min={float(cp.min(cp.linalg.norm(accelerations, axis=1))):.6f}, max={float(cp.max(cp.linalg.norm(accelerations, axis=1))):.6f}")
print(f"First 3 accelerations:")
for i in range(min(3, n)):
    acc = accelerations[i].get()
    print(f"  Particle {i}: [{acc[0]:.6f}, {acc[1]:.6f}, {acc[2]:.6f}]")

print(f"\ndu_dt: min={float(cp.min(du_dt)):.6f}, max={float(cp.max(du_dt)):.6f}")
print(f"First 5 du_dt: {du_dt[:5].get()}")

print("\n[OK] TreeSPH kernels executed successfully!")
