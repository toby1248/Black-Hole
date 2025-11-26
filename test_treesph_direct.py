"""
Direct test of TreeSPH kernels without simulation wrapper.
"""
import numpy as np
import cupy as cp
from tde_sph.gpu.octree_gpu import GPUOctree
from tde_sph.gpu.treesph_kernels import compute_density_treesph, compute_hydro_treesph

# Simple 2-particle test
n = 2
positions = cp.array([[0, 0, 0], [1, 0, 0]], dtype=cp.float32)
velocities = cp.zeros((n, 3), dtype=cp.float32)
masses = cp.ones(n, dtype=cp.float32) * 1e-4
smoothing_lengths = cp.ones(n, dtype=cp.float32) * 0.6  # h=0.6, so q = 1.0/0.6 = 1.67 < 2.0
densities = cp.ones(n, dtype=cp.float32)
pressures = cp.ones(n, dtype=cp.float32)
sound_speeds = cp.ones(n, dtype=cp.float32)

print("Direct TreeSPH Kernel Test")
print("="*70)
print("Setup:")
print(f"  Particles: {n}")
print(f"  Separation: 1.0")
print(f"  Smoothing length: 0.6")
print(f"  q = r/h = 1.0/0.6 = 1.67 < 2.0 (within support)")
print(f"  Masses: {masses.get()}")
print(f"  Pressures: {pressures.get()}")
print(f"  Densities: {densities.get()}")

# Build octree and find neighbours
print("\nBuilding octree and finding neighbours...")
octree = GPUOctree()
octree.build(positions, smoothing_lengths)

support_radius = 2.0
max_neighbours = 128
neighbour_lists, neighbour_counts = octree.find_neighbours(
    positions, smoothing_lengths, support_radius, max_neighbours
)

counts = neighbour_counts.get()
print(f"Neighbour counts: {counts}")
for i in range(n):
    neighs = neighbour_lists[i, :counts[i]].get()
    print(f"  Particle {i} neighbours: {neighs}")

# Test density computation
print("\n" + "="*70)
print("Testing density kernel...")
print("="*70)

densities_computed = compute_density_treesph(
    positions, masses, smoothing_lengths,
    neighbour_lists, neighbour_counts
)

print(f"Computed densities: {densities_computed.get()}")

# Test hydro computation
print("\n" + "="*70)
print("Testing hydro kernel...")
print("="*70)

accelerations, du_dt = compute_hydro_treesph(
    positions, velocities, masses, smoothing_lengths,
    densities, pressures, sound_speeds,
    neighbour_lists, neighbour_counts,
    alpha=1.0, beta=2.0
)

print(f"Accelerations:")
for i in range(n):
    acc = accelerations[i].get()
    print(f"  Particle {i}: [{acc[0]:.10f}, {acc[1]:.10f}, {acc[2]:.10f}]")

print(f"\ndu/dt: {du_dt.get()}")

# Check if forces are non-zero
if cp.max(cp.abs(accelerations)) > 1e-10:
    print("\n[OK] TreeSPH hydro kernel produces non-zero forces!")
else:
    print("\n[ERROR] TreeSPH hydro forces are zero!")
