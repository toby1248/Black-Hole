"""
Debug neighbour finding in octree.
"""
import numpy as np
import cupy as cp
from tde_sph.gpu.octree_gpu import GPUOctree

# Simple 2-particle test
n = 2
positions = cp.array([[0, 0, 0], [1, 0, 0]], dtype=cp.float32)  #  1M separation
smoothing_lengths = cp.ones(n, dtype=cp.float32) * 0.5  # h=0.5

print("2-Particle Neighbour Test")
print("="*70)
print(f"Particle 0: pos={positions[0].get()}")
print(f"Particle 1: pos={positions[1].get()}")
print(f"Separation: 1.0")
print(f"Smoothing length: 0.5")
print(f"Support radius: 2h = 1.0")
print("Expected: particles should be neighbours since r=1.0 <= 2h=1.0")

# Build octree
print("\nBuilding octree...")
octree = GPUOctree()
octree.build(positions, smoothing_lengths)
print("[OK] Octree built")

# Find neighbours
support_radius = 2.0  # Standard SPH support
max_neighbours = 128

print(f"\nFinding neighbours (support_radius={support_radius}, max_neighbours={max_neighbours})...")
neighbour_lists, neighbour_counts = octree.find_neighbours(
    positions, smoothing_lengths, support_radius, max_neighbours
)

print(f"neighbour_lists shape: {neighbour_lists.shape}")
print(f"neighbour_counts shape: {neighbour_counts.shape}")

# Check results
counts = neighbour_counts.get()
print(f"\nNeighbour counts:")
for i in range(n):
    count = counts[i]
    neighs = neighbour_lists[i, :count].get()
    print(f"  Particle {i}: {count} neighbours: {neighs}")

if counts[0] > 0 and counts[1] > 0:
    print("\n[OK] Neighbours found correctly!")
else:
    print("\n[ERROR] No neighbours found! This is the problem.")
