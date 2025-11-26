"""Debug test to check if tree structure is correct."""

import cupy as cp
from src.tde_sph.gpu.octree_gpu import GPUOctree

print("="*60)
print("TREE STRUCTURE DIAGNOSTIC")
print("="*60)

# Simple 2-particle test
n = 2
positions = cp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=cp.float32)
masses = cp.array([1.0, 1.0], dtype=cp.float32)

tree = GPUOctree()
tree.build(positions, masses)

print(f"\nN = {n} particles")
print(f"Internal nodes: {tree.n_internal}")
print(f"Bounding box: min={tree.bbox_min.get()}, max={tree.bbox_max.get()}")

print(f"\nMorton codes: {tree.morton_codes.get()}")
print(f"Sorted indices: {tree.sorted_indices.get()}")

print(f"\nInternal children left: {tree.internal_children_left.get()}")
print(f"Internal children right: {tree.internal_children_right.get()}")
print(f"Internal parent: {tree.internal_parent.get()}")
print(f"Leaf parent: {tree.leaf_parent.get()}")

print(f"\nLeaf boxes (min): {tree.leaf_box_min.get()}")
print(f"Leaf boxes (max): {tree.leaf_box_max.get()}")
print(f"Leaf COM: {tree.leaf_com.get()}")
print(f"Leaf mass: {tree.leaf_mass.get()}")

print(f"\nInternal boxes (min): {tree.internal_box_min.get()}")
print(f"Internal boxes (max): {tree.internal_box_max.get()}")
print(f"Internal COM: {tree.internal_com.get()}")
print(f"Internal mass: {tree.internal_mass.get()}")

# Test neighbour search
print("\n" + "="*60)
print("NEIGHBOUR SEARCH TEST")
print("="*60)

smoothing_lengths = cp.array([2.0, 2.0], dtype=cp.float32)
neighbours, counts = tree.find_neighbours(positions, smoothing_lengths, support_radius=2.0, max_neighbours=10)

print(f"Neighbour counts: {counts.get()}")
print(f"Neighbour lists:\n{neighbours.get()}")

# Expected: Each particle should find the other as a neighbour
# Distance = 1.0, search radius = 2.0 * h = 4.0, so they should find each other

# Test gravity
print("\n" + "="*60)
print("GRAVITY TEST")
print("="*60)

forces = tree.compute_gravity(positions, masses, smoothing_lengths, G=1.0, epsilon=0.1)
print(f"Forces:\n{forces.get()}")

# Expected: Particles should attract each other
# F = G * m1 * m2 / r^2 = 1.0 * 1.0 * 1.0 / (1.0 + 0.1)^2 = 1.0 / 1.21 ≈ 0.83
# Force on particle 0 should point toward particle 1 (+x direction)
# Force on particle 1 should point toward particle 0 (-x direction)

print("\n" + "="*60)
print("Expected: neighbours=[1,0], forces≈[0.83,0,0] and [-0.83,0,0]")
print("="*60)
