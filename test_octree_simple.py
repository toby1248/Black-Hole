"""
Simple test for octree implementation.
"""
import numpy as np
print("Importing Barnes-Hut...")
from src.tde_sph.gravity.barnes_hut import BarnesHutGravity
print("Importing neighbour search functions...")
from src.tde_sph.sph.neighbours_cpu import find_neighbours_octree

print("Creating test data...")
n = 5  # Very small test
positions = np.random.randn(n, 3).astype(np.float32)
masses = np.ones(n, dtype=np.float32)
h = np.full(n, 1.0, dtype=np.float32)  # Larger h for testing

print("Building tree...")
solver = BarnesHutGravity()
_ = solver.compute_acceleration(positions, masses, h)

print("Getting tree data...")
tree_data = solver.get_tree_data()
print(f"Tree data keys: {tree_data.keys() if tree_data else 'None'}")

if tree_data:
    print("Calling octree neighbour search...")
    neighbours, _ = find_neighbours_octree(positions, h, tree_data)
    print(f"Found neighbours for {len(neighbours)} particles")
    print(f"Sample: particle 0 has {len(neighbours[0])} neighbours")
    print("SUCCESS!")
else:
    print("ERROR: No tree data!")
