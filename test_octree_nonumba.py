"""
Test octree without Numba to debug.
"""
import numpy as np
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable Numba JIT

print("Importing Barnes-Hut...")
from src.tde_sph.gravity.barnes_hut import BarnesHutGravity
print("Importing neighbour search functions...")
from src.tde_sph.sph.neighbours_cpu import find_neighbours_octree

print("Creating test data...")
n = 5
positions = np.random.randn(n, 3).astype(np.float32)
masses = np.ones(n, dtype=np.float32)
h = np.full(n, 1.0, dtype=np.float32)

print("Building tree...")
solver = BarnesHutGravity()
_ = solver.compute_acceleration(positions, masses, h)

print("Getting tree data...")
tree_data = solver.get_tree_data()
print(f"Tree nodes: {len(tree_data['leaf_particle'])}")

print("Calling octree neighbour search (no Numba)...")
try:
    neighbours, _ = find_neighbours_octree(positions, h, tree_data)
    print(f"SUCCESS! Found neighbours for {len(neighbours)} particles")
    for i in range(min(3, len(neighbours))):
        print(f"  Particle {i}: {len(neighbours[i])} neighbours")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
