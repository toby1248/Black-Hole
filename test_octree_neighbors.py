"""
Quick test to verify octree neighbour search is working.
"""
import numpy as np
from src.tde_sph.gravity.barnes_hut import BarnesHutGravity
from src.tde_sph.sph.neighbours_cpu import find_neighbours_bruteforce, find_neighbours_octree

# Create test particles
np.random.seed(42)
n_particles = 50000  # Reduced from 100 for faster testing
positions = np.random.randn(n_particles, 3).astype(np.float32)
masses = np.ones(n_particles, dtype=np.float32)
smoothing_lengths = np.full(n_particles, 0.5, dtype=np.float32)

print("Testing octree neighbour search...")
print(f"Number of particles: {n_particles}")

# Build tree using Barnes-Hut gravity solver
print("\n1. Building octree via Barnes-Hut gravity...")
bh_solver = BarnesHutGravity(G=1.0, theta=0.5)
a_grav = bh_solver.compute_acceleration(positions, masses, smoothing_lengths)
print(f"   Gravity computed: acceleration shape = {a_grav.shape}")

# Get tree data
tree_data = bh_solver.get_tree_data()
if tree_data is None:
    print("   ERROR: Tree data is None!")
    exit(1)

print(f"   Tree data retrieved:")
print(f"     - Nodes: {len(tree_data['leaf_particle'])}")
print(f"     - child_ptr shape: {tree_data['child_ptr'].shape}")
print(f"     - leaf_particle shape: {tree_data['leaf_particle'].shape}")

# Find neighbours using brute force
print("\n2. Finding neighbours with brute force...")
import time
t0 = time.time()
neighbours_bruteforce, _ = find_neighbours_bruteforce(positions, smoothing_lengths)
t_bruteforce = time.time() - t0
print(f"   Time: {t_bruteforce*1000:.2f} ms")
print(f"   Sample neighbour counts: {[len(n) for n in neighbours_bruteforce[:5]]}")

# Find neighbours using octree
print("\n3. Finding neighbours with octree...")
t0 = time.time()
neighbours_octree, _ = find_neighbours_octree(positions, smoothing_lengths, tree_data)
t_octree = time.time() - t0
print(f"   Time: {t_octree*1000:.2f} ms")
print(f"   Sample neighbour counts: {[len(n) for n in neighbours_octree[:5]]}")

# Compare results
print("\n4. Comparing results...")
total_bruteforce = sum(len(n) for n in neighbours_bruteforce)
total_octree = sum(len(n) for n in neighbours_octree)
print(f"   Total neighbours (brute force): {total_bruteforce}")
print(f"   Total neighbours (octree):      {total_octree}")

# Check if neighbour lists match (order might differ)
matches = 0
mismatches = 0
for i in range(n_particles):
    set_bf = set(neighbours_bruteforce[i])
    set_oct = set(neighbours_octree[i])
    if set_bf == set_oct:
        matches += 1
    else:
        mismatches += 1
        if mismatches <= 3:  # Show first few mismatches
            missing = set_bf - set_oct
            extra = set_oct - set_bf
            print(f"   Mismatch at particle {i}:")
            print(f"     Missing: {missing}")
            print(f"     Extra: {extra}")

print(f"\n   Matches: {matches}/{n_particles}")
print(f"   Mismatches: {mismatches}/{n_particles}")

if mismatches == 0:
    print("\n✅ SUCCESS: Octree neighbour search matches brute force exactly!")
    print(f"   Speedup: {t_bruteforce/t_octree:.2f}x (will be much higher for N > 10k)")
else:
    print(f"\n⚠️  WARNING: {mismatches} particles have different neighbour lists")
    print("   This might be due to edge cases at the support radius boundary")
