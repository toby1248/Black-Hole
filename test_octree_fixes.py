"""Test script to verify octree BSOD fixes."""

import sys
import cupy as cp
import numpy as np

# Test GPU availability
try:
    print("Testing GPU availability...")
    device = cp.cuda.Device(0)
    print(f"  ✓ GPU found: {device.compute_capability}")
    print(f"  ✓ Memory: {device.mem_info[1] / 1e9:.1f} GB total")
except Exception as e:
    print(f"  ✗ GPU not available: {e}")
    sys.exit(1)

# Import octree
try:
    print("\nImporting octree module...")
    from src.tde_sph.gpu.octree_gpu import GPUOctree
    print("  ✓ Import successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 1: Small dataset (N=100)
print("\n" + "="*60)
print("TEST 1: Small dataset (N=100)")
print("="*60)
try:
    n = 100
    positions = cp.random.randn(n, 3).astype(cp.float32) * 10.0
    masses = cp.ones(n, dtype=cp.float32)
    
    print(f"Building octree for {n} particles...")
    tree = GPUOctree()
    tree.build(positions, masses)
    print("  ✓ Tree construction successful")
    
    print(f"Tree structure: {tree.n_internal} internal nodes, {tree.n_particles} leaves")
    
    print("Testing neighbour search...")
    smoothing_lengths = cp.ones(n, dtype=cp.float32) * 2.0
    neighbours, counts = tree.find_neighbours(positions, smoothing_lengths)
    print(f"  ✓ Neighbour search successful")
    print(f"  Average neighbours per particle: {counts.mean():.1f}")
    
    print("Testing gravity computation...")
    smoothing_lengths_grav = cp.ones(n, dtype=cp.float32) * 2.0
    forces = tree.compute_gravity(positions, masses, smoothing_lengths_grav)
    print(f"  ✓ Gravity computation successful")
    print(f"  Average force magnitude: {cp.linalg.norm(forces, axis=1).mean():.6f}")
    
    print("\n✓ TEST 1 PASSED")
    
except Exception as e:
    print(f"\n✗ TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Medium dataset (N=1000)
print("\n" + "="*60)
print("TEST 2: Medium dataset (N=1000)")
print("="*60)
try:
    n = 1000
    positions = cp.random.randn(n, 3).astype(cp.float32) * 10.0
    masses = cp.ones(n, dtype=cp.float32)
    
    print(f"Building octree for {n} particles...")
    tree = GPUOctree()
    tree.build(positions, masses)
    print("  ✓ Tree construction successful")
    
    print("Testing neighbour search...")
    smoothing_lengths = cp.ones(n, dtype=cp.float32) * 1.0
    neighbours, counts = tree.find_neighbours(positions, smoothing_lengths, max_neighbours=100)
    print(f"  ✓ Neighbour search successful")
    print(f"  Average neighbours per particle: {counts.mean():.1f}")
    
    print("Testing gravity computation...")
    smoothing_lengths_grav = cp.ones(n, dtype=cp.float32) * 1.0
    forces = tree.compute_gravity(positions, masses, smoothing_lengths_grav, G=1.0, epsilon=0.1)
    print(f"  ✓ Gravity computation successful")
    print(f"  Average force magnitude: {cp.linalg.norm(forces, axis=1).mean():.6f}")
    
    print("\n✓ TEST 2 PASSED")
    
except Exception as e:
    print(f"\n✗ TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Large dataset (N=10000)
print("\n" + "="*60)
print("TEST 3: Large dataset (N=10000)")
print("="*60)
try:
    n = 10000
    positions = cp.random.randn(n, 3).astype(cp.float32) * 10.0
    masses = cp.ones(n, dtype=cp.float32)
    
    print(f"Building octree for {n} particles...")
    tree = GPUOctree()
    tree.build(positions, masses)
    print("  ✓ Tree construction successful")
    
    print("Testing neighbour search...")
    smoothing_lengths = cp.ones(n, dtype=cp.float32) * 0.5
    neighbours, counts = tree.find_neighbours(positions, smoothing_lengths, max_neighbours=50)
    print(f"  ✓ Neighbour search successful")
    print(f"  Average neighbours per particle: {counts.mean():.1f}")
    
    print("Testing gravity computation...")
    smoothing_lengths_grav = cp.ones(n, dtype=cp.float32) * 0.5
    forces = tree.compute_gravity(positions, masses, smoothing_lengths_grav, G=1.0, epsilon=0.1)
    print(f"  ✓ Gravity computation successful")
    print(f"  Average force magnitude: {cp.linalg.norm(forces, axis=1).mean():.6f}")
    
    print("\n✓ TEST 3 PASSED")
    
except Exception as e:
    print(f"\n✗ TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Edge case - single particle
print("\n" + "="*60)
print("TEST 4: Edge case - single particle")
print("="*60)
try:
    n = 1
    positions = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    masses = cp.array([1.0], dtype=cp.float32)
    
    print(f"Building octree for {n} particle...")
    tree = GPUOctree()
    tree.build(positions, masses)
    print("  ✓ Tree construction handled gracefully")
    
    print("Testing neighbour search...")
    smoothing_lengths = cp.array([1.0], dtype=cp.float32)
    neighbours, counts = tree.find_neighbours(positions, smoothing_lengths)
    print(f"  ✓ Neighbour search successful (count={counts[0]})")
    
    print("Testing gravity computation...")
    smoothing_lengths_grav = cp.array([1.0], dtype=cp.float32)
    forces = tree.compute_gravity(positions, masses, smoothing_lengths_grav)
    print(f"  ✓ Gravity computation successful (force={forces[0]})")
    
    print("\n✓ TEST 4 PASSED")
    
except Exception as e:
    print(f"\n✗ TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Stress test - multiple iterations
print("\n" + "="*60)
print("TEST 5: Stress test - 100 iterations")
print("="*60)
try:
    n = 1000
    positions = cp.random.randn(n, 3).astype(cp.float32) * 10.0
    masses = cp.ones(n, dtype=cp.float32)
    smoothing_lengths = cp.ones(n, dtype=cp.float32) * 1.0
    
    print(f"Running 100 iterations with {n} particles...")
    for i in range(100):
        tree = GPUOctree()
        tree.build(positions, masses)
        neighbours, counts = tree.find_neighbours(positions, smoothing_lengths)
        forces = tree.compute_gravity(positions, masses, smoothing_lengths)
        
        # Small perturbation
        positions += cp.random.randn(n, 3).astype(cp.float32) * 0.01
        
        if (i + 1) % 20 == 0:
            print(f"  ✓ Iteration {i+1}/100 complete")
    
    print("\n✓ TEST 5 PASSED")
    
except Exception as e:
    print(f"\n✗ TEST 5 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
print("\nOctree implementation is stable and ready for use.")
print("BSOD fixes have been successfully applied.")
