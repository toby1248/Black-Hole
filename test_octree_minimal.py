"""Minimal test to diagnose octree hang."""

import sys
import cupy as cp
import numpy as np

print("="*60)
print("MINIMAL OCTREE DIAGNOSTIC TEST")
print("="*60)

# Test GPU
try:
    print("\n1. Testing GPU availability...")
    device = cp.cuda.Device(0)
    print(f"   ✓ GPU: {device.compute_capability}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Import octree
try:
    print("\n2. Importing octree module...")
    from src.tde_sph.gpu.octree_gpu import GPUOctree
    print("   ✓ Import successful")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create octree object
try:
    print("\n3. Creating GPUOctree object...")
    tree = GPUOctree()
    print("   ✓ Object created")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with 2 particles (minimal tree)
try:
    print("\n4. Testing with N=2 particles...")
    positions = cp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=cp.float32)
    masses = cp.array([1.0, 1.0], dtype=cp.float32)
    
    print("   - Positions created")
    print("   - Calling tree.build()...")
    
    tree.build(positions, masses)
    
    print("   ✓ Tree build completed!")
    print(f"   - Internal nodes: {tree.n_internal}")
    print(f"   - Leaves: {tree.n_particles}")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with 10 particles
try:
    print("\n5. Testing with N=10 particles...")
    positions = cp.random.randn(10, 3).astype(cp.float32)
    masses = cp.ones(10, dtype=cp.float32)
    
    print("   - Calling tree.build()...")
    tree.build(positions, masses)
    
    print("   ✓ Tree build completed!")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with 100 particles
try:
    print("\n6. Testing with N=100 particles...")
    positions = cp.random.randn(100, 3).astype(cp.float32) * 10.0
    masses = cp.ones(100, dtype=cp.float32)
    
    print("   - Calling tree.build()...")
    tree.build(positions, masses)
    
    print("   ✓ Tree build completed!")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
