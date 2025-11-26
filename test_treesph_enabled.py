"""
Verify GPU TreeSPH kernels are available and not disabled.
"""
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

print("=" * 60)
print("Verifying GPU TreeSPH Kernels")
print("=" * 60)

# Test 1: Check HAS_CUDA
print("\n1. Checking CUDA availability...")
try:
    from tde_sph.gpu import HAS_CUDA
    print(f"   HAS_CUDA: {HAS_CUDA}")
    if not HAS_CUDA:
        print("   ⚠ CUDA not available - GPU kernels will not be tested")
        sys.exit(0)
except ImportError as e:
    print(f"   ✗ Failed to import HAS_CUDA: {e}")
    sys.exit(1)

# Test 2: Check TreeSPH kernel imports
print("\n2. Checking TreeSPH kernel imports...")
try:
    from tde_sph.gpu import (
        compute_density_treesph,
        compute_hydro_treesph,
        GPUOctree,
        find_neighbours_octree_gpu
    )
    
    kernels_status = {
        'compute_density_treesph': compute_density_treesph is not None,
        'compute_hydro_treesph': compute_hydro_treesph is not None,
        'GPUOctree': GPUOctree is not None,
        'find_neighbours_octree_gpu': find_neighbours_octree_gpu is not None
    }
    
    print("   TreeSPH Kernel Status:")
    all_available = True
    for name, available in kernels_status.items():
        status = "✓ Available" if available else "✗ NOT AVAILABLE"
        print(f"     {name}: {status}")
        if not available:
            all_available = False
    
    if all_available:
        print("\n   ✓ All GPU TreeSPH kernels are available!")
    else:
        print("\n   ⚠ Some GPU TreeSPH kernels are missing")
        
except ImportError as e:
    print(f"   ✗ Failed to import TreeSPH kernels: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check if kernels can be used
print("\n3. Testing GPU TreeSPH kernel functionality...")
if all_available:
    try:
        import cupy as cp
        import numpy as np
        
        # Create minimal test data
        n = 100
        pos_cpu = np.random.randn(n, 3).astype(np.float32)
        mass_cpu = np.ones(n, dtype=np.float32)
        h_cpu = np.ones(n, dtype=np.float32) * 0.1
        
        pos_gpu = cp.asarray(pos_cpu)
        mass_gpu = cp.asarray(mass_cpu)
        h_gpu = cp.asarray(h_cpu)
        
        # Build octree (theta is passed to __init__, not build())
        octree = GPUOctree(theta=0.5)
        octree.build(pos_gpu, mass_gpu)
        print("   ✓ GPU Octree built successfully")
        
        # Find neighbours - note: pass octree as 'tree' kwarg, not positional
        neighbour_lists, neighbour_counts, _ = find_neighbours_octree_gpu(
            pos_gpu, h_gpu, masses=mass_gpu, 
            support_radius=2.0, max_neighbours=64, tree=octree
        )
        print(f"   ✓ GPU neighbour search completed")
        print(f"     - Neighbour lists shape: {neighbour_lists.shape}")
        print(f"     - Neighbour counts range: [{int(cp.min(neighbour_counts))}, {int(cp.max(neighbour_counts))}]")
        
        # Test density kernel
        rho_gpu = compute_density_treesph(
            pos_gpu, mass_gpu, h_gpu,
            neighbour_lists, neighbour_counts
        )
        print(f"   ✓ compute_density_treesph executed successfully")
        print(f"     - Density range: [{float(cp.min(rho_gpu)):.3e}, {float(cp.max(rho_gpu)):.3e}]")
        
        # Test hydro kernel
        vel_gpu = cp.random.randn(n, 3).astype(cp.float32)
        pressure_gpu = cp.ones(n, dtype=cp.float32)
        cs_gpu = cp.ones(n, dtype=cp.float32)
        
        acc_hydro, du_dt = compute_hydro_treesph(
            pos_gpu, vel_gpu, mass_gpu, h_gpu,
            rho_gpu, pressure_gpu, cs_gpu,
            neighbour_lists, neighbour_counts,
            alpha=1.0, beta=2.0
        )
        print(f"   ✓ compute_hydro_treesph executed successfully")
        print(f"     - Acceleration range: [{float(cp.min(acc_hydro)):.3e}, {float(cp.max(acc_hydro)):.3e}]")
        print(f"     - du_dt range: [{float(cp.min(du_dt)):.3e}, {float(cp.max(du_dt)):.3e}]")
        
        print("\n   ✓ All GPU TreeSPH kernels functional!")
        
    except Exception as e:
        print(f"   ✗ GPU TreeSPH kernel test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⊘ Skipped (kernels not available)")

print("\n" + "=" * 60)
print("GPU TreeSPH Kernel Verification Complete")
print("=" * 60)
