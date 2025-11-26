"""
Test the NaN safety fixes in GPU hydro kernel.
"""
import sys
from pathlib import Path
import numpy as np

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

print("=" * 70)
print("Testing NaN Safety Fixes in GPU Hydro")
print("=" * 70)

try:
    import cupy as cp
    from tde_sph.gpu import HAS_CUDA, compute_hydro_treesph, GPUOctree, find_neighbours_octree_gpu
    
    if not HAS_CUDA or compute_hydro_treesph is None:
        print("CUDA or compute_hydro_treesph not available")
        sys.exit(1)
    
    print(f"\n✓ CUDA and TreeSPH available")
    
    # Test 1: Extreme case - all zero density/pressure
    print(f"\n{'='*70}")
    print("Test 1: All zero density and pressure")
    print("="*70)
    
    n = 100
    pos = cp.random.randn(n, 3).astype(cp.float32) * 10.0
    vel = cp.random.randn(n, 3).astype(cp.float32) * 0.1
    mass = cp.ones(n, dtype=cp.float32)
    h = cp.ones(n, dtype=cp.float32) * 1.0
    
    # Extreme case: all zeros
    rho = cp.zeros(n, dtype=cp.float32)
    pressure = cp.zeros(n, dtype=cp.float32)
    cs = cp.zeros(n, dtype=cp.float32)
    
    octree = GPUOctree(theta=0.5)
    octree.build(pos, mass)
    
    neighbour_lists, neighbour_counts, _ = find_neighbours_octree_gpu(
        pos, h, masses=mass, support_radius=2.0, max_neighbours=64, tree=octree
    )
    
    print(f"  Inputs: all zero")
    print(f"  Neighbour counts range: [{int(cp.min(neighbour_counts))}, {int(cp.max(neighbour_counts))}]")
    
    acc, dudt = compute_hydro_treesph(
        pos, vel, mass, h, rho, pressure, cs,
        neighbour_lists, neighbour_counts,
        alpha=1.0, beta=2.0
    )
    
    acc_finite = cp.all(cp.isfinite(acc))
    dudt_finite = cp.all(cp.isfinite(dudt))
    
    print(f"  ✓ Computation completed")
    print(f"  Accelerations finite: {acc_finite}")
    print(f"  du_dt finite: {dudt_finite}")
    
    if not acc_finite or not dudt_finite:
        print(f"  ✗ FAIL: NaN/inf detected even with safety fixes!")
        sys.exit(1)
    else:
        print(f"  ✓ PASS: No NaN/inf with all-zero inputs")
    
    # Test 2: NaN in input
    print(f"\n{'='*70}")
    print("Test 2: NaN in sound speed input")
    print("="*70)
    
    rho = cp.ones(n, dtype=cp.float32) * 0.1
    pressure = cp.ones(n, dtype=cp.float32) * 0.01
    cs = cp.ones(n, dtype=cp.float32) * 0.1
    cs[0:5] = cp.nan  # Inject NaN
    
    print(f"  NaN count in cs: {int(cp.sum(cp.isnan(cs)))}")
    
    acc, dudt = compute_hydro_treesph(
        pos, vel, mass, h, rho, pressure, cs,
        neighbour_lists, neighbour_counts,
        alpha=1.0, beta=2.0
    )
    
    acc_finite = cp.all(cp.isfinite(acc))
    dudt_finite = cp.all(cp.isfinite(dudt))
    
    print(f"  ✓ Computation completed")
    print(f"  Accelerations finite: {acc_finite}")
    print(f"  du_dt finite: {dudt_finite}")
    
    if not acc_finite or not dudt_finite:
        print(f"  ✗ FAIL: NaN/inf detected even with safety fixes!")
        sys.exit(1)
    else:
        print(f"  ✓ PASS: NaN inputs handled safely")
    
    # Test 3: Very small densities
    print(f"\n{'='*70}")
    print("Test 3: Very small densities (1e-15)")
    print("="*70)
    
    rho = cp.ones(n, dtype=cp.float32) * 1e-15
    pressure = cp.ones(n, dtype=cp.float32) * 1e-20
    cs = cp.ones(n, dtype=cp.float32) * 1e-8
    
    print(f"  Density: {float(rho[0]):.3e}")
    print(f"  Pressure: {float(pressure[0]):.3e}")
    
    acc, dudt = compute_hydro_treesph(
        pos, vel, mass, h, rho, pressure, cs,
        neighbour_lists, neighbour_counts,
        alpha=1.0, beta=2.0
    )
    
    acc_finite = cp.all(cp.isfinite(acc))
    dudt_finite = cp.all(cp.isfinite(dudt))
    
    print(f"  ✓ Computation completed")
    print(f"  Accelerations finite: {acc_finite}")
    print(f"  du_dt finite: {dudt_finite}")
    
    if not acc_finite or not dudt_finite:
        print(f"  ✗ FAIL: NaN/inf detected with tiny densities!")
        sys.exit(1)
    else:
        print(f"  ✓ PASS: Tiny densities handled safely")
    
    print(f"\n{'='*70}")
    print("✅ ALL TESTS PASSED - NaN safety fixes working!")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
