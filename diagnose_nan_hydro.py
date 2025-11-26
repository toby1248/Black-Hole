"""
Diagnose NaN in GPU hydro accelerations by checking intermediate values.
"""
import sys
from pathlib import Path
import numpy as np

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

print("=" * 70)
print("Diagnosing NaN in GPU Hydro Accelerations")
print("=" * 70)

try:
    import cupy as cp
    from tde_sph.gpu import HAS_CUDA
    
    if not HAS_CUDA:
        print("CUDA not available - cannot diagnose GPU issue")
        sys.exit(1)
        
    print(f"\n✓ CUDA available")
    
    # Create test data that might trigger the NaN
    n = 100
    
    # Simulate particles with varying densities (including very small ones)
    pos = cp.random.randn(n, 3).astype(cp.float32)
    vel = cp.random.randn(n, 3).astype(cp.float32) * 0.1
    mass = cp.ones(n, dtype=cp.float32)
    h = cp.ones(n, dtype=cp.float32) * 0.1
    
    # Critical: Include some zero/near-zero densities
    rho = cp.abs(cp.random.randn(n).astype(cp.float32)) * 0.1
    rho[0:5] = 0.0  # Some exactly zero
    rho[5:10] = 1e-12  # Some very small
    
    # Pressure and sound speed might also be zero
    pressure = rho * 0.1  # Some will be zero if rho is zero
    cs = cp.sqrt(pressure / rho)  # Will be NaN if rho=0
    
    print(f"\nTest data created:")
    print(f"  Density range: [{float(cp.min(rho)):.3e}, {float(cp.max(rho)):.3e}]")
    print(f"  Pressure range: [{float(cp.min(pressure)):.3e}, {float(cp.max(pressure)):.3e}]")
    print(f"  Sound speed range: [{float(cp.min(cs)):.3e}, {float(cp.max(cs)):.3e}]")
    print(f"  Zero densities: {int(cp.sum(rho == 0.0))}")
    print(f"  NaN sound speeds: {int(cp.sum(cp.isnan(cs)))}")
    
    # Build octree and get neighbours
    from tde_sph.gpu import GPUOctree, find_neighbours_octree_gpu
    
    octree = GPUOctree(theta=0.5)
    octree.build(pos, mass)
    
    neighbour_lists, neighbour_counts, _ = find_neighbours_octree_gpu(
        pos, h, masses=mass, support_radius=2.0, max_neighbours=64, tree=octree
    )
    
    print(f"\n✓ Octree built and neighbours found")
    print(f"  Neighbour counts range: [{int(cp.min(neighbour_counts))}, {int(cp.max(neighbour_counts))}]")
    
    # Try to compute hydro with problematic data
    from tde_sph.gpu import compute_hydro_treesph
    
    if compute_hydro_treesph is None:
        print("\n✗ compute_hydro_treesph not available")
        sys.exit(1)
    
    print(f"\n✓ Attempting hydro computation with problematic data...")
    
    try:
        acc, dudt = compute_hydro_treesph(
            pos, vel, mass, h, rho, pressure, cs,
            neighbour_lists, neighbour_counts,
            alpha=1.0, beta=2.0
        )
        
        # Check results
        acc_finite = cp.all(cp.isfinite(acc))
        dudt_finite = cp.all(cp.isfinite(dudt))
        
        print(f"\nResults:")
        print(f"  Accelerations finite: {acc_finite}")
        print(f"  du_dt finite: {dudt_finite}")
        
        if not acc_finite:
            nan_count = int(cp.sum(~cp.isfinite(acc)))
            print(f"  ✗ {nan_count} NaN/inf accelerations detected!")
            
            # Find which particles have NaN
            bad_mask = ~cp.isfinite(acc).any(axis=1)
            bad_indices = cp.where(bad_mask)[0]
            print(f"  Particles with NaN: {bad_indices[:10].tolist()}")
            
            # Check their properties
            for idx in bad_indices[:5]:
                i = int(idx)
                print(f"\n  Particle {i}:")
                print(f"    rho: {float(rho[i]):.3e}")
                print(f"    P: {float(pressure[i]):.3e}")
                print(f"    cs: {float(cs[i]):.3e}")
                print(f"    neighbours: {int(neighbour_counts[i])}")
        
        if not dudt_finite:
            nan_count = int(cp.sum(~cp.isfinite(dudt)))
            print(f"  ✗ {nan_count} NaN/inf du_dt detected!")
            
    except Exception as e:
        print(f"\n✗ Hydro computation failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"\n✗ Diagnostic failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Diagnosis complete")
print("=" * 70)
