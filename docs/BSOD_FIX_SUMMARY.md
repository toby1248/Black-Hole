# BSOD Fix Summary

## Problem
Running the GPU octree code caused a Blue Screen of Death (BSOD) due to multiple critical bugs:

1. **CPU-GPU Transfer Loop**: `_compute_internal_properties()` had a Python loop that called `.get()` repeatedly, causing massive PCIe overhead
2. **Stack Overflow**: Kernels used 64-element stacks, insufficient for deep trees (N=10000+ particles)
3. **Infinite Loops**: No iteration limits in tree traversal loops
4. **Missing Bounds Checks**: Array accesses without validation could cause memory corruption
5. **Race Conditions**: Multiple threads updating same internal nodes without proper synchronization

## Fixes Applied

### 1. GPU-Parallel Property Computation
**Before**: CPU-side loop in `_compute_internal_properties()` walking up tree with `.get()` calls

**After**: New GPU kernel `compute_internal_properties_bottomup()` that:
- Computes bbox, COM, and mass in one pass on GPU
- Uses atomic flags to coordinate bottom-up propagation
- Each thread starts from a leaf and walks up
- Eliminates all CPU-GPU transfers during tree construction

### 2. Increased Stack Sizes
**Before**: 64-element stacks in `NEIGHBOUR_SEARCH_KERNEL` and `BARNES_HUT_KERNEL`

**After**: 128-element stacks (MAX_STACK constant)

### 3. Iteration Limits
Added `MAX_ITERATIONS` checks to all tree traversal loops:
- Neighbour search: `n_particles * 10` iterations max
- Gravity computation: `n_particles * 10` iterations max
- Property computation: 64 iterations max

### 4. Bounds Checking
Added validation before all array accesses:
- `if (node < 0) continue;`
- `if (leaf_id < 0 || leaf_id >= n_particles) continue;`
- `if (other_idx < 0 || other_idx >= n_particles) continue;`
- `if (left >= 0 && left < n_internal + n_particles && stack_ptr < MAX_STACK)`

### 5. Stack Overflow Protection
- Check `stack_ptr >= MAX_STACK` before continuing traversal
- Gracefully truncate search if stack limit reached

### 6. Edge Case Handling
- Handle `n_particles < 2` (degenerate single-particle case)
- Added padding to bounding box: `max((bbox_max - bbox_min) * 0.01, 1e-6)`
- Explicit synchronization after tree build: `cp.cuda.Stream.null.synchronize()`

### 7. Parent Array Computation
- `internal_parent` array correctly populated by Karras kernel
- Used in bottom-up property computation to walk up tree

## Modified Files
- `src/tde_sph/gpu/octree_gpu.py`: All kernel fixes and method updates

## Testing Recommendations

1. **Start Small**: Test with N=100, 1000, 10000 particles
2. **Monitor GPU**: Watch GPU memory and utilization
3. **Check Logs**: Look for "Warning: Only 1 particle" or stack overflow messages
4. **Verify Results**: Compare neighbour counts and forces with brute-force CPU version
5. **Stress Test**: Run long simulations (1000+ steps) to check stability

## Expected Improvements
- **No BSOD**: System stability restored
- **Performance**: O(N log N) complexity vs O(N²) brute-force
- **Scalability**: Can handle 100k+ particles without crash
- **Memory Efficiency**: All data stays on GPU, minimal PCIe transfers

## Rollback Plan
If issues persist, the brute-force algorithms are still available:
- `compute_gravity_bruteforce_gpu()` in `gravity/gpu_gravity.py`
- CPU-based neighbour search in `sph/neighbours.py`

To disable octree: Set `use_gpu=False` in simulation config or use CPU pipeline.
