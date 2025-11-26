# GPU Octree Implementation Summary

## Overview
This document summarizes the implementation of GPU-accelerated octree-based algorithms for gravity calculation and neighbour search in the TDE-SPH simulation framework.

## Implementation Details

### 1. Karras Tree Construction (octree_gpu.py)
- **Replaced** simplified bottom-up tree construction with full Karras (2012) parallel radix tree algorithm
- **Features**:
  - Morton code computation for space-filling curve ordering
  - Parallel tree construction using GPU kernels
  - Binary radix tree with one particle per leaf
  - Efficient bounding box computation from bottom-up
  - Center of mass and mass aggregation for internal nodes

### 2. Barnes-Hut Gravity Solver (barnes_hut_gpu.py)
- **Uses** simulation configuration parameters:
  - `G` from gravity solver
  - `theta` from gravity solver (Barnes-Hut opening angle)
  - `epsilon` from gravity solver (softening parameter)
- **Features**:
  - Tree traversal using stack-based algorithm
  - Multipole Acceptance Criterion (MAC) for far-field approximation
  - Direct particle-particle interactions for near-field
  - Forces computed entirely on GPU

### 3. Neighbour Search (neighbours_gpu.py)
- **Uses** simulation configuration parameters:
  - `support_radius` (default 2.0 for SPH kernel support)
  - `max_neighbours` (default 128)
- **Features**:
  - Octree-based neighbour search with sphere-box intersection tests
  - Stack-based tree traversal
  - Efficient pruning of distant tree nodes
  - Results kept on GPU until needed

  - Minimizes PCIe transfers by keeping data on GPU
  - Only updates changed fields (positions, velocities) from CPU
  - Only transfers results back to CPU when needed
  - Tracks whether data is already on GPU to avoid redundant transfers

### 5. Simulation Integration (simulation.py)
  2. Build/reuse octree for current timestep
  3. Update smoothing lengths using GPU
  6. Transfer density to CPU for EOS update (thermodynamics)
  7. Transfer pressure and sound speed back to GPU
  8. Compute gravity forces using Barnes-Hut octree
  9. Compute hydro forces using GPU kernels
  
- **Data Flow Optimization**:
  - Data stays on GPU between steps
  - Only transfer what's needed for CPU-only operations (EOS)

## Performance Benefits

### Reduced PCIe Transfers
- **Before**: Full CPU↔GPU transfer every timestep
- **After**: Only incremental updates and final results transferred

### Algorithm Improvements
- **Gravity**: O(N²) brute force → O(N log N) Barnes-Hut octree
- **Neighbour Search**: O(N²) brute force → O(N log N) octree traversal

### Memory Efficiency
- Octree cached and shared between gravity and neighbour search
- Persistent GPU arrays reused across timesteps
- Minimal host memory usage

## Configuration Parameters

The implementation uses the following simulation parameters:

### From Simulation Config:
- `barnes_hut_theta` (default: 0.5) - Opening angle for multipole approximation
- `target_neighbours` (default: 50) - Target neighbour count for adaptive h
- `artificial_viscosity_alpha` - Viscosity parameter
- `artificial_viscosity_beta` - Viscosity parameter

### From Gravity Solver:
- `G` - Gravitational constant
- `theta` - Barnes-Hut opening angle (overrides config if present)
- `epsilon` - Softening parameter multiplier

### Fixed Constants:
- `support_radius = 2.0` - SPH kernel support radius
- `max_neighbours = 128` - Maximum neighbours per particle
- `max_depth = 21` - Maximum octree depth (for 63-bit Morton codes)

## File Structure

```
src/tde_sph/
│   ├── kernels.py              # CUDA kernels for SPH
│   └── __init__.py             # Export octree functions
├── gravity/
│   ├── barnes_hut_gpu.py      # GPU Barnes-Hut solver
│   └── __init__.py             # Export GPU solver
├── sph/
│   ├── neighbours_gpu.py      # GPU neighbour search
│   └── __init__.py             # Export GPU neighbour functions
└── core/
    └── simulation.py           # Integrated GPU pipeline

## Testing

To test the implementation:

1. Ensure CuPy is installed: `uv add cupy-cuda12x` (or appropriate CUDA version)
2. Run existing tests: `uv run pytest tests/test_gpu.py`
3. Verify octree statistics are logged during simulation
4. Check GPU memory usage is reasonable
5. Compare results with CPU version for correctness

## Future Enhancements

Possible improvements:
1. GPU-accelerated EOS to eliminate CPU transfers
2. Adaptive tree rebuild (only when particles move significantly)
3. Multi-GPU support for very large simulations
4. GPU-based time integration
5. Memory pool optimization for frequent allocations

## References

- Karras, T. (2012) "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
- Bédorf, J. et al. (2012) "A sparse octree gravitational N-body code that runs entirely on the GPU processor"
- Keller, A. et al. (2023) "Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations"
