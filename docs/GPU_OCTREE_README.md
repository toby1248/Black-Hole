# GPU-Accelerated Octree Implementation

This directory contains a fully GPU-accelerated octree implementation using CuPy for CUDA. The implementation provides unified tree construction for both:
1. **TreeSPH neighbour search** (O(N log N) instead of O(N²))
2. **Barnes-Hut gravitational force calculations** (O(N log N) instead of O(N²))

## Key Features

- **Unified Tree**: Build once, use for both neighbour search and gravity calculations
- **Minimal PCIe Transfers**: All tree operations stay on GPU
- **RTX 4090 Optimized**: Designed to maximize GPU throughput and minimize bandwidth bottlenecks
- **CuPy Implementation**: Raw CUDA kernels for maximum performance
- **Compatible Interface**: Drop-in replacement for CPU versions

## Files

```
octree_gpu.py           - Core GPU octree implementation with CUDA kernels
barnes_hut_gpu.py       - GPU Barnes-Hut gravity solver
neighbours_gpu.py       - GPU neighbour search functions
test_gpu_octree.py      - Comprehensive test suite
requirements_gpu.txt    - Python package requirements
```

## Algorithm Details

### Octree Construction

Based on research from:
- **Karras (2012)**: Parallel radix tree construction on GPU
- **Bédorf et al. (2012)**: GPU octree for N-body simulations
- **Keller et al. (2023)**: Cornerstone method for scalable octree construction

#### Implementation Steps:

1. **Morton Code Generation**
   - Convert 3D positions to 63-bit Morton codes (Z-order curve)
   - Parallel computation on GPU using custom CUDA kernel
   - 21 bits per dimension for spatial indexing

2. **Parallel Radix Sort**
   - Sort particles by Morton code using CuPy's GPU sort
   - Groups spatially nearby particles together

3. **Leaf Node Construction**
   - Identify unique Morton codes
   - Build leaf nodes containing particles
   - Compute bounding boxes and center of mass

4. **Tree Traversal Structures**
   - Store leaf metadata (bounding boxes, COM, mass)
    - Enable efficient GPU traversal for neighbour search and gravity

### Neighbour Search

- Sphere-box overlap testing for culling
- GPU-parallel search through octree
- Configurable support radius (default 2.0 × smoothing length)
- Returns neighbour lists in compatible format

### Barnes-Hut Gravity

- Multipole Acceptance Criterion (MAC) with configurable θ
- Softening parameter based on smoothing length
- GPU-parallel force computation
- Returns accelerations directly

## Installation

### Requirements

- NVIDIA GPU with CUDA support (tested on RTX 4090)
- CUDA Toolkit 11.x or 12.x
- Python 3.8+

### Setup

```bash
# Check CUDA version
nvidia-smi

# Install NumPy
pip install numpy

# Install CuPy (choose version matching your CUDA)
# For CUDA 12.x:
pip install cupy-cuda12x

# For CUDA 11.x:
pip install cupy-cuda11x

# Verify installation
python -c "import cupy; print(f'CuPy OK: {cupy.cuda.runtime.getDeviceCount()} GPU(s)')"
```

## Usage

### Basic Octree Construction

```python
import cupy as cp
from octree_gpu import GPUOctree

# Particle data (on GPU)
positions_gpu = cp.random.rand(10000, 3).astype(cp.float32)
masses_gpu = cp.ones(10000, dtype=cp.float32)

# Build tree
tree = GPUOctree(theta=0.5)
tree.build(positions_gpu, masses_gpu)

# Get statistics
stats = tree.get_tree_statistics()
print(f"Built tree with {stats['n_occupied_leaves']} leaves")
```

### GPU Barnes-Hut Gravity

```python
import numpy as np
from barnes_hut_gpu import BarnesHutGravityGPU

# Particle data (on CPU - will be transferred to GPU)
positions = np.random.rand(10000, 3).astype(np.float32)
masses = np.ones(10000, dtype=np.float32)
smoothing_lengths = np.ones(10000, dtype=np.float32) * 0.01

# Create solver
solver = BarnesHutGravityGPU(G=1.0, theta=0.5, epsilon=0.1)

# Compute acceleration (returned on CPU)
accel = solver.compute_acceleration(positions, masses, smoothing_lengths)

# Get tree for neighbour search reuse
tree_data = solver.get_tree_data()
```

### GPU Neighbour Search

```python
from neighbours_gpu import find_neighbours_octree_gpu_integrated

# Reuse tree from gravity solver
neighbour_lists, _ = find_neighbours_octree_gpu_integrated(
    positions,
    smoothing_lengths,
    tree_data,  # From gravity solver
    support_radius=2.0,
    max_neighbours=128
)

# Access neighbours for particle i
neighbours_of_particle_0 = neighbour_lists[0]
```

### Unified Workflow (Optimal)

```python
from barnes_hut_gpu import BarnesHutGravityGPU
from neighbours_gpu import find_neighbours_octree_gpu_integrated

# Setup
solver = BarnesHutGravityGPU(G=1.0, theta=0.5)

# Step 1: Compute gravity (builds tree)
accel = solver.compute_acceleration(positions, masses, smoothing_lengths)

# Step 2: Find neighbours (reuses tree)
tree_data = solver.get_tree_data()
neighbours, _ = find_neighbours_octree_gpu_integrated(
    positions, smoothing_lengths, tree_data
)

# Tree built once, used twice - minimal PCIe overhead!
```

## Performance Optimization

### PCIe Transfer Minimization

The implementation minimizes expensive CPU↔GPU transfers by:

1. **Persistent GPU Arrays**: Reusing GPU memory allocations across time steps
2. **Unified Tree**: Building tree once for both gravity and neighbours
3. **GPU-Side Operations**: All tree construction and traversal on GPU
4. **Lazy Transfers**: Only transferring final results to CPU

### Memory Management

```python
# Check GPU memory usage
stats = solver.get_statistics()
print(f"GPU memory: {stats['gpu_memory_mb']:.1f} MB")
```

### Optimal Parameters

- `theta=0.5`: Standard Barnes-Hut (lower = more accurate, slower)
- `epsilon=0.1`: Softening parameter (prevents singularities)
- `max_neighbours=128`: Typical for SPH simulations
- `support_radius=2.0`: Standard for cubic spline kernel

## Testing

Run the comprehensive test suite:

```bash
python test_gpu_octree.py
```

Tests include:
1. Basic octree construction
2. GPU gravity solver
3. GPU neighbour search
4. Unified tree reuse
5. CPU vs GPU comparison
6. Performance scaling

## Integration with Simulation

The GPU solvers are designed as drop-in replacements:

```python
# In simulation.py or similar

# Option 1: Use GPU gravity solver
from barnes_hut_gpu import BarnesHutGravityGPU
gravity_solver = BarnesHutGravityGPU(G=1.0)

# Option 2: Mix GPU gravity with CPU SPH (works but suboptimal)
# Better: Move entire simulation to GPU for maximum performance

# The tree from GPU gravity can be used for neighbour search
tree_data = gravity_solver.get_tree_data()
if tree_data and tree_data.get('gpu'):
    from neighbours_gpu import find_neighbours_octree_gpu_integrated
    neighbours, _ = find_neighbours_octree_gpu_integrated(
        positions, smoothing_lengths, tree_data
    )
```

## Performance Expectations

For RTX 4090 with N particles:

| N | Tree Build | Gravity | Neighbour Search | Total |
|---|---|---|---|---|
| 1,000 | ~1 ms | ~2 ms | ~1 ms | ~4 ms |
| 10,000 | ~3 ms | ~10 ms | ~5 ms | ~18 ms |
| 100,000 | ~15 ms | ~60 ms | ~30 ms | ~105 ms |
| 1,000,000 | ~100 ms | ~400 ms | ~200 ms | ~700 ms |

*Note: Actual performance depends on particle distribution and parameters*

Scaling is O(N log N) for both gravity and neighbour search.

## Known Limitations

1. **CUDA Required**: No CPU fallback (by design for performance)
2. **Memory**: Limited by GPU VRAM (24GB on RTX 4090 = ~30-50M particles)
3. **Potential Not Implemented**: `compute_potential()` returns zeros (rarely needed)
4. **Tree Structure**: Simplified compared to full hierarchical octree

## Future Optimizations

Potential improvements for even better performance:

1. **Hierarchical Tree Traversal**: Implement full internal nodes for better MAC
2. **Shared Memory Optimization**: Use shared memory for neighbour search
3. **Warp-Level Primitives**: Exploit warp-level operations
4. **Multi-GPU Support**: Distribute particles across multiple GPUs
5. **Persistent Kernels**: Use CUDA graphs for reduced launch overhead

## References

1. Barnes & Hut (1986) - *A Hierarchical O(N log N) Force-Calculation Algorithm*
2. Karras (2012) - *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees*
3. Bédorf et al. (2012) - *A sparse octree gravitational N-body code*
4. Keller et al. (2023) - *Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations*

## License

Same as parent project.

## Contact

For issues or questions about the GPU implementation, please create an issue in the repository.
