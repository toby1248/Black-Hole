# CUDA Implementation Guide — Standalone Module Replacements

## Executive Summary

**Goal**: Implement GPU-accelerated versions of compute-intensive TDE-SPH modules as drop-in replacements for existing CPU implementations.

**Status**: Phase 4 (CUDA implementation) — Not started as of 2025-11-18. Phases 1-3 (Newtonian SPH, GR framework, advanced thermodynamics) are complete.

**Priority Order**:
1. Newtonian Gravity (`gravity/newtonian_cuda.py`) — Highest impact, 50-200x speedup potential
2. Neighbour Search (`sph/neighbours_gpu.py`) — 10-50x speedup
3. SPH Hydro Forces (`sph/hydro_forces_cuda.py`) — 20-100x speedup
4. Time Integration (`integration/leapfrog_cuda.py`) — Optional, 5-20x speedup

**Key Principle**: CUDA modules must be **standalone, drop-in replacements** that:
- Implement the same abstract interfaces (`GravitySolver`, `TimeIntegrator`, etc.)
- Accept identical inputs and produce numerically equivalent outputs (FP32 tolerance)
- Can be selected via configuration without changing calling code

**Read First**:
1. This file (complete implementation guide)
2. `SUMMARY.md` (project status and test baseline)
3. `CLAUDE.md` (high-level project requirements)
4. `src/tde_sph/core/interfaces.py` (required interfaces)

---

## Scope

You are implementing CUDA-backed versions of modules that already exist and are functionally complete in the Python/NumPy codebase.

Your work must produce **drop-in replacements** for these modules:

- Same public classes, functions, and semantics.
- Same configuration surface (read from the same config structures).
- Same units, coordinate conventions, and array layouts at the interface boundary.

Each CUDA module must be usable as a standalone backend that can be swapped for the existing module via configuration or a small dispatcher, without changing call sites.

## Priority Modules for Phase 4

The following modules have been identified as high-impact targets for CUDA acceleration:

### 1. **Newtonian Gravity** (Highest Priority)
- **CPU Module**: `src/tde_sph/gravity/newtonian.py`
- **CUDA Target**: `src/tde_sph/gravity/newtonian_cuda.py`
- **Interface**: `GravitySolver` (from `tde_sph.core.interfaces`)
- **Key Methods**:
  - `compute_acceleration(positions, masses, smoothing_lengths, metric=None) -> NDArrayFloat`
  - `compute_potential(positions, masses, smoothing_lengths) -> NDArrayFloat`
- **Complexity**: O(N²) pairwise gravity
- **Expected Speedup**: 50-200x for N > 10,000
- **Key Challenge**: Memory-efficient tiled N-body kernel

### 2. **Neighbour Search**
- **CPU Module**: `src/tde_sph/sph/neighbours_cpu.py`
- **CUDA Target**: `src/tde_sph/sph/neighbours_gpu.py`
- **API**: `find_neighbours(positions, smoothing_lengths, ...) -> List[NDArray]`
- **Complexity**: O(N²) or O(N log N) with spatial grid
- **Expected Speedup**: 10-50x
- **Key Challenge**: Dynamic neighbour list construction on GPU

### 3. **SPH Hydrodynamic Forces**
- **CPU Module**: `src/tde_sph/sph/hydro_forces.py`
- **CUDA Target**: `src/tde_sph/sph/hydro_forces_cuda.py`
- **API**: `compute_hydro_acceleration(positions, velocities, masses, ...) -> Tuple[accel, du_dt]`
- **Complexity**: O(N × k) where k = avg neighbours
- **Expected Speedup**: 20-100x
- **Key Challenge**: Efficient neighbour iteration and kernel gradient evaluation

### 4. **Time Integration** (Optional)
- **CPU Module**: `src/tde_sph/integration/leapfrog.py`
- **CUDA Target**: `src/tde_sph/integration/leapfrog_cuda.py`
- **Interface**: `TimeIntegrator` (from `tde_sph.core.interfaces`)
- **Key Methods**:
  - `step(particles, dt, forces, **kwargs) -> None`
  - `estimate_timestep(particles, cfl_factor, **kwargs) -> float`
- **Expected Speedup**: 5-20x (if all data stays on GPU)
- **Key Challenge**: GPU-resident particle state management

## Constraints

1. **Do NOT change** the public interfaces defined in `tde_sph/core/interfaces.py`.
   - These abstract base classes define the contract: `GravitySolver`, `TimeIntegrator`, `EOS`, `Metric`, etc.
   - Your CUDA implementations must inherit from and implement these interfaces exactly.

2. **Do NOT modify** high-level orchestration in `tde_sph/core/simulation.py` except for minimal, explicit backend selection.

3. **Treat the Python/NumPy code as the reference implementation**; CUDA is an optimization layer.
   - Your CUDA implementation should match CPU outputs within FP32 numerical tolerances.

4. **Default to FP32 on device**; use FP64 only where clearly justified and documented.
   - Document any precision upgrades in module-level `NOTES.md`.

5. **Keep the existing package layout**; new CUDA modules should live alongside their CPU counterparts:
   ```
   src/tde_sph/
   ├── gravity/
   │   ├── newtonian.py         (CPU - exists)
   │   └── newtonian_cuda.py    (GPU - to implement)
   ├── sph/
   │   ├── neighbours_cpu.py    (CPU - exists)
   │   ├── neighbours_gpu.py    (GPU - to implement)
   │   ├── hydro_forces.py      (CPU - exists)
   │   └── hydro_forces_cuda.py (GPU - to implement)
   └── integration/
       ├── leapfrog.py          (CPU - exists)
       └── leapfrog_cuda.py     (GPU - to implement, optional)
   ```

6. **No hidden global state**; backend choice must be explicit.
   - Use configuration flags or factory functions to select CPU vs CUDA.
   - Example: `backend = "cuda"` in config, then dispatch to appropriate module.


## Vector Arithmetic and Linear Algebra

Where feasible, recast computations as large vector and matrix operations to exploit GPU linear algebra hardware:

### Data Layout Conventions

Represent particle state as **dense contiguous arrays** (already present in CPU code):

- **Positions, velocities, accelerations**: `float32[N, 3]` (AoS layout)
  - Alternative: SoA with three separate `float32[N]` arrays (x, y, z) for better coalescing
  - Choose based on profiling; document choice in `NOTES.md`

- **Masses and scalars**: `float32[N]` or `float32[N, 1]`
  - Masses, densities, pressures, internal energies, smoothing lengths, etc.

- **Neighbour lists**:
  - CPU uses `List[NDArray[int32]]` (variable length per particle)
  - GPU options:
   1. Flattened array + offset array: `neighbours_flat[total_neighbours], offsets[N+1]`
   2. Fixed-size 2D array: `neighbours[N, max_neighbours]` with sentinel values
   - Choose based on memory constraints and neighbour count distribution

### Newtonian Gravity and N-body Interactions

1. **Use batched vector arithmetic** and reductions rather than many tiny kernels.

2. **Explore formulations** that map cleanly to cuBLAS-like primitives (batched GEMV/GEMM) or custom kernels with GEMM-style tiling.

3. **For direct particle–particle gravity**:
   - Consider expressing blocks of the interaction matrix as tiles handled via shared memory.
   - **Classic N-body kernel** (recommended starting point):
     - Thread block processes tile of particles
     - Load positions/masses into shared memory
     - Each thread computes forces for one particle from all particles in tile
     - Use warp-level reductions for partial sums

   - **Matrix-style blocked formulation** (evaluate if beneficial):
     - Express force calculation as blocked matrix operations
     - Potentially leverage cuBLAS for certain subroutines
     - **Caution**: O(N²) memory may be prohibitive for N > 50,000

   - **Document your decision**:
     - If you reject a matrix-style approach, document why (e.g., memory footprint, O(N²) cost, worse cache behavior).
     - If you implement both, compare performance and keep faster as default.

### SPH Neighbour Interactions

1. **Store neighbour lists** as dense or batched structures that allow coalesced access and vectorized operations.

2. **Implement density, pressure, and force loops as fused kernels** over these arrays:
   - Combine density estimation + pressure computation + force accumulation where possible
   - Minimize separate kernel launches (each launch has ~10μs overhead)

3. **Kernel function evaluations**:
   - Pre-compute kernel values/gradients if neighbour lists are persistent
   - Or compute on-the-fly if memory-constrained
   - Document trade-off in `NOTES.md`


## Workflow

### 1. Baseline Review

**Before writing any CUDA code**, thoroughly understand the existing implementation:

a. **Read the global implementation plan**:
   - This file (`INSTRUCTIONS.md`)
   - Project overview in `CLAUDE.md`
   - Summary of current status in `SUMMARY.md`

b. **For each target module**, understand:
   - **Existing Python implementation**: Read the CPU version line-by-line
     - Example: `src/tde_sph/gravity/newtonian.py`
   - **Abstract interface**: Check which ABC it implements
     - Example: `GravitySolver` in `src/tde_sph/core/interfaces.py`
   - **Existing tests**: Look for tests that exercise this module
     - Example: Search for tests using `NewtonianGravity` class

c. **Identify clean interface boundaries**:
   - Where does data enter the module? (function parameters, constructor)
   - Where does data leave the module? (return values, in-place modifications)
   - Are there side effects or stateful operations?

### 2. Design the Replacement Module

**Plan before implementing**:

a. **Mirror the public API** exactly:
   - Same class name (with `_cuda` suffix) or class name without suffix but in `*_cuda.py` file
   - Same constructor signature (add GPU-specific params as keyword-only with defaults)
   - Same public method signatures (match parameter names and types)
   - Inherit from the same abstract base class

b. **Decide internal GPU data layout**:
   - **Contiguous SoA or AoS** arrays optimized for vector arithmetic and coalesced loads
   - **Pre-allocated device buffers** reused across steps (minimize allocations)
   - **Memory management strategy**:
     - Allocate on first call, reuse on subsequent calls
     - Resize only when needed (e.g., particle count changes)
     - Deallocate explicitly or use context managers

c. **Choose CUDA framework**:
   - **CuPy**: NumPy-like API, easiest for direct translations, good for prototyping
   - **Numba CUDA**: More control over kernels, better for custom optimizations
   - **PyCUDA**: Most control, more boilerplate
   - **Recommendation**: Start with CuPy for array operations, use Numba for custom kernels

### 3. Implement CUDA Kernels and/or Linear Algebra Paths

**Implement efficiently**:

a. **Move hot loops** into CUDA kernels:
   - SPH density estimation
   - Pressure/force computation
   - Neighbour search
   - Gravity force summation
   - Integration steps

b. **For Newtonian gravity**, prototype both approaches and compare:
   - **Classic N-body kernel** (start here):
     - Shared-memory tiling (e.g., 256-thread blocks, 32-64 particle tiles)
     - Warp-level reductions for partial force sums
     - Coalesced global memory access patterns
   - **Vectorized / block-matrix formulation** (if memory permits):
     - Compute interaction matrix in blocks
     - Leverage cuBLAS for GEMV operations if beneficial
     - Benchmark against classic kernel

c. **Minimize host–device transfers**:
   - Keep particle data on GPU across multiple timesteps
   - Only transfer results when needed (e.g., for I/O, diagnostics)
   - Use pinned memory for faster transfers when needed

d. **Minimize kernel launch overheads**:
   - Fuse multiple operations into single kernels where possible
   - Use streams for overlapping compute and transfers (advanced)

### 4. Integrate as a Backend

**Make it usable**:

a. **Expose a module-level class** conforming to the existing interface:
   ```python
   # Example: src/tde_sph/gravity/newtonian_cuda.py
   from ..core.interfaces import GravitySolver

   class NewtonianGravityCUDA(GravitySolver):
       def __init__(self, G: float = 1.0, device_id: int = 0):
           # CUDA-specific initialization
           ...
   ```

b. **Implement a backend selection mechanism**:
   - Option 1: Configuration-based dispatch
   - Option 2: Factory function
   - Option 3: Drop-in replacement (import swapping)

   Example factory:
   ```python
   # src/tde_sph/gravity/__init__.py
   def get_gravity_solver(backend="cpu", **kwargs):
       if backend == "cpu":
           from .newtonian import NewtonianGravity
           return NewtonianGravity(**kwargs)
       elif backend == "cuda":
           from .newtonian_cuda import NewtonianGravityCUDA
           return NewtonianGravityCUDA(**kwargs)
   ```

c. **Ensure compatibility** with both Newtonian and GR modes:
   - If module is used in GR context, ensure metric parameter is handled correctly
   - Test with different metrics (Minkowski, Schwarzschild)

### 5. Testing and Validation

**Rigorous comparison against CPU reference**:

a. **Create CPU vs CUDA comparison tests**:
   ```python
   # Example: tests/test_gravity_cuda.py
   def test_newtonian_cuda_vs_cpu():
       # Same inputs
       positions = ...
       masses = ...
       smoothing_lengths = ...

       # CPU computation
       solver_cpu = NewtonianGravity(G=1.0)
       accel_cpu = solver_cpu.compute_acceleration(positions, masses, smoothing_lengths)

       # GPU computation
       solver_cuda = NewtonianGravityCUDA(G=1.0)
       accel_cuda = solver_cuda.compute_acceleration(positions, masses, smoothing_lengths)

       # Compare within FP32 tolerances
       np.testing.assert_allclose(accel_cuda, accel_cpu, rtol=1e-5, atol=1e-7)
   ```

b. **Test coverage**:
   - Small N (e.g., 100 particles) for quick CI tests
   - Medium N (e.g., 10,000 particles) for realistic workloads
   - Edge cases: single particle, two particles, extreme mass ratios

c. **Numerical tolerance guidelines**:
   - FP32 arithmetic: `rtol=1e-5, atol=1e-7` (relative + absolute)
   - Document any cases requiring looser tolerances
   - If CUDA uses FP64 for certain operations, tighten tolerances accordingly

d. **Cover both implementation paths** if multiple exist:
   - Test classic kernel path
   - Test linear algebra path (if implemented)
   - Test backend selection mechanism



## Deliverables per Module

For each ported module, provide the following:

### 1. CUDA Implementation File

**File**: `src/tde_sph/<module>/<name>_cuda.py`

Example: `src/tde_sph/gravity/newtonian_cuda.py`

**Requirements**:
- Inherits from appropriate abstract base class (e.g., `GravitySolver`)
- Implements all required abstract methods
- Matches CPU module's public API exactly
- Includes comprehensive docstrings (NumPy style)
- CUDA-specific parameters (e.g., `device_id`, `block_size`) should be keyword-only with sensible defaults

**Example structure**:
```python
"""
CUDA-accelerated Newtonian gravity solver.

Drop-in replacement for gravity/newtonian.py with GPU acceleration.
"""

from typing import Optional
import numpy as np
import cupy as cp  # or numba.cuda
from ..core.interfaces import GravitySolver, Metric, NDArrayFloat


class NewtonianGravityCUDA(GravitySolver):
    """
    CUDA implementation of Newtonian self-gravity.

    See gravity/newtonian.py for physics documentation.
    This version uses GPU kernels for O(N²) force computation.
    """

    def __init__(self, G: float = 1.0, *, device_id: int = 0, block_size: int = 256):
        # Implementation
        ...
```

### 2. Integration and Backend Selection

**Update module's `__init__.py`** to expose backend selection:

Example: `src/tde_sph/gravity/__init__.py`
```python
from .newtonian import NewtonianGravity

# Try to import CUDA version, gracefully handle if not available
try:
    from .newtonian_cuda import NewtonianGravityCUDA
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    NewtonianGravityCUDA = None

def get_gravity_solver(backend="cpu", **kwargs):
    """Factory function for gravity solvers."""
    if backend == "cpu":
        return NewtonianGravity(**kwargs)
    elif backend == "cuda":
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend requested but not available")
        return NewtonianGravityCUDA(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

### 3. Unit Tests

**File**: `tests/test_<module>_cuda.py`

Example: `tests/test_gravity_cuda.py`

**Requirements**:
- Test CPU vs CUDA output equivalence
- Test with various particle counts (small, medium)
- Test edge cases
- Use pytest fixtures for common setup
- Skip tests if CUDA not available (`pytest.mark.skipif`)

**Example structure**:
```python
import pytest
import numpy as np

# Conditional import
try:
    import cupy as cp
    from tde_sph.gravity.newtonian_cuda import NewtonianGravityCUDA
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

from tde_sph.gravity.newtonian import NewtonianGravity


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestNewtonianGravityCUDA:

    def test_acceleration_vs_cpu_small(self):
        """Test CUDA vs CPU for 100 particles."""
        # Test implementation
        ...

    def test_acceleration_vs_cpu_medium(self):
        """Test CUDA vs CPU for 10,000 particles."""
        ...

    def test_potential_vs_cpu(self):
        """Test potential computation."""
        ...
```

### 4. Module Documentation (NOTES.md)

**File**: `src/tde_sph/<module>/NOTES.md`

Example: `src/tde_sph/gravity/NOTES.md`

**Required sections**:

```markdown
# Newtonian Gravity - CUDA Implementation Notes

## Data Layouts

- **Positions**: `float32[N, 3]` on device (AoS layout)
- **Masses**: `float32[N]` on device
- **Accelerations**: `float32[N, 3]` output on device

## Kernel Architecture

### compute_acceleration_kernel

**Thread configuration**:
- Block size: 256 threads
- Grid size: (N + 255) // 256 blocks

**Shared memory**:
- Position tile: `float[TILE_SIZE, 3]`
- Mass tile: `float[TILE_SIZE]`
- Total: ~4 KB per block (for TILE_SIZE=256)

**Algorithm**:
1. Each thread handles one target particle
2. Loop over tiles of source particles
3. Load tile into shared memory (coalesced)
4. Compute pairwise forces within tile
5. Accumulate force components in registers
6. Write result to global memory

**Computational complexity**: O(N²)
**Memory bandwidth**: ~12 bytes read + 12 bytes write per particle pair

## Matrix-Style Formulation

**Status**: Evaluated and rejected

**Reasoning**:
- O(N²) memory requirement: ~4N² bytes for interaction matrix
- For N=100,000: ~40 GB GPU memory (exceeds typical GPU capacity)
- Classic kernel approach has same O(N²) computation but O(N) memory
- cuBLAS operations don't provide significant benefit for this specific problem

## Precision

- **Default**: FP32 throughout
- **Accumulation**: Force components accumulated in FP32 registers
- **Known limitations**: ~6-7 digits of precision, acceptable for N < 1,000,000

## Performance Characteristics

**Benchmark results** (example for NVIDIA RTX 3090):
- N=1,000: 0.5 ms (20x speedup vs CPU)
- N=10,000: 45 ms (80x speedup vs CPU)
- N=100,000: 4.5 s (120x speedup vs CPU)

**Bottlenecks**:
- Compute-bound for N > 10,000
- Memory-bound for N < 1,000

## Known Issues and Limitations

- Maximum N limited by GPU memory (~1M particles on 24GB GPU)
- Performance degrades if particles very unevenly distributed in space (future: spatial tree methods)

## Future Optimizations

1. Implement Barnes-Hut tree for O(N log N) scaling
2. Use FP64 accumulation option for very long simulations
3. Multi-GPU support for N > 1M
```

### 5. Additional Guidelines

**File Paths and Output**:
- ✅ Use relative paths only
- ✅ Export raw data to `output/` folder
- ✅ Export visualizations and summaries to `results/` folder
- ❌ Never use absolute file paths in code

**Error Handling**:
- Catch and mitigate errors from extreme interactions (e.g., particles too close)
- Identify and flag bad data (NaN, Inf)
- Throw warnings for debugging
- Filter visualization scaling for extreme outliers
- Example:
  ```python
  if np.any(np.isnan(accel)) or np.any(np.isinf(accel)):
      warnings.warn("NaN or Inf detected in acceleration computation")
      # Clip or sanitize values
  ```

---

## Implementation Checklist

For each CUDA module implementation, verify:

- [ ] **CPU module reviewed**: Fully understand reference implementation
- [ ] **Interface compliance**: Inherits from correct ABC, implements all abstract methods
- [ ] **API parity**: Public methods match CPU signatures exactly
- [ ] **CUDA framework chosen**: CuPy, Numba, or PyCUDA (document choice)
- [ ] **Data layout documented**: Shapes, dtypes, memory layout (AoS vs SoA)
- [ ] **Kernel implemented**: Core computation moved to GPU
- [ ] **Memory management**: Pre-allocation, reuse, minimal host-device transfers
- [ ] **Error handling**: NaN/Inf detection, warnings, graceful degradation
- [ ] **Numerical validation**: CPU vs CUDA comparison tests pass (FP32 tolerance)
- [ ] **Edge cases tested**: Small N, large N, degenerate cases
- [ ] **Backend selection**: Factory or config-based dispatch implemented
- [ ] **NOTES.md created**: Data layouts, kernel design, matrix formulation decision
- [ ] **Performance benchmarked**: Speedup measured and documented (optional but recommended)

---

## Quick Reference: Module Mapping

| Module Type | CPU Implementation | CUDA Target | Interface |
|-------------|-------------------|-------------|-----------|
| Gravity | `gravity/newtonian.py` | `gravity/newtonian_cuda.py` | `GravitySolver` |
| Neighbours | `sph/neighbours_cpu.py` | `sph/neighbours_gpu.py` | Function API |
| Hydro Forces | `sph/hydro_forces.py` | `sph/hydro_forces_cuda.py` | Function API |
| Integration | `integration/leapfrog.py` | `integration/leapfrog_cuda.py` | `TimeIntegrator` |

---

## Dependencies and Requirements

**Python Packages** (for CUDA implementation):
```bash
# Core
numpy>=1.20
cupy-cuda11x>=10.0  # or cupy-cuda12x, match your CUDA version
# OR
numba>=0.56  # includes CUDA support

# Testing
pytest>=7.0
pytest-benchmark  # optional, for performance testing
```

**System Requirements**:
- CUDA Toolkit 11.0+ or 12.0+
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
- GPU with 4GB+ memory (8GB+ recommended for N > 50,000)

**Graceful Fallback**:
All CUDA imports should be wrapped in try/except to allow CPU-only operation:
```python
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
```

---

## Tips for Agent Implementers

1. **Start small**: Implement basic kernel first, optimize later
2. **Profile early**: Use `cupy.cuda.profiler` or `nvprof` to identify bottlenecks
3. **Validate continuously**: Compare against CPU after every major change
4. **Document decisions**: Capture design choices in NOTES.md as you go
5. **Test edge cases**: Single particle, two particles, extreme mass ratios
6. **Benchmark systematically**: N=100, 1000, 10000, 100000 (if memory permits)
7. **Watch for common pitfalls**:
   - Forgetting to synchronize: `cp.cuda.Stream.null.synchronize()`
   - Device-host copies in hot loops
   - Incorrect thread/block dimensions
   - Shared memory bank conflicts
   - Warp divergence in conditionals

---

# Reviewer Sub-Agent Prompt - Not for programming agents
**ROLE:** Unbiased AI-to-AI code reviewer agent. Succinct and blunt.

**SCOPE:** Review code in the directory containing this file. To you nothing else exists.

**GOAL:** Verify the high level functionality and logic of the module

**TASKS:**
  1. Identify correctness and robustness issues.
  2. Flag missing context or hidden assumptions.
  3. Suggest minimal, concrete improvements.
  4. Add to or create the local NOTES.md




