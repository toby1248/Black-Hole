goal: CUDA-accelerated backend for existing TDE-SPH modules (drop-in, standalone replacements)
version: 1.1
date_created: 2025-11-18
last_updated: 2025-11-18
owner: TDE-SPH CUDA Team
status: Planned
tags: [cuda, backend, optimisation, SPH, gravity, linear-algebra, Python]
Role and Scope
You are working in a module directory that already has a working Python/NumPy implementation.

Your mission is to implement a CUDA-backed module that can fully replace this directory’s existing runtime behaviour without changing how other modules call into it.

Public API must remain identical (class names, method signatures, free functions).

Behaviour must match the CPU implementation within agreed numerical tolerances.

CUDA-specific data structures and kernels must remain internal to this module.

Core Expectations
Module-level replacement

The module you implement (e.g. hydro_forces_cuda.py, newtonian_cuda.py) must be usable as a standalone substitute:

Existing code should be able to switch imports from CPU to CUDA versions without further changes.

No changes to global interfaces or orchestration code from inside this module.

Vector arithmetic and matrix-based layouts

Express particle data as large contiguous vectors and matrices:

Positions, velocities, accelerations: float32[N, 3] (or SoA with three float32[N] arrays).

Masses and scalar fields: float32[N] or float32[N, 1].

Design kernels to operate on these arrays using:

Vector arithmetic in a data-parallel style.

Blocked loads into shared memory to emulate matrix tiles.

For Newtonian gravity in particular:

Investigate whether the force calculation can be structured as:

Blocked interaction matrices processed tile-by-tile (matrix-style N-body).

Or batched vector operations that resemble matrix–vector products.

Decide and document whether a linear-algebra-style formulation (possibly using cuBLAS or custom GEMM-like kernels) is beneficial for your target N:

If yes, implement and describe the scheme.

If no, justify briefly (e.g. O(N²) memory/compute, performance vs classic pairwise kernel).

Respect core interfaces

Conform strictly to abstract base classes in tde_sph/core/interfaces.py.

Do not change these ABCs here.

Add optional helpers only if they do not break any existing callers.

Configuration and modes

Use the existing config structures for:

CPU vs CUDA backend selection.

Newtonian vs GR or other physics modes.

Avoid new global flags; backend selection should be explicit and local.

Numerics, precision, and stability

Default to FP32 on the GPU; upcast to FP64 only for clearly identified sensitive operations.

Be explicit where dtypes change between host and device.

Ensure that vectorised and matrix-style implementations remain numerically stable for long integrations (especially gravity).

Performance discipline

Minimise host–device transfers and small kernel launches.

Prefer:

Persistent device arrays for particle state.

Fused kernels for SPH loops and neighbour interactions.

Tiled N-body kernels for Newtonian gravity.

Profile both “pure kernel” and “linear algebra-style” variants when both exist; keep the faster one as the default and leave the other as an optional code path if it is still useful.

Testing Expectations
For each major public method, create tests that:

Instantiate both CPU and CUDA implementations.

Run identical, small test problems.

Compare outputs (arrays, scalars) within reasonable tolerances.

Include tests that specifically exercise:

Vectorised kernels over large arrays.

Any matrix-style or cuBLAS-based implementation you introduce.

Ensure tests are small enough for CI but sensitive enough to catch indexing errors, race conditions, and precision regressions.

Local Agent Tasks (per module)
Inside this directory:

Map each public class/method/function in the CPU module to the CUDA-backed version.

Design data layouts and kernels with vector arithmetic and large-array operations in mind.

For Newtonian gravity modules, explicitly evaluate:

Classic pairwise kernels with shared-memory tiling.

Matrix-style or batched linear algebra formulations.
And record your decision and rationale.

Implement kernels and integration glue for this module only.

Add or extend tests comparing CPU and CUDA implementations.

Document:

Data layouts and dtypes.

Where linear algebra primitives are used.

Any intentional approximations or limitations.

