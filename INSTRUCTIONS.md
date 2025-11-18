CUDA Implementation Agents — Standalone Module Replacements


Scope

You are implementing CUDA-backed versions of modules that already exist and are functionally complete in the Python/NumPy codebase.

Your work must produce drop-in replacements for these modules:

Same public classes, functions, and semantics.

Same configuration surface (read from the same config structures).

Same units, coordinate conventions, and array layouts at the interface boundary.


Each CUDA module must be usable as a standalone backend that can be swapped for the existing module via configuration or a small dispatcher, without changing call sites.

Constraints

Do not change the public interfaces defined in tde_sph/core/interfaces.py.

Do not modify high-level orchestration in tde_sph/core/simulation.py except for minimal, explicit backend selection.

Treat the Python/NumPy code as the reference implementation; CUDA is an optimisation layer.

Default to FP32 on device; use FP64 only where clearly justified and documented.

Keep the existing package layout; new CUDA modules should live alongside their CPU counterparts (e.g. sph/hydro_forces_cuda.py, gravity/newtonian_cuda.py).

No hidden global state; backend choice must be explicit.


Vector Arithmetic and Linear Algebra

Where feasible, recast computations as large vector and matrix operations to exploit GPU linear algebra hardware:

Represent particle state as dense arrays:

Positions, velocities, accelerations: float32[N, 3].

Masses and other scalars: float32[N] or float32[N, 1].


In Newtonian gravity and neighbour interactions:

Use batched vector arithmetic and reductions rather than many tiny kernels.

Explore formulations that map cleanly to cuBLAS-like primitives (batched GEMV/GEMM) or custom kernels with GEMM-style tiling.


For direct particle–particle gravity:

Consider expressing blocks of the interaction matrix as tiles handled via shared memory.

Evaluate whether a “matrix-style” blocked formulation outperforms naive pairwise loops for your N-regime and GPU memory limits.

If you reject a matrix-style approach, document why (e.g. memory footprint, O(N²) cost, worse cache behaviour).



For SPH:

Store neighbour lists and kernel weights as dense or batched structures that allow coalesced access and vectorised operations.

Implement density, pressure, and force loops as fused kernels over these arrays.


Workflow

1. Baseline review

Read the global implementation plan and instructions files.

For each target module, understand the existing Python implementation and tests.

Identify clean interface boundaries where a CUDA-backed version can be slotted in.



2. Design the replacement module

Mirror the public API of the CPU module (functions, class names, constructor signatures).

Decide the internal GPU data layout:

Contiguous SoA or AoS arrays optimised for vector arithmetic and coalesced loads.

Pre-allocated device buffers reused across steps.




3. Implement CUDA kernels and/or linear algebra paths

Move hot loops (SPH density, forces, neighbour search, gravity, integration steps) into CUDA kernels.

For Newtonian gravity, prototype both:

A classic N-body kernel with shared-memory tiling and warp-level reductions.

A vectorised / block-matrix formulation if it fits in memory and can leverage efficient linear algebra primitives.


Minimise host–device transfers and kernel launch overheads.



4. Integrate as a backend

Expose a module-level class or factory conforming to the existing interface.

Implement a small, explicit mechanism to choose CPU vs CUDA backends (e.g. config flag, simple dispatcher).

Ensure Newtonian and GR modes both work with the CUDA backend where required.



5. Testing and validation

Add tests that compare CUDA against CPU implementations:

Same inputs, numerically close outputs (tolerances appropriate to FP32).


Cover both “vectorised linear algebra” paths and “pure kernel” paths where both exist.

Include small-N regression tests suitable for CI.




Deliverables per module

For each ported module, provide:

A CUDA-backed implementation file (e.g. *_cuda.py) that can act as a direct module-level replacement.

Minimal integration glue to select CPU or CUDA backends.

Unit tests comparing CPU vs CUDA behaviour.

Short notes (e.g. NOTES.md) describing:

Data layout, kernel entry points, and expected shapes/dtypes.

Whether a matrix/linear-algebra formulation was used, evaluated, or rejected, and why.

Any known numerical differences vs the CPU version



- Don't use absolute filepaths. Export raw data to the `output` folder and all visualisations and summary documents to `results`
- Catch and mitigate errors from extreme interactions. Identify and flag bad data and throw warnings for debug. Filter visualisation scaling for extreme outliers

# Reviewer Sub-Agent Prompt - Not for programming agents
**ROLE:** Unbiased AI-to-AI code reviewer agent. Succinct and blunt.

**SCOPE:** Review code in the directory containing this file. To you nothing else exists.

**GOAL:** Verify the high level functionality and logic of the module

**TASKS:**
  1. Identify correctness and robustness issues.
  2. Flag missing context or hidden assumptions.
  3. Suggest minimal, concrete improvements.
  4. Add to or create the local NOTES.md




