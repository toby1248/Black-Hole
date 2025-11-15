# CLAUDE Instructions — metric module

Role: Implement spacetime metrics and geodesic helpers.

DO:
- Work ONLY inside `tde_sph/metric`.
- Implement Minkowski, Schwarzschild and Kerr metrics, including metric tensors, inverse metrics, Christoffel symbols and geodesic-acceleration helpers.
- Provide numerically stable, well-tested building blocks used by gravity/integration modules.

DO NOT:
- Implement self-gravity or SPH forces.
- Hard-code simulation loops or I/O.
- Open or modify anything under the `prompts/` folder.
