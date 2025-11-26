"""
Gravity module: Newtonian and relativistic gravity solvers.
"""

from .newtonian import NewtonianGravity
from .relativistic_orbit import RelativisticGravitySolver
from .pseudo_newtonian import PseudoNewtonianGravity
from .barnes_hut import BarnesHutGravity

# GPU Barnes-Hut solver
try:
    from .barnes_hut_gpu import BarnesHutGravityGPU
except ImportError:
    BarnesHutGravityGPU = None

__all__ = [
    "NewtonianGravity",
    "RelativisticGravitySolver",
    "PseudoNewtonianGravity",
    "BarnesHutGravity",
    "BarnesHutGravityGPU",
]

