"""
Global energy diagnostics for TDE-SPH simulations (TASK-022).

Implements comprehensive energy tracking for both Newtonian and GR modes,
including kinetic, potential (BH + self-gravity), internal (thermal + radiation),
and cumulative radiated energies.

Supports REQ-009: energy accounting & luminosity.

References:
    Liptai & Price (2019) - GRSPH energy tracking
    Tejeda et al. (2017) - GR energy definitions
    Price (2012) - SPH energy conservation
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float32]


@dataclass
class EnergyComponents:
    """
    Container for energy components at a single snapshot.

    All energies in code units (G = c = M_BH = 1 for GR).

    Attributes
    ----------
    time : float
        Simulation time.
    kinetic : float
        Total kinetic energy (Newtonian) or 4-velocity-based (GR).
    potential_bh : float
        Potential energy from black hole gravity.
    potential_self : float
        Potential energy from self-gravity.
    internal_thermal : float
        Internal thermal energy (gas component).
    internal_radiation : float
        Internal radiation energy (if separated in EOS).
    radiated_cumulative : float
        Cumulative energy radiated away.
    total : float
        Total energy (sum of all components - radiated).
    conservation_error : float
        Relative energy conservation error: (E - E_0) / E_0.
    """
    time: float = 0.0
    kinetic: float = 0.0
    potential_bh: float = 0.0
    potential_self: float = 0.0
    internal_thermal: float = 0.0
    internal_radiation: float = 0.0
    radiated_cumulative: float = 0.0
    total: float = 0.0
    conservation_error: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'time': self.time,
            'kinetic': self.kinetic,
            'potential_bh': self.potential_bh,
            'potential_self': self.potential_self,
            'internal_thermal': self.internal_thermal,
            'internal_radiation': self.internal_radiation,
            'radiated_cumulative': self.radiated_cumulative,
            'total': self.total,
            'conservation_error': self.conservation_error
        }


class EnergyDiagnostics:
    """
    Global energy bookkeeping for TDE-SPH simulations.

    Tracks kinetic, potential (BH + self-gravity), internal (thermal + radiation),
    and cumulative radiated energies. Handles both Newtonian and GR modes.

    Parameters
    ----------
    mode : str
        Simulation mode: "Newtonian" or "GR".
    bh_mass : float, optional
        Black hole mass in code units (default 1.0).
    metric : optional
        Metric object for GR simulations (default None).
    initial_energy : float, optional
        Initial total energy for conservation tracking (default None, auto-computed).

    Attributes
    ----------
    mode : str
        Simulation mode.
    bh_mass : float
        Black hole mass.
    metric : optional
        Spacetime metric for GR.
    initial_energy : float
        Initial total energy (set on first call).
    energy_history : list
        Time series of EnergyComponents.

    Notes
    -----
    Energy definitions:

    **Newtonian mode**:
    - E_kin = Σ (1/2) m v²
    - E_pot_BH = -Σ m G M_BH / r
    - E_pot_self = (1/2) Σᵢ Σⱼ≠ᵢ (-G m_i m_j / |r_i - r_j|)
    - E_int = Σ m u

    **GR mode**:
    - E_kin = Hamiltonian kinetic term from 4-velocity
    - E_pot_BH = effective potential from metric (coordinate-dependent)
    - E_pot_self = Newtonian approximation (hybrid model)
    - E_int = Σ m u (proper internal energy)

    Conservation error:
    - ΔE/E₀ = (E_total(t) - E_total(t=0)) / E_total(t=0)
    - Expect |ΔE/E₀| < 0.001 for good integrators in adiabatic runs
    """

    def __init__(
        self,
        mode: str = "Newtonian",
        bh_mass: float = 1.0,
        metric: Optional[object] = None,
        initial_energy: Optional[float] = None
    ):
        """
        Initialize energy diagnostics.

        Parameters
        ----------
        mode : str
            "Newtonian" or "GR".
        bh_mass : float
            Black hole mass (default 1.0 in code units).
        metric : optional
            Metric object for GR mode.
        initial_energy : float, optional
            Initial total energy (auto-computed if None).
        """
        self.mode = mode
        self.bh_mass = bh_mass
        self.metric = metric
        self.initial_energy = initial_energy
        self.energy_history = []

        # Validate mode
        if mode not in ["Newtonian", "GR"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'Newtonian' or 'GR'.")

        if mode == "GR" and metric is None:
            raise ValueError("GR mode requires a metric object.")

    def compute_kinetic_energy(
        self,
        masses: NDArrayFloat,
        velocities: NDArrayFloat
    ) -> float:
        """
        Compute total kinetic energy.

        Parameters
        ----------
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        velocities : NDArrayFloat, shape (N, 3)
            Particle velocities.

        Returns
        -------
        E_kin : float
            Total kinetic energy.

        Notes
        -----
        Newtonian: E_kin = Σ (1/2) m v²
        GR: E_kin ≈ Σ (1/2) m v² (coordinate velocity approximation)
        For full GR, should use 4-velocity formulation (future enhancement).
        """
        v_squared = np.sum(velocities**2, axis=1)
        E_kin = 0.5 * np.sum(masses * v_squared)
        return float(E_kin)

    def compute_potential_bh(
        self,
        masses: NDArrayFloat,
        positions: NDArrayFloat
    ) -> float:
        """
        Compute potential energy from black hole gravity.

        Parameters
        ----------
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.

        Returns
        -------
        E_pot_BH : float
            Black hole potential energy.

        Notes
        -----
        Newtonian: E_pot_BH = -Σ m G M_BH / r = -Σ m M_BH / r (G=1)
        GR: E_pot_BH ≈ Newtonian for coordinate potential (approximate)
        Full GR should use proper Hamiltonian formulation (future).
        """
        r = np.sqrt(np.sum(positions**2, axis=1))
        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)

        if self.mode == "Newtonian":
            # Φ = -G M_BH / r = -M_BH / r (G=1)
            E_pot_BH = -np.sum(masses * self.bh_mass / r_safe)
        else:  # GR
            # Approximate with Newtonian potential (hybrid model assumption)
            # Full GR would use metric-dependent effective potential
            E_pot_BH = -np.sum(masses * self.bh_mass / r_safe)

        return float(E_pot_BH)

    def compute_potential_self(
        self,
        masses: NDArrayFloat,
        positions: NDArrayFloat,
        softening: Optional[NDArrayFloat] = None
    ) -> float:
        """
        Compute self-gravity potential energy.

        Parameters
        ----------
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.
        softening : NDArrayFloat, shape (N,), optional
            Softening lengths (default None, uses 1e-3).

        Returns
        -------
        E_pot_self : float
            Self-gravity potential energy.

        Notes
        -----
        E_pot_self = (1/2) Σᵢ Σⱼ≠ᵢ (-G m_i m_j / sqrt(r_ij² + ε²))
        where ε is softening length.

        For N particles, this is O(N²). For large N, consider tree approximation.
        Currently using direct summation for accuracy.
        """
        N = len(masses)

        if N == 0:
            return 0.0

        if softening is None:
            softening = np.ones(N, dtype=np.float32) * 1e-3

        # Compute pairwise distances
        # r_ij = positions[i] - positions[j]
        # For efficiency, use vectorized operations
        # This is O(N²) in memory and compute - acceptable for N < 10^6

        E_pot_self = 0.0

        # Vectorized approach: compute upper triangle of distance matrix
        for i in range(N):
            r_ij = positions[i+1:] - positions[i]
            dist = np.sqrt(np.sum(r_ij**2, axis=1) + softening[i]**2)
            E_pot_self += -np.sum(masses[i] * masses[i+1:] / dist)

        # Factor of 1 (not 1/2) because we only summed upper triangle
        return float(E_pot_self)

    def compute_internal_energy(
        self,
        masses: NDArrayFloat,
        internal_energies: NDArrayFloat,
        eos: Optional[object] = None
    ) -> Tuple[float, float]:
        """
        Compute total internal energy (thermal + radiation if separated).

        Parameters
        ----------
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        internal_energies : NDArrayFloat, shape (N,)
            Specific internal energies u.
        eos : optional
            EOS object (for separating gas/radiation if RadiationGasEOS).

        Returns
        -------
        E_thermal : float
            Thermal (gas) internal energy.
        E_radiation : float
            Radiation internal energy (0 if not separated).

        Notes
        -----
        E_int_total = Σ m u

        If using RadiationGasEOS, can separate into gas and radiation components.
        Otherwise, all internal energy is thermal.
        """
        E_total = np.sum(masses * internal_energies)

        # Check if EOS supports gas/radiation separation
        if eos is not None and hasattr(eos, 'gas_pressure_fraction'):
            # Estimate thermal vs radiation split (approximate)
            # For now, return total as thermal (future: implement proper split)
            E_thermal = E_total
            E_radiation = 0.0
        else:
            E_thermal = E_total
            E_radiation = 0.0

        return float(E_thermal), float(E_radiation)

    def compute_all_energies(
        self,
        time: float,
        masses: NDArrayFloat,
        positions: NDArrayFloat,
        velocities: NDArrayFloat,
        internal_energies: NDArrayFloat,
        radiated_cumulative: float = 0.0,
        eos: Optional[object] = None,
        softening: Optional[NDArrayFloat] = None
    ) -> EnergyComponents:
        """
        Compute all energy components at current snapshot.

        Parameters
        ----------
        time : float
            Current simulation time.
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        positions : NDArrayFloat, shape (N, 3)
            Particle positions.
        velocities : NDArrayFloat, shape (N, 3)
            Particle velocities.
        internal_energies : NDArrayFloat, shape (N,)
            Specific internal energies.
        radiated_cumulative : float, optional
            Cumulative radiated energy (default 0.0).
        eos : optional
            EOS object.
        softening : NDArrayFloat, optional
            Softening lengths.

        Returns
        -------
        energies : EnergyComponents
            All energy components.

        Notes
        -----
        Computes:
        - E_kin: kinetic energy
        - E_pot_BH: BH potential energy
        - E_pot_self: self-gravity potential energy
        - E_int_thermal, E_int_rad: internal energies
        - E_total = E_kin + E_pot_BH + E_pot_self + E_int - E_radiated
        - ΔE/E₀: conservation error
        """
        # Compute individual components
        E_kin = self.compute_kinetic_energy(masses, velocities)
        E_pot_BH = self.compute_potential_bh(masses, positions)
        E_pot_self = self.compute_potential_self(masses, positions, softening)
        E_thermal, E_radiation = self.compute_internal_energy(masses, internal_energies, eos)

        # Total energy
        E_total = E_kin + E_pot_BH + E_pot_self + E_thermal + E_radiation - radiated_cumulative

        # Set initial energy on first call
        if self.initial_energy is None:
            self.initial_energy = E_total

        # Conservation error
        if self.initial_energy != 0:
            conservation_error = (E_total - self.initial_energy) / self.initial_energy
        else:
            conservation_error = 0.0

        # Create energy components object
        energies = EnergyComponents(
            time=time,
            kinetic=E_kin,
            potential_bh=E_pot_BH,
            potential_self=E_pot_self,
            internal_thermal=E_thermal,
            internal_radiation=E_radiation,
            radiated_cumulative=radiated_cumulative,
            total=E_total,
            conservation_error=conservation_error
        )

        return energies

    def log_energy_history(self, energies: EnergyComponents):
        """
        Append energy components to history.

        Parameters
        ----------
        energies : EnergyComponents
            Energy components to log.
        """
        self.energy_history.append(energies)

    def get_energy_history_arrays(self) -> Dict[str, NDArrayFloat]:
        """
        Get energy history as arrays for plotting.

        Returns
        -------
        history : dict
            Dictionary with keys: 'time', 'kinetic', 'potential_bh', 'potential_self',
            'internal_thermal', 'internal_radiation', 'radiated_cumulative', 'total',
            'conservation_error', each containing np.array of time series.
        """
        if not self.energy_history:
            return {key: np.array([]) for key in [
                'time', 'kinetic', 'potential_bh', 'potential_self',
                'internal_thermal', 'internal_radiation', 'radiated_cumulative',
                'total', 'conservation_error'
            ]}

        history = {
            'time': np.array([e.time for e in self.energy_history]),
            'kinetic': np.array([e.kinetic for e in self.energy_history]),
            'potential_bh': np.array([e.potential_bh for e in self.energy_history]),
            'potential_self': np.array([e.potential_self for e in self.energy_history]),
            'internal_thermal': np.array([e.internal_thermal for e in self.energy_history]),
            'internal_radiation': np.array([e.internal_radiation for e in self.energy_history]),
            'radiated_cumulative': np.array([e.radiated_cumulative for e in self.energy_history]),
            'total': np.array([e.total for e in self.energy_history]),
            'conservation_error': np.array([e.conservation_error for e in self.energy_history])
        }

        return history

    def compute_energy_conservation_error(self) -> float:
        """
        Get current energy conservation error.

        Returns
        -------
        error : float
            Relative energy conservation error: (E_current - E_initial) / E_initial.

        Notes
        -----
        Returns 0.0 if no energies have been logged yet.
        """
        if not self.energy_history:
            return 0.0

        return self.energy_history[-1].conservation_error

    def clear_history(self):
        """Clear energy history."""
        self.energy_history = []

    def __repr__(self) -> str:
        """String representation."""
        n_snapshots = len(self.energy_history)
        error = self.compute_energy_conservation_error()
        return f"EnergyDiagnostics(mode={self.mode}, snapshots={n_snapshots}, ΔE/E={error:.2e})"
