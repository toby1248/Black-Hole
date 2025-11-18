"""
Energy diagnostics and thermodynamic state tracking for TDE-SPH simulations.

Implements REQ-009: comprehensive energy accounting (kinetic, potential, internal)
and global thermodynamic quantities for validation and analysis.

This module provides tools to:
- Compute energy components (kinetic, potential, internal)
- Track energy conservation over time
- Compute global quantities (mass, momentum, angular momentum)
- Provide thermodynamic statistics (temperature, pressure, entropy proxies)
- Support both Newtonian and GR modes

Usage:
    >>> diagnostics = EnergyDiagnostics()
    >>> energy_dict = diagnostics.compute(particles, gravity_solver, eos)
    >>> print(f"Total energy: {energy_dict['E_total']:.3e}")
"""

import numpy as np
from typing import Dict, Any, Optional, List
import numpy.typing as npt
from .interfaces import GravitySolver, EOS, Metric

# Type aliases
NDArrayFloat = npt.NDArray[np.float32]


class EnergyDiagnostics:
    """
    Comprehensive energy and thermodynamic state diagnostics.

    Computes all energy components, global conserved quantities, and
    thermodynamic statistics for TDE-SPH simulations.

    Supports both Newtonian and GR modes.

    Attributes
    ----------
    history : List[Dict[str, Any]]
        Time-series of diagnostic snapshots.
    E_initial : Optional[float]
        Initial total energy (for conservation tracking).

    Methods
    -------
    compute(particles, gravity_solver, eos, metric=None)
        Compute all diagnostics for current state.
    compute_kinetic_energy(particles)
        Compute total kinetic energy.
    compute_potential_energy(particles, gravity_solver, metric=None)
        Compute total gravitational potential energy.
    compute_internal_energy(particles, eos)
        Compute internal energy (gas + radiation if applicable).
    compute_global_quantities(particles)
        Compute conserved quantities (mass, momentum, angular momentum).
    append_to_history(time, diagnostics)
        Add current diagnostics to time-series.
    get_time_series(quantity)
        Extract time-series of a specific quantity.
    energy_conservation_metric()
        Compute ΔE/E₀ conservation metric.
    """

    def __init__(self):
        """Initialize energy diagnostics tracker."""
        self.history: List[Dict[str, Any]] = []
        self.E_initial: Optional[float] = None

    def compute(
        self,
        particles: Dict[str, NDArrayFloat],
        gravity_solver: GravitySolver,
        eos: EOS,
        metric: Optional[Metric] = None,
        bh_mass: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute all energy and thermodynamic diagnostics.

        Parameters
        ----------
        particles : Dict[str, NDArrayFloat]
            Particle data dictionary with keys:
            - 'positions': shape (N, 3)
            - 'velocities': shape (N, 3)
            - 'masses': shape (N,)
            - 'internal_energy': shape (N,)
            - 'density': shape (N,)
            - 'smoothing_length': shape (N,)
        gravity_solver : GravitySolver
            Gravity solver instance.
        eos : EOS
            Equation of state instance.
        metric : Optional[Metric]
            Spacetime metric (for GR mode).
        bh_mass : float, optional
            Black hole mass in code units (default: 1.0).

        Returns
        -------
        diagnostics : Dict[str, Any]
            Dictionary containing:
            - Energy components: E_kinetic, E_potential, E_internal_total,
              E_internal_gas, E_internal_radiation, E_total
            - Global quantities: total_mass, linear_momentum, angular_momentum
            - Thermodynamic stats: T_min, T_max, T_mean, T_median, P_mean, etc.
            - GR quantities (if metric provided): N_inside_ISCO, etc.
        """
        diagnostics = {}

        # Energy components
        diagnostics['E_kinetic'] = self.compute_kinetic_energy(particles)
        diagnostics['E_potential'] = self.compute_potential_energy(
            particles, gravity_solver, metric, bh_mass
        )

        internal_energy_dict = self.compute_internal_energy(particles, eos)
        diagnostics.update(internal_energy_dict)

        # Total energy
        diagnostics['E_total'] = (
            diagnostics['E_kinetic'] +
            diagnostics['E_potential'] +
            diagnostics['E_internal_total']
        )

        # Global quantities
        global_quantities = self.compute_global_quantities(particles)
        diagnostics.update(global_quantities)

        # Thermodynamic statistics
        thermo_stats = self.compute_thermodynamic_statistics(particles, eos)
        diagnostics.update(thermo_stats)

        # GR-specific diagnostics
        if metric is not None:
            gr_diagnostics = self.compute_gr_diagnostics(particles, metric, bh_mass)
            diagnostics.update(gr_diagnostics)

        # Set initial energy if not set
        if self.E_initial is None:
            self.E_initial = diagnostics['E_total']

        # Energy conservation metric
        if self.E_initial is not None and self.E_initial != 0:
            diagnostics['energy_conservation'] = (
                (diagnostics['E_total'] - self.E_initial) / abs(self.E_initial)
            )
        else:
            diagnostics['energy_conservation'] = 0.0

        return diagnostics

    def compute_kinetic_energy(self, particles: Dict[str, NDArrayFloat]) -> float:
        """
        Compute total kinetic energy E_kin = Σ(½ m v²).

        Parameters
        ----------
        particles : Dict[str, NDArrayFloat]
            Particle data with 'masses' and 'velocities'.

        Returns
        -------
        E_kinetic : float
            Total kinetic energy.
        """
        masses = particles['masses']
        velocities = particles['velocities']

        v_squared = np.sum(velocities**2, axis=1)
        E_kinetic = 0.5 * np.sum(masses * v_squared)

        return float(E_kinetic)

    def compute_potential_energy(
        self,
        particles: Dict[str, NDArrayFloat],
        gravity_solver: GravitySolver,
        metric: Optional[Metric] = None,
        bh_mass: float = 1.0
    ) -> float:
        """
        Compute total gravitational potential energy.

        Includes:
        - Self-gravity potential (from gravity solver)
        - BH gravitational potential (Newtonian or GR)

        Parameters
        ----------
        particles : Dict[str, NDArrayFloat]
            Particle data.
        gravity_solver : GravitySolver
            Gravity solver for self-gravity potential.
        metric : Optional[Metric]
            Spacetime metric (for GR BH potential).
        bh_mass : float
            Black hole mass.

        Returns
        -------
        E_potential : float
            Total potential energy.

        Notes
        -----
        - Self-gravity contribution from gravity_solver.compute_potential()
        - BH contribution: -G M_BH m / r (Newtonian) or metric-dependent (GR)
        - Factor of 1/2 for self-gravity to avoid double-counting
        """
        positions = particles['positions']
        masses = particles['masses']
        smoothing_lengths = particles['smoothing_length']

        # Self-gravity potential
        phi_self = gravity_solver.compute_potential(
            positions, masses, smoothing_lengths
        )
        E_self = 0.5 * np.sum(masses * phi_self)  # Factor 1/2 to avoid double counting

        # BH potential
        r = np.linalg.norm(positions, axis=1)
        r_safe = np.maximum(r, 1e-10)  # Avoid singularity

        if metric is None:
            # Newtonian BH potential: φ = -GM/r (with G=1 in code units)
            phi_bh = -bh_mass / r_safe
        else:
            # GR effective potential (simplified)
            # For Schwarzschild: φ_eff ≈ -M/r + L²/(2r²) - M L²/r³
            # Here we use simple Newtonian as placeholder
            # Full GR would require proper Hamiltonian computation
            phi_bh = -bh_mass / r_safe

        E_bh = np.sum(masses * phi_bh)

        E_potential = E_self + E_bh

        return float(E_potential)

    def compute_internal_energy(
        self,
        particles: Dict[str, NDArrayFloat],
        eos: EOS
    ) -> Dict[str, float]:
        """
        Compute internal energy components.

        For IdealGas: only total internal energy
        For RadiationGas: separates gas and radiation components

        Parameters
        ----------
        particles : Dict[str, NDArrayFloat]
            Particle data with 'masses', 'internal_energy', 'density'.
        eos : EOS
            Equation of state instance.

        Returns
        -------
        energy_dict : Dict[str, float]
            Dictionary with:
            - 'E_internal_total': Total internal energy
            - 'E_internal_gas': Gas component (if RadiationGas)
            - 'E_internal_radiation': Radiation component (if RadiationGas)
        """
        masses = particles['masses']
        u = particles['internal_energy']

        # Total internal energy
        E_internal_total = np.sum(masses * u)

        result = {'E_internal_total': float(E_internal_total)}

        # Check if EOS supports energy partitioning (RadiationGas)
        if hasattr(eos, 'gas_energy') and hasattr(eos, 'radiation_energy'):
            density = particles['density']

            u_gas = eos.gas_energy(density, u)
            u_rad = eos.radiation_energy(density, u)

            E_internal_gas = np.sum(masses * u_gas)
            E_internal_radiation = np.sum(masses * u_rad)

            result['E_internal_gas'] = float(E_internal_gas)
            result['E_internal_radiation'] = float(E_internal_radiation)
        else:
            # IdealGas: all internal energy is gas
            result['E_internal_gas'] = float(E_internal_total)
            result['E_internal_radiation'] = 0.0

        return result

    def compute_global_quantities(
        self,
        particles: Dict[str, NDArrayFloat]
    ) -> Dict[str, Any]:
        """
        Compute global conserved quantities.

        Parameters
        ----------
        particles : Dict[str, NDArrayFloat]
            Particle data.

        Returns
        -------
        quantities : Dict[str, Any]
            Dictionary with:
            - 'total_mass': Total mass
            - 'linear_momentum': Linear momentum vector [3]
            - 'angular_momentum': Angular momentum vector [3]
            - 'center_of_mass': Center of mass position [3]
        """
        masses = particles['masses']
        positions = particles['positions']
        velocities = particles['velocities']

        # Total mass
        total_mass = np.sum(masses)

        # Linear momentum
        linear_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)

        # Center of mass
        com = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass

        # Angular momentum L = Σ m (r × v)
        angular_momentum = np.sum(
            masses[:, np.newaxis] * np.cross(positions, velocities),
            axis=0
        )

        return {
            'total_mass': float(total_mass),
            'linear_momentum': linear_momentum.astype(np.float32),
            'angular_momentum': angular_momentum.astype(np.float32),
            'center_of_mass': com.astype(np.float32)
        }

    def compute_thermodynamic_statistics(
        self,
        particles: Dict[str, NDArrayFloat],
        eos: EOS
    ) -> Dict[str, float]:
        """
        Compute thermodynamic statistics.

        Parameters
        ----------
        particles : Dict[str, NDArrayFloat]
            Particle data.
        eos : EOS
            Equation of state.

        Returns
        -------
        stats : Dict[str, float]
            Dictionary with:
            - 'T_min', 'T_max', 'T_mean', 'T_median': Temperature statistics
            - 'P_min', 'P_max', 'P_mean': Pressure statistics
            - 'density_min', 'density_max', 'density_mean': Density statistics
        """
        density = particles['density']
        internal_energy = particles['internal_energy']

        # Temperature
        T = eos.temperature(density, internal_energy)

        # Pressure
        P = eos.pressure(density, internal_energy)

        stats = {
            'T_min': float(np.min(T)),
            'T_max': float(np.max(T)),
            'T_mean': float(np.mean(T)),
            'T_median': float(np.median(T)),
            'P_min': float(np.min(P)),
            'P_max': float(np.max(P)),
            'P_mean': float(np.mean(P)),
            'density_min': float(np.min(density)),
            'density_max': float(np.max(density)),
            'density_mean': float(np.mean(density))
        }

        # Add radiation-specific diagnostics if available
        if hasattr(eos, 'beta_parameter'):
            beta = eos.beta_parameter(density, internal_energy)
            stats['beta_min'] = float(np.min(beta))
            stats['beta_max'] = float(np.max(beta))
            stats['beta_mean'] = float(np.mean(beta))

        return stats

    def compute_gr_diagnostics(
        self,
        particles: Dict[str, NDArrayFloat],
        metric: Metric,
        bh_mass: float
    ) -> Dict[str, Any]:
        """
        Compute GR-specific diagnostics.

        Parameters
        ----------
        particles : Dict[str, NDArrayFloat]
            Particle data.
        metric : Metric
            Spacetime metric.
        bh_mass : float
            Black hole mass.

        Returns
        -------
        gr_diagnostics : Dict[str, Any]
            Dictionary with:
            - 'N_inside_ISCO': Number of particles inside ISCO
            - 'mass_inside_ISCO': Mass inside ISCO
            - 'r_ISCO': ISCO radius
        """
        positions = particles['positions']
        masses = particles['masses']

        r = np.linalg.norm(positions, axis=1)

        # ISCO radius (Schwarzschild: r_ISCO = 6M)
        # For Kerr, this depends on spin and is more complex
        r_ISCO = 6.0 * bh_mass  # Schwarzschild approximation

        inside_ISCO = r < r_ISCO
        N_inside_ISCO = np.sum(inside_ISCO)
        mass_inside_ISCO = np.sum(masses[inside_ISCO]) if N_inside_ISCO > 0 else 0.0

        return {
            'N_inside_ISCO': int(N_inside_ISCO),
            'mass_inside_ISCO': float(mass_inside_ISCO),
            'r_ISCO': float(r_ISCO)
        }

    def append_to_history(self, time: float, diagnostics: Dict[str, Any]) -> None:
        """
        Append diagnostics snapshot to time-series history.

        Parameters
        ----------
        time : float
            Simulation time.
        diagnostics : Dict[str, Any]
            Diagnostics dictionary from compute().
        """
        snapshot = {'time': time, **diagnostics}
        self.history.append(snapshot)

    def get_time_series(self, quantity: str) -> Dict[str, np.ndarray]:
        """
        Extract time-series of a specific quantity.

        Parameters
        ----------
        quantity : str
            Quantity name (e.g., 'E_total', 'T_mean').

        Returns
        -------
        time_series : Dict[str, np.ndarray]
            Dictionary with 'time' and quantity arrays.

        Raises
        ------
        ValueError
            If quantity not found in history or history is empty.
        """
        if not self.history:
            raise ValueError("No diagnostic history available")

        if quantity not in self.history[0]:
            available = list(self.history[0].keys())
            raise ValueError(f"Quantity '{quantity}' not found. Available: {available}")

        times = np.array([snap['time'] for snap in self.history])
        values = np.array([snap[quantity] for snap in self.history])

        return {'time': times, quantity: values}

    def energy_conservation_metric(self) -> float:
        """
        Compute overall energy conservation metric from history.

        Returns
        -------
        conservation : float
            Maximum |ΔE/E₀| over simulation history.

        Notes
        -----
        Returns 0.0 if history is empty or E_initial is zero.
        """
        if not self.history or self.E_initial is None or self.E_initial == 0:
            return 0.0

        E_values = np.array([snap['E_total'] for snap in self.history])
        dE = E_values - self.E_initial
        relative_error = np.abs(dE) / abs(self.E_initial)

        return float(np.max(relative_error))

    def reset_history(self) -> None:
        """Clear diagnostic history and reset initial energy."""
        self.history = []
        self.E_initial = None

    def __repr__(self) -> str:
        """String representation."""
        n_snapshots = len(self.history)
        return f"EnergyDiagnostics(snapshots={n_snapshots})"
