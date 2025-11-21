#!/usr/bin/env python3
"""
Simulation Thread for Background Execution (TASK2).

Runs SPH simulations in a background QThread, emitting progress signals
with comprehensive diagnostic data for real-time GUI updates.

Features:
- Thread-safe simulation execution
- Progress signals with particle statistics, performance metrics
- Coordinate/metric data for GR simulations
- Graceful stop support

Author: TDE-SPH Development Team
Date: 2025-11-21
"""

import time
from typing import Dict, Optional, Any
import numpy as np

try:
    from PyQt6.QtCore import QThread, pyqtSignal
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal
    PYQT_VERSION = 5


# Check for CUDA availability
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


class SimulationThread(QThread):
    """
    Background thread for running SPH simulations.

    Emits comprehensive progress data including particle statistics,
    performance metrics, and coordinate/metric data for GR simulations.

    Signals:
        progress_updated (dict): Emitted with diagnostic data dict
        simulation_finished (): Emitted when simulation completes
        error_occurred (str): Emitted on simulation error
        step_completed (int, float): Emitted after each step (step_num, sim_time)

    Usage:
        thread = SimulationThread(config, simulation)
        thread.progress_updated.connect(self.update_diagnostics)
        thread.simulation_finished.connect(self.on_finished)
        thread.start()
    """

    # Signals
    progress_updated = pyqtSignal(dict)
    simulation_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    step_completed = pyqtSignal(int, float)  # step_number, simulation_time

    def __init__(self, config: Dict[str, Any], simulation: Optional[Any] = None, parent=None):
        """
        Initialize SimulationThread.

        Parameters:
            config: Simulation configuration dictionary
            simulation: Optional pre-configured Simulation object
            parent: Parent QObject
        """
        super().__init__(parent)

        self.config = config
        self.simulation = simulation
        self._stop_requested = False

        # Performance tracking
        self._start_time: float = 0.0
        self._step_count: int = 0
        self._last_report_time: float = 0.0
        self._report_interval: float = 0.5  # Report every 0.5 seconds

    def request_stop(self):
        """Request graceful stop of simulation."""
        self._stop_requested = True

    def run(self):
        """Run simulation in background thread."""
        try:
            self._start_time = time.time()
            self._step_count = 0
            self._stop_requested = False

            # Initialize simulation if not provided
            if self.simulation is None:
                self.simulation = self._create_simulation()

            # Run simulation loop
            self._run_simulation_loop()

            self.simulation_finished.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _create_simulation(self) -> Any:
        """Create simulation from config."""
        # Import here to avoid circular imports
        try:
            from tde_sph.core.simulation import Simulation
            return Simulation.from_config(self.config)
        except ImportError:
            # Return mock simulation for testing
            return MockSimulation(self.config)

    def _run_simulation_loop(self):
        """Execute the main simulation loop."""
        sim = self.simulation
        t_end = self.config.get('integration', {}).get('t_end', 100.0)

        while not self._stop_requested:
            # Advance one step
            try:
                sim.step()
            except AttributeError:
                # Mock simulation - just increment time
                if hasattr(sim, 'current_time'):
                    sim.current_time += sim.dt

            self._step_count += 1

            # Emit step completed
            current_time = getattr(sim, 'current_time', self._step_count * 0.01)
            self.step_completed.emit(self._step_count, current_time)

            # Report progress at intervals
            wall_time = time.time() - self._start_time
            if wall_time - self._last_report_time >= self._report_interval:
                self._report_progress()
                self._last_report_time = wall_time

            # Check end condition
            if current_time >= t_end:
                break

        # Final progress report
        self._report_progress()

    def _report_progress(self):
        """Collect and emit comprehensive diagnostic data."""
        sim = self.simulation
        wall_time = time.time() - self._start_time

        # Get particles object
        particles = getattr(sim, 'particles', None)

        # Build comprehensive stats dict
        stats = self._build_stats_dict(sim, particles, wall_time)

        # Emit signal
        self.progress_updated.emit(stats)

    def _build_stats_dict(self, sim: Any, particles: Any, wall_time: float) -> Dict:
        """
        Build comprehensive statistics dictionary for GUI.

        Parameters:
            sim: Simulation object
            particles: ParticleSystem object
            wall_time: Elapsed wall-clock time in seconds

        Returns:
            Dictionary with all diagnostic data
        """
        stats = {}

        # Performance metrics
        stats['performance'] = {
            'wall_time': wall_time,
            'sim_time': getattr(sim, 'current_time', 0.0),
            'step_count': self._step_count,
            'dt': getattr(sim, 'dt', 0.0),
            'steps_per_sec': self._step_count / wall_time if wall_time > 0 else 0.0,
            'gpu_available': HAS_CUDA,
            'memory_mb': self._estimate_memory_usage(particles),
        }

        # Particle statistics
        if particles is not None:
            stats['particle_stats'] = self._compute_particle_stats(particles)

        # Basic stats (for legacy widgets)
        if particles is not None:
            stats['n_particles'] = particles.n_particles
            stats['total_mass'] = float(np.sum(particles.masses))

        # Energy data
        if particles is not None:
            stats['energies'] = self._compute_energies(sim, particles)

        # Coordinate/metric data for GR simulations
        metric_type = getattr(sim, 'metric_type', None)
        if metric_type and metric_type != 'Minkowski':
            stats['coordinate_metric'] = self._compute_coordinate_data(sim, particles)

        return stats

    def _compute_particle_stats(self, particles: Any) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for all particle quantities.

        Returns:
            Dict with quantity name -> {'min': ..., 'max': ..., 'mean': ..., 'std': ...}
        """
        stats = {}

        # Define quantities to compute stats for
        quantities = [
            ('density', 'density'),
            ('pressure', 'pressure'),
            ('temperature', 'temperature'),
            ('sound_speed', 'sound_speed'),
            ('smoothing_length', 'smoothing_length'),
            ('internal_energy', 'internal_energy'),
        ]

        for name, attr in quantities:
            data = getattr(particles, attr, None)
            if data is not None and len(data) > 0:
                stats[name] = {
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                }

        # Velocity magnitude (computed from velocities array)
        velocities = getattr(particles, 'velocities', None)
        if velocities is not None and len(velocities) > 0:
            vmag = np.linalg.norm(velocities, axis=1)
            stats['velocity_magnitude'] = {
                'min': float(np.min(vmag)),
                'max': float(np.max(vmag)),
                'mean': float(np.mean(vmag)),
                'std': float(np.std(vmag)),
            }

        return stats

    def _compute_energies(self, sim: Any, particles: Any) -> Dict[str, float]:
        """Compute energy components."""
        energies = {}

        # Kinetic energy
        if hasattr(particles, 'kinetic_energy'):
            energies['kinetic'] = float(particles.kinetic_energy())

        # Thermal/internal energy
        if hasattr(particles, 'thermal_energy'):
            energies['internal'] = float(particles.thermal_energy())

        # Potential energy (from simulation)
        energies['potential'] = getattr(sim, 'potential_energy', 0.0)

        # Total energy
        energies['total'] = (
            energies.get('kinetic', 0.0) +
            energies.get('potential', 0.0) +
            energies.get('internal', 0.0)
        )

        # Conservation error
        initial_energy = getattr(sim, 'initial_total_energy', energies['total'])
        if initial_energy != 0:
            energies['error'] = (energies['total'] - initial_energy) / abs(initial_energy)
        else:
            energies['error'] = 0.0

        return energies

    def _compute_coordinate_data(self, sim: Any, particles: Any) -> Dict[str, Any]:
        """
        Compute coordinate and metric data for GR simulations.

        Returns:
            Dict with GR-specific information
        """
        data = {
            'metric_type': getattr(sim, 'metric_type', 'Unknown'),
            'coordinate_system': getattr(sim, 'coordinate_system', 'Cartesian'),
            'bh_mass': getattr(sim, 'black_hole_mass', 0.0),
            'bh_spin': getattr(sim, 'black_hole_spin', 0.0),
        }

        # Compute particle distances from black hole
        if particles is not None:
            positions = getattr(particles, 'positions', None)
            if positions is not None and len(positions) > 0:
                # Distance from origin (black hole position)
                distances = np.linalg.norm(positions, axis=1)
                data['r_min'] = float(np.min(distances))
                data['r_max'] = float(np.max(distances))
                data['r_mean'] = float(np.mean(distances))

                # ISCO radius (depends on spin)
                spin = data['bh_spin']
                if abs(spin) < 1e-6:
                    # Schwarzschild ISCO
                    isco_radius = 6.0
                else:
                    # Kerr ISCO (simplified)
                    # r_isco = M * (3 + Z2 - sqrt((3-Z1)(3+Z1+2*Z2)))
                    # For prograde orbits with spin a
                    z1 = 1 + (1 - spin**2)**(1/3) * ((1 + spin)**(1/3) + (1 - spin)**(1/3))
                    z2 = np.sqrt(3 * spin**2 + z1**2)
                    isco_radius = 3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))

                data['isco_radius'] = float(isco_radius)
                data['particles_within_isco'] = int(np.sum(distances < isco_radius))

        return data

    def _estimate_memory_usage(self, particles: Any) -> float:
        """
        Estimate memory usage in MB.

        Parameters:
            particles: ParticleSystem object

        Returns:
            Estimated memory usage in megabytes
        """
        if particles is None:
            return 0.0

        n = getattr(particles, 'n_particles', 0)
        if n == 0:
            return 0.0

        # Each particle has multiple arrays (all float32 = 4 bytes)
        # positions: 3, velocities: 3, masses: 1, density: 1, pressure: 1
        # sound_speed: 1, smoothing_length: 1, internal_energy: 1, temperature: 1
        floats_per_particle = 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # = 13
        bytes_per_particle = floats_per_particle * 4  # float32

        total_bytes = n * bytes_per_particle
        return total_bytes / (1024 * 1024)  # Convert to MB


class MockSimulation:
    """
    Mock simulation for testing SimulationThread without actual physics.

    Generates synthetic data that changes over time.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize mock simulation."""
        self.config = config
        self.current_time = 0.0
        self.dt = config.get('integration', {}).get('timestep', 0.01)

        # Create mock particles
        n_particles = config.get('particles', {}).get('count', 1000)
        self.particles = MockParticleSystem(n_particles)

        # GR parameters
        self.metric_type = config.get('simulation', {}).get('mode', 'newtonian')
        if self.metric_type == 'newtonian':
            self.metric_type = 'Minkowski'
        elif self.metric_type == 'schwarzschild':
            self.metric_type = 'Schwarzschild'
        elif self.metric_type == 'kerr':
            self.metric_type = 'Kerr'

        self.coordinate_system = 'Boyer-Lindquist'
        self.black_hole_mass = config.get('black_hole', {}).get('mass', 1e6)
        self.black_hole_spin = config.get('black_hole', {}).get('spin', 0.0)

        self.potential_energy = -1.0
        self.initial_total_energy = None

    def step(self):
        """Advance mock simulation by one timestep."""
        self.current_time += self.dt

        # Update mock particle data with time-varying values
        self.particles.update(self.current_time)

        # Update potential energy with small variation
        self.potential_energy = -1.0 + np.sin(self.current_time * 0.1) * 0.1

        # Set initial energy on first step
        if self.initial_total_energy is None:
            self.initial_total_energy = (
                self.particles.kinetic_energy() +
                self.potential_energy +
                self.particles.thermal_energy()
            )


class MockParticleSystem:
    """Mock particle system for testing."""

    def __init__(self, n_particles: int):
        """Initialize mock particles."""
        self.n_particles = n_particles

        # Initialize arrays
        self.positions = np.random.randn(n_particles, 3).astype(np.float32) * 10
        self.velocities = np.random.randn(n_particles, 3).astype(np.float32) * 0.1
        self.masses = np.ones(n_particles, dtype=np.float32) / n_particles
        self.density = np.ones(n_particles, dtype=np.float32) * 1e-7
        self.pressure = np.ones(n_particles, dtype=np.float32) * 1e-10
        self.temperature = np.ones(n_particles, dtype=np.float32) * 5000
        self.sound_speed = np.ones(n_particles, dtype=np.float32) * 0.05
        self.smoothing_length = np.ones(n_particles, dtype=np.float32) * 0.1
        self.internal_energy = np.ones(n_particles, dtype=np.float32) * 1e6

    def update(self, time: float):
        """Update mock data with time variation."""
        # Add small variations to simulate evolution
        self.density *= (1.0 + np.random.randn(self.n_particles).astype(np.float32) * 0.01)
        self.temperature *= (1.0 + np.random.randn(self.n_particles).astype(np.float32) * 0.01)
        self.pressure = self.density * self.temperature * 1e-4  # Ideal gas-like

    def kinetic_energy(self) -> float:
        """Compute kinetic energy."""
        v_squared = np.sum(self.velocities**2, axis=1)
        return float(0.5 * np.sum(self.masses * v_squared))

    def thermal_energy(self) -> float:
        """Compute thermal energy."""
        return float(np.sum(self.masses * self.internal_energy))
