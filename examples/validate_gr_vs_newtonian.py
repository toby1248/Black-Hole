#!/usr/bin/env python3
"""
Validation script for TASK-020: Compare relativistic vs Newtonian trajectories.

This script runs a test particle on a bound orbit around a black hole in both
GR (Schwarzschild) and Newtonian modes, then compares:
- Orbital period
- Periapsis precession
- Energy conservation
- Trajectory differences

References:
- Tejeda et al. (2017) - Hybrid GR approach
- Liptai & Price (2019) - GRSPH validation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tde_sph.sph import ParticleSystem
from tde_sph.gravity import NewtonianGravity, RelativisticGravitySolver
from tde_sph.metric import SchwarzschildMetric
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator
from tde_sph.core import SimulationConfig, Simulation
import tde_sph.config.loaders as config_loaders


def setup_circular_orbit(radius: float, bh_mass: float = 1.0) -> ParticleSystem:
    """
    Set up a single test particle on a circular orbit.

    Parameters
    ----------
    radius : float
        Orbital radius (in units of M_BH)
    bh_mass : float
        Black hole mass (code units, typically 1.0)

    Returns
    -------
    particles : ParticleSystem
        Single particle with circular orbit initial conditions
    """
    # Circular orbit velocity: v = sqrt(GM/r)
    v_circ = np.sqrt(bh_mass / radius)

    # Initial position: particle at (r, 0, 0)
    pos = np.array([[radius, 0.0, 0.0]], dtype=np.float32)

    # Initial velocity: particle moving in y-direction
    vel = np.array([[0.0, v_circ, 0.0]], dtype=np.float32)

    # Negligible mass (test particle)
    mass = np.array([1e-10], dtype=np.float32)

    # Minimal internal energy
    u = np.array([1e-5], dtype=np.float32)

    # Smoothing length (not used for test particle, but required)
    h = np.array([0.1 * radius], dtype=np.float32)

    particles = ParticleSystem(
        n_particles=1,
        positions=pos,
        velocities=vel,
        masses=mass,
        internal_energy=u,
        smoothing_length=h
    )

    return particles


def run_test_orbit(mode: str, n_orbits: int = 3, radius: float = 10.0) -> dict:
    """
    Run a test orbit in either Newtonian or GR mode.

    Parameters
    ----------
    mode : str
        "Newtonian" or "GR"
    n_orbits : int
        Number of orbits to simulate
    radius : float
        Initial orbital radius (in M_BH)

    Returns
    -------
    results : dict
        Trajectory data and diagnostics
    """
    bh_mass = 1.0

    # Estimate orbital period: T = 2π * sqrt(r³/GM)
    T_orbit = 2.0 * np.pi * np.sqrt(radius**3 / bh_mass)
    t_end = n_orbits * T_orbit

    print(f"\n{'='*60}")
    print(f"Running {mode} mode")
    print(f"  Orbital radius: {radius:.2f} M")
    print(f"  Expected period: {T_orbit:.4f} (code units)")
    print(f"  Simulation time: {t_end:.4f} ({n_orbits} orbits)")
    print(f"{'='*60}")

    # Setup particles
    particles = setup_circular_orbit(radius, bh_mass)

    # Setup physics modules
    if mode == "Newtonian":
        metric = None
        gravity = NewtonianGravity(bh_mass=bh_mass, bh_position=np.zeros(3, dtype=np.float32))
        config = SimulationConfig(
            mode="Newtonian",
            t_start=0.0,
            t_end=t_end,
            dt_initial=0.01 * T_orbit,
            cfl_factor=0.3,
            snapshot_interval=0.1 * T_orbit,
            log_interval=0.1 * T_orbit,
            output_dir=f"output/validation_{mode.lower()}",
            verbose=False
        )
    else:  # GR mode
        metric = SchwarzschildMetric(mass=bh_mass)
        gravity = RelativisticGravitySolver(
            bh_mass=bh_mass,
            bh_position=np.zeros(3, dtype=np.float32),
            metric=metric,
            newtonian_self_gravity=False  # Pure test particle
        )
        config = SimulationConfig(
            mode="GR",
            metric_type="schwarzschild",
            bh_mass=bh_mass,
            t_start=0.0,
            t_end=t_end,
            dt_initial=0.01 * T_orbit,
            cfl_factor_gr=0.1,  # Stricter for GR
            snapshot_interval=0.1 * T_orbit,
            log_interval=0.1 * T_orbit,
            output_dir=f"output/validation_{mode.lower()}",
            verbose=False
        )

    eos = IdealGas(gamma=5.0/3.0)
    integrator = LeapfrogIntegrator()

    # Create simulation
    sim = Simulation(
        particles=particles,
        gravity_solver=gravity,
        eos=eos,
        integrator=integrator,
        config=config,
        metric=metric
    )

    # Storage for trajectory
    times = []
    positions = []
    velocities = []
    energies = []

    # Run simulation
    print(f"Integrating {mode} trajectory...")
    step_count = 0
    while sim.state.time < t_end:
        times.append(sim.state.time)
        positions.append(sim.particles.positions[0].copy())
        velocities.append(sim.particles.velocities[0].copy())

        # Compute energy
        E = sim.compute_energies()
        energies.append(E['total'])

        sim.step()
        step_count += 1

        if step_count % 100 == 0:
            print(f"  t = {sim.state.time:.4f} / {t_end:.4f} ({100*sim.state.time/t_end:.1f}%)")

    print(f"  Complete! {step_count} steps")

    # Convert to arrays
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    energies = np.array(energies)

    # Compute orbital radius
    r = np.linalg.norm(positions, axis=1)

    # Compute angular momentum (should be conserved)
    L = np.cross(positions, velocities)
    L_mag = np.linalg.norm(L, axis=1)

    # Energy drift
    E_drift = np.abs(energies - energies[0]) / np.abs(energies[0])

    results = {
        'mode': mode,
        'times': times,
        'positions': positions,
        'velocities': velocities,
        'energies': energies,
        'radius': r,
        'angular_momentum': L_mag,
        'energy_drift': E_drift,
        'steps': step_count,
        'period_expected': T_orbit
    }

    return results


def analyze_orbit(results: dict) -> dict:
    """
    Analyze orbital characteristics.

    Parameters
    ----------
    results : dict
        Output from run_test_orbit()

    Returns
    -------
    analysis : dict
        Orbital diagnostics
    """
    times = results['times']
    r = results['radius']
    pos = results['positions']

    # Find periapsis passages (local minima in radius)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(-r, distance=50)  # Minima are peaks of -r

    if len(peaks) > 1:
        # Measure orbital period from periapsis crossings
        periods = np.diff(times[peaks])
        T_measured = np.mean(periods)
        T_std = np.std(periods)
    else:
        T_measured = np.nan
        T_std = np.nan

    # Measure periapsis precession (change in angle at periapsis)
    if len(peaks) > 2:
        periapsis_angles = []
        for pk in peaks:
            x, y = pos[pk, 0], pos[pk, 1]
            angle = np.arctan2(y, x)
            periapsis_angles.append(angle)

        # Unwrap angles
        periapsis_angles = np.unwrap(periapsis_angles)

        # Precession per orbit (radians)
        if len(periapsis_angles) > 1:
            precession_rate = np.diff(periapsis_angles).mean()
        else:
            precession_rate = 0.0
    else:
        precession_rate = 0.0

    # Energy conservation
    E_drift_max = np.max(results['energy_drift'])
    E_drift_final = results['energy_drift'][-1]

    # Angular momentum conservation
    L = results['angular_momentum']
    L_drift_max = np.max(np.abs(L - L[0]) / L[0])

    analysis = {
        'T_expected': results['period_expected'],
        'T_measured': T_measured,
        'T_std': T_std,
        'T_error': np.abs(T_measured - results['period_expected']) / results['period_expected'] if not np.isnan(T_measured) else np.nan,
        'precession_per_orbit_rad': precession_rate,
        'precession_per_orbit_deg': np.degrees(precession_rate),
        'energy_drift_max': E_drift_max,
        'energy_drift_final': E_drift_final,
        'angular_momentum_drift_max': L_drift_max,
        'n_periapsis': len(peaks)
    }

    return analysis


def plot_comparison(newtonian_results: dict, gr_results: dict, output_path: Path):
    """
    Create comparison plots.

    Parameters
    ----------
    newtonian_results : dict
        Newtonian trajectory results
    gr_results : dict
        GR trajectory results
    output_path : Path
        Directory to save plots
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Figure 1: Trajectories
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Newtonian trajectory
    ax = axes[0]
    pos_n = newtonian_results['positions']
    ax.plot(pos_n[:, 0], pos_n[:, 1], 'b-', linewidth=1, alpha=0.7, label='Newtonian')
    ax.plot(0, 0, 'ko', markersize=10, label='Black Hole')
    ax.set_xlabel('x (M)')
    ax.set_ylabel('y (M)')
    ax.set_title('Newtonian Trajectory')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # GR trajectory
    ax = axes[1]
    pos_gr = gr_results['positions']
    ax.plot(pos_gr[:, 0], pos_gr[:, 1], 'r-', linewidth=1, alpha=0.7, label='GR (Schwarzschild)')
    ax.plot(0, 0, 'ko', markersize=10, label='Black Hole')
    ax.set_xlabel('x (M)')
    ax.set_ylabel('y (M)')
    ax.set_title('GR Trajectory (Schwarzschild)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'trajectories.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / 'trajectories.png'}")
    plt.close()

    # Figure 2: Orbital radius
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(newtonian_results['times'], newtonian_results['radius'], 'b-',
            linewidth=1, alpha=0.7, label='Newtonian')
    ax.plot(gr_results['times'], gr_results['radius'], 'r-',
            linewidth=1, alpha=0.7, label='GR (Schwarzschild)')
    ax.set_xlabel('Time (code units)')
    ax.set_ylabel('Orbital Radius (M)')
    ax.set_title('Orbital Radius vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'radius.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / 'radius.png'}")
    plt.close()

    # Figure 3: Energy conservation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(newtonian_results['times'], newtonian_results['energy_drift'], 'b-',
                linewidth=1, alpha=0.7, label='Newtonian')
    ax.semilogy(gr_results['times'], gr_results['energy_drift'], 'r-',
                linewidth=1, alpha=0.7, label='GR (Schwarzschild)')
    ax.set_xlabel('Time (code units)')
    ax.set_ylabel('Fractional Energy Drift |ΔE/E₀|')
    ax.set_title('Energy Conservation')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'energy_conservation.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / 'energy_conservation.png'}")
    plt.close()

    # Figure 4: Overlay comparison
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(pos_n[:, 0], pos_n[:, 1], 'b-', linewidth=2, alpha=0.6, label='Newtonian')
    ax.plot(pos_gr[:, 0], pos_gr[:, 1], 'r-', linewidth=2, alpha=0.6, label='GR (Schwarzschild)')
    ax.plot(0, 0, 'ko', markersize=12, label='Black Hole', zorder=10)
    ax.set_xlabel('x (M)')
    ax.set_ylabel('y (M)')
    ax.set_title('Trajectory Comparison: Newtonian vs GR')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'comparison_overlay.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / 'comparison_overlay.png'}")
    plt.close()


def main():
    """
    Main validation script for TASK-020.
    """
    print("\n" + "="*60)
    print("TASK-020: GR vs Newtonian Trajectory Validation")
    print("="*60)

    # Run both simulations
    radius = 10.0  # 10 M_BH (well outside ISCO at 6 M)
    n_orbits = 3

    newtonian_results = run_test_orbit("Newtonian", n_orbits=n_orbits, radius=radius)
    gr_results = run_test_orbit("GR", n_orbits=n_orbits, radius=radius)

    # Analyze orbits
    print("\n" + "="*60)
    print("Orbital Analysis")
    print("="*60)

    newtonian_analysis = analyze_orbit(newtonian_results)
    gr_analysis = analyze_orbit(gr_results)

    print("\nNewtonian Mode:")
    print(f"  Expected period: {newtonian_analysis['T_expected']:.4f}")
    print(f"  Measured period: {newtonian_analysis['T_measured']:.4f} ± {newtonian_analysis['T_std']:.6f}")
    print(f"  Period error: {100*newtonian_analysis['T_error']:.4f}%")
    print(f"  Precession: {newtonian_analysis['precession_per_orbit_deg']:.6f}°/orbit")
    print(f"  Energy drift (max): {newtonian_analysis['energy_drift_max']:.2e}")
    print(f"  Energy drift (final): {newtonian_analysis['energy_drift_final']:.2e}")
    print(f"  L conservation: {newtonian_analysis['angular_momentum_drift_max']:.2e}")

    print("\nGR Mode (Schwarzschild):")
    print(f"  Expected period: {gr_analysis['T_expected']:.4f}")
    print(f"  Measured period: {gr_analysis['T_measured']:.4f} ± {gr_analysis['T_std']:.6f}")
    print(f"  Period error: {100*gr_analysis['T_error']:.4f}%")
    print(f"  Precession: {gr_analysis['precession_per_orbit_deg']:.6f}°/orbit")
    print(f"  Energy drift (max): {gr_analysis['energy_drift_max']:.2e}")
    print(f"  Energy drift (final): {gr_analysis['energy_drift_final']:.2e}")
    print(f"  L conservation: {gr_analysis['angular_momentum_drift_max']:.2e}")

    # Comparison
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    period_difference = gr_analysis['T_measured'] - newtonian_analysis['T_measured']
    print(f"  Period difference: {period_difference:.6f} ({100*period_difference/newtonian_analysis['T_measured']:.4f}%)")
    print(f"  GR precession: {gr_analysis['precession_per_orbit_deg']:.6f}°/orbit")
    print(f"  Newtonian precession: {newtonian_analysis['precession_per_orbit_deg']:.6f}°/orbit")

    # Expected GR precession at r=10M: Δφ ≈ 6πM/r ≈ 1.88 radians/orbit ≈ 107.7°
    # (This is the Schwarzschild perihelion precession formula)
    expected_precession_rad = 6.0 * np.pi * 1.0 / radius
    expected_precession_deg = np.degrees(expected_precession_rad)
    print(f"  Expected GR precession (analytic): {expected_precession_deg:.2f}°/orbit")

    # Create plots
    print("\n" + "="*60)
    print("Creating Plots")
    print("="*60)
    output_path = Path("output/validation")
    plot_comparison(newtonian_results, gr_results, output_path)

    # Write summary
    summary_file = output_path / "VALIDATION_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("TASK-020: GR vs Newtonian Trajectory Validation\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Orbital radius: {radius:.2f} M_BH\n")
        f.write(f"  Number of orbits: {n_orbits}\n")
        f.write(f"  Black hole mass: 1.0 (code units)\n\n")

        f.write(f"Newtonian Results:\n")
        f.write(f"  Measured period: {newtonian_analysis['T_measured']:.4f} ± {newtonian_analysis['T_std']:.6f}\n")
        f.write(f"  Period error: {100*newtonian_analysis['T_error']:.4f}%\n")
        f.write(f"  Precession: {newtonian_analysis['precession_per_orbit_deg']:.6f}°/orbit\n")
        f.write(f"  Energy conservation: ΔE/E = {newtonian_analysis['energy_drift_final']:.2e}\n\n")

        f.write(f"GR Results (Schwarzschild):\n")
        f.write(f"  Measured period: {gr_analysis['T_measured']:.4f} ± {gr_analysis['T_std']:.6f}\n")
        f.write(f"  Period error: {100*gr_analysis['T_error']:.4f}%\n")
        f.write(f"  Precession: {gr_analysis['precession_per_orbit_deg']:.6f}°/orbit\n")
        f.write(f"  Energy conservation: ΔE/E = {gr_analysis['energy_drift_final']:.2e}\n\n")

        f.write(f"Comparison:\n")
        f.write(f"  Period difference: {100*period_difference/newtonian_analysis['T_measured']:.4f}%\n")
        f.write(f"  GR precession (measured): {gr_analysis['precession_per_orbit_deg']:.4f}°/orbit\n")
        f.write(f"  GR precession (expected): {expected_precession_deg:.2f}°/orbit\n")
        f.write(f"  Precession error: {100*abs(gr_analysis['precession_per_orbit_rad']-expected_precession_rad)/expected_precession_rad:.2f}%\n")

    print(f"  Saved: {summary_file}")

    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_path.absolute()}")

    # Check success criteria
    success = True
    if newtonian_analysis['energy_drift_final'] > 0.01:
        print("\n⚠️  WARNING: Newtonian energy drift exceeds 1%")
        success = False
    if gr_analysis['energy_drift_final'] > 0.01:
        print("\n⚠️  WARNING: GR energy drift exceeds 1%")
        success = False
    if abs(gr_analysis['precession_per_orbit_rad'] - expected_precession_rad) / expected_precession_rad > 0.2:
        print("\n⚠️  WARNING: GR precession differs from analytic prediction by >20%")
        success = False

    if success:
        print("\n✅ All validation checks passed!")
        print("   TASK-020 COMPLETE")
        return 0
    else:
        print("\n❌ Some validation checks failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
