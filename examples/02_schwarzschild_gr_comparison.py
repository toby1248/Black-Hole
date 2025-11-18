#!/usr/bin/env python3
"""
Example 2: Schwarzschild GR vs Newtonian Comparison (TASK-039)

Compares a test particle orbit in Schwarzschild spacetime vs Newtonian gravity.

Demonstrates:
- Schwarzschild metric implementation
- GR orbit precession (periapsis advance)
- Comparison with Newtonian predictions
- Relativistic corrections near ISCO

Physics:
- Test particle on eccentric orbit (e = 0.3-0.5)
- Schwarzschild metric (non-rotating BH, spin a = 0)
- ISCO radius: r_ISCO = 6M (Schwarzschild)
- Periapsis precession: Δϕ ≈ 6πGM / (c²a(1-e²)) per orbit (GR)

Expected results:
- GR orbit precesses, Newtonian does not
- Precession rate increases closer to ISCO
- Orbital period differs (GR vs Newtonian)
- Energy conservation in both modes

References:
- Tejeda et al. (2017) - Hybrid relativistic approach
- Liptai & Price (2019) - GRSPH test particle orbits
- Misner, Thorne, Wheeler (1973) - Gravitation textbook

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tde_sph.metric import SchwarzschildMetric, MinkowskiMetric
from tde_sph.core.energy_diagnostics import EnergyDiagnostics
import tde_sph.sph as sph


def setup_eccentric_orbit(
    semi_major_axis: float = 10.0,
    eccentricity: float = 0.3,
    bh_mass: float = 1.0
):
    """
    Set up a test particle on an eccentric orbit.

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis in units of M_BH.
    eccentricity : float
        Orbital eccentricity (0 = circular, 1 = parabolic).
    bh_mass : float
        Black hole mass (code units).

    Returns
    -------
    position : np.ndarray, shape (3,)
        Initial position at apoapsis.
    velocity : np.ndarray, shape (3,)
        Initial velocity (purely tangential at apoapsis).
    r_periapsis : float
        Periapsis distance.
    r_apoapsis : float
        Apoapsis distance.

    Notes
    -----
    Orbital parameters:
    - r_periapsis = a(1 - e)
    - r_apoapsis = a(1 + e)
    - v_apoapsis = sqrt(GM(1-e) / (a(1+e)))  (Newtonian)
    """
    a = semi_major_axis
    e = eccentricity

    # Orbital radii
    r_peri = a * (1 - e)
    r_apo = a * (1 + e)

    # Check ISCO constraint for Schwarzschild
    r_isco = 6.0 * bh_mass
    if r_peri < r_isco:
        print(f"Warning: Periapsis {r_peri:.3f} < ISCO {r_isco:.3f}")
        print("  Orbit is unstable! Particle will plunge into BH.")

    # Initial position at apoapsis (along +x axis)
    position = np.array([r_apo, 0.0, 0.0], dtype=np.float32)

    # Velocity at apoapsis (Newtonian approximation for initial conditions)
    # v = sqrt(GM(1-e) / a(1+e))
    v_apo = np.sqrt(bh_mass * (1 - e) / (a * (1 + e)))

    # Velocity along +y direction (purely tangential at apoapsis)
    velocity = np.array([0.0, v_apo, 0.0], dtype=np.float32)

    print(f"Eccentric orbit setup:")
    print(f"  Semi-major axis: {a:.3f}")
    print(f"  Eccentricity: {e:.3f}")
    print(f"  Periapsis: {r_peri:.3f} (ISCO = {r_isco:.3f})")
    print(f"  Apoapsis: {r_apo:.3f}")
    print(f"  Initial velocity: {v_apo:.4f}")

    return position, velocity, r_peri, r_apo


def compute_schwarzschild_acceleration(position, velocity, metric):
    """
    Compute acceleration in Schwarzschild spacetime.

    Uses geodesic equation with Schwarzschild Christoffel symbols.

    Parameters
    ----------
    position : np.ndarray, shape (3,)
        Position vector (x, y, z).
    velocity : np.ndarray, shape (3,)
        Velocity vector (vx, vy, vz).
    metric : SchwarzschildMetric
        Schwarzschild metric object.

    Returns
    -------
    acceleration : np.ndarray, shape (3,)
        Coordinate acceleration (dvx/dt, dvy/dt, dvz/dt).

    Notes
    -----
    Geodesic equation: d²x^μ/dτ² + Γ^μ_νλ dx^ν/dτ dx^λ/dτ = 0
    For coordinate time t: dv^i/dt = -Γ^i_jk v^j v^k (simplified)

    This is a simplified implementation. Full GR requires careful treatment
    of coordinate vs proper time.
    """
    # Get metric components and Christoffel symbols
    r = np.linalg.norm(position)
    M = metric.mass

    # Schwarzschild metric component: g_tt = -(1 - 2M/r)
    # Christoffel symbols for radial geodesic (approximate)
    # Γ^r_tt = M/r² (1 - 2M/r)
    # Γ^r_rr = -M / (r² (1 - 2M/r))
    # Γ^r_θθ = -(r - 2M)
    # Γ^r_φφ = -(r - 2M) sin²θ

    # Simplified: use effective potential approach
    # a_eff = -GM/r² (1 - 3M/r) r_hat + angular momentum terms

    r_safe = np.maximum(r, 2.1 * M)  # Avoid horizon
    r_hat = position / r_safe

    # Radial acceleration (approximate, includes first-order GR correction)
    a_radial = -(M / r_safe**2) * (1 - 3 * M / r_safe)

    # Add angular momentum conservation (centrifugal term)
    v_perp = velocity - np.dot(velocity, r_hat) * r_hat
    L = np.linalg.norm(np.cross(position, velocity))  # Angular momentum magnitude
    if L > 1e-10:
        a_angular = (L**2) / (r_safe**3)  # Centrifugal acceleration (outward)
        a_total_mag = a_radial + a_angular
    else:
        a_total_mag = a_radial

    acceleration = a_total_mag * r_hat

    return acceleration


def integrate_orbit(
    position_init,
    velocity_init,
    bh_mass=1.0,
    mode="Newtonian",
    n_orbits=3,
    n_steps_per_orbit=1000
):
    """
    Integrate test particle orbit using simple leapfrog.

    Parameters
    ----------
    position_init : np.ndarray, shape (3,)
        Initial position.
    velocity_init : np.ndarray, shape (3,)
        Initial velocity.
    bh_mass : float
        Black hole mass.
    mode : str
        "Newtonian" or "GR" (Schwarzschild).
    n_orbits : int
        Number of orbits to simulate.
    n_steps_per_orbit : int
        Time steps per orbital period.

    Returns
    -------
    trajectory : dict
        Contains 'time', 'positions', 'velocities', 'distances'.
    """
    print(f"\nIntegrating {mode} orbit...")

    # Estimate orbital period (Newtonian approximation)
    r0 = np.linalg.norm(position_init)
    v0 = np.linalg.norm(velocity_init)
    # For ellipse: T = 2π sqrt(a³/GM), approximate a ~ r0
    T_orb = 2 * np.pi * np.sqrt(r0**3 / bh_mass)
    print(f"  Estimated orbital period: {T_orb:.3f}")

    dt = T_orb / n_steps_per_orbit
    n_steps = n_orbits * n_steps_per_orbit

    print(f"  Time step: {dt:.6f}")
    print(f"  Total steps: {n_steps}")

    # Initialize arrays
    positions = np.zeros((n_steps, 3), dtype=np.float32)
    velocities = np.zeros((n_steps, 3), dtype=np.float32)
    times = np.zeros(n_steps, dtype=np.float32)

    positions[0] = position_init
    velocities[0] = velocity_init

    # Setup metric
    if mode == "GR":
        metric = SchwarzschildMetric(mass=bh_mass)
    else:
        metric = None

    # Leapfrog integration
    for i in range(n_steps - 1):
        r = positions[i]
        v = velocities[i]

        # Compute acceleration
        if mode == "Newtonian":
            # a = -GM r / |r|³
            r_norm = np.linalg.norm(r)
            r_safe = np.maximum(r_norm, 1e-6)
            accel = -bh_mass * r / (r_safe**3)
        else:  # GR (Schwarzschild)
            accel = compute_schwarzschild_acceleration(r, v, metric)

        # Leapfrog step
        v_half = v + 0.5 * accel * dt
        r_new = r + v_half * dt

        # Recompute acceleration at new position
        if mode == "Newtonian":
            r_norm_new = np.linalg.norm(r_new)
            r_safe_new = np.maximum(r_norm_new, 1e-6)
            accel_new = -bh_mass * r_new / (r_safe_new**3)
        else:
            accel_new = compute_schwarzschild_acceleration(r_new, v_half, metric)

        v_new = v_half + 0.5 * accel_new * dt

        positions[i + 1] = r_new
        velocities[i + 1] = v_new
        times[i + 1] = times[i] + dt

    distances = np.linalg.norm(positions, axis=1)

    return {
        'time': times,
        'positions': positions,
        'velocities': velocities,
        'distances': distances,
        'dt': dt
    }


def measure_precession(trajectory):
    """
    Measure periapsis precession angle.

    Parameters
    ----------
    trajectory : dict
        Trajectory data from integrate_orbit.

    Returns
    -------
    precession_angle : float
        Total precession angle (radians).
    n_periapses : int
        Number of periapsis passages detected.
    """
    distances = trajectory['distances']
    positions = trajectory['positions']

    # Find periapsis passages (local minima in distance)
    periapsis_indices = []
    for i in range(1, len(distances) - 1):
        if distances[i] < distances[i - 1] and distances[i] < distances[i + 1]:
            periapsis_indices.append(i)

    if len(periapsis_indices) < 2:
        return 0.0, 0

    # Compute angle of each periapsis position
    angles = []
    for idx in periapsis_indices:
        pos = positions[idx]
        angle = np.arctan2(pos[1], pos[0])
        angles.append(angle)

    # Total precession: difference between last and first periapsis angle
    # Unwrap angles to handle 2π jumps
    angles = np.unwrap(angles)
    precession_total = angles[-1] - angles[0]

    return precession_total, len(periapsis_indices)


def demo_schwarzschild_comparison():
    """
    Compare Schwarzschild GR orbit with Newtonian orbit.
    """
    print("\n" + "=" * 60)
    print("Schwarzschild GR vs Newtonian Orbit Comparison")
    print("=" * 60 + "\n")

    # Setup output
    output_dir = Path("output_schwarzschild_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Orbital parameters
    semi_major_axis = 15.0  # Above ISCO (6M)
    eccentricity = 0.4
    bh_mass = 1.0
    n_orbits = 5

    # Setup initial conditions
    position_init, velocity_init, r_peri, r_apo = setup_eccentric_orbit(
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        bh_mass=bh_mass
    )

    # Integrate Newtonian orbit
    traj_newton = integrate_orbit(
        position_init,
        velocity_init,
        bh_mass=bh_mass,
        mode="Newtonian",
        n_orbits=n_orbits,
        n_steps_per_orbit=500
    )

    # Integrate GR orbit
    traj_gr = integrate_orbit(
        position_init,
        velocity_init,
        bh_mass=bh_mass,
        mode="GR",
        n_orbits=n_orbits,
        n_steps_per_orbit=500
    )

    # Measure precession
    precession_gr, n_peri_gr = measure_precession(traj_gr)
    precession_newton, n_peri_newton = measure_precession(traj_newton)

    print(f"\nPeriapsis precession:")
    print(f"  Newtonian: {np.degrees(precession_newton):.4f}° ({n_peri_newton} periapses)")
    print(f"  GR (Schwarzschild): {np.degrees(precession_gr):.4f}° ({n_peri_gr} periapses)")
    print(f"  Difference: {np.degrees(precession_gr - precession_newton):.4f}°")

    # Theoretical GR precession (approximate)
    # Δϕ ≈ 6πGM / (c²a(1-e²)) per orbit (in natural units, G=c=1)
    # Δϕ ≈ 6πM / (a(1-e²))
    precession_theory = (6 * np.pi * bh_mass) / (semi_major_axis * (1 - eccentricity**2))
    precession_theory_total = precession_theory * (n_peri_gr - 1)  # Total over all orbits

    print(f"  Theoretical (per orbit): {np.degrees(precession_theory):.4f}°")
    print(f"  Theoretical (total): {np.degrees(precession_theory_total):.4f}°")

    # Plot trajectories
    print("\nPlotting trajectories...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: XY trajectories
    ax1 = axes[0, 0]
    ax1.plot(traj_newton['positions'][:, 0], traj_newton['positions'][:, 1],
             'b-', label='Newtonian', alpha=0.7, linewidth=1.5)
    ax1.plot(traj_gr['positions'][:, 0], traj_gr['positions'][:, 1],
             'r-', label='GR (Schwarzschild)', alpha=0.7, linewidth=1.5)
    ax1.plot(0, 0, 'ko', markersize=10, label='Black Hole')

    # Mark ISCO
    isco_circle = plt.Circle((0, 0), 6 * bh_mass, color='gray', fill=False,
                             linestyle='--', label='ISCO (6M)')
    ax1.add_patch(isco_circle)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Orbital Trajectories (XY plane)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Distance vs time
    ax2 = axes[0, 1]
    ax2.plot(traj_newton['time'], traj_newton['distances'],
             'b-', label='Newtonian', linewidth=1.5)
    ax2.plot(traj_gr['time'], traj_gr['distances'],
             'r-', label='GR (Schwarzschild)', linewidth=1.5)
    ax2.axhline(y=6 * bh_mass, color='gray', linestyle='--', label='ISCO')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Distance from BH')
    ax2.set_title('Radial Distance Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Orbital velocity
    ax3 = axes[1, 0]
    v_newton = np.linalg.norm(traj_newton['velocities'], axis=1)
    v_gr = np.linalg.norm(traj_gr['velocities'], axis=1)
    ax3.plot(traj_newton['time'], v_newton, 'b-', label='Newtonian', linewidth=1.5)
    ax3.plot(traj_gr['time'], v_gr, 'r-', label='GR (Schwarzschild)', linewidth=1.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Velocity magnitude')
    ax3.set_title('Orbital Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Trajectory difference (overlay zoom)
    ax4 = axes[1, 1]
    # Plot only first 2 orbits for clarity
    n_half = len(traj_newton['positions']) // (n_orbits // 2) if n_orbits > 1 else len(traj_newton['positions'])
    ax4.plot(traj_newton['positions'][:n_half, 0], traj_newton['positions'][:n_half, 1],
             'b-', label='Newtonian', alpha=0.7, linewidth=2)
    ax4.plot(traj_gr['positions'][:n_half, 0], traj_gr['positions'][:n_half, 1],
             'r-', label='GR (Schwarzschild)', alpha=0.7, linewidth=2)
    ax4.plot(0, 0, 'ko', markersize=8, label='Black Hole')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title(f'Precession Detail (first {n_orbits//2} orbits)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')

    plt.tight_layout()
    plt.savefig(output_dir / "schwarzschild_vs_newtonian.png", dpi=150)
    print(f"  Saved: {output_dir}/schwarzschild_vs_newtonian.png")

    print("\n" + "=" * 60)
    print("Schwarzschild Comparison Complete!")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}/")
    print("- schwarzschild_vs_newtonian.png: Trajectory comparison plots")

    print("\nKey Results:")
    print(f"- GR orbit shows periapsis precession: {np.degrees(precession_gr):.2f}°")
    print(f"- Newtonian orbit has minimal precession: {np.degrees(precession_newton):.2f}°")
    print(f"- Theoretical GR precession: {np.degrees(precession_theory_total):.2f}° (total)")
    print(f"- Agreement within ~{100 * abs(1 - precession_gr / precession_theory_total):.1f}%")

    print("\nNote: This uses a simplified GR integrator. For accurate")
    print("results near ISCO, use the full Hamiltonian formulation.")


if __name__ == "__main__":
    demo_schwarzschild_comparison()
