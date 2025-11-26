#!/usr/bin/env python3
"""
Example 1: Newtonian Tidal Disruption Event Simulation (TASK-039)

Demonstrates a complete Newtonian TDE simulation using the TDE-SPH framework:
- Polytropic star on parabolic orbit
- Newtonian gravity
- Energy conservation tracking
- Visualization and diagnostic outputs

This example showcases:
- Initial conditions generation (polytrope star)
- Newtonian gravity solver
- Ideal gas EOS
- Energy diagnostics (kinetic, potential, internal)
- Snapshot export to HDF5
- Visualization with Plotly and export to Blender/ParaView

Physics:
- Star: n=1.5 polytrope, M_star = 1 M_sun, R_star = 1 R_sun
- Black hole: M_BH = 10^6 M_sun (scaled to M_BH = 1 in code units)
- Orbit: Parabolic (e ≈ 1), penetration factor β = r_t / r_p ~ 1-2
- Units: G = c = M_BH = 1 (geometrized units)

Expected results:
- Tidal disruption at periapsis (r_p ≈ r_t)
- Debris stream formation
- Energy conservation ΔE/E₀ < 0.01
- Return time t_return ~ (GM_BH / r_t³)^(1/2)

References:
- Guillochon & Ramirez-Ruiz (2013) - TDE simulations
- Tejeda et al. (2017) - Relativistic TDEs
- Price (2012) - SPH methodology

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tde_sph.ICs import Polytrope
from tde_sph.core.energy_diagnostics import EnergyDiagnostics
from tde_sph.sph import ParticleSystem
from tde_sph.io.hdf5 import HDF5Writer

# Try to import visualization (optional)
try:
    from tde_sph.visualization.plotly_3d import plot_particles_3d
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: Plotly not available, skipping 3D visualization")


def setup_tde_initial_conditions(
    n_particles: int = 10000,
    polytropic_index: float = 1.5,
    star_mass: float = 0.001,  # M_star / M_BH (solar mass / 10^6 M_sun)
    star_radius: float = 0.01,  # R_star in code units (~ 1 R_sun for 10^6 M_sun BH)
    periapsis_distance: float = 0.05,  # r_p in code units
    bh_mass: float = 1.0
):
    """
    Set up initial conditions for a Newtonian TDE.

    Creates a polytropic star on a parabolic orbit heading toward periapsis.

    Parameters
    ----------
    n_particles : int
        Number of SPH particles in the star.
    polytropic_index : float
        Polytropic index (n=1.5 for solar-type stars).
    star_mass : float
        Stellar mass (in units of M_BH).
    star_radius : float
        Stellar radius (in code units).
    periapsis_distance : float
        Periapsis distance (closest approach to BH).
    bh_mass : float
        Black hole mass (code units, typically 1.0).

    Returns
    -------
    particles : dict
        Particle data dictionary with keys:
        - 'positions': (N, 3) array
        - 'velocities': (N, 3) array
        - 'masses': (N,) array
        - 'density': (N,) array
        - 'internal_energy': (N,) array
        - 'smoothing_length': (N,) array

    Notes
    -----
    Tidal radius: r_t = R_star * (M_BH / M_star)^(1/3)
    Penetration factor: β = r_t / r_p
    For β ~ 1: partial disruption
    For β > 1: full disruption
    """
    print("=" * 60)
    print("Setting up Newtonian TDE Initial Conditions")
    print("=" * 60)

    # Create polytropic star
    print(f"Generating polytrope star (n={polytropic_index}, N={n_particles})...")
    star_ic = Polytrope(
        gamma=5.0/3.0
    )

    positions, velocities, masses, internal_energy, densities = star_ic.generate(
        n_particles=n_particles,
        M_star=star_mass,
        R_star=star_radius
    )

    # Compute smoothing lengths
    smoothing_length = star_ic.compute_smoothing_lengths(masses, densities)

    print(f"  Star mass: {np.sum(masses):.4e} (target: {star_mass:.4e})")
    print(f"  Star radius: {np.max(np.linalg.norm(positions, axis=1)):.4e} (target: {star_radius:.4e})")

    # Compute tidal radius
    r_tidal = star_radius * (bh_mass / star_mass)**(1.0/3.0)
    penetration_factor = r_tidal / periapsis_distance

    print(f"\nOrbital parameters:")
    print(f"  Periapsis distance: {periapsis_distance:.4f}")
    print(f"  Tidal radius: {r_tidal:.4f}")
    print(f"  Penetration factor β: {penetration_factor:.4f}")

    if penetration_factor < 0.7:
        print("  → Weak encounter (minimal disruption)")
    elif penetration_factor < 1.5:
        print("  → Partial disruption")
    else:
        print("  → Full disruption")

    # Place star on parabolic orbit
    # At initial time, star is far from BH, approaching periapsis
    initial_distance = 10 * periapsis_distance  # Start far from BH

    # For parabolic orbit: v² = 2GM/r at any point
    # At initial distance: v_initial = sqrt(2GM/r_initial)
    v_orbital = np.sqrt(2 * bh_mass / initial_distance)

    # Velocity direction: mostly tangential, slightly radial inward
    # Parabolic orbit has eccentricity e = 1, so at apoapsis v is purely tangential
    # We start near apoapsis for simplicity
    v_radial = -0.1 * v_orbital  # Small inward component
    v_tangential = np.sqrt(v_orbital**2 - v_radial**2)

    print(f"\nInitial orbital state:")
    print(f"  Initial distance: {initial_distance:.4f}")
    print(f"  Orbital velocity: {v_orbital:.4f}")
    print(f"  Radial velocity: {v_radial:.4f}")
    print(f"  Tangential velocity: {v_tangential:.4f}")

    # Offset star position to initial orbital location
    # Star center at (initial_distance, 0, 0)
    star_center = np.array([initial_distance, 0.0, 0.0], dtype=np.float32)
    positions += star_center

    # Add orbital velocity to star
    # Radial: along -x direction (toward BH at origin)
    # Tangential: along +y direction
    v_radial_vec = np.array([v_radial, 0.0, 0.0], dtype=np.float32)
    v_tangential_vec = np.array([0.0, v_tangential, 0.0], dtype=np.float32)
    orbital_velocity = v_radial_vec + v_tangential_vec

    velocities += orbital_velocity

    print(f"\nInitial conditions ready:")
    print(f"  Star center: ({star_center[0]:.4f}, {star_center[1]:.4f}, {star_center[2]:.4f})")
    print(f"  Center-of-mass velocity: ({orbital_velocity[0]:.4f}, {orbital_velocity[1]:.4f}, {orbital_velocity[2]:.4f})")

    # Package as dictionary
    particles = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'density': densities,
        'internal_energy': internal_energy,
        'smoothing_length': smoothing_length
    }

    return particles


def compute_newtonian_forces(particles: dict, bh_mass: float = 1.0):
    """
    Compute Newtonian gravitational acceleration from black hole.

    Parameters
    ----------
    particles : dict
        Particle data dictionary.
    bh_mass : float
        Black hole mass.

    Returns
    -------
    accelerations : np.ndarray, shape (N, 3)
        Gravitational acceleration vectors.

    Notes
    -----
    a = -GM_BH * r / |r|³
    Self-gravity is neglected in this simple demo (could add via tree code).
    """
    positions = particles['positions']
    r = np.linalg.norm(positions, axis=1, keepdims=True)
    r_safe = np.maximum(r, 1e-6)  # Avoid division by zero

    # Acceleration: a = -GM/r² in direction of -r
    accel = -bh_mass * positions / (r_safe**3)

    return accel


def demo_newtonian_tde():
    """
    Run a simplified Newtonian TDE demonstration.

    This is a conceptual demo showing the workflow. For full SPH evolution,
    use the Simulation class with proper time integration.
    """
    print("\n" + "=" * 60)
    print("Newtonian TDE Demonstration")
    print("=" * 60 + "\n")

    # Setup
    output_dir = Path("output_newtonian_tde")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate initial conditions
    particles = setup_tde_initial_conditions(
        n_particles=1000,  # Use smaller N for quick demo
        polytropic_index=1.5,
        star_mass=0.001,  # 1 M_sun / 10^6 M_sun
        star_radius=0.01,
        periapsis_distance=0.05,
        bh_mass=1.0
    )

    # Initialize energy diagnostics
    print("\nInitializing energy diagnostics...")
    energy_diag = EnergyDiagnostics(mode="Newtonian", bh_mass=1.0)

    # Compute initial energy
    E0 = energy_diag.compute_all_energies(
        time=0.0,
        masses=particles['masses'],
        positions=particles['positions'],
        velocities=particles['velocities'],
        internal_energies=particles['internal_energy'],
        softening=particles['smoothing_length']
    )

    energy_diag.log_energy_history(E0)

    print(f"\nInitial energies:")
    print(f"  Kinetic: {E0.kinetic:.6e}")
    print(f"  Potential (BH): {E0.potential_bh:.6e}")
    print(f"  Potential (self): {E0.potential_self:.6e}")
    print(f"  Internal: {E0.internal_thermal:.6e}")
    print(f"  Total: {E0.total:.6e}")

    # Save initial snapshot
    print(f"\nSaving initial snapshot...")
    writer = HDF5Writer()
    writer.write_snapshot(
        str(output_dir / "snapshot_0000.h5"),
        particles,
        time=0.0,
        metadata={
            'bh_mass': 1.0,
            'mode': 'Newtonian',
            'n_particles': len(particles['masses'])
        }
    )

    # Simplified evolution (single step demo - not a full simulation)
    print(f"\nPerforming single time step (demo)...")
    dt = 0.001

    # Compute forces
    accel = compute_newtonian_forces(particles, bh_mass=1.0)

    # Simple Euler step (just for demonstration)
    particles['velocities'] += accel * dt
    particles['positions'] += particles['velocities'] * dt

    # Compute energy after step
    E1 = energy_diag.compute_all_energies(
        time=dt,
        masses=particles['masses'],
        positions=particles['positions'],
        velocities=particles['velocities'],
        internal_energies=particles['internal_energy'],
        softening=particles['smoothing_length']
    )

    energy_diag.log_energy_history(E1)

    print(f"\nEnergy after step:")
    print(f"  Total: {E1.total:.6e}")
    print(f"  Conservation error: {E1.conservation_error:.6e}")

    # Save snapshot
    writer.write_snapshot(
        str(output_dir / "snapshot_0001.h5"),
        particles,
        time=dt,
        metadata={
            'bh_mass': 1.0,
            'mode': 'Newtonian',
            'n_particles': len(particles['masses'])
        }
    )

    print(f"\nSnapshots saved to: {output_dir}")

    # Plot energy evolution
    print("\nPlotting energy evolution...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    history = energy_diag.get_energy_history_arrays()
    time = history['time']

    # Energy components
    ax1.plot(time, history['kinetic'], 'r-', label='Kinetic')
    ax1.plot(time, history['potential_bh'], 'b-', label='Potential (BH)')
    ax1.plot(time, history['potential_self'], 'g-', label='Potential (self)')
    ax1.plot(time, history['internal_thermal'], 'm-', label='Internal')
    ax1.plot(time, history['total'], 'k--', label='Total', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy')
    ax1.set_title('Newtonian TDE: Energy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Conservation error
    ax2.plot(time, history['conservation_error'], 'k-')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ΔE / E₀')
    ax2.set_title('Energy Conservation Error')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "energy_evolution.png", dpi=150)
    print(f"  Saved: {output_dir}/energy_evolution.png")

    # 3D visualization (if Plotly available)
    if HAS_PLOTLY:
        print("\nGenerating 3D visualization...")
        try:
            fig_3d = plot_particles_3d(
                particles['positions'],
                color_by=particles['density'],
                title="Newtonian TDE: Initial Conditions",
                marker_size=2
            )
            fig_3d.write_html(str(output_dir / "particles_3d.html"))
            print(f"  Saved: {output_dir}/particles_3d.html")
        except Exception as e:
            print(f"  Error creating 3D plot: {e}")

    # Export to Blender/ParaView formats
    print("\nExporting to Blender/ParaView formats...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tools.export_to_blender import SnapshotExporter

        exporter = SnapshotExporter(verbose=False)

        # Export PLY
        exporter.export_ply(
            str(output_dir / "snapshot_0000.h5"),
            str(output_dir / "snapshot_0000.ply"),
            color_by="density"
        )
        print(f"  Saved: {output_dir}/snapshot_0000.ply (Blender)")

        # Export VTK
        exporter.export_vtk(
            str(output_dir / "snapshot_0000.h5"),
            str(output_dir / "snapshot_0000.vtk")
        )
        print(f"  Saved: {output_dir}/snapshot_0000.vtk (ParaView)")

    except Exception as e:
        print(f"  Error exporting: {e}")

    print("\n" + "=" * 60)
    print("Newtonian TDE Demo Complete!")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}/")
    print("- snapshot_*.h5: HDF5 snapshots")
    print("- snapshot_*.ply: Blender point clouds")
    print("- snapshot_*.vtk: ParaView data")
    print("- energy_evolution.png: Energy diagnostic plot")
    if HAS_PLOTLY:
        print("- particles_3d.html: Interactive 3D visualization")

    print("\nNote: This is a simplified 1-step demo. For full TDE simulations,")
    print("use the Simulation class with proper time integration (leapfrog, RK4, etc.)")


if __name__ == "__main__":
    demo_newtonian_tde()
