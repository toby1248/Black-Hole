#!/usr/bin/env python3
"""
Command-line entrypoint for TDE-SPH simulations.

This script demonstrates a complete Newtonian TDE simulation workflow:
1. Generate polytropic stellar model
2. Place on parabolic orbit around black hole
3. Evolve with SPH + gravity
4. Save snapshots
5. Visualize results

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --particles 10000 --tend 5.0
    python scripts/run_simulation.py --help
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path if running from repository root
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from tde_sph.core import Simulation, SimulationConfig
from tde_sph.sph import ParticleSystem
from tde_sph.gravity import NewtonianGravity
from tde_sph.eos import IdealGas
from tde_sph.integration import LeapfrogIntegrator
from tde_sph.ICs import Polytrope
from tde_sph.visualization import quick_plot
from tde_sph.io import read_snapshot


def setup_tde_orbit(
    stellar_mass: float = 1.0,
    stellar_radius: float = 1.0,
    bh_mass: float = 1e6,
    periapsis: float = 1.0,  # In units of tidal radius
) -> tuple:
    """
    Setup initial conditions for a star on parabolic orbit around BH.

    Parameters
    ----------
    stellar_mass : float
        Stellar mass in solar masses.
    stellar_radius : float
        Stellar radius in solar radii.
    bh_mass : float
        Black hole mass in solar masses.
    periapsis : float
        Periapsis distance in units of tidal radius R_t = R_star (M_BH/M_star)^(1/3).

    Returns
    -------
    position : ndarray, shape (3,)
        Initial center-of-mass position.
    velocity : ndarray, shape (3,)
        Initial center-of-mass velocity.
    """
    # Tidal radius (dimensionless, G=1)
    R_t = stellar_radius * (bh_mass / stellar_mass) ** (1.0 / 3.0)

    # Periapsis distance
    r_p = periapsis * R_t

    # For parabolic orbit: v = sqrt(2 * G * M / r)
    # At periapsis, velocity is tangential
    G = 1.0  # Dimensionless units
    v_p = np.sqrt(2.0 * G * bh_mass / r_p)

    # Initial position: star approaching periapsis from -x direction
    # Place star at apoapsis (far away) for parabolic orbit
    # For simplicity, start near periapsis but slightly before
    r_init = 3.0 * r_p  # Start 3× periapsis distance

    position = np.array([-r_init, 0.0, 0.0], dtype=np.float32)

    # Velocity for parabolic orbit at r_init
    v_init = np.sqrt(2.0 * G * bh_mass / r_init)
    velocity = np.array([0.0, v_init, 0.0], dtype=np.float32)

    print(f"Orbit setup:")
    print(f"  Tidal radius R_t = {R_t:.3e}")
    print(f"  Periapsis r_p = {r_p:.3e} ({periapsis:.1f} R_t)")
    print(f"  Initial distance r_0 = {r_init:.3e}")
    print(f"  Initial velocity v_0 = {v_init:.3e}")
    print(f"  Specific energy E = v²/2 - GM/r = {0.5 * v_init**2 - G * bh_mass / r_init:.3e} (should be ~0 for parabolic)")

    return position, velocity


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run TDE-SPH simulation of stellar tidal disruption",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Simulation parameters
    parser.add_argument("--particles", "-n", type=int, default=5000,
                        help="Number of SPH particles")
    parser.add_argument("--tend", "-t", type=float, default=2.0,
                        help="End time (dimensionless)")
    parser.add_argument("--dt-init", type=float, default=0.001,
                        help="Initial timestep")
    parser.add_argument("--snapshot-interval", type=float, default=0.1,
                        help="Snapshot save interval")
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                        help="Output directory for snapshots")

    # Stellar parameters
    parser.add_argument("--stellar-mass", type=float, default=1.0,
                        help="Stellar mass (solar masses)")
    parser.add_argument("--stellar-radius", type=float, default=1.0,
                        help="Stellar radius (solar radii)")
    parser.add_argument("--gamma", type=float, default=5.0/3.0,
                        help="Adiabatic index (5/3 for ideal gas, 4/3 for radiation)")

    # Black hole parameters
    parser.add_argument("--bh-mass", type=float, default=1e6,
                        help="Black hole mass (solar masses)")
    parser.add_argument("--periapsis", type=float, default=1.0,
                        help="Periapsis in units of tidal radius")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--visualize", action="store_true",
                        help="Show visualization after simulation")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    # Print banner
    if not args.quiet:
        print("=" * 70)
        print("TDE-SPH: Relativistic SPH framework for Tidal Disruption Events")
        print("=" * 70)
        print()

    # 1. Generate stellar initial conditions
    print(f"Generating {args.particles}-particle polytropic star (γ={args.gamma:.3f})...")
    polytrope = Polytrope(gamma=args.gamma, random_seed=args.seed)

    pos, vel, mass, u, rho = polytrope.generate(
        n_particles=args.particles,
        M_star=args.stellar_mass,
        R_star=args.stellar_radius,
        position=np.zeros(3, dtype=np.float32),  # Centered at origin initially
        velocity=np.zeros(3, dtype=np.float32),
    )

    # 2. Setup orbital motion
    print(f"\nSetting up parabolic orbit around {args.bh_mass:.1e} M_sun black hole...")
    cm_position, cm_velocity = setup_tde_orbit(
        stellar_mass=args.stellar_mass,
        stellar_radius=args.stellar_radius,
        bh_mass=args.bh_mass,
        periapsis=args.periapsis,
    )

    # Shift star to orbit
    pos += cm_position
    vel += cm_velocity

    # 3. Create particle system
    print(f"\nInitializing {args.particles} SPH particles...")
    particles = ParticleSystem(
        positions=pos,
        velocities=vel,
        masses=mass,
        internal_energy=u,
    )

    # Compute smoothing lengths
    h = polytrope.compute_smoothing_lengths(mass, rho, eta=1.2)
    particles.smoothing_lengths = h

    print(f"  Total mass: {np.sum(mass):.3e}")
    print(f"  Radius: {np.max(np.linalg.norm(pos - cm_position, axis=1)):.3e}")
    print(f"  Mean smoothing length: {np.mean(h):.3e}")

    # 4. Setup physics modules
    print("\nInitializing physics modules...")
    gravity = NewtonianGravity()
    eos = IdealGas(gamma=args.gamma)
    integrator = LeapfrogIntegrator(cfl_factor=0.3)

    # 5. Configure simulation
    config = SimulationConfig(
        t_start=0.0,
        t_end=args.tend,
        dt_initial=args.dt_init,
        snapshot_interval=args.snapshot_interval,
        output_dir=args.output_dir,
        mode="Newtonian",
        bh_mass=args.bh_mass,
        random_seed=args.seed,
        verbose=not args.quiet,
    )

    # 6. Create simulation
    sim = Simulation(
        particles=particles,
        gravity_solver=gravity,
        eos=eos,
        integrator=integrator,
        config=config,
    )

    # 7. Run simulation
    print("\nStarting simulation...\n")
    sim.run()

    # 8. Visualize (optional)
    if args.visualize:
        print("\nGenerating visualization...")
        # Load final snapshot
        final_snap = Path(args.output_dir) / f"snapshot_{sim.state.snapshot_count-1:04d}.h5"
        if final_snap.exists():
            data = read_snapshot(str(final_snap))
            positions = data['particles']['positions']
            density = data['particles']['density']

            fig = quick_plot(
                positions,
                color_by=density,
                title=f"TDE at t={sim.state.time:.2f} ({args.particles} particles)",
            )
            fig.show()
            print("Visualization complete.")
        else:
            print(f"Warning: Could not find final snapshot at {final_snap}")

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Snapshots: {sim.state.snapshot_count}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
