#!/usr/bin/env python3
"""
Example: Polytrope IC generation and Leapfrog integration

Demonstrates creating a polytropic star and evolving it with simple forces.
This is a minimal working example showing how to use the ICs and integration modules.
"""

import sys
sys.path.insert(0, '/home/user/Black-Hole/src')

import numpy as np

# Plotting optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping plots")

from tde_sph.ICs.polytrope import Polytrope
from tde_sph.integration.leapfrog import LeapfrogIntegrator


class SimpleParticleSystem:
    """Minimal particle system for demonstration."""
    def __init__(self, positions, velocities, masses, internal_energy, densities):
        self.positions = positions
        self.velocities = velocities
        self.masses = masses
        self.internal_energy = internal_energy
        self.densities = densities
        # Compute smoothing lengths (approximate)
        self.smoothing_lengths = 1.2 * (masses / densities)**(1.0/3.0)
        # Compute sound speeds (ideal gas, γ=5/3)
        gamma = 5.0/3.0
        self.sound_speeds = np.sqrt(gamma * (gamma - 1.0) * internal_energy)


def main():
    print("="*70)
    print(" TDE-SPH Integration & IC Example")
    print("="*70)

    # Generate polytropic star
    print("\n1. Generating γ=5/3 polytrope with 5000 particles...")
    poly = Polytrope(gamma=5.0/3.0, eta=1.2, random_seed=42)

    positions, velocities, masses, u, rho = poly.generate(
        n_particles=5000,
        M_star=1.0,
        R_star=1.0,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    )

    print(f"   Generated {len(masses)} particles")
    print(f"   Total mass: {np.sum(masses):.6f}")
    print(f"   Stellar radius: {np.max(np.linalg.norm(positions, axis=1)):.4f}")

    # Create particle system
    particles = SimpleParticleSystem(positions, velocities, masses, u, rho)

    # Set up simple external force (uniform gravity in -z direction)
    print("\n2. Setting up uniform gravitational field...")
    g_external = np.float32(0.1)  # Acceleration in -z

    # Create integrator
    print("\n3. Initializing leapfrog integrator...")
    integrator = LeapfrogIntegrator(cfl_factor=0.3)

    # Estimate initial timestep
    dt = integrator.estimate_timestep(particles)
    print(f"   Initial timestep: {dt:.6f}")

    # Evolve for a few steps
    n_steps = 10
    print(f"\n4. Evolving for {n_steps} steps...")

    center_of_mass_z = []
    mean_velocity_z = []
    times = []
    t = 0.0

    for step in range(n_steps):
        # Compute forces
        accel_gravity = np.zeros_like(particles.positions)
        accel_gravity[:, 2] = -g_external  # Downward gravity

        forces = {
            'gravity': accel_gravity,
            'hydro': np.zeros_like(accel_gravity),  # No hydrodynamics yet
            'du_dt': np.zeros(len(masses), dtype=np.float32)
        }

        # Take integration step
        integrator.step(particles, dt, forces)

        # Record diagnostics
        t += dt
        times.append(t)
        com_z = np.sum(particles.masses * particles.positions[:, 2]) / np.sum(particles.masses)
        v_z = np.sum(particles.masses * particles.velocities[:, 2]) / np.sum(particles.masses)
        center_of_mass_z.append(com_z)
        mean_velocity_z.append(v_z)

        if step % 2 == 0:
            print(f"   Step {step:3d}: t={t:.4f}, CoM_z={com_z:.6f}, <v_z>={v_z:.6f}")

        # Update timestep
        dt = integrator.estimate_timestep(
            particles,
            accelerations=accel_gravity
        )

    print("\n5. Integration complete!")

    # Plot results
    if not HAS_MATPLOTLIB:
        print("\n6. Skipping plots (matplotlib not installed)")
        print("\n" + "="*70)
        print(" Example complete!")
        print("="*70)
        return

    print("\n6. Generating diagnostic plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Initial particle distribution (x-y slice)
    ax = axes[0, 0]
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=rho,
        s=1,
        cmap='viridis',
        alpha=0.6
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Initial Particle Distribution (x-y)')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Density')

    # Radial density profile
    ax = axes[0, 1]
    r_initial = np.linalg.norm(positions, axis=1)
    ax.scatter(r_initial, rho, s=1, alpha=0.3)
    ax.set_xlabel('Radius')
    ax.set_ylabel('Density')
    ax.set_title('Initial Radial Density Profile')
    ax.set_yscale('log')

    # Center of mass evolution
    ax = axes[1, 0]
    ax.plot(times, center_of_mass_z, 'b-', linewidth=2)
    # Analytic solution: z(t) = -0.5 * g * t²
    t_analytic = np.array(times)
    z_analytic = -0.5 * g_external * t_analytic**2
    ax.plot(times, z_analytic, 'r--', linewidth=2, label='Analytic')
    ax.set_xlabel('Time')
    ax.set_ylabel('Center of Mass Z')
    ax.set_title('CoM Evolution Under Uniform Gravity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean velocity evolution
    ax = axes[1, 1]
    ax.plot(times, mean_velocity_z, 'b-', linewidth=2)
    # Analytic: v(t) = -g * t
    v_analytic = -g_external * t_analytic
    ax.plot(times, v_analytic, 'r--', linewidth=2, label='Analytic')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Velocity Z')
    ax.set_title('Mean Velocity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Black-Hole/examples/integration_test.png', dpi=150)
    print("   Saved plot to examples/integration_test.png")

    # 3D visualization
    print("\n7. Generating 3D visualization...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample for clarity
    subsample = np.random.choice(len(positions), size=min(2000, len(positions)), replace=False)

    scatter = ax.scatter(
        particles.positions[subsample, 0],
        particles.positions[subsample, 1],
        particles.positions[subsample, 2],
        c=particles.densities[subsample],
        s=2,
        cmap='plasma',
        alpha=0.6
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Particle Distribution (t={t:.4f})')
    plt.colorbar(scatter, ax=ax, label='Density', shrink=0.6)

    plt.savefig('/home/user/Black-Hole/examples/particle_distribution_3d.png', dpi=150)
    print("   Saved 3D plot to examples/particle_distribution_3d.png")

    print("\n" + "="*70)
    print(" Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()
